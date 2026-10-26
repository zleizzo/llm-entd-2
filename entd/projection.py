"""Linear codes for the Hellinger embedding s = sqrt(p), and for the hidden state h.

- exact:  P = I_V (store the full sqrt-distribution; tiny vocabularies only)
- pca:    uncentered PCA of s (top-d eigenvectors of E[s s^T]); nested, so truncation = smaller PCA
- random: Gaussian projection, columns scaled so truncation to d' is itself a proper projection
Distances: ||s_q - s_i||^2 = ||z_q||^2 + ||z_i||^2 - 2 <z_q, z_i>, with per-position ||z||^2 stored.
int8 storage: symmetric per-dim scale from a clip quantile fit on a sample.
"""
import torch


class MomentAccumulator:
    """Accumulates sum of outer products (and sum, count) for uncentered/centered PCA."""

    def __init__(self, dim, device):
        self.M = torch.zeros(dim, dim, device=device, dtype=torch.float32)
        self.s = torch.zeros(dim, device=device, dtype=torch.float32)
        self.n = 0

    def add(self, X):   # X: (n, dim) float32
        self.M.addmm_(X.T, X)
        self.s += X.sum(0)
        self.n += X.shape[0]

    def pca(self, d, centered=False, exact_max_dim=4096):
        """Top-d eigenpairs (descending) of the (un)centered second-moment matrix.
        Small matrices: exact CPU double eigh. Large ones (V ~ 50k: a 20 GB double eigh takes an hour or more on CPU):
        randomized subspace iteration on the accumulator's device, which needs only V x (d + oversample) extra memory."""
        M = self.M / self.n
        mean = self.s / self.n
        if centered:
            M = M - torch.outer(mean, mean)
        if M.shape[0] <= exact_max_dim or d > M.shape[0] // 4:
            evals, evecs = torch.linalg.eigh(M.cpu().double())   # ascending
            idx = torch.argsort(evals, descending=True)
            evals, evecs = evals[idx][:d], evecs[:, idx][:, :d]
        else:
            evals, evecs = top_eigh(M, d)
        return evecs.float(), evals.float(), (mean.cpu() if centered else None)


def top_eigh(M, k, iters=40, oversample=64, seed=0):
    """Top-k eigenpairs (descending) of a symmetric PSD matrix M by randomized subspace iteration + Rayleigh-Ritz.
    Runs on M's device in M's dtype (fp32 matmul/QR); the small (k+oversample)^2 eigenproblem is solved in double
    on CPU. Error of the j-th eigenvector decays like (lambda_{k+oversample} / lambda_j)^iters. Returns CPU tensors."""
    n = M.shape[0]
    p = min(n, k + oversample)
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(n, p, generator=g).to(M.device, M.dtype)
    Q, _ = torch.linalg.qr(Q)
    for _ in range(iters):
        Q, _ = torch.linalg.qr(M @ Q)
    MQ = M @ Q
    B = (Q.T @ MQ).cpu().double()
    B = 0.5 * (B + B.T)
    ev, U = torch.linalg.eigh(B)                      # ascending
    ev, U = ev.flip(0)[:k], U.flip(1)[:, :k]
    return ev, Q.cpu().double() @ U


def random_projection(V, d, seed):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(V, d, generator=g)
    return G / (d ** 0.5)


class Code:
    """Linear code z = (x - mean) @ P, optional int8 quantizer."""

    def __init__(self, P, kind, mean=None, scale=None, evals=None):
        self.P, self.kind, self.mean, self.scale, self.evals = P, kind, mean, scale, evals

    @property
    def d(self):
        return self.P.shape[1]

    def to(self, device):
        return Code(self.P.to(device), self.kind, None if self.mean is None else self.mean.to(device),
                    None if self.scale is None else self.scale.to(device), self.evals)

    def encode(self, X, d=None):
        if self.mean is not None:
            X = X - self.mean
        P = self.P if d is None else self.P[:, :d]
        z = X @ P
        if d is not None and self.kind == "random":
            z = z * ((self.P.shape[1] / d) ** 0.5)
        return z

    def fit_int8(self, Z, q):
        """Z: (n, d) sample of codes -> per-dim symmetric scale from quantile q of |z|."""
        clip = torch.quantile(Z.abs().float().cpu(), q, dim=0).clamp_min(1e-8)
        self.scale = clip / 127.0
        return self.scale

    def quantize(self, Z):
        return torch.round(Z / self.scale.to(Z.device)[None, :]).clamp(-127, 127).to(torch.int8)

    def dequantize(self, Q):
        return Q.float() * self.scale.to(Q.device)[None, :]

    def state(self):
        return {"P": self.P.cpu(), "kind": self.kind, "mean": None if self.mean is None else self.mean.cpu(),
                "scale": None if self.scale is None else self.scale.cpu(), "evals": self.evals}

    @staticmethod
    def from_state(st):
        return Code(st["P"], st["kind"], st.get("mean"), st.get("scale"), st.get("evals"))


def exact_code(V):
    return Code(torch.eye(V), "exact")


# ------------------------------------------------------------------ fidelity

def knn_sets(d2, k_max, exclude_self=True):
    """d2: (nq, n) distances; returns (nq, k_max) indices of nearest (excluding the diagonal if asked)."""
    if exclude_self:
        d2 = d2.clone()
        nq = d2.shape[0]
        d2[torch.arange(nq), torch.arange(nq)] = float("inf")
    return torch.topk(d2, k_max, dim=1, largest=False).indices


def pairwise_d2(A, B):
    an = (A * A).sum(1)
    bn = (B * B).sum(1)
    return (an[:, None] + bn[None, :] - 2.0 * (A @ B.T)).clamp_min(0.0)


def recall_at_k(idx_exact, idx_test, ks):
    """Mean |top-k exact ∩ top-k test| / k for each k, per query -> dict k -> (nq,) tensor."""
    out = {}
    for k in ks:
        a = idx_exact[:, :k]
        b = idx_test[:, :k]
        # membership test via sorting
        a_s, _ = torch.sort(a, dim=1)
        pos = torch.searchsorted(a_s, b)
        pos = pos.clamp(max=k - 1)
        hit = torch.gather(a_s, 1, pos) == b
        out[k] = hit.float().mean(1)
    return out
