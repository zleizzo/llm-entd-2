"""Brute-force nearest-neighbor search streamed over a store, with an elementwise threshold prefilter.

search(): exact top-K (squared distances) + per-radius counts over the full source.
radius_hist(): per-radius next-token histograms over an in-memory sub-store.
"""
import numpy as np
import torch

INF = float("inf")


def _merge_flat(best_d, best_i, qi, vals, pidx, K):
    """Merge flat survivors (qi, vals, pidx) into per-row sorted top-K (best_d, best_i); only touched rows."""
    rows = torch.unique(qi)
    nr = rows.shape[0]
    loc = torch.searchsorted(rows, qi)
    sub_d, sub_i = best_d[rows], best_i[rows]
    fin = torch.isfinite(sub_d)
    oq = torch.arange(nr, device=best_d.device)[:, None].expand(nr, K)[fin]
    aq = torch.cat([oq, loc])
    ad = torch.cat([sub_d[fin], vals])
    ai = torch.cat([sub_i[fin], pidx])
    o1 = torch.argsort(ad)
    aq, ad, ai = aq[o1], ad[o1], ai[o1]
    o2 = torch.argsort(aq, stable=True)
    aq, ad, ai = aq[o2], ad[o2], ai[o2]
    first = torch.searchsorted(aq, torch.arange(nr, device=aq.device))
    rank = torch.arange(aq.shape[0], device=aq.device) - first[aq]
    keep = rank < K
    nd = torch.full((nr, K), INF, device=best_d.device, dtype=best_d.dtype)
    ni = torch.full((nr, K), -1, device=best_d.device, dtype=best_i.dtype)
    nd[aq[keep], rank[keep]] = ad[keep]
    ni[aq[keep], rank[keep]] = ai[keep]
    best_d[rows] = nd
    best_i[rows] = ni
    return best_d, best_i


def search(src, Zq, qnorm2, K, radii2=None, exclude=None, keep=None, chunk=262144, device=None, verbose=False):
    """
    src: object with N, znorm2 (N,) numpy, z_chunk(a, b, device) -> (b-a, d) float32.
    Zq: (Q, d) float32 tensor on device; qnorm2: (Q,) float32 on device.
    radii2: list of squared radii -> counts (Q, R) over the full source (respecting masks).
    exclude: (Q, 2) int64 numpy [lo, hi) source-index ranges excluded per query.
    keep: (N,) bool numpy; False positions are ignored.
    Returns best_d2 (Q, K), best_idx (Q, K) (-1 if fewer than K), counts (Q, R)  -- all CPU tensors.
    """
    device = device or Zq.device
    Q = Zq.shape[0]
    N = src.N
    best_d = torch.full((Q, K), INF, device=device)
    best_i = torch.full((Q, K), -1, dtype=torch.int64, device=device)
    R = 0 if radii2 is None else len(radii2)
    counts = torch.zeros((Q, R), dtype=torch.int64, device=device)
    r2 = torch.tensor(list(radii2 or []), device=device, dtype=torch.float32)
    if exclude is not None:
        lo = torch.from_numpy(np.ascontiguousarray(exclude[:, 0])).to(device)
        hi = torch.from_numpy(np.ascontiguousarray(exclude[:, 1])).to(device)
    n_dense = n_flat = 0
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        C = b - a
        Zc = src.z_chunk(a, b, device)
        nc = torch.from_numpy(np.ascontiguousarray(src.znorm2[a:b])).to(device).float()
        d2 = qnorm2[:, None] + nc[None, :] - 2.0 * (Zq @ Zc.T)
        d2.clamp_(min=0.0)
        if keep is not None:
            kc = torch.from_numpy(np.ascontiguousarray(keep[a:b])).to(device)
            d2.masked_fill_(~kc[None, :], INF)
        if exclude is not None:
            pos = torch.arange(a, b, device=device)
            m = (pos[None, :] >= lo[:, None]) & (pos[None, :] < hi[:, None])
            d2.masked_fill_(m, INF)
        for ri in range(R):
            counts[:, ri] += (d2 < r2[ri]).sum(1)
        thr = best_d[:, -1]
        surv = d2 < thr[:, None]
        n_surv = int(surv.sum())
        if n_surv == 0:
            continue
        if n_surv > Q * K // 4:
            kk = min(K, C)
            vals, idx = torch.topk(d2, kk, dim=1, largest=False)
            cat_d = torch.cat([best_d, vals], 1)
            cat_i = torch.cat([best_i, idx + a], 1)
            best_d, order = torch.topk(cat_d, K, dim=1, largest=False)
            best_i = torch.gather(cat_i, 1, order)
            n_dense += 1
        else:
            qi, ci = surv.nonzero(as_tuple=True)
            best_d, best_i = _merge_flat(best_d, best_i, qi, d2[qi, ci], ci + a, K)
            n_flat += 1
        if verbose and (a // chunk) % 50 == 0:
            print(f"  [knn] {b}/{N} dense={n_dense} flat={n_flat}", flush=True)
    best_i[~torch.isfinite(best_d)] = -1
    return best_d.cpu(), best_i.cpu(), counts.cpu()


def search_dense(Zs, ns, Zq, qn, K, exclude_mask=None):
    """Reference: full distance matrix. Zs (N,d), ns (N,), Zq (Q,d), qn (Q,) on the same device.
    exclude_mask: optional (Q, N) bool, True = exclude."""
    d2 = (qn[:, None] + ns[None, :] - 2.0 * (Zq @ Zs.T)).clamp_min(0.0)
    if exclude_mask is not None:
        d2 = d2.masked_fill(exclude_mask, INF)
    kk = min(K, Zs.shape[0])
    vals, idx = torch.topk(d2, kk, dim=1, largest=False)
    if kk < K:
        vals = torch.cat([vals, torch.full((vals.shape[0], K - kk), INF, device=vals.device)], 1)
        idx = torch.cat([idx, torch.full((idx.shape[0], K - kk), -1, device=idx.device, dtype=idx.dtype)], 1)
    idx[~torch.isfinite(vals)] = -1
    return vals, idx


def radius_hist(Zs, ns, nexts, Zq, qn, radii2, V, sub_pos=None, exclude=None, qbatch=128):
    """Per-radius histogram of next tokens among sub-store positions within each radius of each query.
    Zs (M,d), ns (M,), nexts (M,) long on device. sub_pos (M,) long: original store index of each row,
    used with exclude (Q,2) [lo,hi) ranges. Returns hist (Q,R,V) int32 CPU and counts (Q,R) int64 CPU."""
    device = Zs.device
    Q, R = Zq.shape[0], len(radii2)
    r2 = torch.tensor(radii2, device=device, dtype=torch.float32)
    hist = torch.zeros((Q, R, V), dtype=torch.int32)
    counts = torch.zeros((Q, R), dtype=torch.int64)
    for a in range(0, Q, qbatch):
        b = min(a + qbatch, Q)
        d2 = (qn[a:b, None] + ns[None, :] - 2.0 * (Zq[a:b] @ Zs.T)).clamp_min(0.0)
        if exclude is not None:
            lo = torch.as_tensor(exclude[a:b, 0], device=device)[:, None]
            hi = torch.as_tensor(exclude[a:b, 1], device=device)[:, None]
            m = (sub_pos[None, :] >= lo) & (sub_pos[None, :] < hi)
            d2 = d2.masked_fill(m, INF)
        hb = torch.zeros(((b - a) * R * V,), dtype=torch.int32, device=device)
        for ri in range(R):
            qi, mi = (d2 < r2[ri]).nonzero(as_tuple=True)
            if qi.numel() == 0:
                continue
            flat = (qi * R + ri) * V + nexts[mi]
            hb.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        hb = hb.view(b - a, R, V)
        hist[a:b] = hb.cpu()
        counts[a:b] = hb.sum(-1).long().cpu()
    return hist, counts
