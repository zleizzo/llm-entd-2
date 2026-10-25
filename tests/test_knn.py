"""Streaming prefiltered search == naive full-matrix top-K (with masks and exclusions)."""
import numpy as np
import torch

from entd.knn import search, search_dense, radius_hist
from entd.store import ArraySource


def _setup(N=50_000, d=16, Q=37, seed=0):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(N, d, generator=g)
    Z = Z / Z.norm(dim=1, keepdim=True)
    Zq = torch.randn(Q, d, generator=g)
    Zq = Zq / Zq.norm(dim=1, keepdim=True)
    return Z, Zq


def test_search_matches_dense():
    Z, Zq = _setup()
    N, Q = Z.shape[0], Zq.shape[0]
    T = 100
    K = 50
    src = ArraySource(Z.numpy(), (Z * Z).sum(1).numpy())
    ex = np.stack([np.arange(Q) * T, np.arange(Q) * T + T], 1)
    keep = np.ones(N, dtype=bool)
    keep[::7] = False
    radii2 = [0.5, 1.0, 1.9]
    bd, bi, cnt = search(src, Zq, (Zq * Zq).sum(1), K, radii2, exclude=ex, keep=keep, chunk=4096)
    pos = torch.arange(N)
    exm = (pos[None, :] >= torch.as_tensor(ex[:, 0])[:, None]) & (pos[None, :] < torch.as_tensor(ex[:, 1])[:, None])
    exm |= ~torch.as_tensor(keep)[None, :]
    rd, ri = search_dense(Z, (Z * Z).sum(1), Zq, (Zq * Zq).sum(1), K, exclude_mask=exm)
    assert torch.allclose(bd, rd, atol=1e-4), (bd - rd).abs().max()
    # indices may differ only under exact ties
    same = (bi == ri) | (torch.isclose(bd, rd, atol=1e-6) & (bd == torch.gather(rd, 1, torch.argsort(rd, 1))))
    assert same.float().mean() > 0.999
    d2 = (((Zq[:, None, :] - Z[None, :, :]) ** 2).sum(-1)).masked_fill(exm, float("inf"))
    for ri_, r2 in enumerate(radii2):
        assert torch.equal(cnt[:, ri_], (d2 < r2).sum(1))


def test_search_small_chunks_and_fewer_than_K():
    Z, Zq = _setup(N=300, d=8, Q=5)
    src = ArraySource(Z.numpy())
    K = 400
    bd, bi, _ = search(src, Zq, (Zq * Zq).sum(1), K, None, chunk=64)
    assert (bi[:, :300] >= 0).all() and (bi[:, 300:] == -1).all()
    assert torch.isinf(bd[:, 300:]).all()
    rd, ri = search_dense(Z, (Z * Z).sum(1), Zq, (Zq * Zq).sum(1), K)
    assert torch.allclose(bd[:, :300], rd[:, :300], atol=1e-5)


def test_radius_hist():
    Z, Zq = _setup(N=2000, d=8, Q=9)
    V = 11
    nexts = torch.randint(0, V, (2000,))
    sub_pos = torch.arange(2000)
    ex = np.array([[0, 100]] * 9)
    hist, cnt = radius_hist(Z, (Z * Z).sum(1), nexts, Zq, (Zq * Zq).sum(1), [0.8, 1.5], V, sub_pos=sub_pos, exclude=ex, qbatch=4)
    d2 = ((Zq[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    d2[:, :100] = float("inf")
    for ri, r2 in enumerate([0.8, 1.5]):
        m = d2 < r2
        assert torch.equal(cnt[:, ri], m.sum(1))
        ref = torch.stack([torch.bincount(nexts[m[i]], minlength=V) for i in range(9)])
        assert torch.equal(hist[:, ri].long(), ref)


if __name__ == "__main__":
    test_search_matches_dense()
    test_search_small_chunks_and_fewer_than_K()
    test_radius_hist()
    print("knn tests ok")
