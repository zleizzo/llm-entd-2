"""Projection identities, fidelity helpers, T statistic under its null, suffix-array baseline."""
import numpy as np
import torch

from entd.metrics import entropy, js, kl, window_bootstrap
from entd.projection import Code, MomentAccumulator, exact_code, knn_sets, pairwise_d2, recall_at_k
from entd.suffix import NgramSet, ReversedSuffixIndex, _sa_prefix_doubling, build_sa


def test_exact_code_distances():
    V = 65
    g = torch.Generator().manual_seed(0)
    P = torch.softmax(torch.randn(200, V, generator=g) * 3, -1)
    S = torch.sqrt(P)
    code = exact_code(V)
    Z = code.encode(S)
    assert torch.allclose(pairwise_d2(Z, Z), pairwise_d2(S, S), atol=1e-5)
    # full-rank PCA is a rotation: distances preserved, recall 1
    acc = MomentAccumulator(V, torch.device("cpu"))
    acc.add(S)
    Pm, ev, _ = acc.pca(V)
    Zp = Code(Pm, "pca").encode(S)
    assert torch.allclose(pairwise_d2(Zp, Zp), pairwise_d2(S, S), atol=1e-4)
    ex = knn_sets(pairwise_d2(S, S), 10)
    r = recall_at_k(ex, knn_sets(pairwise_d2(Zp, Zp), 10), [1, 5, 10])
    assert all(float(v.mean()) > 0.999 for v in r.values())
    # int8 roundtrip is close
    code = Code(Pm, "pca")
    code.fit_int8(Zp, 0.9999)
    Zi = code.dequantize(code.quantize(Zp))
    assert (Zi - Zp).abs().max() < 0.02


def test_T_statistic_null():
    """Under y_i ~ p_i the statistic T has mean 0 and the closed-form variance."""
    g = torch.Generator().manual_seed(1)
    V, K, R = 50, 512, 400
    logpq = torch.log_softmax(torch.randn(V, generator=g) * 2, -1)
    logp = torch.log_softmax(torch.randn(K, V, generator=g) * 2 + logpq[None, :], -1)
    p = torch.exp(logp)
    H = entropy(p)
    KL = (p * (logp - logpq[None, :])).sum(1)
    m1 = (p * logpq[None, :]).sum(1)
    var = (p * logpq[None, :] ** 2).sum(1) - m1 ** 2
    Ts = []
    for _ in range(R):
        y = torch.multinomial(p, 1, generator=g)[:, 0]
        Ts.append(float((-logpq[y]).mean() - (H + KL).mean()))
    Ts = np.array(Ts)
    se = float(torch.sqrt(var.sum()) / K)
    assert abs(Ts.mean()) < 4 * se / np.sqrt(R), (Ts.mean(), se)
    assert 0.7 < Ts.std() / se < 1.3, (Ts.std(), se)


def test_js_props():
    g = torch.Generator().manual_seed(2)
    p = torch.softmax(torch.randn(4, 20, generator=g), -1)
    assert torch.allclose(js(p, p), torch.zeros(4), atol=1e-6)
    e = torch.zeros(4, 20)
    e[:, 0] = 1.0
    assert (js(e, p) <= np.log(2) + 1e-6).all()
    assert torch.isfinite(kl(e, p)).all()


def test_suffix_index():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 5, size=3000).astype(np.uint16)
    assert np.array_equal(build_sa(t), _sa_prefix_doubling(t))
    idx = ReversedSuffixIndex(t)
    ctx = t[100:130]
    n, lo, hi = idx.longest_suffix(ctx, 8, 2)
    e, nxt = idx.match_positions(lo, hi)
    # every matched end position e has t[e-n+1:e+1] == ctx[-n:]
    for ee in e:
        assert np.array_equal(t[ee - n + 1: ee + 1], ctx[-n:])
    assert 129 in e   # the query's own position matches
    # count check by brute force
    pat = ctx[-n:]
    brute = sum(1 for s in range(len(t) - n + 1) if np.array_equal(t[s: s + n], pat) and s + n < len(t))
    assert brute == len(e)
    ns = NgramSet(t, 10)
    assert ns.contains_any(t[500:520])
    assert not ns.contains_any(np.full(12, 7, dtype=np.uint16))


def test_window_bootstrap():
    m, lo, hi = window_bootstrap(np.r_[np.ones(50), np.zeros(50)], np.repeat(np.arange(10), 10), n_boot=200)
    assert abs(m - 0.5) < 1e-9 and lo < 0.5 < hi


if __name__ == "__main__":
    test_exact_code_distances()
    test_T_statistic_null()
    test_js_props()
    test_suffix_index()
    test_window_bootstrap()
    print("projection/metrics/suffix tests ok")
