"""Divergences, calibration statistics and aggregation helpers (torch, batched over the last dim)."""
import numpy as np
import torch

EPS = 1e-12


def kl(p, q):
    return (p * (torch.log(p + EPS) - torch.log(q + EPS))).sum(-1)


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv(p, q):
    return 0.5 * (p - q).abs().sum(-1)


def hell2(p, q):
    """||sqrt p - sqrt q||^2 = 2 H^2  (our squared-distance convention)."""
    return ((torch.sqrt(p) - torch.sqrt(q)) ** 2).sum(-1)


def entropy(p):
    return -(p * torch.log(p + EPS)).sum(-1)


def hist_from_ids(ids, V, k=None):
    """ids: (Q, K) long (−1 = missing) -> (Q, V) empirical histograms over the first k columns."""
    ids = ids[:, :k] if k is not None else ids
    Q = ids.shape[0]
    out = torch.zeros((Q, V), device=ids.device)
    valid = ids >= 0
    src = valid.float()
    out.scatter_add_(1, ids.clamp_min(0), src)
    n = valid.sum(1, keepdim=True).clamp_min(1)
    return out / n, valid.sum(1)


def naive_null_js(pq, k, draws):
    """JS(e*, p_q) with e* a k-sample histogram drawn from p_q itself. pq: (B, V). Returns (B, draws)."""
    B, V = pq.shape
    s = torch.multinomial(pq, k * draws, replacement=True).view(B, draws, k)
    cnt = torch.zeros((B * draws, V), device=pq.device)
    cnt.scatter_add_(1, s.view(B * draws, k), torch.ones((B * draws, k), device=pq.device))
    e = (cnt / k).view(B, draws, V)
    return js(e, pq[:, None, :].expand(B, draws, V))


def ece_top1(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def window_bootstrap(values, groups, n_boot=1000, seed=0, q=(0.025, 0.975)):
    """Mean of `values` with a CI from resampling whole groups (windows) with replacement."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed)
    G = len(ug)
    picks = rng.integers(0, G, size=(n_boot, G))
    boots = sums[picks].sum(1) / cnts[picks].sum(1)
    return float(values.mean()), float(np.quantile(boots, q[0])), float(np.quantile(boots, q[1]))


def summarize(df, value_cols, by, group_col="wid", n_boot=500):
    """Per-group means with window-bootstrap CIs -> long-format DataFrame."""
    import pandas as pd
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(by, key))
        base["n"] = len(g)
        for c in value_cols:
            if c not in g:
                continue
            m, lo, hi = window_bootstrap(g[c].values, g[group_col].values, n_boot=n_boot)
            rows.append({**base, "metric": c, "mean": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
