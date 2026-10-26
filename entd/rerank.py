"""Exact re-ranking and the exact (per-neighbor) null for a subset of queries.

For a query with candidates (store positions), re-run the model on the candidates' windows to get their
exact distributions p_i, sort by exact Hellinger distance, and compute, for every k in the grid:
  total      d(e_k, p_q)        empirical histogram of the k nearest vs the query's prediction
  geometry   d(pbar_k, p_q)     mean of the neighbors' own predictions vs the query's prediction
  calib      d(e_k, pbar_k)     empirical histogram vs what the model predicted *for those neighbors*
  null       the same with y_i* ~ p_i resampled (consistency resampling)
  T_k        mean_i[-log p_q(y_i)] - mean_i[H(p_i) + KL(p_i || p_q)], with closed-form null variance
"""
import numpy as np
import torch
import torch.nn.functional as F

from .metrics import entropy, hell2, js, tv
from .util import gather_windows


@torch.no_grad()
def dists_for_windows(model, stream, W, T, wids, offs, device, batch=64):
    """Exact log-distributions at (window, offset) pairs. Returns (n, V) float32 on device."""
    uniq, inv = np.unique(wids, return_inverse=True)
    out = None
    for i in range(0, len(uniq), batch):
        u = uniq[i:i + batch]
        x = torch.from_numpy(gather_windows(stream, W[u], T)[:, :T]).to(device)
        logits, _, _ = model(x)
        sel = np.nonzero((inv >= i) & (inv < i + batch))[0]
        # gather the needed (window, offset) rows first, then log_softmax: identical result, ~T x less work than a
        # log_softmax over every position (this stage visits ~2M windows at tier 1)
        rows = logits[torch.as_tensor(inv[sel] - i, device=device), torch.as_tensor(offs[sel], device=device)]
        rows = F.log_softmax(rows.float(), -1)
        if out is None:
            out = torch.empty((len(wids), rows.shape[-1]), device=device, dtype=torch.float32)
        out[torch.as_tensor(sel, device=device)] = rows
    return out


@torch.no_grad()
def rerank_query(model, stream_q, W_q, wid_q, off_q, stream_s, W_s, T, cand_pos, next_ids, d2_proj,
                 k_grid, null_draws, leak_L, device, q_tail=None, lambdas=(), unigram=None):
    """Returns (per_k list of dict, per_cand dict of arrays, query dict)."""
    V = model.config.vocab_size
    logpq = dists_for_windows(model, stream_q, W_q, T, np.array([wid_q]), np.array([off_q]), device)[0]
    pq = torch.exp(logpq)
    y_q = int(stream_q[W_q[wid_q] + off_q + 1])
    K = len(cand_pos)
    wids, offs = cand_pos // T, cand_pos % T
    logp = dists_for_windows(model, stream_s, W_s, T, wids, offs, device)      # (K, V)
    p = torch.exp(logp)
    d2 = ((torch.sqrt(p) - torch.sqrt(pq)[None, :]) ** 2).sum(1)                # exact Hellinger^2 * 2
    order = torch.argsort(d2)
    d2, p, logp = d2[order], p[order], logp[order]
    order_np = order.cpu().numpy()
    cand_pos, wids, offs = cand_pos[order_np], wids[order_np], offs[order_np]
    next_ids = next_ids[order_np]
    d2_proj = d2_proj[order_np]
    y = torch.as_tensor(next_ids, device=device, dtype=torch.long)
    # per-candidate terms
    logpq_y = logpq[y]
    H_i = entropy(p)
    KL_i = (p * (logp - logpq[None, :])).sum(1)
    m1 = (p * logpq[None, :]).sum(1)
    var_i = (p * logpq[None, :] ** 2).sum(1) - m1 ** 2
    same_w = wids == wid_q if stream_q is stream_s else np.zeros(K, dtype=bool)
    leak = np.zeros(K, dtype=bool)
    if q_tail is not None and leak_L > 0:
        spos = W_s[wids] + offs
        idx = spos[:, None] + np.arange(-leak_L + 1, 1)[None, :]
        valid = (idx >= 0).all(1)
        tails = np.asarray(stream_s[np.clip(idx, 0, len(stream_s) - 1)])
        leak = valid & (tails == q_tail[None, :]).all(1) & (idx[:, -1] != (W_q[wid_q] + off_q) if stream_q is stream_s else True)
    # cumulative quantities
    oh = torch.zeros((K, V), device=device)
    oh[torch.arange(K, device=device), y] = 1.0
    ce = torch.cumsum(oh, 0)
    cp = torch.cumsum(p, 0)
    c_logpq_y = torch.cumsum(logpq_y, 0)
    c_H_KL = torch.cumsum(H_i + KL_i, 0)
    c_var = torch.cumsum(var_i, 0)
    # null draws y* ~ p_i
    D = null_draws
    samples = torch.multinomial(p, D, replacement=True) if D > 0 else None   # (K, D)
    per_k = []
    for k in [k for k in k_grid if k <= K]:
        e = ce[k - 1] / k
        pbar = cp[k - 1] / k
        rec = {"k": k, "d2_exact_k": float(d2[k - 1]), "d2_proj_k": float(d2_proj[k - 1]),
               "n_distinct_w": int(len(np.unique(wids[:k]))), "n_leak": int(leak[:k].sum()),
               "n_same_w": int(same_w[:k].sum()),
               "js_total": float(js(e, pq)), "tv_total": float(tv(e, pq)), "hell2_total": float(hell2(e, pq)),
               "js_geom": float(js(pbar, pq)), "hell2_geom": float(hell2(pbar, pq)),
               "js_calib": float(js(e, pbar)), "tv_calib": float(tv(e, pbar)),
               "top1_agree": int(e.argmax() == pq.argmax()),
               "pq_at_argmax_e": float(pq[e.argmax()]), "e_at_argmax_pq": float(e[pq.argmax()]),
               "T": float(c_logpq_y[k - 1] / k * -1 - c_H_KL[k - 1] / k),
               "T_se": float(torch.sqrt(c_var[k - 1]) / k),
               "e_at_yq": float(e[y_q]), "pbar_at_yq": float(pbar[y_q])}
        if samples is not None:
            cnt = torch.zeros((D, V), device=device)
            cnt.scatter_add_(1, samples[:k].T, torch.ones((D, k), device=device))
            es = cnt / k
            jt = js(es, pq[None, :].expand(D, V))
            jc = js(es, pbar[None, :].expand(D, V))
            rec.update({"null_js_total_mean": float(jt.mean()), "null_js_total_sd": float(jt.std()),
                        "null_js_calib_mean": float(jc.mean()), "null_js_calib_sd": float(jc.std()),
                        "null_js_total_pct": float((jt < rec["js_total"]).float().mean()),
                        "null_js_calib_pct": float((jc < rec["js_calib"]).float().mean())})
        if unigram is not None:
            e_s = (k * e + unigram) / (k + 1)
            rec["nll_e_smoothed"] = float(-torch.log(e_s[y_q]))
        for lam in lambdas:
            rec[f"nll_lam_{lam}"] = float(-torch.log(lam * e[y_q] + (1 - lam) * pq[y_q]))
        per_k.append(rec)
    per_cand = {"pos": cand_pos, "d2_exact": d2.cpu().numpy(), "d2_proj": d2_proj, "y": next_ids,
                "logpq_y": logpq_y.cpu().numpy(), "H_i": H_i.cpu().numpy(), "KL_i": KL_i.cpu().numpy(),
                "var_i": var_i.cpu().numpy(), "leak": leak, "same_w": same_w}
    qrec = {"H_q": float(entropy(pq)), "nll_model": float(-logpq[y_q]), "argmax_q": int(pq.argmax()),
            "y_q": y_q, "pq_max": float(pq.max())}
    return per_k, per_cand, qrec
