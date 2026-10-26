"""Evaluation orchestrator.

Usage: python -m entd.eval configs/x.yaml [--stages queries,search,proj,radius,rerank,baselines,fidelity,anchors,report]
Each stage writes to work_dir/results/ and can be re-run independently (later stages read earlier outputs).
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import ensure_tables, load_stream, prepare, last_seen_at
from .knn import radius_hist, search
from .metrics import ece_top1, entropy, hell2, hist_from_ids, js, naive_null_js, tv
from .model import checkpoint_path, load_checkpoint
from .projection import Code, MomentAccumulator, knn_sets, pairwise_d2, random_projection, recall_at_k
from .rerank import dists_for_windows, rerank_query
from .store import ArraySource, Store
from .suffix import NgramSet, ReversedSuffixIndex
from .util import Timer, gather_windows, load_json, save_json

SETS = ("heldout", "train")


class Ctx:
    def __init__(self, cfg):
        from .util import pick_device
        self.cfg, self.ev = cfg, cfg["eval"]
        self.meta = prepare(cfg)
        self.tables = ensure_tables(cfg, self.meta)
        self.T, self.V = self.tables["T"], self.meta["vocab_size"]
        self.device = pick_device(self.ev["device"])
        self.wd = cfg["work_dir"]
        self.rd = os.path.join(self.wd, "results")
        os.makedirs(self.rd, exist_ok=True)
        self.model = load_checkpoint(cfg, self.meta, self.device)
        d = cfg["data"]["dir"]
        self.stream = {"train": load_stream(d, "train"), "val": load_stream(d, "val")}
        self.W = {"train": self.tables["windows_train"], "val": self.tables["windows_val"]}
        self.store = {"train": Store(os.path.join(self.wd, "store_train")), "val": Store(os.path.join(self.wd, "store_val"))}
        self.split = self.tables["split_train"]
        ls = os.path.join(self.wd, "last_seen_step.npy")
        if not os.path.exists(ls):
            # model.checkpoint from another run (e.g. its epoch-1 checkpoint): last-seen steps up to that
            # checkpoint's step, taken from that run's schedule.
            ckp = checkpoint_path(cfg)
            sp = os.path.join(os.path.dirname(ckp), "schedule.npy")
            if os.path.exists(sp):
                step = int(torch.load(ckp, map_location="cpu").get("step", 0))
                np.save(ls, last_seen_at(np.load(sp), step, len(self.W["train"])))
                print(f"[eval] last_seen_step derived from {sp} at step {step}", flush=True)
        self.last_seen = np.load(ls) if os.path.exists(ls) else None
        self.k_grid = [k for k in self.ev["k_grid"] if k <= self.ev["K"]]
        self.radii = list(self.ev["radii"])
        self.radii2 = [r * r for r in self.radii]
        self.rng = np.random.default_rng(self.ev["seed"])
        self.n_w = len(self.W["train"])
        self.eot = self.meta.get("eot")
        st = torch.load(os.path.join(self.wd, "projection.pt"))
        self.code = Code.from_state(st["code"]).to(self.device)
        self.hcode = None if st.get("hcode") is None else Code.from_state(st["hcode"]).to(self.device)

    def stream_split(self, set_name):
        return "val" if set_name == "heldout" else "train"

    def pos_mask(self, wmask):
        return np.repeat(np.asarray(wmask, dtype=bool), self.T)

    def path(self, name):
        return os.path.join(self.rd, name)

    def unigram(self):
        keep = self.pos_mask(self.split == 0)
        u = np.bincount(np.asarray(self.store["train"].next)[keep], minlength=self.V).astype(np.float64)
        return torch.tensor(u / u.sum(), dtype=torch.float32, device=self.device)


# ----------------------------------------------------------------------------- queries

def stage_queries(ctx):
    ev, T = ctx.ev, ctx.T
    rows = []
    pools = {"heldout": ("val", np.arange(len(ctx.W["val"])), ev["n_queries_heldout"]),
             "train": ("train", np.nonzero(ctx.split == 0)[0], ev["n_queries_train"])}
    dedup = NgramSet(np.asarray(ctx.stream["train"]), ev["dedup_ngram"]) if ev["dedup_ngram"] else None
    qid = 0
    for set_name, (split, pool, n_q) in pools.items():
        wids = ctx.rng.choice(pool, size=n_q, replace=n_q > len(pool))
        offs = ctx.rng.integers(0, T, size=n_q)
        stream, W = ctx.stream[split], ctx.W[split]
        for w, o in zip(wids, offs):
            sp = int(W[w] + o)
            tok = int(stream[sp])
            rec = {"qid": qid, "set": set_name, "wid": int(w), "off": int(o), "pos": int(w) * T + int(o),
                   "stream_pos": sp, "tok": tok, "y": int(stream[sp + 1]),
                   "is_eot": bool(ctx.eot is not None and tok == ctx.eot),
                   "off_bin": int(min(3, int(math.log2(o + 1) // 2))),   # 0:[0,3] 1:[4,15] 2:[16,63] 3:64+
                   "last_seen": int(ctx.last_seen[w]) if (set_name == "train" and ctx.last_seen is not None) else -1,
                   "dup_train": False}
            if set_name == "heldout" and dedup is not None:
                rec["dup_train"] = dedup.contains_any(np.asarray(stream[W[w]: W[w] + T + 1]))
            rows.append(rec)
            qid += 1
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.path("queries.parquet"))
    print(f"[queries] {len(df)} queries; heldout dup_train={int(df[df.set=='heldout'].dup_train.sum())}", flush=True)
    return df


def load_queries(ctx):
    return pd.read_parquet(ctx.path("queries.parquet"))


def query_vectors(ctx, q):
    """Code vectors of the queries from their stores."""
    Zq = torch.empty((len(q), ctx.code.d), device=ctx.device)
    for set_name in SETS:
        m = (q.set == set_name).values
        if m.any():
            st = ctx.store[ctx.stream_split(set_name)]
            Zq[torch.as_tensor(np.nonzero(m)[0], device=ctx.device)] = st.z_rows(q.pos.values[m], ctx.device)
    return Zq, (Zq * Zq).sum(1)


def exclude_ranges(ctx, q):
    ex = np.zeros((len(q), 2), dtype=np.int64)
    m = (q.set == "train").values
    ex[m, 0] = q.pos.values[m] // ctx.T * ctx.T
    ex[m, 1] = ex[m, 0] + ctx.T
    ex[~m] = [-1, -1]
    return ex


def query_dists(ctx, q):
    """Exact log p_q for a query batch (DataFrame slice) -> (B, V) on device."""
    out = torch.empty((len(q), ctx.V), device=ctx.device)
    for set_name in SETS:
        m = (q.set == set_name).values
        if m.any():
            split = ctx.stream_split(set_name)
            lp = dists_for_windows(ctx.model, ctx.stream[split], ctx.W[split], ctx.T, q.wid.values[m], q.off.values[m], ctx.device)
            out[torch.as_tensor(np.nonzero(m)[0], device=ctx.device)] = lp
    return out


# ----------------------------------------------------------------------------- search

def store_variants(ctx):
    trained = ctx.split == 0
    v = {"full": trained, "untrained": ctx.split == 1}
    rng = np.random.default_rng(ctx.ev["seed"] + 7)
    for f in ctx.ev["coverage_fracs"]:
        v[f"cov{f}"] = trained & (rng.random(ctx.n_w) < f)
    return v


def stage_search(ctx, q):
    Zq, qn = query_vectors(ctx, q)
    ex = exclude_ranges(ctx, q)
    for name, wmask in store_variants(ctx).items():
        keep = ctx.pos_mask(wmask)
        with Timer(f"search {name} ({int(wmask.sum())} windows)"):
            bd, bi, cnt = search(ctx.store["train"], Zq, qn, ctx.ev["K"], ctx.radii2, exclude=ex, keep=keep,
                                 chunk=ctx.ev["chunk"], device=ctx.device)
        np.savez(ctx.path(f"search_{name}.npz"), d2=bd.numpy(), idx=bi.numpy(), counts=cnt.numpy())


def load_search(ctx, name):
    z = np.load(ctx.path(f"search_{name}.npz"))
    return z["d2"], z["idx"], z["counts"]


# ----------------------------------------------------------------------------- projected-space metrics

def leak_flags(ctx, q, idx, L):
    """(B, Kl) bool: neighbor's last-L tokens (stream level) equal the query's last-L tokens."""
    T = ctx.T
    B, Kl = idx.shape
    st = ctx.stream["train"]
    W = ctx.W["train"]
    spos = W[np.clip(idx, 0, None) // T] + idx % T
    nidx = spos[:, :, None] + np.arange(-L + 1, 1)[None, None, :]
    valid = (nidx >= 0).all(2) & (idx >= 0)
    ntails = np.asarray(st[np.clip(nidx, 0, len(st) - 1)])
    qt = np.zeros((B, L), dtype=np.int64)
    qvalid = np.zeros(B, dtype=bool)
    for i, (set_name, sp) in enumerate(zip(q.set.values, q.stream_pos.values)):
        s = ctx.stream[ctx.stream_split(set_name)]
        lo = sp - L + 1
        if lo >= 0:
            qt[i] = np.asarray(s[lo: sp + 1])
            qvalid[i] = True
    return valid & qvalid[:, None] & (ntails == qt[:, None, :]).all(2)


def stage_proj(ctx, q):
    ev, T, V = ctx.ev, ctx.T, ctx.V
    uni = ctx.unigram()
    nexts = ctx.store["train"].next
    lams = list(ev["lambdas"])
    Kl = min(ev["K"], 1024)
    rows = []
    for variant in store_variants(ctx).keys():
        bd, bi, _ = load_search(ctx, variant)
        for a in range(0, len(q), ev["query_batch"]):
            qb = q.iloc[a: a + ev["query_batch"]]
            B = len(qb)
            logpq = query_dists(ctx, qb)
            pq = torch.exp(logpq)
            yq = torch.as_tensor(qb.y.values, device=ctx.device)
            idx = bi[a: a + B]
            ids = np.where(idx >= 0, np.asarray(nexts)[np.clip(idx, 0, None)], -1)
            ids_t = torch.as_tensor(ids, device=ctx.device)
            wids_n = np.where(idx >= 0, idx // T, -1)
            leak = leak_flags(ctx, qb, idx[:, :Kl], ev["leak_L"]) if variant == "full" else None
            Hq = entropy(pq)
            nll_model = -logpq[torch.arange(B, device=ctx.device), yq]
            base = {"qid": qb.qid.values, "set": qb.set.values, "wid": qb.wid.values, "variant": variant,
                    "H_q": Hq.cpu().numpy(), "nll_model": nll_model.cpu().numpy(),
                    "pq_max": pq.max(1).values.cpu().numpy(),
                    "top1_correct": (pq.argmax(1) == yq).cpu().numpy()}
            for k in ctx.k_grid:
                e, nv = hist_from_ids(ids_t, V, k)
                ok = nv >= k
                j = js(e, pq)
                nn = naive_null_js(pq, k, ev["naive_null_draws"]) if ev["naive_null_draws"] > 0 else None
                e_s = (k * e + uni[None, :]) / (k + 1)
                rec = dict(base)
                rec.update({"k": k, "n_valid": nv.cpu().numpy(), "d2_k": bd[a: a + B, k - 1],
                            "js_total": j.cpu().numpy(), "tv_total": tv(e, pq).cpu().numpy(),
                            "hell2_total": hell2(e, pq).cpu().numpy(),
                            "top1_agree": (e.argmax(1) == pq.argmax(1)).cpu().numpy(),
                            "e_at_yq": e[torch.arange(B, device=ctx.device), yq].cpu().numpy(),
                            "nll_e_smoothed": (-torch.log(e_s[torch.arange(B, device=ctx.device), yq])).cpu().numpy(),
                            "n_distinct_w": np.array([len(np.unique(w[:k][w[:k] >= 0])) for w in wids_n]),
                            "n_leak": leak[:, :k].sum(1) if (leak is not None and k <= Kl) else np.full(B, -1)})
                if nn is not None:
                    rec.update({"null_js_mean": nn.mean(1).cpu().numpy(), "null_js_sd": nn.std(1).cpu().numpy(),
                                "null_js_pct": (nn < j[:, None]).float().mean(1).cpu().numpy()})
                for lam in lams:
                    mix = lam * e[torch.arange(B, device=ctx.device), yq] + (1 - lam) * pq[torch.arange(B, device=ctx.device), yq]
                    rec[f"nll_lam_{lam}"] = (-torch.log(mix)).cpu().numpy()
                df = pd.DataFrame(rec)
                df.loc[~ok.cpu().numpy(), ["js_total", "tv_total", "hell2_total"]] = np.nan
                rows.append(df)
        print(f"[proj] variant {variant} done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(ctx.path("proj_metrics.parquet"))
    return out


# ----------------------------------------------------------------------------- radius

def stage_radius(ctx, q):
    ev, T, V = ctx.ev, ctx.T, ctx.V
    rng = np.random.default_rng(ev["seed"] + 11)
    sub_w = np.nonzero((ctx.split == 0) & (rng.random(ctx.n_w) < ev["substore_frac"]))[0]
    sub_pos = (sub_w[:, None] * T + np.arange(T)[None, :]).reshape(-1)
    st = ctx.store["train"]
    Zs = st.z_rows(sub_pos, ctx.device)
    ns = (Zs * Zs).sum(1)
    nexts = torch.as_tensor(np.asarray(st.next)[sub_pos].astype(np.int64), device=ctx.device)
    sub_pos_t = torch.as_tensor(sub_pos, device=ctx.device)
    Zq, qn = query_vectors(ctx, q)
    ex = exclude_ranges(ctx, q)
    with Timer(f"radius hist over sub-store ({len(sub_w)} windows)"):
        hist, cnt_sub = radius_hist(Zs, ns, nexts, Zq, qn, ctx.radii2, V, sub_pos=sub_pos_t, exclude=ex)
    _, _, cnt_full = load_search(ctx, "full")
    rows = []
    for a in range(0, len(q), ev["query_batch"]):
        qb = q.iloc[a: a + ev["query_batch"]]
        B = len(qb)
        pq = torch.exp(query_dists(ctx, qb))
        for ri, r in enumerate(ctx.radii):
            h = hist[a: a + B, ri].to(ctx.device).float()
            c = cnt_sub[a: a + B, ri].to(ctx.device).float()
            e = h / c.clamp_min(1)[:, None]
            rec = {"qid": qb.qid.values, "set": qb.set.values, "wid": qb.wid.values, "radius": r,
                   "count_full": cnt_full[a: a + B, ri], "count_sub": c.cpu().numpy(),
                   "js_total": js(e, pq).cpu().numpy(), "tv_total": tv(e, pq).cpu().numpy(),
                   "top1_agree": (e.argmax(1) == pq.argmax(1)).cpu().numpy()}
            df = pd.DataFrame(rec)
            df.loc[df.count_sub < 1, ["js_total", "tv_total"]] = np.nan
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(ctx.path("radius_metrics.parquet"))
    return out


# ----------------------------------------------------------------------------- rerank (exact)

def stage_rerank(ctx, q):
    ev, T = ctx.ev, ctx.T
    bd, bi, _ = load_search(ctx, "full")
    uni = ctx.unigram()
    nexts = np.asarray(ctx.store["train"].next)
    per_k, per_c, per_q = [], [], []
    L = ev["leak_L"]
    for set_name in SETS:
        sel = np.nonzero((q.set == set_name).values)[0][: ev["rerank_queries"]]
        split = ctx.stream_split(set_name)
        with Timer(f"rerank {set_name} ({len(sel)} queries x {ev['rerank_K']})"):
            for n, i in enumerate(sel):
                row = q.iloc[i]
                idx = bi[i, : ev["rerank_K"]]
                idx = idx[idx >= 0]
                if len(idx) == 0:
                    continue
                s = ctx.stream[split]
                lo = row.stream_pos - L + 1
                q_tail = np.asarray(s[lo: row.stream_pos + 1]) if lo >= 0 else None
                pk, pc, qr = rerank_query(ctx.model, s, ctx.W[split], int(row.wid), int(row.off), ctx.stream["train"],
                                          ctx.W["train"], T, idx, nexts[idx].astype(np.int64), bd[i, : len(idx)],
                                          ctx.k_grid, ev["null_draws"], L, ctx.device, q_tail=q_tail,
                                          lambdas=ev["lambdas"], unigram=uni)
                for r in pk:
                    r.update({"qid": int(row.qid), "set": set_name, "wid": int(row.wid)})
                    r.update(qr)
                per_k.extend(pk)
                dfc = pd.DataFrame(pc)
                dfc["qid"] = int(row.qid)
                dfc["set"] = set_name
                dfc["rank"] = np.arange(len(dfc))
                per_c.append(dfc)
                if n % 50 == 0:
                    print(f"  [rerank] {set_name} {n}/{len(sel)}", flush=True)
    pd.DataFrame(per_k).to_parquet(ctx.path("rerank_metrics.parquet"))
    pd.concat(per_c, ignore_index=True).to_parquet(ctx.path("rerank_candidates.parquet"))


# ----------------------------------------------------------------------------- baselines

def stage_baselines(ctx, q):
    ev, T, V = ctx.ev, ctx.T, ctx.V
    st = ctx.store["train"]
    keep_pos = ctx.pos_mask(ctx.split == 0)
    nexts = np.asarray(st.next).astype(np.int64)
    uni = ctx.unigram()
    # bucket keys over trained positions: argmax x entropy bin
    nb = ev["entropy_bins"]
    ent = np.asarray(st.entropy).astype(np.float32)
    edges = np.quantile(ent[keep_pos], np.linspace(0, 1, nb + 1)[1:-1])
    ebin = np.digitize(ent, edges)
    key = np.where(keep_pos, np.asarray(st.argmax).astype(np.int64) * nb + ebin, -1)
    order = np.argsort(key, kind="stable")
    ks = key[order]

    def bucket_hist(a, b_lo, b_hi, self_pos=None, y_self=None):
        lo = np.searchsorted(ks, a * nb + b_lo)
        hi = np.searchsorted(ks, a * nb + b_hi, side="right")
        ids = nexts[order[lo:hi]]
        h = np.bincount(ids, minlength=V).astype(np.float64)
        n = hi - lo
        if self_pos is not None and lo <= np.searchsorted(ks, key[self_pos]) < hi and key[self_pos] >= 0:
            h[y_self] -= 1
            n -= 1
        return h, n

    # n-gram index over the train stream (reversed suffix array)
    with Timer("suffix array"):
        sidx = ReversedSuffixIndex(np.asarray(ctx.stream["train"]))
    stream_keep = np.zeros(len(ctx.stream["train"]), dtype=bool)
    stream_keep[: len(keep_pos)] = keep_pos   # contiguous windows: stream pos == store pos
    # hidden-state subset
    h_sub = np.load(os.path.join(ctx.wd, "h_sub.npy"), mmap_mode="r")
    h_wids = np.load(os.path.join(ctx.wd, "h_sub_wids.npy"))
    h_rank = np.full(ctx.n_w, -1, dtype=np.int64)
    h_rank[h_wids] = np.arange(len(h_wids))
    hn = np.concatenate([(np.asarray(h_sub[a: a + 1_000_000], dtype=np.float32) ** 2).sum(1)
                         for a in range(0, h_sub.shape[0], 1_000_000)])
    hsrc = ArraySource(h_sub, hn)
    h_trained = (ctx.split[h_wids] == 0)
    keep_h = np.repeat(h_trained, T)
    # query hidden states
    hq = torch.empty((len(q), h_sub.shape[1]), device=ctx.device)
    for a in range(0, len(q), 256):
        qb = q.iloc[a: a + 256]
        for set_name in SETS:
            m = (qb.set == set_name).values
            if not m.any():
                continue
            split = ctx.stream_split(set_name)
            x = torch.from_numpy(gather_windows(ctx.stream[split], ctx.W[split][qb.wid.values[m]], T)[:, :T]).to(ctx.device)
            with torch.no_grad():
                _, _, h = ctx.model(x, return_h=True)
            hq[a + torch.as_tensor(np.nonzero(m)[0], device=ctx.device)] = h[torch.arange(int(m.sum()), device=ctx.device),
                                                                              torch.as_tensor(qb.off.values[m], device=ctx.device)].float()
    ex_h = np.full((len(q), 2), -1, dtype=np.int64)
    m = (q.set == "train").values & (h_rank[q.wid.values] >= 0)
    ex_h[m, 0] = h_rank[q.wid.values[m]] * T
    ex_h[m, 1] = ex_h[m, 0] + T
    Kh = max(ctx.k_grid)
    with Timer("hidden-state kNN"):
        _, bi_h, _ = search(hsrc, hq, (hq * hq).sum(1), Kh, None, exclude=ex_h, keep=keep_h, chunk=ev["chunk"], device=ctx.device)
    bi_h = bi_h.numpy()
    pos_h = np.where(bi_h >= 0, h_wids[np.clip(bi_h, 0, None) // T] * T + bi_h % T, -1)
    # distribution kNN restricted to the same (h-subset ∩ trained) windows
    wm = np.zeros(ctx.n_w, dtype=bool)
    wm[h_wids] = True
    wm &= ctx.split == 0
    Zq, qn = query_vectors(ctx, q)
    with Timer("distribution kNN on h-subset windows"):
        _, bi_d, _ = search(st, Zq, qn, Kh, None, exclude=exclude_ranges(ctx, q), keep=ctx.pos_mask(wm),
                            chunk=ev["chunk"], device=ctx.device)
    bi_d = bi_d.numpy()

    rows_q, rows_k = [], []
    for a in range(0, len(q), ev["query_batch"]):
        qb = q.iloc[a: a + ev["query_batch"]]
        B = len(qb)
        pq = torch.exp(query_dists(ctx, qb))
        pq_np = pq.cpu().numpy()
        ids_h = torch.as_tensor(np.where(pos_h[a: a + B] >= 0, nexts[np.clip(pos_h[a: a + B], 0, None)], -1), device=ctx.device)
        ids_d = torch.as_tensor(np.where(bi_d[a: a + B] >= 0, nexts[np.clip(bi_d[a: a + B], 0, None)], -1), device=ctx.device)
        for k in ctx.k_grid:
            eh, nh = hist_from_ids(ids_h, V, k)
            ed, nd = hist_from_ids(ids_d, V, k)
            rows_k.append(pd.DataFrame({"qid": qb.qid.values, "set": qb.set.values, "wid": qb.wid.values, "k": k,
                                        "hknn_js": np.where((nh >= k).cpu().numpy(), js(eh, pq).cpu().numpy(), np.nan),
                                        "dknn_sub_js": np.where((nd >= k).cpu().numpy(), js(ed, pq).cpu().numpy(), np.nan),
                                        "hknn_top1": (eh.argmax(1) == pq.argmax(1)).cpu().numpy(),
                                        "dknn_sub_top1": (ed.argmax(1) == pq.argmax(1)).cpu().numpy()}))
        for i in range(B):
            row = qb.iloc[i]
            p = torch.as_tensor(pq_np[i], device=ctx.device)
            a_q = int(pq_np[i].argmax())
            b_q = int(np.digitize(float(entropy(p)), edges))
            is_tr = row.set == "train"
            self_pos = int(row.pos) if is_tr else None
            hA, nA = bucket_hist(a_q, 0, nb - 1, self_pos, int(row.y))
            hB, nB = bucket_hist(a_q, b_q, b_q, self_pos, int(row.y))
            split = ctx.stream_split(row.set)
            ctxt = np.asarray(ctx.stream[split][ctx.W[split][row.wid]: ctx.W[split][row.wid] + row.off + 1])
            rec = {"qid": int(row.qid), "set": row.set, "wid": int(row.wid),
                   "unigram_js": float(js(uni, p)),
                   "bucketA_js": float(js(torch.as_tensor(hA / max(nA, 1), dtype=torch.float32, device=ctx.device), p)) if nA > 0 else np.nan,
                   "bucketA_n": int(nA),
                   "bucketAE_js": float(js(torch.as_tensor(hB / max(nB, 1), dtype=torch.float32, device=ctx.device), p)) if nB > 0 else np.nan,
                   "bucketAE_n": int(nB)}
            for mc in ev["ngram_min_counts"]:
                n, lo, hi = sidx.longest_suffix(ctxt, ev["ngram_max_n"], mc + (1 if is_tr else 0))
                hN, nN = sidx.next_hist(lo, hi, V, stream_keep)
                if is_tr and n > 0:
                    hN[int(row.y)] -= 1
                    nN -= 1
                rec.update({f"ngram{mc}_js": float(js(torch.as_tensor(hN / max(nN, 1), dtype=torch.float32, device=ctx.device), p)) if nN > 0 else np.nan,
                            f"ngram{mc}_n": int(n), f"ngram{mc}_count": int(nN),
                            f"ngram{mc}_nll": float(-np.log((hN[int(row.y)] + 1e-3) / (nN + 1e-3 * V))) if nN > 0 else np.nan})
            rows_q.append(rec)
        print(f"  [baselines] {a + B}/{len(q)}", flush=True)
    pd.DataFrame(rows_q).to_parquet(ctx.path("baselines_q.parquet"))
    pd.concat(rows_k, ignore_index=True).to_parquet(ctx.path("baselines_knn.parquet"))


# ----------------------------------------------------------------------------- fidelity

def stage_fidelity(ctx):
    ev, T, V = ctx.ev, ctx.T, ctx.V
    dev = ctx.device
    P_ex = np.load(os.path.join(ctx.wd, "exact_p.npy"), mmap_mode="r")
    pos = np.load(os.path.join(ctx.wd, "exact_pos.npy"))
    S = torch.sqrt(torch.as_tensor(np.asarray(P_ex, dtype=np.float32), device=dev))
    n = S.shape[0]
    nq = min(ev["fidelity_queries"], n)
    ks = [k for k in (1, 10, 100) if k < n]
    kmax = max(ks)
    Hq = entropy(S[:nq] ** 2).cpu().numpy()
    terc = np.digitize(Hq, np.quantile(Hq, [1 / 3, 2 / 3]))
    d2_exact = pairwise_d2(S[:nq], S)
    ex_idx = knn_sets(d2_exact, kmax)
    out = {"n_sample": int(n), "n_queries": int(nq), "ks": ks, "recall": {}}

    def rec(name, Zq, Zs):
        idx = knn_sets(pairwise_d2(Zq, Zs), kmax)
        r = recall_at_k(ex_idx, idx, ks)
        out["recall"][name] = {str(k): {"all": float(v.mean()),
                                         **{f"H_terc{t}": float(v[torch.as_tensor(terc == t, device=dev)].mean()) for t in range(3)}}
                               for k, v in r.items()}

    sweep = [d for d in ev["fidelity_d_sweep"] if d <= ctx.code.d] + ([ctx.code.d] if ctx.code.d not in ev["fidelity_d_sweep"] else [])
    code = ctx.code
    if code.kind == "exact":
        # the store is exact; fit a PCA on the sample so the compression path is still validated
        acc = MomentAccumulator(V, dev)
        acc.add(S)
        Pm, evals, _ = acc.pca(V)
        code = Code(Pm, "pca", evals=evals).to(dev)
        out["pca_energy_cum"] = {str(d): float(evals[:d].sum() / evals.sum()) for d in sweep}
        rec("exact_store", S[:nq], S)
    for d in sweep:
        Z = code.encode(S, d)
        rec(f"{code.kind}_d{d}_fp32", Z[:nq], Z)
        Zh = Z.half().float()
        rec(f"{code.kind}_d{d}_fp16", Zh[:nq], Zh)
        qcode = Code(code.P[:, :d], code.kind)
        qcode.fit_int8(Z, ctx.cfg["embed"]["int8_clip_quantile"])
        Zi = qcode.dequantize(qcode.quantize(Z))
        rec(f"{code.kind}_d{d}_int8", Zi[:nq], Zi)
    if code.kind != "random":
        R = random_projection(V, ctx.code.d, ev["seed"]).to(dev)
        for d in sweep:
            Z = (S @ R[:, :d]) * ((ctx.code.d / d) ** 0.5)
            rec(f"random_d{d}_fp32", Z[:nq], Z)
    # stored rows (end-to-end check of embed.py)
    Zst = ctx.store["train"].z_rows(pos, dev)
    rec("stored_z", Zst[:nq], Zst)
    # plain p (L2 on probabilities) with the same code dims as an alternative metric
    Pp = S ** 2
    rec("L2_on_p_exact", Pp[:nq], Pp)
    # hidden-state code decoded through the head, and raw h
    hp = os.path.join(ctx.wd, "h_sub.npy")
    if os.path.exists(hp) and ctx.hcode is not None:
        h_sub = np.load(hp, mmap_mode="r")
        h_wids = np.load(os.path.join(ctx.wd, "h_sub_wids.npy"))
        h_rank = np.full(ctx.n_w, -1, dtype=np.int64)
        h_rank[h_wids] = np.arange(len(h_wids))
        rr = h_rank[pos // T] * T + pos % T
        if (rr >= 0).all():
            H = torch.as_tensor(np.asarray(h_sub[rr], dtype=np.float32), device=dev)
            rec("raw_h_L2", H[:nq], H)
            WU = ctx.model.unembed_weight().float()
            for d in [d for d in sweep if d <= ctx.hcode.d]:
                zh = (H - ctx.hcode.mean) @ ctx.hcode.P[:, :d]
                Hh = ctx.hcode.mean + zh @ ctx.hcode.P[:, :d].T
                Sh = torch.sqrt(torch.softmax(Hh @ WU.T, -1))
                rec(f"hcode_d{d}_decoded", Sh[:nq], Sh)
            # sanity: full-rank decode must reproduce exact distributions
            Sfull = torch.sqrt(torch.softmax(H @ WU.T, -1))
            out["h_decode_max_abs_err"] = float((Sfull - S).abs().max())
    if ctx.code.evals is not None and "pca_energy_cum" not in out:
        ev_ = ctx.code.evals
        out["pca_energy_cum"] = {str(d): float(ev_[:d].sum() / ev_.sum()) for d in sweep}
    save_json(out, ctx.path("fidelity.json"))
    for name, r in out["recall"].items():
        print(f"  [fidelity] {name:28s} " + "  ".join(f"r@{k}={r[str(k)]['all']:.3f}" for k in ks), flush=True)


# ----------------------------------------------------------------------------- global anchors

def stage_anchors(ctx, q):
    out = {}
    for set_name in SETS:
        split = ctx.stream_split(set_name)
        wids = np.unique(q[q.set == set_name].wid.values)[:512]
        nll, ent, conf, corr = [], [], [], []
        for a in range(0, len(wids), 64):
            tok = gather_windows(ctx.stream[split], ctx.W[split][wids[a: a + 64]], ctx.T)
            x = torch.from_numpy(tok[:, : ctx.T]).to(ctx.device)
            y = torch.from_numpy(tok[:, 1:]).to(ctx.device)
            with torch.no_grad():
                logits, _, _ = ctx.model(x)
            lp = torch.log_softmax(logits.float(), -1)
            p = torch.exp(lp)
            nll.append(-torch.gather(lp, 2, y[..., None])[..., 0].flatten().cpu().numpy())
            ent.append(entropy(p).flatten().cpu().numpy())
            conf.append(p.max(-1).values.flatten().cpu().numpy())
            corr.append((p.argmax(-1) == y).flatten().cpu().numpy())
        nll, ent, conf, corr = map(np.concatenate, (nll, ent, conf, corr))
        out[set_name] = {"mean_nll": float(nll.mean()), "mean_entropy": float(ent.mean()),
                         "top1_acc": float(corr.mean()), "top1_ece": ece_top1(conf, corr), "n_positions": int(len(nll))}
    save_json(out, ctx.path("anchors.json"))
    print("[anchors]", out, flush=True)


# ----------------------------------------------------------------------------- main

ALL = ["queries", "search", "proj", "radius", "rerank", "baselines", "fidelity", "anchors", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stages", default="all")
    args = ap.parse_args()
    cfg = load_config(args.config)
    stages = ALL if args.stages == "all" else args.stages.split(",")
    ctx = Ctx(cfg)
    q = None
    if "queries" in stages:
        with Timer("queries"):
            q = stage_queries(ctx)
    if q is None and os.path.exists(ctx.path("queries.parquet")):
        q = load_queries(ctx)
    for s in stages:
        if s == "queries":
            continue
        with Timer(s):
            if s == "search":
                stage_search(ctx, q)
            elif s == "proj":
                stage_proj(ctx, q)
            elif s == "radius":
                stage_radius(ctx, q)
            elif s == "rerank":
                stage_rerank(ctx, q)
            elif s == "baselines":
                stage_baselines(ctx, q)
            elif s == "fidelity":
                stage_fidelity(ctx)
            elif s == "anchors":
                stage_anchors(ctx, q)
            elif s == "report":
                from .report import main as report_main
                report_main(cfg)
            else:
                raise ValueError(s)


if __name__ == "__main__":
    main()
