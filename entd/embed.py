"""Embedding pass: final model over every window -> sqrt(softmax) -> linear code -> store.

Usage: python -m entd.embed configs/x.yaml [--split train|val|both]
Writes work_dir/projection.pt, work_dir/store_train/, work_dir/store_val/,
work_dir/h_sub.npy (+ h_sub_wids.npy), work_dir/exact_p.npy (+ exact_pos.npy).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config
from .data import ensure_tables, load_stream, prepare
from .model import load_checkpoint
from .projection import Code, MomentAccumulator, exact_code, random_projection

DTYPES_T = {"float32": torch.float32, "float16": torch.float16}
from .store import Store
from .util import Timer, gather_windows, pick_device


@torch.no_grad()
def forward_dist(model, x, return_h=False):
    """x: (B, T) long -> logp (B, T, V) float32, h (B, T, n_embd) float32 or None."""
    logits, _, h = model(x, return_h=return_h)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp, (h.float() if h is not None else None)


@torch.no_grad()
def fit_code(cfg, meta, model, stream, W, T, device):
    ec = cfg["embed"]
    V = meta["vocab_size"]
    n_embd = model.config.n_embd
    if ec["code"] == "exact" and ec["store_dtype"] == "int8":
        raise ValueError("exact code with int8 storage is pointless; use float32/float16")
    rng = np.random.default_rng(ec["seed"])
    wids = np.sort(rng.choice(len(W), size=min(ec["pca_sample_windows"], len(W)), replace=False))
    accS, accH = MomentAccumulator(V, device), MomentAccumulator(n_embd, device)
    B = ec["batch_windows"]
    for i in range(0, len(wids), B):
        x = torch.from_numpy(gather_windows(stream, W[wids[i:i + B]], T)[:, :T]).to(device)
        logp, h = forward_dist(model, x, return_h=True)
        s = torch.exp(0.5 * logp).view(-1, V)
        accS.add(s)
        accH.add(h.view(-1, n_embd))
    d = min(ec["d"], V)
    if ec["code"] == "exact":
        code = exact_code(V)
        d = V
    elif ec["code"] == "pca":
        P, evals, _ = accS.pca(d)
        code = Code(P, "pca", evals=evals)
        print(f"[embed] PCA(s) d={d}: captured energy {float(evals.sum() / (accS.M.trace() / accS.n)):.4f}", flush=True)
    elif ec["code"] == "random":
        code = Code(random_projection(V, d, ec["seed"]), "random")
    else:
        raise ValueError(ec["code"])
    Ph, evals_h, mean_h = accH.pca(min(d, n_embd), centered=True)
    hcode = Code(Ph, "pca", mean=mean_h, evals=evals_h)
    if ec["store_dtype"] == "int8":
        code = code.to(device)
        zs = []
        for i in range(0, min(len(wids), 4 * B), B):
            x = torch.from_numpy(gather_windows(stream, W[wids[i:i + B]], T)[:, :T]).to(device)
            logp, _ = forward_dist(model, x)
            zs.append(code.encode(torch.exp(0.5 * logp).view(-1, V)).cpu())
        code.fit_int8(torch.cat(zs), ec["int8_clip_quantile"])
    return code, hcode


@torch.no_grad()
def embed_split(cfg, meta, model, code, split, tables, device):
    ec = cfg["embed"]
    wd = cfg["work_dir"]
    V = meta["vocab_size"]
    T = tables["T"]
    stream = load_stream(cfg["data"]["dir"], split)
    W = tables[f"windows_{split}"]
    n_w = len(W)
    N = n_w * T
    code = code.to(device)
    extra = {"split": split, "V": V, "code": code.kind, "n_windows": int(n_w),
             "scale": None if code.scale is None else code.scale.tolist()}
    store = Store.create(os.path.join(wd, f"store_{split}"), N, code.d, T, ec["store_dtype"], extra)
    print(f"[embed] {split}: {n_w} windows, {N} positions, d={code.d}, dtype={ec['store_dtype']}", flush=True)

    # subsets (train split only): hidden states for a window subset, exact distributions for a position sample
    h_rank, h_store, exact_pos, exact_store = None, None, None, None
    if split == "train":
        rng = np.random.default_rng(ec["seed"] + 1)
        n_embd = model.config.n_embd
        h_mask = rng.random(n_w) < ec["h_subset_frac"]
        h_wids = np.nonzero(h_mask)[0]
        if len(h_wids):
            h_rank = np.full(n_w, -1, dtype=np.int64)
            h_rank[h_wids] = np.arange(len(h_wids))
            np.save(os.path.join(wd, "h_sub_wids.npy"), h_wids)
            h_store = np.lib.format.open_memmap(os.path.join(wd, "h_sub.npy"), mode="w+", dtype=np.float16,
                                                shape=(len(h_wids) * T, n_embd))
        cand_w = h_wids if len(h_wids) else np.arange(n_w)
        cand = (cand_w[:, None] * T + np.arange(T)[None, :]).reshape(-1)
        n_ex = min(ec["exact_sample"], len(cand))
        exact_pos = np.sort(rng.choice(cand, size=n_ex, replace=False))
        np.save(os.path.join(wd, "exact_pos.npy"), exact_pos)
        exact_store = np.lib.format.open_memmap(os.path.join(wd, "exact_p.npy"), mode="w+", dtype=np.float16,
                                                shape=(n_ex, V))

    B = ec["batch_windows"]
    ex_ptr = 0
    for i in range(0, n_w, B):
        wids = np.arange(i, min(i + B, n_w))
        tok = gather_windows(stream, W[wids], T)
        x = torch.from_numpy(tok[:, :T]).to(device)
        logp, h = forward_dist(model, x, return_h=h_store is not None)
        p = torch.exp(logp)
        s = torch.sqrt(p).view(-1, V)
        z = code.encode(s)
        a, b = int(wids[0]) * T, int(wids[-1] + 1) * T
        if store.dtype == "int8":
            q = code.quantize(z)
            store.z[a:b] = q.cpu().numpy()
            zdq = code.dequantize(q)
            store.znorm2[a:b] = (zdq * zdq).sum(1).cpu().numpy()
        else:
            zq = z.to(DTYPES_T[store.dtype])            # what gets stored; its norm (not z's) is what search uses
            store.z[a:b] = zq.cpu().numpy()
            zq = zq.float()
            store.znorm2[a:b] = (zq * zq).sum(1).cpu().numpy()   # on GPU: a CPU torch op here cost 300 ms/batch with
                                                                  # 128 OMP threads on a 16-vCPU pod
        store.next[a:b] = tok[:, 1:T + 1].reshape(-1).astype(np.uint16)
        store.argmax[a:b] = p.argmax(-1).view(-1).cpu().numpy().astype(np.uint16)
        store.entropy[a:b] = (-(p * logp).sum(-1)).view(-1).cpu().numpy().astype(np.float16)
        if h_store is not None:
            for j, w in enumerate(wids):
                if h_rank[w] >= 0:
                    r = h_rank[w] * T
                    h_store[r:r + T] = h[j].cpu().numpy().astype(np.float16)
        if exact_store is not None:
            while ex_ptr < len(exact_pos) and exact_pos[ex_ptr] < b:
                pos = exact_pos[ex_ptr]
                exact_store[ex_ptr] = p.view(-1, V)[pos - a].cpu().numpy().astype(np.float16)
                ex_ptr += 1
        if (i // B) % 50 == 0:
            print(f"  [{split}] window {i}/{n_w}", flush=True)
    store.flush()
    store.set_meta(complete=True)
    if h_store is not None:
        h_store.flush()
    if exact_store is not None:
        exact_store.flush()
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = prepare(cfg)
    tables = ensure_tables(cfg, meta)
    device = pick_device(cfg["embed"]["device"])
    model = load_checkpoint(cfg, meta, device)
    model.eval()
    T = tables["T"]
    proj_path = os.path.join(cfg["work_dir"], "projection.pt")
    if os.path.exists(proj_path):
        st = torch.load(proj_path)
        code = Code.from_state(st["code"])
        print(f"[embed] loaded projection {code.kind} d={code.d}", flush=True)
    else:
        with Timer("fit code"):
            code, hcode = fit_code(cfg, meta, model, load_stream(cfg["data"]["dir"], "train"),
                                   tables["windows_train"], T, device)
        torch.save({"code": code.state(), "hcode": None if hcode is None else hcode.state()}, proj_path)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        with Timer(f"embed {split}"):
            embed_split(cfg, meta, model, code, split, tables, device)


if __name__ == "__main__":
    main()
