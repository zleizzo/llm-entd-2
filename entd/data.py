"""Corpora -> uint16 token streams; fixed-window tables; training schedule.

Every corpus becomes a flat uint16 token stream (`train.bin`, `val.bin`) plus `meta.json`.
Windows are explicit: `windows_<split>.npy` holds the stream offset of each window; a window
holds T+1 tokens (T inputs, T shifted targets). Position j of window i has store index
i*T + j, context = window[:j+1], and actual next token = window[j+1].
"""
import json
import os
from urllib.request import urlretrieve

import numpy as np

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ----------------------------------------------------------------------------- streams

def load_meta(d):
    with open(os.path.join(d, "meta.json")) as f:
        return json.load(f)


def load_stream(d, split):
    return np.memmap(os.path.join(d, f"{split}.bin"), dtype=np.uint16, mode="r")


def prepare(cfg):
    d = cfg["data"]["dir"]
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "meta.json")) and os.path.exists(os.path.join(d, "train.bin")):
        return load_meta(d)
    kind = cfg["data"]["kind"]
    print(f"[data] preparing {kind} -> {d}", flush=True)
    if kind == "shakespeare_char":
        prepare_shakespeare_char(d)
    elif kind == "tinystories":
        prepare_tinystories(d, cfg)
    elif kind == "fineweb_edu":
        prepare_fineweb_edu(d, cfg)
    elif kind == "pythia_pile":
        prepare_pythia_pile(d, cfg)
    else:
        raise ValueError(f"unknown data kind {kind}")
    return load_meta(d)


def prepare_shakespeare_char(d):
    fp = os.path.join(d, "input.txt")
    if not os.path.exists(fp):
        urlretrieve(SHAKESPEARE_URL, fp)
    text = open(fp, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = len(ids)
    ids[: int(n * 0.9)].tofile(os.path.join(d, "train.bin"))
    ids[int(n * 0.9):].tofile(os.path.join(d, "val.bin"))
    meta = {"kind": "shakespeare_char", "vocab_size": len(chars), "vocab_real": len(chars),
            "itos": chars, "eot": None, "tokenizer": "char"}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)


def _write_bin_from_hf(dset, path, n_shards):
    total = int(np.sum(dset["len"], dtype=np.uint64))
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
    idx = 0
    for b in range(n_shards):
        batch = dset.shard(num_shards=n_shards, index=b, contiguous=True).with_format("numpy")
        ab = np.concatenate(batch["ids"]).astype(np.uint16)
        arr[idx: idx + len(ab)] = ab
        idx += len(ab)
        if b % 16 == 0:
            print(f"  shard {b}/{n_shards} ({idx/1e6:.1f}M tokens)", flush=True)
    arr.flush()
    return total


def prepare_tinystories(d, cfg):
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories")

    def process(ex):
        ids = enc.encode_ordinary(ex["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tok = ds.map(process, remove_columns=["text"], num_proc=cfg["data"]["num_proc"], desc="tokenizing")
    n_tr = _write_bin_from_hf(tok["train"], os.path.join(d, "train.bin"), 256)
    n_va = _write_bin_from_hf(tok["validation"], os.path.join(d, "val.bin"), 8)
    meta = {"kind": "tinystories", "vocab_size": 50304, "vocab_real": 50257, "tokenizer": "gpt2",
            "eot": enc.eot_token, "n_train": int(n_tr), "n_val": int(n_va)}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)


def prepare_fineweb_edu(d, cfg):
    """First n_train_tokens of FineWeb-Edu sample-10BT -> train.bin, next n_val_tokens -> val.bin.
    NOTE: Large-version only; not exercised by the MVP."""
    import tiktoken
    from datasets import load_dataset
    from multiprocessing import Pool

    enc = tiktoken.get_encoding("gpt2")
    n_tr = int(cfg["data"]["n_train_tokens"] or 5_000_000_000)
    n_va = int(cfg["data"]["n_val_tokens"] or 100_000_000)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    def tok(text):
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        return np.array(ids, dtype=np.uint16)

    targets = [("train.bin", n_tr), ("val.bin", n_va)]
    it = (ex["text"] for ex in ds)
    with Pool(cfg["data"]["num_proc"]) as pool:
        chunks = pool.imap(tok, it, chunksize=64)
        for fname, n in targets:
            arr = np.memmap(os.path.join(d, fname), dtype=np.uint16, mode="w+", shape=(n,))
            idx = 0
            while idx < n:
                ids = next(chunks)
                take = min(len(ids), n - idx)
                arr[idx: idx + take] = ids[:take]
                idx += take
                if (idx // 1_000_000) != ((idx - take) // 1_000_000) and (idx // 1_000_000) % 100 == 0:
                    print(f"  {fname}: {idx/1e6:.0f}M tokens", flush=True)
            arr.flush()
    meta = {"kind": "fineweb_edu", "vocab_size": 50304, "vocab_real": 50257, "tokenizer": "gpt2",
            "eot": enc.eot_token, "n_train": n_tr, "n_val": n_va}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)


def prepare_pythia_pile(d, cfg):
    """Slice of the preshuffled Pile in Pythia's exact training order (Large Option B).

    Expects cfg.data.pythia = {bin_prefix: path to pile_*_document (without .bin/.idx),
    mmap_module: path to a GPT-NeoX/pythia `mmap_dataset.py`, start_sample, n_samples,
    val_bin: path to a pre-tokenized Pile validation uint16 stream}.
    Each sample is 2049 tokens; windows_train.npy is written with stride 2049 (T = 2048).
    NOTE: Large-version only; not exercised by the MVP.
    """
    import importlib.util
    p = cfg["data"]["pythia"]
    spec = importlib.util.spec_from_file_location("mmap_dataset", p["mmap_module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ds = mod.MMapIndexedDataset(p["bin_prefix"], skip_warmup=True)
    s0, n = int(p["start_sample"]), int(p["n_samples"])
    T = 2048
    arr = np.memmap(os.path.join(d, "train.bin"), dtype=np.uint16, mode="w+", shape=(n * (T + 1),))
    for i in range(n):
        sample = np.asarray(ds[s0 + i], dtype=np.uint16)
        assert len(sample) == T + 1, len(sample)
        arr[i * (T + 1): (i + 1) * (T + 1)] = sample
    arr.flush()
    np.save(os.path.join(d, "windows_train.npy"), np.arange(n, dtype=np.int64) * (T + 1))
    np.save(os.path.join(d, "step_seen_train.npy"), np.arange(n, dtype=np.int64) + s0)  # sample index ~ step*1024
    val = np.memmap(p["val_bin"], dtype=np.uint16, mode="r")
    np.asarray(val).tofile(os.path.join(d, "val.bin"))
    meta = {"kind": "pythia_pile", "vocab_size": 50304, "vocab_real": 50277, "tokenizer": "gpt-neox",
            "eot": 0, "n_train": int(n * (T + 1)), "block_size": T, "stride": T + 1}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)


# ----------------------------------------------------------------------------- windows

def window_offsets(n_tokens, T, stride=None):
    stride = stride or T
    n_w = (n_tokens - 1 - T) // stride + 1
    return np.arange(n_w, dtype=np.int64) * stride


def ensure_tables(cfg, meta):
    """Window tables and the trained/untrained split, persisted in work_dir."""
    wd, d = cfg["work_dir"], cfg["data"]["dir"]
    T = cfg["data"]["block_size"]
    out = {}
    for split in ("train", "val"):
        fp = os.path.join(wd, f"windows_{split}.npy")
        if not os.path.exists(fp):
            pre = os.path.join(d, f"windows_{split}.npy")   # corpus-provided (pythia)
            if os.path.exists(pre):
                w = np.load(pre)
            else:
                w = window_offsets(len(load_stream(d, split)), T)
            np.save(fp, w)
        out[f"windows_{split}"] = np.load(fp)
    fp = os.path.join(wd, "split_train.npy")
    if not os.path.exists(fp):
        rng = np.random.default_rng(cfg["data"]["split_seed"])
        n_w = len(out["windows_train"])
        split = (rng.random(n_w) < cfg["data"]["untrained_frac"]).astype(np.int8)  # 1 = untrained store
        np.save(fp, split)
    out["split_train"] = np.load(fp)
    out["T"] = T
    return out


def make_schedule(trained_wids, n_windows, epochs, batch_size, seed):
    """(n_steps, batch_size) window ids, and last step at which each window was seen (-1 = never)."""
    rng = np.random.default_rng(seed)
    steps = []
    for _ in range(epochs):
        perm = rng.permutation(trained_wids)
        n_b = len(perm) // batch_size
        steps.append(perm[: n_b * batch_size].reshape(n_b, batch_size))
    sched = np.concatenate(steps).astype(np.int64)
    return sched, last_seen_at(sched, len(sched), n_windows)


def last_seen_at(sched, step, n_windows):
    """Last step < `step` at which each window was seen (-1 = not yet), for a checkpoint taken after `step` steps."""
    last_seen = np.full(n_windows, -1, dtype=np.int64)
    for s in range(min(step, len(sched))):
        last_seen[sched[s]] = s
    return last_seen


def ensure_schedule(cfg, tables):
    wd = cfg["work_dir"]
    fp = os.path.join(wd, "schedule.npy")
    if not os.path.exists(fp):
        trained = np.nonzero(tables["split_train"] == 0)[0]
        sched, last = make_schedule(trained, len(tables["windows_train"]), cfg["train"]["epochs"],
                                    cfg["train"]["batch_size"], cfg["train"]["seed"])
        np.save(fp, sched)
        np.save(os.path.join(wd, "last_seen_step.npy"), last)
    return np.load(fp), np.load(os.path.join(wd, "last_seen_step.npy"))
