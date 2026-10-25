import copy
import os

import yaml

DEFAULTS = {
    "name": "run",
    "work_dir": None,
    "data": {
        "kind": "shakespeare_char",   # shakespeare_char | tinystories | fineweb_edu | pythia_pile
        "dir": None,
        "block_size": 256,            # T: tokens of context per window; window holds T+1 tokens
        "untrained_frac": 0.1,        # fraction of train windows held out of training but embedded
        "split_seed": 0,
        "n_train_tokens": None,       # fineweb_edu only
        "n_val_tokens": None,
        "num_proc": 8,
        "pythia": {},                 # pythia_pile only (see data.py)
    },
    "model": {
        "kind": "gpt",                # gpt | hf
        "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.0, "bias": False,
        "hf_name": None,
    },
    "train": {
        "epochs": 1, "batch_size": 64, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 100,
        "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
        "eval_interval": 250, "eval_batches": 20, "log_interval": 10,
        "seed": 1337, "device": "auto", "dtype": "auto", "compile": False,
    },
    "embed": {
        "code": "pca",                # exact | pca | random
        "d": 128,
        "store_dtype": "float16",     # float32 | float16 | int8
        "h_subset_frac": 0.1,         # windows whose hidden states are stored
        "exact_sample": 20000,        # positions with full exact distributions stored
        "pca_sample_windows": 512,
        "batch_windows": 32,
        "int8_clip_quantile": 0.9999,
        "seed": 0,
        "device": "auto",
    },
    "eval": {
        "n_queries_heldout": 1500,
        "n_queries_train": 1500,
        "k_grid": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "K": 2048,
        "radii": [0.1, 0.2, 0.3, 0.5, 0.7],   # on d = ||sqrt p - sqrt q||_2
        "rerank_queries": 200,                 # per query set
        "rerank_K": 512,
        "coverage_fracs": [0.1, 0.01],
        "substore_frac": 0.1,
        "leak_L": 32,
        "dedup_ngram": 50,
        "ngram_max_n": 32,
        "ngram_min_counts": [5, 64],
        "null_draws": 200,
        "naive_null_draws": 20,
        "lambdas": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        "entropy_bins": 8,
        "fidelity_queries": 1000,
        "fidelity_d_sweep": [8, 16, 32, 64, 128],
        "query_batch": 128,
        "chunk": 262144,
        "seed": 0,
        "device": "auto",
    },
}


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_update(copy.deepcopy(DEFAULTS), user)
    cfg["_path"] = path
    cfg["work_dir"] = cfg.get("work_dir") or os.path.join("runs", cfg["name"])
    if cfg["data"]["dir"] is None:
        cfg["data"]["dir"] = os.path.join("data", cfg["data"]["kind"])
    os.makedirs(cfg["work_dir"], exist_ok=True)
    return cfg
