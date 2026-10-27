import json
import os
import time

import numpy as np
import torch


def pick_device(name="auto"):
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        # TF32 matmuls for fp32 inference (embed/eval/rerank forward passes): ~8x faster on A100 than strict fp32,
        # and far finer than the bf16 autocast the model was trained with. Training uses autocast anyway.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device, dtype="auto"):
    """bf16 autocast on CUDA; fp32 elsewhere (MPS bf16 is slow / flaky for small models)."""
    if device.type == "cuda" and dtype in ("auto", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def gather_windows(stream, offsets, T):
    """stream: 1-D array-like of tokens; offsets: (B,) int64 -> (B, T+1) int64 numpy."""
    idx = np.asarray(offsets, dtype=np.int64)[:, None] + np.arange(T + 1, dtype=np.int64)[None, :]
    return np.asarray(stream[idx], dtype=np.int64)


def to_t(a, device, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] {time.time() - self.t0:.1f}s", flush=True)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
