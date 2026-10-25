"""On-disk embedding store: one row per corpus position (store index = window_id * T + offset)."""
import json
import os

import numpy as np
import torch

DTYPES = {"float32": np.float32, "float16": np.float16, "int8": np.int8}


class Store:
    def __init__(self, path, mode="r"):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.N, self.d, self.T = self.meta["N"], self.meta["d"], self.meta["T"]
        self.dtype = self.meta["dtype"]
        mm = "r+" if mode == "r+" else "r"
        self.z = np.load(os.path.join(path, "z.npy"), mmap_mode=mm)
        self.znorm2 = np.load(os.path.join(path, "znorm2.npy"), mmap_mode=mm)
        self.next = np.load(os.path.join(path, "next.npy"), mmap_mode=mm)
        self.argmax = np.load(os.path.join(path, "argmax.npy"), mmap_mode=mm)
        self.entropy = np.load(os.path.join(path, "entropy.npy"), mmap_mode=mm)
        sc = self.meta.get("scale")
        self.scale = None if sc is None else torch.tensor(sc, dtype=torch.float32)

    @staticmethod
    def create(path, N, d, T, dtype, extra=None):
        os.makedirs(path, exist_ok=True)
        np.lib.format.open_memmap(os.path.join(path, "z.npy"), mode="w+", dtype=DTYPES[dtype], shape=(N, d))
        np.lib.format.open_memmap(os.path.join(path, "znorm2.npy"), mode="w+", dtype=np.float32, shape=(N,))
        np.lib.format.open_memmap(os.path.join(path, "next.npy"), mode="w+", dtype=np.uint16, shape=(N,))
        np.lib.format.open_memmap(os.path.join(path, "argmax.npy"), mode="w+", dtype=np.uint16, shape=(N,))
        np.lib.format.open_memmap(os.path.join(path, "entropy.npy"), mode="w+", dtype=np.float16, shape=(N,))
        meta = {"N": int(N), "d": int(d), "T": int(T), "dtype": dtype, "complete": False}
        meta.update(extra or {})
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
        return Store(path, mode="r+")

    def set_meta(self, **kw):
        self.meta.update(kw)
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(self.meta, f)

    def flush(self):
        for a in (self.z, self.znorm2, self.next, self.argmax, self.entropy):
            a.flush()

    def _decode(self, arr, device):
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        if self.dtype == "int8":
            return t.float() * self.scale.to(device)[None, :]
        return t.float()

    def z_chunk(self, a, b, device):
        return self._decode(self.z[a:b], device)

    def z_rows(self, idx, device):
        idx = np.asarray(idx, dtype=np.int64)
        order = np.argsort(idx)
        rows = self.z[idx[order]]
        out = np.empty_like(rows)
        out[order] = rows
        return self._decode(out, device)

    def window_of(self, pos):
        return np.asarray(pos) // self.T


class ArraySource:
    """In-memory / memmap (N, d) float source with the same accessor interface as Store."""

    def __init__(self, z, znorm2=None):
        self.z = z
        self.N, self.d = z.shape
        self.znorm2 = znorm2 if znorm2 is not None else (np.asarray(z, dtype=np.float32) ** 2).sum(1)
        self.dtype = str(z.dtype)
        self.scale = None

    def z_chunk(self, a, b, device):
        return torch.from_numpy(np.ascontiguousarray(self.z[a:b])).to(device).float()

    def z_rows(self, idx, device):
        return torch.from_numpy(np.ascontiguousarray(self.z[np.asarray(idx)])).to(device).float()
