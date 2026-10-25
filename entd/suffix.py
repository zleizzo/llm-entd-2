"""Suffix array over the REVERSED token stream: longest-suffix n-gram baseline and n-gram dedup.

Reversed stream r[i] = t[N-1-i]. A suffix of r starting at s matching (ctx[-1], ctx[-2], ..., ctx[-n])
means the original segment ending at e = N-1-s equals ctx[-n:], and the original next token is t[e+1].
Extending the pattern by one token *appends* in reversed space, so ranges nest and each n costs two
binary searches.
"""
import bisect

import numpy as np


def _sa_prefix_doubling(t):
    t = np.asarray(t, dtype=np.int64)
    n = len(t)
    rank = t.copy()
    k = 1
    sa = np.argsort(rank, kind="stable")
    while True:
        nxt = np.arange(n) + k
        r2 = np.where(nxt < n, rank[np.minimum(nxt, n - 1)], -1)
        key = rank * (n + 2) + (r2 + 1)
        sa = np.argsort(key, kind="stable")
        ks = key[sa]
        newrank = np.zeros(n, dtype=np.int64)
        newrank[sa[1:]] = np.cumsum(ks[1:] != ks[:-1])
        rank = newrank
        if rank.max() == n - 1:
            break
        k *= 2
    return sa


def build_sa(tokens):
    tokens = np.asarray(tokens)
    try:
        from pydivsufsort import divsufsort
        return np.asarray(divsufsort(np.ascontiguousarray(tokens.astype(np.uint16))), dtype=np.int64)
    except ImportError:
        return _sa_prefix_doubling(tokens)


class ReversedSuffixIndex:
    def __init__(self, tokens, sa=None):
        self.t = np.asarray(tokens)
        self.N = len(self.t)
        self.r = np.ascontiguousarray(self.t[::-1])
        self.sa = build_sa(self.r) if sa is None else sa

    # --- symbol at depth for a suffix (sentinel -1 past the end sorts first)
    def _sym(self, s, depth):
        j = s + depth
        return int(self.r[j]) if j < self.N else -1

    def _lower(self, lo, hi, depth, val):
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sym(int(self.sa[mid]), depth) < val:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _upper(self, lo, hi, depth, val):
        while lo < hi:
            mid = (lo + hi) // 2
            if self._sym(int(self.sa[mid]), depth) <= val:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def longest_suffix(self, ctx, max_n, min_count):
        """ctx: 1-D tokens (context ending at the query position, inclusive).
        Returns (n, lo, hi): longest n <= max_n with count >= min_count (n=0 -> whole array)."""
        lo, hi = 0, len(self.sa)
        best = (0, lo, hi)
        for n in range(1, min(max_n, len(ctx)) + 1):
            v = int(ctx[-n])
            lo2 = self._lower(lo, hi, n - 1, v)
            hi2 = self._upper(lo2, hi, n - 1, v)
            if hi2 - lo2 < min_count:
                break
            lo, hi = lo2, hi2
            best = (n, lo, hi)
        return best

    def match_positions(self, lo, hi):
        """Original end positions e of the matched segments and their next tokens (where defined)."""
        s = self.sa[lo:hi]
        e = self.N - 1 - s
        ok = e + 1 < self.N
        return e[ok], self.t[e[ok] + 1]

    def next_hist(self, lo, hi, V, pos_keep=None):
        e, nxt = self.match_positions(lo, hi)
        if pos_keep is not None:
            m = pos_keep[e]
            e, nxt = e[m], nxt[m]
        return np.bincount(nxt, minlength=V).astype(np.float64), len(nxt)


# ------------------------------------------------------------------ n-gram hashing (dedup)

_B = np.uint64(1000003)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def ngram_hashes(tokens, n):
    """Rolling polynomial hash (mod 2^64) of every n-gram; returns (len-n+1,) uint64."""
    t = np.asarray(tokens, dtype=np.uint64)
    with np.errstate(over="ignore"):
        H = np.zeros(len(t) + 1, dtype=np.uint64)
        acc = np.uint64(0)
        # cumulative prefix hash via loop-free doubling is awkward; use np.cumsum-free iteration in blocks
        for i in range(len(t)):
            acc = acc * _B + t[i]
            H[i + 1] = acc
        Bn = np.uint64(1)
        for _ in range(n):
            Bn = Bn * _B
        return H[n:] - H[:-n] * Bn


def ngram_hashes_fast(tokens, n):
    """Vectorized variant: hash of n-gram = sum_t tok[i+t] * B^(n-1-t) mod 2^64 via n shifted adds."""
    t = np.asarray(tokens, dtype=np.uint64)
    L = len(t) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = np.zeros(L, dtype=np.uint64)
        for k in range(n):
            h = h * _B + t[k:k + L]
    return h


class NgramSet:
    def __init__(self, tokens, n):
        self.n = n
        self.h = np.sort(ngram_hashes_fast(tokens, n))

    def contains_any(self, seq):
        hs = ngram_hashes_fast(seq, self.n)
        if len(hs) == 0:
            return False
        pos = np.searchsorted(self.h, hs)
        pos = np.minimum(pos, len(self.h) - 1)
        return bool(np.any(self.h[pos] == hs))
