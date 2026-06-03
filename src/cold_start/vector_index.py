"""
Python binding for the C++ vector index (ctypes, no pybind11 needed).

If `vector_index/libvecindex.so` isn't built, transparently falls back to a numpy
implementation so the pipeline still runs (CI, machines without a compiler). Build
the fast path with: bash vector_index/build.sh
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_SO = Path(__file__).resolve().parents[2] / "vector_index" / "libvecindex.so"


class _CppIndex:
    def __init__(self, dim: int, lib):
        self.dim = dim
        self.lib = lib
        self.h = lib.vi_create(dim)

    def add(self, ids: np.ndarray, vecs: np.ndarray):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self.lib.vi_add_batch(
            self.h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(ids),
        )

    def search(self, query: np.ndarray, k: int):
        q = np.ascontiguousarray(query, dtype=np.float32)
        out_ids = np.empty(k, dtype=np.int64)
        out_sims = np.empty(k, dtype=np.float32)
        self.lib.vi_search(
            self.h,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k,
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_sims.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        mask = out_ids >= 0
        return out_ids[mask], out_sims[mask]

    def __len__(self):
        return self.lib.vi_size(self.h)


class _NumpyIndex:
    """Fallback. Same interface, cosine via normalised dot product."""

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[int] = []
        self._vecs: list[np.ndarray] = []

    def add(self, ids, vecs):
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)
        self._ids.extend(int(i) for i in ids)
        self._vecs.extend(vecs)

    def search(self, query, k):
        if not self._ids:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        M = np.vstack(self._vecs)
        q = np.asarray(query, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-12)
        sims = M @ q
        kk = min(k, len(sims))
        top = np.argsort(-sims)[:kk]
        return np.array([self._ids[i] for i in top]), sims[top].astype(np.float32)

    def __len__(self):
        return len(self._ids)


def _load_lib():
    if not _SO.exists():
        return None
    try:
        lib = ctypes.CDLL(str(_SO))
    except OSError:
        # .so exists but can't load here (e.g. Linux .so on Windows). Fall back.
        return None
    lib.vi_create.restype = ctypes.c_void_p
    lib.vi_create.argtypes = [ctypes.c_int]
    lib.vi_add_batch.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.vi_search.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float),
    ]
    lib.vi_size.argtypes = [ctypes.c_void_p]
    lib.vi_size.restype = ctypes.c_int
    return lib


# Resolve the backend ONCE at import. This is the source of truth -- the label
# reflects what actually loaded, not merely whether the .so file is present.
_LIB = _load_lib()


def VectorIndex(dim: int):
    """Factory: returns the C++ index if the .so loaded, else numpy fallback."""
    return _CppIndex(dim, _LIB) if _LIB is not None else _NumpyIndex(dim)


def backend_name() -> str:
    return "cpp (libvecindex.so)" if _LIB is not None else "numpy fallback"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = VectorIndex(8)
    vecs = rng.normal(size=(100, 8)).astype(np.float32)
    idx.add(np.arange(100), vecs)
    ids, sims = idx.search(vecs[7], k=3)
    print("backend:", backend_name())
    print("query item 7 -> nearest ids:", ids.tolist(), "sims:", np.round(sims, 3).tolist())
    assert ids[0] == 7, "nearest neighbour of a point should be itself"
    print("OK")
