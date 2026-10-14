#!/usr/bin/env python3
"""Generate SciPy float64 reference results for every matrix in data/.

For each <name>.mtx we write, into data/ref/:
  <name>.x.npy / .y.npy / .B.npy / .Y.npy   (NumPy, for Python use)
  <name>.x.bin / .y.bin / .B.bin / .Y.bin   (float64 little-endian, row-major)
  <name>.json                               (flat header read by the C++ side)

where y = A x and Y = A B with a deterministic, mixed-magnitude x (values
spanning ~1e-6..1e6 to expose cancellation) and a seeded dense B with k=32.

Run:
    python3 tools/reference.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REF_DIR = os.path.join(DATA_DIR, "ref")

K = 32
X_SEED = 7
B_SEED = 11


def make_x(n: int) -> np.ndarray:
    """Deterministic vector with magnitudes spanning ~1e-6 .. 1e6 and random signs."""
    rng = np.random.default_rng(X_SEED)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    exp = rng.uniform(-6.0, 6.0, size=n)
    mag = 10.0 ** exp
    return (sign * mag).astype(np.float64)


def make_B(n: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(B_SEED)
    return (rng.uniform(-1.0, 1.0, size=(n, k))).astype(np.float64)


def write_bin(path: str, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype="<f8").tofile(path)


def process(mtx_path: str) -> None:
    name = os.path.splitext(os.path.basename(mtx_path))[0]
    A = scipy.io.mmread(mtx_path)
    if not sp.issparse(A):
        A = sp.coo_matrix(np.atleast_2d(np.asarray(A)))
    A = A.tocsr().astype(np.float64)
    A.sum_duplicates()
    A.sort_indices()
    m, n = A.shape
    nnz = int(A.nnz)

    x = make_x(n)
    B = make_B(n, K)
    y = A.dot(x).astype(np.float64).ravel()
    Y = np.asarray(A.dot(B), dtype=np.float64)

    # NOTE: file names must not differ only by case - macOS/Windows are
    # case-insensitive filesystems, where y.bin and Y.bin would collide.
    os.makedirs(REF_DIR, exist_ok=True)
    np.save(os.path.join(REF_DIR, f"{name}.x.npy"), x)
    np.save(os.path.join(REF_DIR, f"{name}.y.npy"), y)
    np.save(os.path.join(REF_DIR, f"{name}.bmat.npy"), B)
    np.save(os.path.join(REF_DIR, f"{name}.ymat.npy"), Y)

    write_bin(os.path.join(REF_DIR, f"{name}.x.bin"), x)
    write_bin(os.path.join(REF_DIR, f"{name}.y.bin"), y)
    write_bin(os.path.join(REF_DIR, f"{name}.bmat.bin"), B)
    write_bin(os.path.join(REF_DIR, f"{name}.ymat.bin"), Y)

    header = {
        "name": name,
        "m": int(m),
        "n": int(n),
        "k": int(K),
        "nnz": nnz,
        "dtype": "float64",
        "endian": "little",
        "layout": "row-major",
        "x": f"{name}.x.bin",
        "y": f"{name}.y.bin",
        "bmat": f"{name}.bmat.bin",
        "ymat": f"{name}.ymat.bin",
    }
    with open(os.path.join(REF_DIR, f"{name}.json"), "w", encoding="ascii") as f:
        json.dump(header, f, indent=1)
    print(f"[ref] {name:24s} m={m:7d} n={n:7d} nnz={nnz:9d} k={K}")


def main() -> int:
    mtxs = sorted(glob.glob(os.path.join(DATA_DIR, "*.mtx")))
    if not mtxs:
        print("no .mtx files in data/; run tools/gen_matrices.py first", file=sys.stderr)
        return 1
    for p in mtxs:
        process(p)
    print("[ref] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
