#!/usr/bin/env python3
"""Check value ranges in .npy files without requiring numpy (parses v1/v2 .npy format)."""
from __future__ import annotations

import ast
import struct
import sys
from pathlib import Path


def _read_npy_header(f) -> tuple[str, tuple[int, ...], int]:
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("Not a .npy file")
    major, minor = struct.unpack("BB", f.read(2))
    if major == 1:
        hlen = struct.unpack("<H", f.read(2))[0]
    elif major == 2:
        hlen = struct.unpack("<I", f.read(4))[0]
    else:
        raise ValueError(f"Unsupported numpy format version {major}.{minor}")
    header = f.read(hlen).decode("latin1")
    # Header padded so offset from file start is multiple of 16
    consumed = 6 + 2 + (2 if major == 1 else 4) + hlen
    pad = (16 - (consumed % 16)) % 16
    if pad:
        f.read(pad)
    meta = ast.literal_eval(header.strip())
    descr: str = meta["descr"]
    shape = meta["shape"]
    if not isinstance(shape, tuple):
        shape = (int(shape),)
    else:
        shape = tuple(int(x) for x in shape)
    itemsize = int(descr[2:]) if len(descr) > 2 and descr[2:].isdigit() else 4
    if descr == "<f4" or descr == "|f4":
        itemsize = 4
    elif descr == "<f8" or descr == "|f8":
        itemsize = 8
    else:
        # fallback: try to infer from suffix
        if "f4" in descr:
            itemsize = 4
        elif "f8" in descr:
            itemsize = 8
        else:
            raise ValueError(f"Unsupported descr {descr!r} (need float32/64)")
    return descr, shape, itemsize


def npy_min_max(path: Path) -> tuple[float, float]:
    with path.open("rb") as f:
        descr, shape, itemsize = _read_npy_header(f)
        n = 1
        for s in shape:
            n *= s
        fmt = "<f" if itemsize == 4 else "<d"
        chunk_elems = 256 * 1024
        gmin, gmax = float("inf"), float("-inf")
        remaining = n
        while remaining > 0:
            take = min(chunk_elems, remaining)
            nbytes = take * itemsize
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError("Unexpected EOF")
            vals = struct.unpack(f"{take}{fmt[1]}", raw)
            for v in vals:
                if v != v:  # nan
                    continue
                if v < gmin:
                    gmin = v
                if v > gmax:
                    gmax = v
            remaining -= take
        if gmin == float("inf"):
            gmin, gmax = 0.0, 0.0
        return gmin, gmax


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "ml_data_v4" / "dent"
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    paths = sorted(root.rglob("*.npy"))
    # Skip augmented copies for "base" stats, but also report them separately
    base = [p for p in paths if "_aug_noise" not in p.stem and "_aug_scale_" not in p.stem]
    sample = base[: min(250, len(base))]
    if not sample:
        print(f"No base .npy under {root}")
        return
    mins, maxs = [], []
    for p in sample:
        lo, hi = npy_min_max(p)
        mins.append(lo)
        maxs.append(hi)
    import statistics

    print(f"Root: {root}")
    print(f"Sampled {len(sample)} base dent .npy files (skipped _aug_noise / _aug_scale_)")
    print(f"Per-file min: global_min={min(mins):.6g}, p1={statistics.quantiles(mins, n=100)[0]:.6g}, median={statistics.median(mins):.6g}")
    print(f"Per-file max: global_max={max(maxs):.6g}, p99={statistics.quantiles(maxs, n=100)[98]:.6g}, median={statistics.median(maxs):.6g}")
    n_below0 = sum(1 for x in mins if x < -1e-6)
    n_above1 = sum(1 for x in maxs if x > 1.0 + 1e-6)
    print(f"Files with min < 0: {n_below0} / {len(sample)}")
    print(f"Files with max > 1: {n_above1} / {len(sample)}")
    # Worst cases
    imin = mins.index(min(mins))
    imax = maxs.index(max(maxs))
    for label, idx in [("lowest min", imin), ("highest max", imax)]:
        p = sample[idx]
        lo, hi = npy_min_max(p)
        print(f"\n{label}: {p.name}")
        print(f"  min={lo:.6g}, max={hi:.6g}")
    print(
        "\nConclusion: clip(0,1) for intensity scaling is appropriate if most data lies in [0,1]. "
        "If many files exceed [0,1], rescale or use a wider clip in dataset_v4.ipynb."
    )


if __name__ == "__main__":
    main()
