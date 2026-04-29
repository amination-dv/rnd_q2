"""Write missing `std_normalized/qc/*.png` under ml_data_v4 dent trees from sibling `.npy`.

After `copy_dents_from_qc_json`, some rows may lack a source std QC PNG on disk; the
paired `d<basename>.npy` is still copied. This script renders each missing PNG from the
numpy matrix using the same visualization as `rnd.utils.data_utils.generate_images`
(inferno, vmin/vmax 0..1, transposed).

Usage:
    python complete_dent_std_qc_pngs.py
    python complete_dent_std_qc_pngs.py --root /path/to/ml_data_v4/dent
    python complete_dent_std_qc_pngs.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _qc_png_path_for_std_npy(std_npy: Path) -> Path:
    stem = std_npy.stem
    if stem.startswith("d"):
        stem = stem[1:]
    return std_npy.parent / "qc" / f"{stem}.png"


def _render_std_npy_to_png(std_npy: Path, png_out: Path) -> None:
    matrix = np.load(std_npy)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array in {std_npy}, got shape {matrix.shape}")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.imshow(
        matrix.T,
        cmap="inferno",
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(ax.images[0], ax=ax)
    ax.axis("off")
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "ml_data_v4" / "dent",
        help="Dent dataset root (contains <iid>/std_normalized/*.npy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list missing PNGs; do not write files.",
    )
    args = parser.parse_args()
    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    missing: list[Path] = []
    for std_npy in sorted(root.glob("*/std_normalized/*.npy")):
        png_out = _qc_png_path_for_std_npy(std_npy)
        if not png_out.is_file():
            missing.append(std_npy)

    print(f"Scanned std .npy under {root}")
    print(f"Missing std qc PNG: {len(missing)}")
    if args.dry_run:
        for p in missing[:30]:
            print(f"  would write {_qc_png_path_for_std_npy(p)}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")
        return 0

    errors: list[tuple[Path, str]] = []
    for std_npy in missing:
        png_out = _qc_png_path_for_std_npy(std_npy)
        try:
            _render_std_npy_to_png(std_npy, png_out)
        except Exception as e:  # noqa: BLE001
            errors.append((std_npy, str(e)))

    print(f"Wrote {len(missing) - len(errors)} PNGs")
    if errors:
        print(f"Failed: {len(errors)}", file=sys.stderr)
        for p, msg in errors[:20]:
            print(f"  {p}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
