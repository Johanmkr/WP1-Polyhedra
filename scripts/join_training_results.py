"""Join train/test accuracy + loss from each HDF5's `training_results/` group
into the aggregated CSVs, keyed by (dataset, arch, seed, noise/target_dim, epoch).

Reads:
  results/composite_label_noise_new_estimator.csv
  results/wbc_label_noise_new_estimator.csv
  results/mnist_capacity_new_estimator.csv

Writes (in place — copies kept as `*.bak`):
  + columns: train_acc, test_acc, train_loss, test_loss, gen_gap_acc, gen_gap_loss
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

CSV_TO_ROOT = {
    "composite_label_noise_new_estimator.csv": (REPO / "outputs" / "composite_label_noise", "noise_level"),
    "wbc_label_noise_new_estimator.csv": (REPO / "outputs" / "wbc_label_noise", "noise_level"),
    "mnist_capacity_new_estimator.csv": (REPO / "outputs" / "mnist_capacity", "target_dim"),
}


def _h5_path(root: Path, key_field: str, key_value, arch_str: str, seed: int) -> Path:
    if key_field == "noise_level":
        cfg = f"n{key_value}_{arch_str}"
    else:
        cfg = f"{int(key_value)}_dim_{arch_str}"
    return root / cfg / f"seed_{seed}.h5"


def _load_curves(h5: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5, "r") as f:
        tr = f["training_results"]
        return {
            "train_acc": tr["eval_train_accuracy"][:],  # eval-mode train acc (proper for gen gap)
            "test_acc": tr["test_accuracy"][:],
            "train_loss": tr["eval_train_loss"][:],
            "test_loss": tr["test_loss"][:],
        }


def join_one(csv_name: str) -> None:
    csv_path = REPO / "results" / csv_name
    root, key_field = CSV_TO_ROOT[csv_name]

    if not csv_path.exists():
        print(f"[skip] missing {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n{csv_name}: {len(df)} rows, key_field={key_field}")

    cache: dict[Path, dict[str, np.ndarray]] = {}

    new_cols = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    missing = 0

    for arch_str, key_val, seed, epoch in zip(
        df["arch_str"], df[key_field], df["seed"], df["epoch"]
    ):
        h5 = _h5_path(root, key_field, key_val, arch_str, int(seed))
        if h5 not in cache:
            try:
                cache[h5] = _load_curves(h5)
            except Exception as exc:
                print(f"[warn] {h5}: {exc!r}", file=sys.stderr)
                cache[h5] = None
        curves = cache[h5]
        e = int(epoch)
        if curves is None or e >= len(curves["train_acc"]):
            for k in new_cols:
                new_cols[k].append(np.nan)
            missing += 1
            continue
        for k in new_cols:
            new_cols[k].append(float(curves[k][e]))

    for k, v in new_cols.items():
        df[k] = v
    df["gen_gap_acc"] = df["train_acc"] - df["test_acc"]
    df["gen_gap_loss"] = df["test_loss"] - df["train_loss"]

    backup = csv_path.with_suffix(".csv.bak")
    if not backup.exists():
        csv_path.replace(backup)
        print(f"  backup → {backup.name}")
    else:
        print(f"  backup exists; overwriting {csv_path.name} only")
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path.name} with cols {list(new_cols)} + gen_gap_*")
    print(f"  missing: {missing} / {len(df)}")
    print(f"  unique HDF5s touched: {len(cache)}")


def main() -> int:
    for name in CSV_TO_ROOT:
        join_one(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
