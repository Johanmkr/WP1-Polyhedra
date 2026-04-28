"""
Phase A.1 sweep runner: evaluate the MI baselines (binning, k-means, KSG,
InfoNCE, MINE-f) at the last-epoch / deepest hidden layer of every label-noise
HDF5, one CSV row per network.

Reuses :func:`run_label_noise_estimator.discover_jobs` for filename parsing
and the standard probe loaders. Per-HDF5 CSVs land at
``mi_baselines_seed_<seed>.csv`` next to the HDF5 (resumable: the script
skips files whose CSV already exists). ``--aggregate`` concatenates them
into ``results/mi_baselines.csv``.

Examples
--------
Smoke run on a single composite HDF5 with the cheap baselines only::

    uv run python scripts/run_mi_baselines.py --datasets composite --limit 1 \\
        --skip-mine --skip-infonce

Full sweep::

    uv run python scripts/run_mi_baselines.py --datasets both

Aggregate::

    uv run python scripts/run_mi_baselines.py --aggregate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from run_label_noise_estimator import DATASETS, discover_jobs  # noqa: E402
from src_experiment.baselines.activations import load_layer_activations  # noqa: E402
from src_experiment.baselines.mi_baselines import (  # noqa: E402
    InfoNCEEstimator,
    MINEEstimator,
    binning_mi,
    kmeans_mi,
    ksg_mi,
)
from src_experiment.probe_loader import (  # noqa: E402
    ProbeBundle,
    make_composite_probe,
    make_wbc_probe,
)


# Sweep grids (planning §A.1.1).
BINNING_BINS: Tuple[int, ...] = (2, 4, 8, 16, 30)
KSG_KS: Tuple[int, ...] = (3, 5, 10)
KMEANS_FIXED_K: Tuple[int, ...] = (16, 64, 256)
MINE_SEEDS_DEFAULT = 5
INFONCE_SEEDS_DEFAULT = 3
SUBSAMPLE_N_DEFAULT = 5_000


# ---------------------------------------------------------------------------
# Probe resolution
# ---------------------------------------------------------------------------
def _resolve_probe(dataset: str) -> ProbeBundle:
    if dataset == "composite":
        return make_composite_probe()  # N=20k, cached
    if dataset == "wbc":
        return make_wbc_probe(mode="full")  # N=569
    raise ValueError(f"unknown dataset {dataset!r}")


# ---------------------------------------------------------------------------
# HDF5 introspection
# ---------------------------------------------------------------------------
def _h5_summary(h5_path: Path) -> dict:
    """Read architecture, last epoch, num_classes, network_id from one HDF5."""
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["metadata"].attrs)
        arch = list(attrs.get("architecture", []))
        epochs = sorted(
            int(k.split("_")[1]) for k in f["epochs"].keys() if k.startswith("epoch_")
        )
    return {
        "architecture": arch,
        "deepest_layer": len(arch),  # 1-indexed last hidden
        "last_epoch": int(epochs[-1]),
        "num_classes": int(attrs.get("inferred_num_classes", 0)),
        "network_id": str(attrs.get("experiment_name", h5_path.stem)),
    }


def _crossref_existing(h5_path: Path, epoch: int, layer: int) -> dict:
    """Pick `bits_ours_raw`/`bits_ours_func`/`rho`/`rho_func` from the
    existing ``new_estimator_<seed>.csv`` if one exists. Returns empty dict
    if not. Joins on (epoch, layer); for the functional column we take the
    smallest non-zero ε row."""
    csv = h5_path.with_name(f"new_estimator_{h5_path.stem}.csv")
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    sub = df[(df["epoch"] == epoch) & (df["layer"] == layer)]
    if sub.empty:
        return {}
    raw = sub.iloc[0]
    out = {
        "bits_ours_raw": float(raw["miller_madow_bits"]),
        "rho": float(raw["rho"]),
    }
    func_rows = sub[sub["epsilon"] > 0]
    if not func_rows.empty:
        f = func_rows.sort_values("epsilon").iloc[0]
        out["bits_ours_func"] = float(f["miller_madow_func_bits"])
        out["rho_func"] = float(f["rho_func"])
        out["epsilon_for_ours_func"] = float(f["epsilon"])
    return out


# ---------------------------------------------------------------------------
# Per-HDF5 evaluation
# ---------------------------------------------------------------------------
def evaluate_one(
    job: dict,
    probe: ProbeBundle,
    args: argparse.Namespace,
) -> pd.DataFrame:
    h5_path: Path = job["h5"]
    info = _h5_summary(h5_path)
    epoch = info["last_epoch"]
    layer = info["deepest_layer"]
    num_classes = info["num_classes"]

    X = probe.X_probe.astype(np.float32, copy=False)
    Y = probe.y_probe.astype(np.int64, copy=False)
    N = X.shape[0]

    # Subsample (deterministic per HDF5) if probe larger than the cap.
    if N > args.subsample_n:
        rng = np.random.default_rng(int(job["seed"]))
        idx = rng.choice(N, size=args.subsample_n, replace=False)
        X = X[idx]
        Y = Y[idx]
        N_eff = args.subsample_n
    else:
        N_eff = N

    T = load_layer_activations(h5_path, epoch, layer, X, kind="pre")
    d_T = T.shape[1]

    row: dict = {
        "network_id": info["network_id"],
        "dataset": job["dataset"],
        "noise_level": job["noise"],
        "arch_str": job["arch"],
        "seed": int(job["seed"]),
        "epoch": int(epoch),
        "layer": int(layer),
        "d_T": int(d_T),
        "num_classes": int(num_classes),
        "probe_N": int(N),
        "N_used": int(N_eff),
    }

    # Binning sweep
    t0 = time.perf_counter()
    for nb in BINNING_BINS:
        out = binning_mi(T, Y, n_bins=nb, num_classes=num_classes)
        row[f"bits_binning_{nb}"] = out["bits"]
        row[f"R_binning_{nb}"] = out["num_regions"]
    row["wall_seconds_binning"] = time.perf_counter() - t0

    # K-means sweep — class-relative settings + fixed grid
    kmeans_settings: List[Tuple[str, int]] = [
        ("KY", num_classes),
        ("2KY", 2 * num_classes),
        ("4KY", 4 * num_classes),
    ]
    seen = {num_classes, 2 * num_classes, 4 * num_classes}
    for K in KMEANS_FIXED_K:
        if K not in seen:
            kmeans_settings.append((str(K), K))
            seen.add(K)
    t0 = time.perf_counter()
    for label, K in kmeans_settings:
        out = kmeans_mi(T, Y, K=K, seed=int(job["seed"]), num_classes=num_classes)
        row[f"bits_kmeans_K{label}"] = out["bits"]
        row[f"R_kmeans_K{label}"] = out["num_regions"]
    row["wall_seconds_kmeans"] = time.perf_counter() - t0

    # KSG sweep
    t0 = time.perf_counter()
    for k in KSG_KS:
        out = ksg_mi(T, Y, k=k)
        row[f"bits_ksg_k{k}"] = out["bits"]
    row["wall_seconds_ksg"] = time.perf_counter() - t0

    # InfoNCE
    if not args.skip_infonce:
        t0 = time.perf_counter()
        nce_bits = []
        nce = InfoNCEEstimator(
            n_iter=args.infonce_iter, batch=args.batch, device=args.device
        )
        for s in range(args.infonce_seeds):
            nce_bits.append(nce.estimate(T, Y, num_classes, seed=s)["bits"])
        row["bits_infonce_mean"] = float(np.mean(nce_bits))
        row["bits_infonce_std"] = float(np.std(nce_bits))
        row["wall_seconds_infonce"] = time.perf_counter() - t0
        row["infonce_seeds"] = args.infonce_seeds

    # MINE
    if not args.skip_mine:
        t0 = time.perf_counter()
        mine_bits = []
        mine = MINEEstimator(
            n_iter=args.mine_iter,
            batch=args.batch,
            device=args.device,
        )
        for s in range(args.mine_seeds):
            mine_bits.append(mine.estimate(T, Y, num_classes, seed=s)["bits"])
        row["bits_mine_mean"] = float(np.mean(mine_bits))
        row["bits_mine_std"] = float(np.std(mine_bits))
        row["wall_seconds_mine"] = time.perf_counter() - t0
        row["mine_seeds"] = args.mine_seeds

    # Cross-ref to existing new-estimator CSV (no-op if missing).
    row.update(_crossref_existing(h5_path, epoch, layer))

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _csv_path_for(h5_path: Path) -> Path:
    return h5_path.with_name(f"mi_baselines_{h5_path.stem}.csv")


def run_jobs(jobs: List[dict], args: argparse.Namespace) -> None:
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        print("nothing to do")
        return

    # Build probes once per dataset (cached).
    probes: dict[str, ProbeBundle] = {}

    total = len(jobs)
    cum = 0.0
    done = 0
    for i, job in enumerate(jobs, start=1):
        tag = f"{job['dataset']}/n{job['noise']}/{job['arch']}/seed_{job['seed']}"
        out_csv = _csv_path_for(job["h5"])
        if out_csv.exists() and not args.force:
            print(f"[{i}/{total}] skip (exists): {tag}")
            continue
        if job["dataset"] not in probes:
            probes[job["dataset"]] = _resolve_probe(job["dataset"])
            print(f"  probe[{job['dataset']}]: {probes[job['dataset']].note}")

        print(f"[{i}/{total}] running: {tag}")
        t0 = time.perf_counter()
        try:
            df = evaluate_one(job, probes[job["dataset"]], args)
        except Exception as exc:
            print(f"  -> FAILED: {exc!r}", file=sys.stderr)
            continue
        df.to_csv(out_csv, index=False)
        dt = time.perf_counter() - t0
        cum += dt
        done += 1
        avg = cum / done
        eta = avg * (total - i)
        print(
            f"  -> {dt:6.2f}s   cum={cum/60:.1f}min   "
            f"ETA {eta/60:.1f} min over {total - i} remaining"
        )

    print(f"\nfinished: {done} ran, {total - done} skipped/failed.")


def aggregate(output: Path) -> None:
    frames: List[pd.DataFrame] = []
    for ds_root in DATASETS.values():
        for csv in ds_root.glob("*/mi_baselines_seed_*.csv"):
            try:
                frames.append(pd.read_csv(csv))
            except Exception as exc:
                print(f"[warn] failed to read {csv}: {exc}", file=sys.stderr)
    if not frames:
        print("no per-HDF5 CSVs found; nothing to aggregate")
        return
    out = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"wrote {len(out)} rows to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS) + ["both"],
        default=["both"],
    )
    p.add_argument("--noise", nargs="+", type=float, default=None)
    p.add_argument("--archs", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "mi_baselines.csv",
    )

    p.add_argument("--subsample-n", type=int, default=SUBSAMPLE_N_DEFAULT,
                   help="cap probe size when running estimators (default 5000)")
    p.add_argument("--skip-mine", action="store_true")
    p.add_argument("--skip-infonce", action="store_true")
    p.add_argument("--mine-seeds", type=int, default=MINE_SEEDS_DEFAULT)
    p.add_argument("--mine-iter", type=int, default=2000)
    p.add_argument("--infonce-seeds", type=int, default=INFONCE_SEEDS_DEFAULT)
    p.add_argument("--infonce-iter", type=int, default=1000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = p.parse_args(argv)

    if args.aggregate:
        aggregate(args.output)
        return 0

    datasets = sorted(DATASETS) if "both" in args.datasets else sorted(set(args.datasets))
    jobs = discover_jobs(
        datasets=datasets,
        noise_filter=args.noise,
        arch_filter=args.archs,
        seed_filter=args.seeds,
    )
    print(f"discovered {len(jobs)} jobs across {datasets}")
    if args.list:
        for j in jobs:
            print(f"  {j['dataset']}/n{j['noise']}/{j['arch']}/seed_{j['seed']}")
        return 0

    run_jobs(jobs, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
