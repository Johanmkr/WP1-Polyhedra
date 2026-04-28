"""
Phase A.2 sweep runner: rebuild each label-noise model from its HDF5 and
evaluate the four standard generalization-gap predictors at the last
epoch — Frobenius, path-norm, spectral margin, sharpness.

Mirrors :mod:`scripts.run_mi_baselines` (same job-discovery API,
resumable per-HDF5 CSVs, ``--aggregate`` step). One row per HDF5 emitted
to ``gen_gap_seed_<seed>.csv`` next to the .h5; aggregate writes
``results/gen_gap_predictors.csv``.

Examples
--------
Smoke run, single composite HDF5::

    uv run python scripts/run_gen_gap_predictors.py --datasets composite --limit 1

Skip sharpness (the slow predictor) for a quick sweep::

    uv run python scripts/run_gen_gap_predictors.py --skip-sharpness

Aggregate after::

    uv run python scripts/run_gen_gap_predictors.py --aggregate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from run_label_noise_estimator import DATASETS, discover_jobs  # noqa: E402
from src_experiment.baselines.gen_gap_predictors import (  # noqa: E402
    frobenius,
    load_neural_net_from_h5,
    path_norm,
    sharpness,
    spectral_margin,
)
from src_experiment.probe_loader import (  # noqa: E402
    ProbeBundle,
    make_composite_probe,
    make_wbc_probe,
)


# ---------------------------------------------------------------------------
# Probe resolution (mirrors run_mi_baselines)
# ---------------------------------------------------------------------------
def _resolve_probe(dataset: str) -> ProbeBundle:
    if dataset == "composite":
        return make_composite_probe()
    if dataset == "wbc":
        return make_wbc_probe(mode="full")
    raise ValueError(f"unknown dataset {dataset!r}")


# ---------------------------------------------------------------------------
# HDF5 introspection
# ---------------------------------------------------------------------------
def _h5_summary(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["metadata"].attrs)
        epochs = sorted(
            int(k.split("_")[1]) for k in f["epochs"].keys() if k.startswith("epoch_")
        )
        # Read final train/test loss & accuracy if present (training_results group).
        gen_gap = {}
        if "training_results" in f:
            tr = f["training_results"]
            for key in ("train_loss", "test_loss", "train_accuracy", "test_accuracy"):
                if key in tr:
                    arr = np.asarray(tr[key][:])
                    gen_gap[f"final_{key}"] = float(arr[-1])
    out = {
        "architecture": [int(x) for x in attrs.get("architecture", [])],
        "last_epoch": int(epochs[-1]),
        "num_classes": int(attrs.get("inferred_num_classes", 0)),
        "network_id": str(attrs.get("experiment_name", h5_path.stem)),
    }
    out.update(gen_gap)
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

    model = load_neural_net_from_h5(h5_path, epoch, device=args.device)
    X = np.asarray(probe.X_probe, dtype=np.float32)
    Y = np.asarray(probe.y_probe, dtype=np.int64)

    row: dict = {
        "network_id": info["network_id"],
        "dataset": job["dataset"],
        "noise_level": job["noise"],
        "arch_str": job["arch"],
        "seed": int(job["seed"]),
        "epoch": int(epoch),
        "num_classes": int(info["num_classes"]),
        "probe_N": int(X.shape[0]),
    }
    for k in ("final_train_loss", "final_test_loss", "final_train_accuracy", "final_test_accuracy"):
        if k in info:
            row[k] = info[k]
    if "final_train_loss" in row and "final_test_loss" in row:
        row["gen_gap_loss"] = row["final_test_loss"] - row["final_train_loss"]
    if "final_train_accuracy" in row and "final_test_accuracy" in row:
        row["gen_gap_acc"] = row["final_train_accuracy"] - row["final_test_accuracy"]

    # 1. Frobenius
    fr = frobenius(model)
    row["frobenius"] = fr["frobenius"]
    row["wall_frobenius"] = fr["wall"]

    # 2. Path-norm
    pn = path_norm(model)
    row["path_norm"] = pn["path_norm"]
    row["log_path_norm"] = pn["log_path_norm"]
    row["wall_path_norm"] = pn["wall"]

    # 3. Spectral margin
    sm = spectral_margin(model, X, Y, device=args.device)
    row["gamma_min"] = sm["gamma_min"]
    row["gamma_mean"] = sm["gamma_mean"]
    row["spectral_prod"] = sm["spectral_prod"]
    row["spectral_margin_ratio"] = sm["spectral_margin_ratio"]
    row["wall_spectral_margin"] = sm["wall"]

    # 4. Sharpness
    if not args.skip_sharpness:
        sh_seeds = []
        sh_lmax: List[float] = []
        sh_top: List[float] = []
        wall = 0.0
        converged_all = True
        for s in range(args.sharpness_seeds):
            sh = sharpness(
                model, X, Y,
                k=args.sharpness_k,
                n_subsample=args.sharpness_subsample,
                seed=s,
                device=args.device,
            )
            sh_lmax.append(sh["lambda_max"])
            sh_top.append(sh["lambda_sum_top"])
            wall += sh["wall"]
            converged_all &= sh["converged"]
            sh_seeds.append(s)
        row["lambda_max_median"] = float(np.median(sh_lmax))
        row["lambda_max_iqr"] = float(np.subtract(*np.percentile(sh_lmax, [75, 25])))
        row["lambda_sum_top_median"] = float(np.median(sh_top))
        row["sharpness_seeds"] = args.sharpness_seeds
        row["sharpness_subsample"] = args.sharpness_subsample
        row["sharpness_k"] = args.sharpness_k
        row["sharpness_converged"] = bool(converged_all)
        row["wall_sharpness"] = wall

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _csv_path_for(h5_path: Path) -> Path:
    return h5_path.with_name(f"gen_gap_{h5_path.stem}.csv")


def run_jobs(jobs: List[dict], args: argparse.Namespace) -> None:
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        print("nothing to do")
        return

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
        for csv in ds_root.glob("*/gen_gap_seed_*.csv"):
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
    p.add_argument("--datasets", nargs="+",
                   choices=sorted(DATASETS) + ["both"], default=["both"])
    p.add_argument("--noise", nargs="+", type=float, default=None)
    p.add_argument("--archs", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--output", type=Path,
                   default=REPO / "results" / "gen_gap_predictors.csv")

    p.add_argument("--skip-sharpness", action="store_true")
    p.add_argument("--sharpness-seeds", type=int, default=3)
    p.add_argument("--sharpness-subsample", type=int, default=1024)
    p.add_argument("--sharpness-k", type=int, default=5)
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
