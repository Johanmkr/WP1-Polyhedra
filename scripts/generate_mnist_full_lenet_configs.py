"""Generate per-(width, noise, seed) YAML configs for the Phase C
mnist_full LeNet sweep (planning §C.3).

Grid: 4 widths × 3 noise levels × 5 seeds = 60 configs.

Hyperparameters chosen to mirror the label-noise sweeps where possible
(save schedule, optimizer) and adapt where the CNN demands it
(batch_size 128 instead of 32, lr 0.01 instead of 0.001 — the smoke run
verified 97 % test-acc after epoch 0 at these settings).

Outputs land in `configs/mnist_full_lenet/n<noise>_<arch_label>/seed_<s>.yaml`
to mirror the existing `outputs/composite_label_noise/n<noise>_<arch>/seed_<s>.h5`
convention. Trained HDF5s will land in
`outputs/mnist_full_lenet/n<noise>_<arch_label>/seed_<s>.h5`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO / "configs" / "mnist_full_lenet"
OUTPUT_ROOT = REPO / "outputs" / "mnist_full_lenet"

# (label, conv_channels, fc_widths) — planning §C.1.
WIDTHS = [
    ("XS", (4, 8),   (60, 42)),
    ("S",  (6, 16),  (120, 84)),
    ("M",  (8, 24),  (180, 126)),
    ("L",  (12, 32), (240, 168)),
]
NOISE_LEVELS = [0.0, 0.2, 0.4]
SEEDS = [101, 102, 103, 104, 105]

EPOCHS = 101
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
SAVE_INTERVAL = 10
SAVE_EPOCHS = [0, 1, 2, 3, 4, 6, 8, 10]


def make_config(label: str, conv_channels, fc_widths,
                noise: float, seed: int) -> dict:
    arch_dir = f"n{noise}_LeNet-{label}"
    return {
        "experiment_name": arch_dir,
        "dataset": "mnist_full",
        "output_dir": str(OUTPUT_ROOT.relative_to(REPO)),
        "arch_type": "lenet5",
        "conv_channels": list(conv_channels),
        "fc_widths": list(fc_widths),
        "kernel_size": 5,
        "pool_size": 2,
        "global_seed": 42,
        "model_seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "SGD",
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "noise": noise,
        "test_size": 0.2,
        "save_interval": SAVE_INTERVAL,
        "save_epochs": SAVE_EPOCHS,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true",
                   help="Print one config path per line; do not write.")
    args = p.parse_args()

    paths: list[Path] = []
    for label, conv, fc in WIDTHS:
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                arch_dir = f"n{noise}_LeNet-{label}"
                cfg_dir = CONFIG_ROOT / arch_dir
                cfg_path = cfg_dir / f"seed_{seed}.yaml"
                if not args.list:
                    cfg_dir.mkdir(parents=True, exist_ok=True)
                    cfg = make_config(label, conv, fc, noise, seed)
                    with cfg_path.open("w") as fh:
                        yaml.safe_dump(cfg, fh, sort_keys=False)
                paths.append(cfg_path)

    if args.list:
        for p in paths:
            print(p)
    else:
        print(f"wrote {len(paths)} configs under {CONFIG_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
