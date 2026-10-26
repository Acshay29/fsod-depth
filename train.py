"""train.py — entry point. Usage: python train.py --config configs/base_a0_split1.yaml [--set key=value ...]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help='overrides, e.g. train.max_iter=5 "model.min_size=[256]"')
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    cfg["no_resume"] = a.no_resume
    if cfg["phase"] == "base":
        from training.base_trainer import train_base
        train_base(cfg)
    else:
        sys.exit(f"phase '{cfg['phase']}' is not implemented yet")


if __name__ == "__main__":
    main()
