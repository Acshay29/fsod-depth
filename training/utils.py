"""training/utils.py — seeding, YAML config with command-line overrides, device choice, atomic checkpoint saving."""
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path, overrides=()):
    """Load YAML; apply overrides like 'train.max_iter=5' (value parsed as YAML: 5, true, [1,2], cpu...)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, sep, val = item.partition("=")
        assert sep, f"override must look like a.b=value, got '{item}'"
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_atomic(obj, path):
    """Write to a temp file then rename, so a disconnect mid-save never corrupts the checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
