"""scripts/aggregate_results.py — result tables from real results.csv files only (nothing is filled in).
Usage: python scripts/aggregate_results.py --out results_tables.md \
         A0=runs/a0/results.csv  OFF=runs/dga_off/results.csv  DEPTH=runs/dga_depth/results.csv  SHUFFLE=...
First label is the reference baseline; every other label is compared with it AND with the label given by --control (default: OFF), seed by seed."""
import argparse
import csv
from collections import defaultdict

import numpy as np


def load(path):
    d = defaultdict(dict)                       # d[K][seed] = (novel_AP50, base_AP50)
    for r in csv.DictReader(open(path)):
        d[int(r["K"])][int(r["seed"])] = (float(r["novel_AP50"]), float(r["base_AP50"]))
    return d


def ms(v):
    return "—" if not v else (f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f} (n=1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=path/to/results.csv")
    ap.add_argument("--out", default="results_tables.md")
    ap.add_argument("--control", default="OFF", help="label of the fairness control (continued A0, no geometry)")
    a = ap.parse_args()
    data = {}
    for x in a.runs:
        lab, p = x.split("=", 1)
        data[lab] = load(p)
    labels = list(data)
    ks = sorted({k for d in data.values() for k in d})
    L = ["## Novel-class AP50 (mean ± std over seeds; n = number of seeds)\n", "| K | " + " | ".join(labels) + " |",
         "|---|" + "---|" * len(labels)]
    for k in ks:
        L.append(f"| {k} | " + " | ".join(ms([v[0] for v in data[lab].get(k, {}).values()]) for lab in labels) + " |")
    L += ["", "## Base-class AP50 (forgetting check)\n", "| K | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for k in ks:
        L.append(f"| {k} | " + " | ".join(ms([v[1] for v in data[lab].get(k, {}).values()]) for lab in labels) + " |")
    L += ["", "## Paired difference in novel AP50 (same seed, same K-shot support set)\n",
          "| K | comparison | mean Δ | seeds won / n | per-seed Δ |", "|---|---|---|---|---|"]
    ref = labels[0]
    pairs = [(l, ref) for l in labels[1:]]
    if a.control in data:
        pairs += [(l, a.control) for l in labels if l not in (a.control, ref)]
    for k in ks:
        for x, y in pairs:
            seeds = sorted(set(data[x].get(k, {})) & set(data[y].get(k, {})))
            if not seeds:
                continue
            dl = [data[x][k][s][0] - data[y][k][s][0] for s in seeds]
            L.append(f"| {k} | {x} − {y} | {np.mean(dl):+.2f} | {sum(v > 0 for v in dl)}/{len(dl)} | "
                     + ", ".join(f"{v:+.2f}" for v in dl) + " |")
    L += ["", "_With few seeds these are descriptive, not significance tests; a gain smaller than the seed std is not evidence._"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
