"""scripts/run_fewshot_sweep.py — fine-tune + evaluate over seeds x shots; resumable; appends to results.csv.
Example (A0):  python scripts/run_fewshot_sweep.py --config configs/fewshot_a0.yaml --base-ckpt <model_final.pth> --out-dir <dir>
Example (DGA / controls): add  --geo-ckpt <geo_final.pth>  and use configs/fewshot_dga.yaml
  (--base-ckpt is still the A0 base model; the geo checkpoint carries the geometry modules + base predictor rows)"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.inference import evaluate_model
from fsod_data.voc_fsod import TEST_SETS, VOC_CLASSES, load_records
from models.fsod.frcnn import count_params
from training.finetune import finetune
from training.finetune_geo import finetune_geo
from training.utils import load_config, pick_device

COLS = ["experiment", "split", "K", "seed", "novel_AP50", "base_AP50", "all_AP50", "novel_AP", "base_AP",
        "all_AP", "novel_AP75", "base_AP75", "all_AP75", "ms_per_image", "params_M", "ft_iters",
        "ft_seconds", "ft_final_loss", "lr", "test_images"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--geo-ckpt", default=None, help="geo_final.pth -> DGA variant or 'continued A0' control")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--limit-test", type=int, default=0, help="evaluate on the first N test images only (testing)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    P = cfg["paths"]
    recs = load_records(P["voc_root"], TEST_SETS, cache=f'{P["cache"]}/voc_test.json')
    if a.limit_test:
        recs = recs[: a.limit_test]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    if not csv_path.exists():
        csv_path.write_text(",".join(COLS) + "\n")

    for seed in a.seeds:
        for k in a.shots:
            tag = f'{cfg["experiment"]}_seed{seed}_{k}shot'
            run = out / tag
            if (run / "eval.json").exists():
                print(f"[skip] {tag} already done")
                continue
            print(f"[run] {tag}")
            if a.geo_ckpt:
                model, info, need_depth = finetune_geo(cfg, a.base_ckpt, a.geo_ckpt, seed, k, run, device)
            else:
                model, info = finetune(cfg, a.base_ckpt, seed, k, run, device)
                need_depth = False
            res = evaluate_model(model, VOC_CLASSES, cfg["split"], recs, P["voc_root"], device,
                                 batch_size=8, workers=cfg["data"]["num_workers"],
                                 depth_dir=cfg["data"]["depth_dir"] if need_depth else None)
            total, _ = count_params(model)
            res.update({"params_M": total / 1e6, "finetune": info, "K": k, "seed": seed,
                        "geo_ckpt": a.geo_ckpt, "geo_mode": info.get("geo_mode", "rgb-only")})
            (run / "eval.json").write_text(json.dumps(res, indent=2))
            row = {"experiment": cfg["experiment"], "split": cfg["split"], "K": k, "seed": seed,
                   "ms_per_image": f'{res["ms_per_image"]:.1f}', "params_M": f'{res["params_M"]:.2f}',
                   "ft_iters": info["iters"], "ft_seconds": f'{info["seconds"]:.0f}',
                   "ft_final_loss": f'{info["final_loss"]:.4f}', "lr": cfg["train"]["lr"],
                   "test_images": res["n_images"]}
            for g in ("novel", "base", "all"):
                for m in ("AP50", "AP", "AP75"):
                    row[f"{g}_{m}"] = f'{res[g][m]:.2f}'
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=COLS).writerow(row)
            print(f"   nAP50 {row['novel_AP50']}  bAP50 {row['base_AP50']}  allAP50 {row['all_AP50']}  "
                  f"(finetune {row['ft_seconds']}s, loss {row['ft_final_loss']})")

    rows = list(csv.DictReader(open(csv_path)))
    print("\nsummary (mean ± std over seeds):")
    for k in sorted({int(r["K"]) for r in rows if r["experiment"] == cfg["experiment"]}):
        v = [float(r["novel_AP50"]) for r in rows if r["experiment"] == cfg["experiment"] and int(r["K"]) == k]
        sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
        print(f"  K={k:<2} novel AP50 {np.mean(v):.2f} ± {sd:.2f}  (n={len(v)})")


if __name__ == "__main__":
    main()
