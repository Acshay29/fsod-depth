"""scripts/train_geo.py — geometry stage (training/geo_trainer.py) + a quick base-class sanity evaluation.
Example:
  python scripts/train_geo.py --config configs/geo_split1.yaml --base-ckpt <A0 model_final.pth> \
         --out-dir <runs>/geo_depth --mode depth
modes: depth | shuffle | const | gray | off   ('off' = continued A0, the fairness control)"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.inference import evaluate_model
from fsod_data.voc_fsod import TEST_SETS, load_records
from training.geo_trainer import NEEDS_DEPTH, model_from_geo_ckpt, train_geo
from training.utils import load_config, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", required=True, choices=["depth", "shuffle", "const", "gray", "off"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=500, help="sanity evaluation on the first N test images (0=skip)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    cfg["geo"]["mode"] = a.mode
    device = pick_device(a.device)
    out = Path(a.out_dir)
    final = train_geo(cfg, a.base_ckpt, out, device, seed=a.seed)

    if a.eval_limit:
        P = cfg["paths"]
        ck = torch.load(a.base_ckpt, map_location="cpu", weights_only=False)
        gk = torch.load(final, map_location="cpu", weights_only=False)
        model = model_from_geo_ckpt(ck, gk)
        recs = load_records(P["voc_root"], TEST_SETS, cache=f'{P["cache"]}/voc_test.json')[: a.eval_limit]
        res = evaluate_model(model, ck["classes"], cfg["split"], recs, P["voc_root"], device, batch_size=8,
                             workers=cfg["data"]["num_workers"],
                             depth_dir=cfg["data"]["depth_dir"] if a.mode in NEEDS_DEPTH else None)
        (out / "eval_base.json").write_text(json.dumps(res, indent=2))
        print(f"[sanity] mode={a.mode}: base-class AP50 {res['all']['AP50']:.2f}  AP {res['all']['AP']:.2f}  "
              f"({res['n_images']} test images, {res['ms_per_image']:.0f} ms/img)")


if __name__ == "__main__":
    main()
