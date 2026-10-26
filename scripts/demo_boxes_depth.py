"""scripts/demo_boxes_depth.py — quick visual demo: detections next to the estimated depth map.
For N VOC07 test images it draws the baseline detector's boxes (score >= --thr) and the cached depth map.
Saves one picture: demo_out/demo.png  (row = image | depth map).
  python scripts/demo_boxes_depth.py --base-ckpt <model_final.pth> --n 4
Note: uses the 15 base classes of the A0 model (few-shot weights are not stored); the depth maps are the cached
Depth Anything V2-Small maps the geometry module uses."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsod_data.voc_fsod import TEST_SETS, load_records
from preprocessing.depth_utils import depth_path, load_depth
from training.geo_trainer import build_geo_model
from training.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--config", default="configs/geo_split1.yaml")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="demo_out/demo.png")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = cfg["paths"]
    ck = torch.load(a.base_ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = build_geo_model(ck, "off").eval()          # plain detector with the A0 base weights
    recs = load_records(P["voc_root"], TEST_SETS, cache=f'{P["cache"]}/voc_test.json')
    recs = [r for r in recs if r["objects"]]
    step = max(1, len(recs) // a.n)
    rows = []
    for r in recs[::step][: a.n]:
        bgr = cv2.imread(str(Path(P["voc_root"]) / r["path"]))
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = model([x])[0]
        vis = bgr.copy()
        for b, s, l in zip(out["boxes"], out["scores"], out["labels"]):
            if float(s) < a.thr:
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(vis, f"{classes[int(l) - 1]} {float(s):.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        d = load_depth(depth_path(cfg["data"]["depth_dir"], r["path"]), expected_hw=(h, w))
        dm = cv2.applyColorMap((np.clip(d, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.putText(vis, "RGB + detections", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(dm, "estimated depth map", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(cv2.resize(np.hstack([vis, dm]), (1000, int(1000 * h / (2 * w)))))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"saved {a.out} ({len(rows)} images)")


if __name__ == "__main__":
    main()
