"""scripts/check_depth.py — verify cached depth: coverage, alignment with images, value range; writes a preview image."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsod_data.voc_fsod import TEST_SETS, TRAINVAL_SETS, load_records
from preprocessing.depth_utils import depth_path, load_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/VOCdevkit")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--depth-dir", default="depth_cache")
    ap.add_argument("--sample", type=int, default=300, help="images to verify (0 = all)")
    ap.add_argument("--preview", default="visualization/depth_preview.png")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    recs = (load_records(a.root, TRAINVAL_SETS, cache=f"{a.cache}/voc_trainval.json")
            + load_records(a.root, TEST_SETS, cache=f"{a.cache}/voc_test.json"))
    have = [r for r in recs if depth_path(a.depth_dir, r["path"]).exists()]
    print(f"coverage: {len(have)}/{len(recs)} images have a depth map")
    if not have:
        sys.exit("no depth maps found; run scripts/precompute_depth.py first")

    rng = random.Random(a.seed)
    sample = have if a.sample == 0 else rng.sample(have, min(a.sample, len(have)))
    problems = []
    for r in sample:
        img = cv2.imread(str(Path(a.root) / r["path"]))
        ih, iw = img.shape[:2]
        try:
            d = load_depth(depth_path(a.depth_dir, r["path"]), expected_hw=(ih, iw))
        except Exception as e:
            problems.append(f"{r['image_id']}: {e}")
            continue
        if not np.isfinite(d).all():
            problems.append(f"{r['image_id']}: non-finite values")
        elif d.std() < 1e-3:
            problems.append(f"{r['image_id']}: flat depth map (std={d.std():.5f})")
        elif d.min() < 0 or d.max() > 1:
            problems.append(f"{r['image_id']}: values outside [0,1]")

    # preview: RGB | depth, both with ground-truth boxes, to eyeball alignment
    rows = []
    for r in rng.sample(have, min(6, len(have))):
        img = cv2.imread(str(Path(a.root) / r["path"]))
        d = load_depth(depth_path(a.depth_dir, r["path"]), expected_hw=img.shape[:2])
        dep = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)  # bright = near
        for o in r["objects"]:
            x1, y1, x2, y2 = (int(round(v)) for v in o["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(dep, (x1, y1), (x2, y2), (0, 255, 0), 2)
        h = 280
        w = int(img.shape[1] * h / img.shape[0])
        pair = np.hstack([cv2.resize(img, (w, h)), cv2.resize(dep, (w, h))])
        pad = np.zeros((h, 2 * 500 - pair.shape[1], 3), np.uint8) if pair.shape[1] < 1000 else None
        rows.append(np.hstack([pair, pad]) if pad is not None else pair[:, :1000])
    Path(a.preview).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.preview, np.vstack(rows))
    print(f"preview written to {a.preview} (left: RGB, right: depth; green = ground-truth boxes)")

    total_gb = sum(p.stat().st_size for p in Path(a.depth_dir).rglob("*.png")) / 2**30
    print(f"depth cache size on disk: {total_gb:.2f} GB")
    for p in problems[:10]:
        print("  PROBLEM:", p)
    if problems:
        print(f"DEPTH CHECK FAILED: {len(problems)} problem(s) in {len(sample)} checked")
        sys.exit(1)
    print(f"DEPTH CHECK PASSED ({len(sample)} maps checked: shape matches image, finite, range OK)")


if __name__ == "__main__":
    main()
