"""scripts/precompute_depth.py — run Depth Anything V2-Small once per image and cache normalised depth."""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsod_data.voc_fsod import TEST_SETS, TRAINVAL_SETS, load_records
from preprocessing.depth_utils import depth_path, normalize_depth, save_depth

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/VOCdevkit")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="depth_cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (for testing)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ids", nargs="+", default=None, help="only these image_ids, e.g. VOC2007/004513")
    a = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = pick_device(a.device)
    print(f"device={device}, model={MODEL_ID}")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

    recs = (load_records(a.root, TRAINVAL_SETS, cache=f"{a.cache}/voc_trainval.json")
            + load_records(a.root, TEST_SETS, cache=f"{a.cache}/voc_test.json"))
    if a.limit:
        recs = recs[:a.limit]
    if a.ids:
        recs = [r for r in recs if r["image_id"] in set(a.ids)]
        assert recs, "none of the --ids matched"
    print(f"{len(recs)} images to cover (existing files are skipped unless --overwrite)")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_f = out_dir / "stats.csv"
    new_file = not stats_f.exists()
    done = skipped = mismatches = nonfinite_imgs = 0
    written_bytes = 0
    t_start = time.time()

    with open(stats_f, "a") as sf:
        if new_file:
            sf.write("image_id,height,width,raw_lo,raw_hi,n_nonfinite,seconds\n")
        for r in tqdm(recs):
            dst = depth_path(out_dir, r["path"])
            if dst.exists() and not a.overwrite:
                skipped += 1
                continue
            t0 = time.time()
            img = Image.open(Path(a.root) / r["path"]).convert("RGB")
            W, H = img.size
            if (H, W) != (r["height"], r["width"]):
                mismatches += 1
                tqdm.write(f"WARNING {r['image_id']}: JPEG is {W}x{H} but XML says "
                           f"{r['width']}x{r['height']} (depth follows the JPEG)")
            inputs = proc(images=img, return_tensors="pt").to(device)
            with torch.inference_mode():
                pred = model(**inputs).predicted_depth  # (1, h, w)
            pred = pred.float().cpu().unsqueeze(1)
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
            d01, lo, hi, n_bad = normalize_depth(pred)
            if n_bad:
                nonfinite_imgs += 1
            save_depth(dst, d01)
            written_bytes += dst.stat().st_size
            sf.write(f"{r['image_id']},{H},{W},{lo:.5f},{hi:.5f},{n_bad},{time.time() - t0:.3f}\n")
            done += 1

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min: wrote {done}, skipped {skipped}, "
          f"images with NaN/inf fixed={nonfinite_imgs}, size mismatches={mismatches}, "
          f"new data={written_bytes / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
