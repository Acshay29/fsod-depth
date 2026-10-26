"""scripts/evaluate_detector.py — evaluate a saved detector on VOC07 test (AP, AP50, AP75; base/novel means)."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.voc_eval import evaluate_voc
from fsod_data.dataset import DetDataset, collate_fn
from fsod_data.voc_fsod import TEST_SETS, load_records, split_classes
from models.fsod.frcnn import build_detector, count_params
from training.utils import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a model_final.pth written by train.py")
    ap.add_argument("--root", default="data/VOCdevkit")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test images (for testing)")
    ap.add_argument("--test-size", type=int, default=0, help="override test image size (default: last min_size)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = pick_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    classes, split, cfg = ck["classes"], ck["split"], ck["cfg"]
    base_all, novel_all = split_classes(split)
    base = [c for c in classes if c in base_all]
    novel = [c for c in classes if c in novel_all]
    M = cfg["model"]
    test_size = a.test_size or M["min_size"][-1]

    model = build_detector(len(classes) + 1, pretrained_backbone=False, min_size=(test_size,),
                           max_size=M["max_size"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    total, _ = count_params(model)

    recs = load_records(a.root, TEST_SETS, cache=f"{a.cache}/voc_test.json")
    if a.limit:
        recs = recs[: a.limit]
    ds = DetDataset(recs, a.root, classes, train=False)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                        collate_fn=collate_fn)
    print(f"evaluating {Path(a.ckpt).parent.name}: {len(recs)} test images, classes={len(classes)} "
          f"(base {len(base)}, novel {len(novel)}), test size {test_size}, device={device}")

    dets, n_img, t_inf = {}, 0, 0.0
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [i.to(device) for i in imgs]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(imgs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf += time.time() - t0
            n_img += len(imgs)
            for o, t in zip(out, tgts):
                dets[t["image_id"]] = [
                    (classes[int(l) - 1], float(s), [float(x) for x in b])
                    for b, s, l in zip(o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu())]

    gts = {r["image_id"]: r["objects"] for r in recs}
    res = evaluate_voc(dets, gts, classes, base=base, novel=novel or None)
    res.update({"params_M": total / 1e6, "ms_per_image": 1000 * t_inf / max(n_img, 1),
                "n_images": n_img, "test_size": test_size, "ckpt": a.ckpt})

    for g in ("base", "novel", "all"):
        if g in res:
            r = res[g]
            print(f"{g:>5}: AP {r['AP']:.2f}  AP50 {r['AP50']:.2f}  AP75 {r['AP75']:.2f}")
    print(f"params {res['params_M']:.1f}M, {res['ms_per_image']:.0f} ms/image (detector only, excludes depth)")
    out_path = Path(a.out) if a.out else Path(a.ckpt).parent / f"eval_test{test_size}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
