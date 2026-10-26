"""scripts/test_baseline_parts.py — unit checks for evaluator, dataset (RGB/depth/box alignment + flip) and detector."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.voc_eval import class_ap, evaluate_voc
from fsod_data.dataset import DetDataset, collate_fn, detector_targets
from fsod_data.voc_fsod import TRAINVAL_SETS, build_base_train, load_records, split_classes
from models.fsod.frcnn import build_detector, count_params


def test_evaluator():
    gts = {"a": [{"name": "cat", "bbox": [10, 10, 50, 50], "difficult": 0}],
           "b": [{"name": "cat", "bbox": [0, 0, 30, 30], "difficult": 1},
                 {"name": "cat", "bbox": [40, 40, 90, 90], "difficult": 0}]}
    perfect = {"a": [("cat", .9, [10, 10, 50, 50])],
               "b": [("cat", .7, [0, 0, 30, 30]), ("cat", .95, [40, 40, 90, 90])]}
    assert abs(class_ap(perfect, gts, "cat") - 1.0) < 1e-9, "perfect detections should give AP=1 (difficult ignored)"
    miss = {"a": [("cat", .9, [10, 10, 50, 50])], "b": []}
    assert abs(class_ap(miss, gts, "cat") - 6 / 11) < 1e-9, "half recall should give VOC07 AP=6/11"
    print("[PASS] evaluator: perfect=1.0, half-recall=6/11, difficult objects ignored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/VOCdevkit")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--depth-dir", default="depth_cache")
    a = ap.parse_args()

    test_evaluator()

    recs = load_records(a.root, TRAINVAL_SETS, cache=f"{a.cache}/voc_trainval.json")
    base, _ = split_classes(1)
    sub = build_base_train(recs, 1)[:4]
    ds = DetDataset(sub, a.root, base, train=False, depth_dir=a.depth_dir)
    dsf = DetDataset(sub, a.root, base, train=True, depth_dir=a.depth_dir, flip_prob=1.0)
    for i in range(len(sub)):
        x, t = ds[i]
        xf, tf = dsf[i]
        _, h, w = x.shape
        assert t["depth"].shape == (h, w), "depth/image size mismatch"
        assert t["labels"].min() >= 1 and t["labels"].max() <= len(base), "label out of range"
        b = t["boxes"]
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all(), "degenerate box"
        assert b[:, 0].min() >= -1 and b[:, 2].max() <= w + 1 and b[:, 3].max() <= h + 1, "box outside image"
        assert torch.equal(xf, x.flip(-1)), "RGB flip wrong"
        assert torch.equal(tf["depth"], t["depth"].flip(-1)), "depth not flipped with RGB"
        assert torch.allclose(tf["boxes"][:, [0, 2]], w - t["boxes"][:, [2, 0]]), "box flip wrong"
        assert torch.equal(tf["boxes"][:, [1, 3]], t["boxes"][:, [1, 3]]), "box y changed by flip"
    print(f"[PASS] dataset: {len(sub)} samples; RGB/depth/box shapes match; joint flip correct")

    model = build_detector(len(base) + 1, pretrained_backbone=False, min_size=256, max_size=384)
    total, trainable = count_params(model)
    batch = collate_fn([ds[i] for i in range(2)])
    imgs, tgts = list(batch[0]), detector_targets(batch[1], "cpu")
    model.train()
    losses = model(imgs, tgts)
    assert all(torch.isfinite(v) for v in losses.values()), f"non-finite loss {losses}"
    model.eval()
    with torch.no_grad():
        out = model(imgs)
    assert len(out) == 2 and {"boxes", "scores", "labels"} <= set(out[0]), "bad eval output"
    dets = {t["image_id"]: [(base[int(l) - 1], float(s), bx.tolist())
                            for bx, s, l in zip(o["boxes"], o["scores"], o["labels"])]
            for o, t in zip(out, batch[1])}
    gts = {r["image_id"]: r["objects"] for r in sub[:2]}
    res = evaluate_voc(dets, gts, base, base=base)
    print(f"[PASS] detector: params {total / 1e6:.1f}M (trainable {trainable / 1e6:.1f}M), "
          f"loss keys {sorted(losses)}, eval pipeline ran (untrained AP50={res['base']['AP50']:.1f})")
    print("ALL BASELINE PARTS OK")


if __name__ == "__main__":
    main()
