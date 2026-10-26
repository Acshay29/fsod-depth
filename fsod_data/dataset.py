"""fsod_data/dataset.py — detection Dataset with optional aligned depth and joint augmentation."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from preprocessing.depth_utils import depth_path, load_depth


def make_label_map(classes):
    """Class name -> contiguous label 1..N (0 is background in torchvision detectors)."""
    return {c: i + 1 for i, c in enumerate(classes)}


def support_to_records(support, classes=None):
    """Turn a K-shot support dict (from build_kshot) into dataset records."""
    recs = []
    for c in (classes or sorted(support)):
        for e in support[c]:
            recs.append({"image_id": e["image_id"], "path": e["path"], "width": e["width"],
                         "height": e["height"],
                         "objects": [{"name": c, "bbox": b, "difficult": 0} for b in e["boxes"]]})
    return recs


class DetDataset(torch.utils.data.Dataset):
    """Returns (image[3,H,W] float in 0..1, target dict).
    target: boxes [N,4] (x1,y1,x2,y2, pixels), labels [N] (1..len(label_classes)), image_id (str),
            and depth [H,W] in 0..1 when depth_dir is given. Flips are applied jointly to RGB, depth, boxes."""

    def __init__(self, records, root, label_classes, train=False, depth_dir=None,
                 flip_prob=0.5, missing_depth="error"):
        assert missing_depth in ("error", "zeros")
        self.records = records
        self.root = Path(root)
        self.cls2idx = make_label_map(label_classes)
        self.depth_dir = depth_dir
        self.flip_prob = flip_prob if train else 0.0
        self.missing_depth = missing_depth

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        bgr = cv2.imread(str(self.root / r["path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / r["path"])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        objs = [o for o in r["objects"] if o["name"] in self.cls2idx]
        boxes = np.asarray([o["bbox"] for o in objs], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray([self.cls2idx[o["name"]] for o in objs], dtype=np.int64)

        depth = None
        if self.depth_dir is not None:
            depth = load_depth(depth_path(self.depth_dir, r["path"]), expected_hw=(h, w))
            if depth is None:
                if self.missing_depth == "error":
                    raise FileNotFoundError(f"missing depth map for {r['image_id']}")
                depth = np.zeros((h, w), dtype=np.float32)

        if self.flip_prob > 0 and random.random() < self.flip_prob:
            img = img[:, ::-1]
            if depth is not None:
                depth = depth[:, ::-1]
            if len(boxes):
                boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]],
                                 axis=1).astype(np.float32)

        target = {"boxes": torch.from_numpy(np.ascontiguousarray(boxes)),
                  "labels": torch.from_numpy(labels), "image_id": r["image_id"]}
        if depth is not None:
            target["depth"] = torch.from_numpy(np.ascontiguousarray(depth))
        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def detector_targets(targets, device):
    """Keep only what torchvision's detector needs (boxes, labels) and move to device."""
    return [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
