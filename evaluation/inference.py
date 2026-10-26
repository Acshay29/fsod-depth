"""evaluation/inference.py — run a detector over VOC records and score it (shared by scripts).
depth_dir=None  -> RGB-only models (A0, 'off', and the const/gray controls);
depth_dir=<dir> -> depth maps are loaded alongside each image and passed to the DGA detector."""
import time

import torch
from torch.utils.data import DataLoader

from evaluation.voc_eval import evaluate_voc
from fsod_data.dataset import DetDataset, collate_fn
from fsod_data.voc_fsod import split_classes


@torch.no_grad()
def evaluate_model(model, classes, split, recs, root, device, batch_size=8, workers=2, depth_dir=None):
    """classes: label order of the model's outputs (label i+1 -> classes[i]). Returns the evaluate_voc dict
    plus ms_per_image and n_images. Base/novel means follow the given split."""
    model.to(device).eval()
    base_all, novel_all = split_classes(split)
    base = [c for c in classes if c in base_all]
    novel = [c for c in classes if c in novel_all]
    ds = DetDataset(recs, root, classes, train=False, depth_dir=depth_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    dets, n_img, t_inf = {}, 0, 0.0
    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        depths = [t["depth"].to(device) for t in tgts] if depth_dir is not None else None
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(imgs, depths=depths) if depth_dir is not None else model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.time() - t0
        n_img += len(imgs)
        for o, t in zip(out, tgts):
            dets[t["image_id"]] = [(classes[int(l) - 1], float(s), [float(x) for x in b])
                                   for b, s, l in zip(o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu())]
    gts = {r["image_id"]: r["objects"] for r in recs}
    res = evaluate_voc(dets, gts, classes, base=base, novel=novel or None)
    res.update({"ms_per_image": 1000 * t_inf / max(n_img, 1), "n_images": n_img})
    return res
