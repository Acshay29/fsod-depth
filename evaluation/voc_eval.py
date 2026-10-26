"""evaluation/voc_eval.py — PASCAL VOC detection evaluation (numpy only).

Conventions (match TFA/FsDet on VOC07 test):
  * boxes are [x1, y1, x2, y2] floats, 0-indexed x1/y1 and VOC xmax/ymax as x2/y2,
    so width = x2 - x1 equals VOC's inclusive width (xmax - xmin + 1). Plain continuous IoU is used.
  * objects flagged `difficult` are ignored (neither TP nor FP).
  * AP uses the VOC07 11-point metric by default; AP over IoU 0.50:0.95 and AP75 are also reported.
"""
import numpy as np


def _iou_one_to_many(box, boxes):
    ix1 = np.maximum(boxes[:, 0], box[0])
    iy1 = np.maximum(boxes[:, 1], box[1])
    ix2 = np.minimum(boxes[:, 2], box[2])
    iy2 = np.minimum(boxes[:, 3], box[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
            ap += p / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def class_ap(dets, gts, cls, iou_thresh=0.5, use_07_metric=True):
    """dets: {image_id: [(class, score, [x1,y1,x2,y2]), ...]}
    gts:  {image_id: [{"name", "bbox", "difficult"}, ...]}"""
    gt_c, npos = {}, 0
    for img, objs in gts.items():
        sel = [o for o in objs if o["name"] == cls]
        if sel:
            gt_c[img] = {"boxes": np.array([o["bbox"] for o in sel], dtype=np.float64),
                         "diff": np.array([bool(o["difficult"]) for o in sel]),
                         "used": np.zeros(len(sel), dtype=bool)}
            npos += int((~gt_c[img]["diff"]).sum())
    flat = [(img, s, b) for img, ds in dets.items() for (c, s, b) in ds if c == cls]
    if not flat:
        return 0.0
    flat.sort(key=lambda x: -x[1])
    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))
    for k, (img, _, b) in enumerate(flat):
        g = gt_c.get(img)
        ovmax, jmax = -1.0, -1
        if g is not None:
            ov = _iou_one_to_many(np.asarray(b, dtype=np.float64), g["boxes"])
            jmax = int(np.argmax(ov))
            ovmax = float(ov[jmax])
        if ovmax >= iou_thresh:
            if g["diff"][jmax]:
                continue  # ignored
            if not g["used"][jmax]:
                tp[k] = 1
                g["used"][jmax] = True
            else:
                fp[k] = 1
        else:
            fp[k] = 1
    fp, tp = np.cumsum(fp), np.cumsum(tp)
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    return voc_ap(rec, prec, use_07_metric)


def evaluate_voc(dets, gts, classes, base=None, novel=None, use_07_metric=True):
    """Returns dict with per-class AP50, AP (0.5:0.95), AP75 and group means (all / base / novel)."""
    thr = np.arange(0.5, 0.96, 0.05)
    per = {c: {"AP50": class_ap(dets, gts, c, 0.5, use_07_metric),
               "AP75": class_ap(dets, gts, c, 0.75, use_07_metric),
               "AP": float(np.mean([class_ap(dets, gts, c, t, use_07_metric) for t in thr]))}
           for c in classes}

    def mean(group, key):
        return float(np.mean([per[c][key] for c in group])) * 100 if group else float("nan")

    out = {"per_class": per}
    for name, group in (("all", classes), ("base", base or []), ("novel", novel or [])):
        if group:
            out[name] = {k: mean(group, k) for k in ("AP", "AP50", "AP75")}
    return out
