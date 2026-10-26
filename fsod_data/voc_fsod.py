"""fsod_data/voc_fsod.py — VOC loading, TFA-style base/novel splits, seeded K-shot sampling."""
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
NOVEL_CLASSES = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}
TRAINVAL_SETS = [("2007", "trainval"), ("2012", "trainval")]
TEST_SETS = [("2007", "test")]
SHOTS = [1, 2, 3, 5, 10]


def split_classes(split):
    novel = NOVEL_CLASSES[split]
    return [c for c in VOC_CLASSES if c not in novel], list(novel)


def _parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        objs.append({
            "name": o.findtext("name"),
            # 0-indexed [x1, y1, x2, y2] (VOC is 1-indexed; same convention as detectron2/TFA)
            "bbox": [float(bb.findtext("xmin")) - 1.0, float(bb.findtext("ymin")) - 1.0,
                     float(bb.findtext("xmax")), float(bb.findtext("ymax"))],
            "difficult": int(o.findtext("difficult") or 0),
        })
    return int(size.findtext("width")), int(size.findtext("height")), objs


def load_records(root, sets, cache=None):
    """Return a list of image records. Paths are relative to the VOCdevkit root."""
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    root = Path(root)
    records = []
    for year, split in sets:
        d = root / f"VOC{year}"
        for i in (d / "ImageSets" / "Main" / f"{split}.txt").read_text().split():
            w, h, objs = _parse_xml(d / "Annotations" / f"{i}.xml")
            records.append({"image_id": f"VOC{year}/{i}", "path": f"VOC{year}/JPEGImages/{i}.jpg",
                            "width": w, "height": h, "objects": objs})
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(records))
    return records


def build_base_train(records, split):
    """Base-training set: only base-class objects kept; images with no base object are dropped."""
    base, _ = split_classes(split)
    out = []
    for r in records:
        objs = [o for o in r["objects"] if o["name"] in base]
        if objs:
            out.append({**r, "objects": objs})
    return out


def build_kshot(records, k, seed, classes=VOC_CLASSES, max_tries=2000):
    """TFA-style K-shot sampling: for each class, pick images so that the total number of
    non-difficult instances of that class is exactly K. Only that class is annotated in the
    support entry. Deterministic for a given (seed, k, class)."""
    by_class = {c: [] for c in classes}
    for idx, r in enumerate(records):
        cnt = Counter(o["name"] for o in r["objects"] if not o["difficult"])
        for c, n in cnt.items():
            if c in by_class:
                by_class[c].append((idx, n))
    out = {}
    for c in classes:
        rng = random.Random(f"{seed}-{k}-{c}")
        for _ in range(max_tries):
            order = by_class[c][:]
            rng.shuffle(order)
            picked, total = [], 0
            for idx, n in order:
                if total + n > k:
                    continue
                picked.append(idx)
                total += n
                if total == k:
                    break
            if total == k:
                break
        else:
            raise RuntimeError(f"could not sample exactly {k} instances of {c}")
        entries = []
        for idx in picked:
            r = records[idx]
            boxes = [o["bbox"] for o in r["objects"] if o["name"] == c and not o["difficult"]]
            entries.append({"image_id": r["image_id"], "path": r["path"], "width": r["width"],
                            "height": r["height"], "class": c, "boxes": boxes})
        out[c] = entries
    return out
