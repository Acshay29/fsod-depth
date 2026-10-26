"""scripts/check_voc.py — verify VOC files: image counts, missing files, broken boxes, per-class counts."""
import argparse, collections, sys
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
EXPECTED = {("2007", "trainval"): 5011, ("2007", "test"): 4952, ("2012", "trainval"): 11540}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/VOCdevkit")
    args = ap.parse_args()
    root = Path(args.root)
    ok = True

    for (year, split), n_exp in EXPECTED.items():
        d = root / f"VOC{year}"
        ids_file = d / "ImageSets" / "Main" / f"{split}.txt"
        if not ids_file.exists():
            print(f"[MISSING] {ids_file}")
            ok = False
            continue
        ids = ids_file.read_text().split()
        missing_img = bad_xml = bad_box = unknown = 0
        counts = collections.Counter()
        for i in ids:
            if not (d / "JPEGImages" / f"{i}.jpg").exists():
                missing_img += 1
            try:
                tree = ET.parse(d / "Annotations" / f"{i}.xml")
            except Exception:
                bad_xml += 1
                continue
            for obj in tree.findall("object"):
                name = obj.findtext("name")
                if name not in CLASSES:
                    unknown += 1
                    continue
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                if x2 <= x1 or y2 <= y1:
                    bad_box += 1
                counts[name] += 1
        good = (len(ids) == n_exp and not (missing_img or bad_xml or bad_box or unknown))
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] VOC{year} {split}: {len(ids)} images (expected {n_exp}), "
              f"missing_jpg={missing_img} bad_xml={bad_xml} bad_box={bad_box} unknown_class={unknown}, "
              f"objects={sum(counts.values())}")
        if split == "trainval" and year == "2007":
            print("   per-class objects:", dict(sorted(counts.items())))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
