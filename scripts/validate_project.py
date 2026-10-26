"""scripts/validate_project.py — pre-flight checks (v1: environment, VOC, splits, leakage, K-shot sets)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsod_data.voc_fsod import (TEST_SETS, TRAINVAL_SETS, VOC_CLASSES, build_base_train,
                                build_kshot, load_records, split_classes)

CTX = {}


def check_env(a):
    import torch, torchvision
    mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if mps else "cpu")
    return (f"python {sys.version.split()[0]}, torch {torch.__version__}, "
            f"torchvision {torchvision.__version__}, device={dev}")


def check_voc_files(a):
    exp = {("2007", "trainval"): 5011, ("2007", "test"): 4952, ("2012", "trainval"): 11540}
    for (y, s), n in exp.items():
        f = Path(a.root) / f"VOC{y}" / "ImageSets" / "Main" / f"{s}.txt"
        assert f.exists(), f"missing {f}"
        got = len(f.read_text().split())
        assert got == n, f"{f}: {got} ids, expected {n}"
    return "VOC07 trainval/test and VOC12 trainval id lists OK"


def check_records(a):
    train = load_records(a.root, TRAINVAL_SETS, cache=f"{a.cache}/voc_trainval.json")
    test = load_records(a.root, TEST_SETS, cache=f"{a.cache}/voc_test.json")
    CTX["train"], CTX["test"] = train, test
    assert len(train) == 16551, f"trainval has {len(train)} images, expected 16551"
    assert len(test) == 4952, f"test has {len(test)} images, expected 4952"
    ids_tr = {r["image_id"] for r in train}
    ids_te = {r["image_id"] for r in test}
    assert len(ids_tr) == len(train), "duplicate image ids in trainval"
    assert not (ids_tr & ids_te), "train/test image overlap"
    for r in train + test:
        for o in r["objects"]:
            assert o["name"] in VOC_CLASSES, f"unknown class {o['name']}"
            x1, y1, x2, y2 = o["bbox"]
            ok = (x2 > x1 and y2 > y1 and x1 >= -1 and y1 >= -1
                  and x2 <= r["width"] + 1 and y2 <= r["height"] + 1)
            assert ok, f"bad box {o['bbox']} in {r['image_id']} ({r['width']}x{r['height']})"
    return f"{len(train)} trainval + {len(test)} test images, boxes valid, no overlap"


def check_splits(a):
    for s in (1, 2, 3):
        base, novel = split_classes(s)
        assert len(base) == 15 and len(novel) == 5, f"split {s}: wrong sizes"
        assert not set(base) & set(novel), f"split {s}: base/novel overlap"
        assert set(base) | set(novel) == set(VOC_CLASSES), f"split {s}: classes missing"
    return "3 splits: 15 base + 5 novel, disjoint, cover all 20 classes"


def check_base_leakage(a):
    msgs = []
    for s in (1, 2, 3):
        _, novel = split_classes(s)
        bt = build_base_train(CTX["train"], s)
        leaked = sum(1 for r in bt for o in r["objects"] if o["name"] in novel)
        assert bt, f"split {s}: empty base-train set"
        assert leaked == 0, f"split {s}: {leaked} novel objects leaked into base training"
        msgs.append(f"split{s}={len(bt)} imgs")
    return "no novel annotations in base training (" + ", ".join(msgs) + ")"


def check_kshot(a):
    files = sorted(Path(a.fewshot).glob("seed*/*shot.json"))
    assert files, "no K-shot files found; run: python scripts/make_fewshot.py"
    train_map = {r["image_id"]: r for r in CTX["train"]}
    test_ids = {r["image_id"] for r in CTX["test"]}
    for f in files:
        k = int(f.stem.replace("shot", ""))
        data = json.loads(f.read_text())
        assert set(data) == set(VOC_CLASSES), f"{f}: class set mismatch"
        for c, entries in data.items():
            n = sum(len(e["boxes"]) for e in entries)
            assert n == k, f"{f}: class {c} has {n} boxes, expected {k}"
            for e in entries:
                assert e["class"] == c, f"{f}: wrong class label"
                assert e["image_id"] in train_map, f"{f}: support image not from trainval"
                assert e["image_id"] not in test_ids, f"{f}: support image is a test image"
                truth = [o["bbox"] for o in train_map[e["image_id"]]["objects"]
                         if o["name"] == c and not o["difficult"]]
                for b in e["boxes"]:
                    assert b in truth, f"{f}: box not in ground truth for {c}"
    f0 = Path(a.fewshot) / "seed0" / "1shot.json"
    if f0.exists():
        assert build_kshot(CTX["train"], 1, 0) == json.loads(f0.read_text()), "sampling not deterministic"
    f1, f2 = Path(a.fewshot) / "seed0" / "5shot.json", Path(a.fewshot) / "seed1" / "5shot.json"
    if f1.exists() and f2.exists():
        d1, d2 = json.loads(f1.read_text()), json.loads(f2.read_text())
        assert any(d1[c] != d2[c] for c in VOC_CLASSES), "seeds 0 and 1 give identical support sets"
    return f"{len(files)} support files: exact K per class, trainval-only, deterministic, seeds differ"


CHECKS = [("environment", check_env), ("VOC files", check_voc_files),
          ("records + boxes", check_records), ("class splits", check_splits),
          ("base-train leakage", check_base_leakage), ("K-shot sets", check_kshot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/VOCdevkit")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--fewshot", default="data/fewshot")
    a = ap.parse_args()
    failed = 0
    for name, fn in CHECKS:
        try:
            print(f"[PASS] {name}: {fn(a)}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print("\nVALIDATION PASSED (v1)" if not failed else f"\n{failed} CHECK(S) FAILED")
    print("Not covered yet (added in later steps): depth maps, model shapes, checkpoints.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
