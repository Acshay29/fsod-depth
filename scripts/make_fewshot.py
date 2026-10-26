"""scripts/make_fewshot.py — generate seeded K-shot support sets from VOC07+12 trainval."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsod_data.voc_fsod import SHOTS, TRAINVAL_SETS, build_kshot, load_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/VOCdevkit")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="data/fewshot")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    a = ap.parse_args()

    print("Loading VOC07+12 trainval annotations (first run parses ~16.5k XML files)...")
    recs = load_records(a.root, TRAINVAL_SETS, cache=f"{a.cache}/voc_trainval.json")
    print(f"  {len(recs)} images")
    n = 0
    for seed in a.seeds:
        for k in a.shots:
            sup = build_kshot(recs, k, seed)
            f = Path(a.out) / f"seed{seed}" / f"{k}shot.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(sup))
            n += 1
        print(f"  seed {seed}: wrote shots {a.shots}")
    print(f"Done: {n} files in {a.out}/")


if __name__ == "__main__":
    main()
