"""scripts/download_voc.py — download + extract PASCAL VOC 2007 (trainval, test) and 2012 (trainval)."""
import argparse, sys, tarfile, urllib.request
from pathlib import Path

OXFORD07 = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
OXFORD12 = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
PJ = "https://pjreddie.com/media/files/"

# tar name -> (mirror base URLs, file that proves it is already extracted)
FILES = {
    "VOCtrainval_06-Nov-2007.tar": ([OXFORD07, PJ], "VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"),
    "VOCtest_06-Nov-2007.tar": ([OXFORD07, PJ], "VOCdevkit/VOC2007/ImageSets/Main/test.txt"),
    "VOCtrainval_11-May-2012.tar": ([OXFORD12, PJ], "VOCdevkit/VOC2012/ImageSets/Main/trainval.txt"),
}


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last = {"p": -1}

    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                p = int(done * 100 / total)
                if p // 5 != last["p"] // 5:
                    print(f"  {dest.name}: {p}% ({done / 2**20:.0f}/{total / 2**20:.0f} MB)")
                    last["p"] = p
    tmp.rename(dest)


def extract(tar_path: Path, out: Path) -> None:
    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(out, filter="data")
        except TypeError:  # older Python without extraction filters
            tar.extractall(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="directory that will contain VOCdevkit/")
    ap.add_argument("--keep-tars", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, (mirrors, marker) in FILES.items():
        if (out / marker).exists():
            print(f"[skip] {name} already extracted")
            continue
        tar_path = out / name
        if not tar_path.exists():
            for base in mirrors:
                try:
                    print(f"[download] {base}{name}")
                    download(base + name, tar_path)
                    break
                except Exception as e:
                    print(f"  failed: {type(e).__name__}: {e}")
            else:
                sys.exit(f"ERROR: could not download {name} from any mirror. "
                         f"Download it manually into {out}/ and re-run.")
        print(f"[extract] {name}")
        extract(tar_path, out)
        if not args.keep_tars:
            tar_path.unlink()
    print("Done. Now run: python scripts/check_voc.py --root", out / "VOCdevkit")


if __name__ == "__main__":
    main()