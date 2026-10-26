"""scripts/check_geo_effect.py — is the depth branch actually wired in, and how much does it change the RoI feature?
Loads A0 base + geo checkpoint, runs a few real test images, and compares the 1024-d RoI feature before / after FiLM
and with real depth vs a constant depth map.   (Only meaningful for modes depth / shuffle.)
  python scripts/check_geo_effect.py --base-ckpt <model_final.pth> --geo-ckpt <geo_final.pth>"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsod_data.dataset import DetDataset, collate_fn
from fsod_data.voc_fsod import TEST_SETS, load_records
from training.geo_trainer import model_from_geo_ckpt
from training.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--geo-ckpt", required=True)
    ap.add_argument("--config", default="configs/geo_split1.yaml")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = cfg["paths"]
    ck = torch.load(a.base_ckpt, map_location="cpu", weights_only=False)
    gk = torch.load(a.geo_ckpt, map_location="cpu", weights_only=False)
    mode = gk["mode"]
    if mode not in ("depth", "shuffle"):
        print(f"mode '{mode}' does not use depth maps; nothing to check")
        return
    model = model_from_geo_ckpt(ck, gk).eval()
    head = model.det.roi_heads.box_head
    fw = head.film.weight.detach()
    print(f"FiLM weights: max |w| = {fw.abs().max():.3e}   mean |w| = {fw.abs().mean():.3e}   (exactly 0 = never trained)")

    recs = load_records(P["voc_root"], TEST_SETS, cache=f'{P["cache"]}/voc_test.json')[: a.n]
    ds = DetDataset(recs, P["voc_root"], ck["classes"], train=False, depth_dir=cfg["data"]["depth_dir"])
    imgs, tgts = collate_fn([ds[i] for i in range(len(ds))])
    depths = [t["depth"] for t in tgts]
    cap = {}
    head.head.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o.detach()))
    head.register_forward_hook(lambda m, i, o: cap.__setitem__("fp", o.detach()))
    with torch.no_grad():
        model(list(imgs), depths=depths)
        f, fp_real, alpha = cap["f"], cap["fp"], head.last_alpha.clone()
        model(list(imgs), depths=[torch.full_like(d, 0.5) for d in depths])
        fp_const = cap["fp"]
    scale = f.abs().mean()
    print(f"RoIs: {f.shape[0]}   gate alpha: mean {alpha.mean():.3f}  min {alpha.min():.3f}  max {alpha.max():.3f}")
    print(f"feature change from FiLM       : mean|f' - f| / mean|f| = {(fp_real - f).abs().mean() / scale:.3e}")
    print(f"real depth vs constant depth   : mean|f'_real - f'_const| / mean|f| = {(fp_real - fp_const).abs().mean() / scale:.3e}")
    change = float((fp_real - f).abs().mean() / scale)
    dep = float((fp_real - fp_const).abs().mean() / scale)
    if change < 1e-3:
        print(f"RESULT: WARNING - FiLM barely changes the features ({change:.1e}): depth branch is effectively OFF")
    elif dep < 1e-4:
        print(f"RESULT: WARNING - features change ({change:.1e}) but hardly depend on depth ({dep:.1e})")
    else:
        print(f"RESULT: depth branch is active (feature change {change:.1e}, depth-dependent {dep:.1e})")


if __name__ == "__main__":
    main()
