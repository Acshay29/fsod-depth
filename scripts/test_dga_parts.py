"""scripts/test_dga_parts.py — unit checks for the depth-geometry module (no dataset, no GPU, ~1 minute on CPU).
Run:  python scripts/test_dga_parts.py"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.fsod.frcnn import build_detector
from models.geo.dga import MODES, DGADetector, load_detector_state
from models.geo.geometry import GEO_DIM, region_descriptors

torch.manual_seed(0)
OK = []


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    OK.append(bool(cond))
    if not cond:
        raise SystemExit(f"FAILED: {name} {extra}")


def make_det():
    d = build_detector(16, pretrained_backbone=False, min_size=(160,), max_size=260,
                       rpn_post_nms_top_n_test=200, rpn_post_nms_top_n_train=300)
    d.roi_heads.score_thresh, d.roi_heads.detections_per_img = 0.0, 50
    return d


def fake_batch(n=2):
    imgs = [torch.rand(3, 200 + 20 * i, 300 - 10 * i) for i in range(n)]
    depths = [torch.rand(int(im.shape[1] * 1.5), int(im.shape[2] * 1.5)) for im in imgs]   # other resolution
    tgts = [{"boxes": torch.tensor([[20., 30., 120., 150.], [150., 40., 280., 190.]]),
             "labels": torch.tensor([3, 7])} for _ in imgs]
    return imgs, depths, tgts


# 1. descriptors ------------------------------------------------------------------------------
dm = [torch.rand(240, 360), torch.rand(1, 100, 100)]
props = [torch.tensor([[10., 10., 100., 80.], [0., 0., 5., 5.], [300., 200., 400., 300.]]), torch.zeros(0, 4)]
g = region_descriptors(dm, props, [(160, 240), (100, 100)])
check("descriptor shape", g.shape == (3, GEO_DIM), str(tuple(g.shape)))
check("descriptor finite (tiny box, box outside image, empty image)", bool(torch.isfinite(g).all()))
flat = region_descriptors([torch.zeros(50, 50)], [torch.tensor([[5., 5., 30., 30.]])], [(50, 50)])
check("flat depth map -> finite", bool(torch.isfinite(flat).all()))
near = torch.zeros(100, 100)
near[20:60, 20:60] = 1.0
gg = region_descriptors([near], [torch.tensor([[20., 20., 60., 60.]])], [(100, 100)])[0]
check("near object: inside-minus-ring contrast is positive", float(gg[5]) > 0.3, f"{float(gg[5]):.2f}")

# 2. zero-init == baseline ----------------------------------------------------------------------
det = make_det().eval()
dga = DGADetector(make_det(), mode="depth")
missing = load_detector_state(dga, det.state_dict())
check("only geo parameters are missing after loading the plain checkpoint",
      all(k.startswith(("roi_heads.box_head.embed", "roi_heads.box_head.film", "roi_heads.box_head.gate"))
          for k in missing), f"({len(missing)} keys)")
dga.eval()
imgs, depths, _ = fake_batch(2)
with torch.no_grad():
    a, b = det(imgs), dga(imgs, depths=depths)
same = all(torch.allclose(x["boxes"], y["boxes"], atol=1e-4) and torch.allclose(x["scores"], y["scores"], atol=1e-5)
           for x, y in zip(a, b))
check("zero-initialised DGA gives the same detections as the plain detector", same)

h = dga.det.roi_heads.box_head
with torch.no_grad():
    h.film.weight.normal_(0, 0.05)
    b2 = dga(imgs, depths=depths)
check("non-zero FiLM changes the detections", not torch.allclose(a[0]["scores"], b2[0]["scores"], atol=1e-5))
with torch.no_grad():                      # gate -> closed
    h.gate[-1].weight.zero_()
    h.gate[-1].bias.fill_(-30.0)
    b3 = dga(imgs, depths=depths)
check("closed gate (alpha ~ 0) restores the baseline output",
      all(torch.allclose(x["scores"], y["scores"], atol=1e-4) for x, y in zip(a, b3)))

# 3. controls run and differ from real depth ---------------------------------------------------
outs = {}
for mode in MODES:
    m = DGADetector(make_det(), mode=mode)
    load_detector_state(m, det.state_dict())
    with torch.no_grad():
        m.det.roi_heads.box_head.film.weight.normal_(0, 0.05)
    torch.manual_seed(1)
    m.eval()
    with torch.no_grad():
        outs[mode] = m(imgs, depths=None if mode in ("const", "gray") else depths)[0]["scores"]
check("all control modes run", len(outs) == len(MODES), str(list(outs)))
check("'const' control differs from 'depth'", not torch.allclose(outs["const"], outs["depth"], atol=1e-6))
try:
    DGADetector(make_det(), mode="depth").eval()(imgs)
    check("depth mode without depth maps raises", False)
except ValueError:
    check("depth mode without depth maps raises a clear error", True)

# 4. training: gradients, freezing, loss goes down ------------------------------------------------
model = DGADetector(make_det(), mode="depth")
load_detector_state(model, det.state_dict())
for n, p in model.named_parameters():
    p.requires_grad = n.startswith(("det.roi_heads.box_head.embed", "det.roi_heads.box_head.film",
                                    "det.roi_heads.box_head.gate", "det.roi_heads.box_predictor"))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=0.001, momentum=0.9)
model.train()
imgs, depths, tgts = fake_batch(2)
losses = []
for it in range(60):
    out = model(imgs, tgts, depths=depths)
    loss = out["loss_classifier"] + out["loss_box_reg"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
tail = sum(losses[-10:]) / 10
check("training loss is finite and falls on a fixed batch (random-weight model, so only a smoke test)",
      all(torch.isfinite(torch.tensor(losses))) and tail < 0.8 * losses[0], f"{losses[0]:.2f} -> {tail:.2f}")
hh = model.det.roi_heads.box_head
check("FiLM weights moved away from zero", float(hh.film.weight.detach().abs().sum()) > 0)
check("backbone / RPN / fc6 / fc7 got no gradients",
      all(p.grad is None for n, p in model.named_parameters()
          if n.startswith(("det.backbone", "det.rpn", "det.roi_heads.box_head.head"))))
check("geo parameters are a small fraction of the model",
      sum(p.numel() for p in model.geo_parameters()) < 0.01 * sum(p.numel() for p in model.parameters()),
      f"({sum(p.numel() for p in model.geo_parameters()) / 1e3:.0f}K params)")

# 5. cost -------------------------------------------------------------------------------------------
props = [torch.rand(512, 4) * 150 + torch.tensor([0., 0., 100., 100.]) for _ in range(2)]
t0 = time.time()
for _ in range(5):
    region_descriptors([torch.rand(480, 640)] * 2, props, [(320, 480)] * 2)
print(f"[info] descriptor time for 2 images x 512 RoIs: {1000 * (time.time() - t0) / 5:.0f} ms (CPU)")

print("\nALL DGA UNIT CHECKS PASSED" if all(OK) else "\nSOME CHECKS FAILED")
