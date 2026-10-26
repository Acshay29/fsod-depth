"""training/geo_trainer.py — stage 'geo': train ONLY the depth-geometry modules (+ the box predictor) on base-class
data, starting from the A0 base checkpoint. Backbone, RPN and the two box-head FC layers stay frozen.

mode: depth | shuffle | const | gray   DGA variants (geometry branch fed with depth / another image's depth /
                                       a constant map / grayscale RGB)
      off                              'continued A0': same schedule and trainable predictor, NO geometry branch
                                       (fairness control: DGA must beat this, not just the original A0 checkpoint)
Resumable: last.pth is written every ckpt_every iterations; re-running the same command continues."""
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from fsod_data.dataset import DetDataset, collate_fn, detector_targets
from fsod_data.voc_fsod import TRAINVAL_SETS, build_base_train, load_records
from models.fsod.frcnn import build_detector
from models.geo.dga import MODES, DGADetector
from training.utils import save_atomic, set_seed

NEEDS_DEPTH = ("depth", "shuffle")
GEO_PREFIX = ("det.roi_heads.box_head.embed", "det.roi_heads.box_head.film", "det.roi_heads.box_head.gate")
PRED_PREFIX = ("det.roi_heads.box_predictor.", "roi_heads.box_predictor.")


def build_geo_model(ck, mode, emb=64, film_scale=0.5):
    """Plain detector carrying the A0 base weights, wrapped in DGA unless mode == 'off'."""
    assert mode == "off" or mode in MODES, f"unknown geo mode '{mode}'"
    bm = ck["cfg"]["model"]
    det = build_detector(len(ck["classes"]) + 1, pretrained_backbone=False, min_size=tuple(bm["min_size"]),
                         max_size=bm["max_size"], predictor="linear")
    det.load_state_dict(ck["model"], strict=True)
    return det if mode == "off" else DGADetector(det, mode=mode, emb=emb, film_scale=film_scale)


def model_from_geo_ckpt(ck, gk):
    """A0 base checkpoint (ck) + geo-stage checkpoint (gk) -> evaluation-ready 15-class model."""
    mode = gk["mode"]
    model = build_geo_model(ck, mode, emb=gk["emb"], film_scale=gk["film_scale"])
    pre = "" if mode == "off" else "det."
    sd = {pre + k: v for k, v in gk["predictor"].items()}
    sd.update(gk["geo"])
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:3]}"
    return model


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def train_geo(cfg, base_ckpt, out_dir, device, seed=0):
    set_seed(seed)
    P, D, G, T = cfg["paths"], cfg["data"], cfg["geo"], cfg["train"]
    mode = G["mode"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "geo_final.pth"
    if final.exists():
        print(f"[skip] {final} already exists")
        return final

    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base_classes, split = ck["classes"], ck["split"]
    assert split == cfg["split"], f"config split {cfg['split']} != base checkpoint split {split}"

    recs = load_records(P["voc_root"], TRAINVAL_SETS, cache=f'{P["cache"]}/voc_trainval.json')
    train_recs = build_base_train(recs, split)
    ds = DetDataset(train_recs, P["voc_root"], base_classes, train=True,
                    depth_dir=D["depth_dir"] if mode in NEEDS_DEPTH else None, flip_prob=D["flip_prob"])
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=T["ims_per_batch"], sampler=RandomSampler(ds, generator=g),
                        num_workers=D["num_workers"], collate_fn=collate_fn, drop_last=True,
                        persistent_workers=D["num_workers"] > 0)

    model = build_geo_model(ck, mode, emb=G.get("emb", 64), film_scale=G.get("film_scale", 0.5))
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(GEO_PREFIX + PRED_PREFIX)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"[geo] mode={mode}  train images={len(ds)}  trainable params={n_train / 1e3:.0f}K")

    n_iter, warm = T["iters"], T["warmup"]
    geo_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(GEO_PREFIX)]
    pred_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(GEO_PREFIX)]
    opt = torch.optim.SGD(pred_params, lr=T["lr"], momentum=T["momentum"], weight_decay=T["weight_decay"])
    opt_g = (torch.optim.AdamW(geo_params, lr=T.get("geo_lr", 3e-4), betas=(0.9, 0.999), eps=1e-10, weight_decay=0.0)
             if geo_params else None)

    def lr_mult(it):
        if it < warm:
            return (it + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, n_iter - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_mult) if opt_g is not None else None
    use_amp = bool(T["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ac_dev = "cuda" if device.type == "cuda" else "cpu"

    start = 0
    last = out_dir / "last.pth"
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        assert st["mode"] == mode, f"{last} was written by mode '{st['mode']}', not '{mode}'"
        cur = dict(model.named_parameters())
        for n, v in st["trainable"].items():
            cur[n].data.copy_(v)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if opt_g is not None and "opt_g" in st:
            opt_g.load_state_dict(st["opt_g"])
            sched_g.load_state_dict(st["sched_g"])
        scaler.load_state_dict(st["scaler"])
        start = st["it"]
        print(f"[resume] {last} at iter {start}")

    def batches():
        while True:
            for b in loader:
                yield b

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("iter,loss,loss_cls,loss_reg,alpha_mean,lr,seconds\n")
    model.to(device).train()
    stream = batches()
    t0, run, alpha_m = time.time(), [], float("nan")
    for it in range(start, n_iter):
        imgs, tgts = next(stream)
        imgs = [i.to(device) for i in imgs]
        tg = detector_targets(tgts, device)
        kw = {}
        if mode != "off":
            kw["depths"] = [t["depth"].to(device) for t in tgts] if mode in NEEDS_DEPTH else None
        with torch.autocast(device_type=ac_dev, enabled=use_amp):
            losses = model(imgs, tg, **kw)
            loss = losses["loss_classifier"] + losses["loss_box_reg"]   # RPN/backbone are frozen
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it} (mode {mode})")
        opt.zero_grad(set_to_none=True)
        if opt_g is not None:
            opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if opt_g is not None:
            scaler.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(params, T["grad_clip"])
        scaler.step(opt)
        if opt_g is not None:
            scaler.step(opt_g)
        scaler.update()
        sched.step()
        if sched_g is not None:
            sched_g.step()
        run.append((float(loss.detach()), float(losses["loss_classifier"].detach()),
                    float(losses["loss_box_reg"].detach())))
        if mode != "off":
            alpha_m = float(model.det.roi_heads.box_head.last_alpha.float().mean())
        if (it + 1) % T["log_every"] == 0:
            m = [sum(x[i] for x in run) / len(run) for i in range(3)]
            run = []
            with open(log_path, "a") as f:
                f.write(f"{it + 1},{m[0]:.4f},{m[1]:.4f},{m[2]:.4f},{alpha_m:.4f},"
                        f"{opt.param_groups[0]['lr']:.6f},{time.time() - t0:.0f}\n")
            print(f"  iter {it + 1}/{n_iter}  loss {m[0]:.4f} (cls {m[1]:.4f} reg {m[2]:.4f})  "
                  f"alpha {alpha_m:.3f}  lr {opt.param_groups[0]['lr']:.5f}  {time.time() - t0:.0f}s", flush=True)
        if (it + 1) % T["ckpt_every"] == 0 and it + 1 < n_iter:
            save_atomic({"it": it + 1, "mode": mode, "trainable": trainable_state(model), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                         "opt_g": opt_g.state_dict() if opt_g is not None else None,
                         "sched_g": sched_g.state_dict() if sched_g is not None else None}, last)

    sd = model.state_dict()
    predictor = {(k[len("det."):] if k.startswith("det.") else k): v.detach().cpu()
                 for k, v in sd.items() if k.startswith(PRED_PREFIX)}
    geo = {k: v.detach().cpu() for k, v in sd.items() if k.startswith(GEO_PREFIX)}
    save_atomic({"mode": mode, "emb": G.get("emb", 64), "film_scale": G.get("film_scale", 0.5),
                 "classes": base_classes, "split": split, "base_ckpt": str(base_ckpt), "iters": n_iter,
                 "seed": seed, "predictor": predictor, "geo": geo, "cfg": cfg}, final)
    if last.exists():
        last.unlink()
    print(f"[done] {final}")
    return final
