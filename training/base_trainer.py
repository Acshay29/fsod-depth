"""training/base_trainer.py — base-class training of the RGB-only Faster R-CNN baseline (A0)."""
import csv
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

from fsod_data.dataset import DetDataset, collate_fn, detector_targets
from fsod_data.voc_fsod import TRAINVAL_SETS, build_base_train, load_records, split_classes
from models.fsod.frcnn import build_detector, count_params
from training.utils import pick_device, save_atomic, set_seed


def _worker_init(_):
    import random
    import numpy as np
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s)
    np.random.seed(s)


def train_base(cfg):
    seed = cfg["seed"]
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))
    P, D, M, T = cfg["paths"], cfg["data"], cfg["model"], cfg["train"]
    out = Path(P["output_dir"]) / f'{cfg["experiment"]}_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))

    # ---- data: base-class annotations only (no novel objects leak into base training)
    recs = load_records(P["voc_root"], TRAINVAL_SETS, cache=f'{P["cache"]}/voc_trainval.json')
    base, _ = split_classes(cfg["split"])
    train_recs = build_base_train(recs, cfg["split"])
    if D.get("subset"):
        train_recs = train_recs[: D["subset"]]
    assert len(train_recs) >= T["ims_per_batch"], "fewer training images than batch size"
    ds = DetDataset(train_recs, P["voc_root"], base, train=True,
                    depth_dir=D["depth_dir"] if D.get("use_depth") else None, flip_prob=D["flip_prob"])
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=T["ims_per_batch"], sampler=RandomSampler(ds, generator=g),
                        num_workers=D["num_workers"], collate_fn=collate_fn, drop_last=True,
                        worker_init_fn=_worker_init, persistent_workers=D["num_workers"] > 0)

    def batches():
        while True:
            for b in loader:
                yield b

    # ---- model / optimiser / schedule
    model = build_detector(len(base) + 1, pretrained_backbone=M["pretrained_backbone"],
                           min_size=tuple(M["min_size"]), max_size=M["max_size"],
                           trainable_layers=M["trainable_layers"]).to(device)
    total, trainable = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=T["lr"], momentum=T["momentum"], weight_decay=T["weight_decay"])

    def lr_lambda(it):
        f = T["gamma"] ** sum(1 for s in T["steps"] if it >= s)
        if it < T["warmup_iters"]:
            f *= 0.001 + 0.999 * it / T["warmup_iters"]
        return f

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = bool(T["amp"]) and device.type == "cuda"
    ac_dev = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume
    start = 0
    last = out / "last.pth"
    if last.exists() and not cfg.get("no_resume"):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["iter"]
        print(f"[resume] {last} at iter {start}")

    print(f"exp={out.name} device={device} amp={use_amp} train_images={len(ds)} classes={len(base)} "
          f"params={total / 1e6:.1f}M (trainable {trainable / 1e6:.1f}M) iters {start}->{T['max_iter']}")

    log_path = out / "train_log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    w = csv.writer(log_f)
    if new_log:
        w.writerow(["iter", "loss", "loss_classifier", "loss_box_reg", "loss_objectness",
                    "loss_rpn_box_reg", "lr", "sec_per_iter", "gpu_mem_gb"])

    def save(it_done):
        save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "iter": it_done}, last)

    model.train()
    stream = batches()
    acc, n_acc, t_log = {}, 0, time.time()
    for it in range(start, T["max_iter"]):
        imgs, tgts = next(stream)
        imgs = [i.to(device) for i in imgs]
        tg = detector_targets(tgts, device)
        with torch.autocast(device_type=ac_dev, enabled=use_amp):
            losses = model(imgs, tg)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it}: {({k: float(v) for k, v in losses.items()})}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        acc["loss"] = acc.get("loss", 0.0) + float(loss.detach())
        for k, v in losses.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        n_acc += 1
        done = it + 1
        if done % T["log_every"] == 0 or done == T["max_iter"]:
            spi = (time.time() - t_log) / n_acc
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            row = [done, acc["loss"] / n_acc] + [acc.get(k, 0.0) / n_acc for k in
                                                 ("loss_classifier", "loss_box_reg", "loss_objectness",
                                                  "loss_rpn_box_reg")] + [opt.param_groups[0]["lr"], spi, mem]
            w.writerow([f"{x:.6g}" if isinstance(x, float) else x for x in row])
            log_f.flush()
            eta_h = spi * (T["max_iter"] - done) / 3600
            print(f"iter {done}/{T['max_iter']} loss {row[1]:.3f} lr {row[6]:.5f} "
                  f"{spi:.2f}s/it gpu {mem:.1f}GB eta {eta_h:.1f}h")
            acc, n_acc, t_log = {}, 0, time.time()
        if done % T["ckpt_every"] == 0 or done == T["max_iter"]:
            save(done)

    save_atomic({"model": model.state_dict(), "classes": base, "split": cfg["split"], "cfg": cfg},
                out / "model_final.pth")
    log_f.close()
    print(f"done. final weights: {out / 'model_final.pth'}")
