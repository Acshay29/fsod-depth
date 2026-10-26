"""training/finetune.py — TFA-style few-shot fine-tuning (RGB-only baseline A0).
Start from the base model, add a cosine predictor for all 20 classes (base rows copied, novel rows random),
freeze everything except that predictor, train on a balanced K-shot set (K instances per class, base + novel)."""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from fsod_data.dataset import DetDataset, collate_fn, detector_targets, support_to_records
from fsod_data.voc_fsod import VOC_CLASSES
from models.fsod.frcnn import build_detector
from models.heads.cosine_predictor import transfer_base_predictor
from training.utils import save_atomic, set_seed


def finetune(cfg, base_ckpt, seed, k, out_dir, device):
    """Returns (model, info). Saves only the small predictor weights to out_dir/predictor.pth."""
    set_seed(seed)
    P, D, M, T = cfg["paths"], cfg["data"], cfg["model"], cfg["train"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base_classes, split, bm = ck["classes"], ck["split"], ck["cfg"]["model"]
    assert split == cfg["split"], f"config split {cfg['split']} != base checkpoint split {split}"

    support = json.loads((Path(P["fewshot_dir"]) / f"seed{seed}" / f"{k}shot.json").read_text())
    recs = support_to_records(support, VOC_CLASSES)
    ds = DetDataset(recs, P["voc_root"], VOC_CLASSES, train=True, flip_prob=D["flip_prob"])
    g = torch.Generator()
    g.manual_seed(seed * 1000 + k)
    loader = DataLoader(ds, batch_size=min(T["ims_per_batch"], len(ds)), sampler=RandomSampler(ds, generator=g),
                        num_workers=D["num_workers"], collate_fn=collate_fn,
                        persistent_workers=D["num_workers"] > 0)

    def batches():
        while True:
            for b in loader:
                yield b

    model = build_detector(len(VOC_CLASSES) + 1, pretrained_backbone=False, min_size=tuple(bm["min_size"]),
                           max_size=bm["max_size"], predictor="cosine", cos_scale=M["cos_scale"])
    state = {n: v for n, v in ck["model"].items() if not n.startswith("roi_heads.box_predictor.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(n.startswith("roi_heads.box_predictor.") for n in missing), \
        f"unexpected weight mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    n_copied = transfer_base_predictor(model.roi_heads.box_predictor, ck["model"], base_classes, VOC_CLASSES)
    assert n_copied == len(base_classes)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("roi_heads.box_predictor.")

    n_iter = T["iters_by_shot"][k]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=T["lr"], momentum=T["momentum"], weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(T["step_frac"] * n_iter)], gamma=T["gamma"])
    use_amp = bool(T["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ac_dev = "cuda" if device.type == "cuda" else "cpu"

    model.to(device).train()
    stream = batches()
    last, t0 = [], time.time()
    for it in range(n_iter):
        imgs, tgts = next(stream)
        imgs = [i.to(device) for i in imgs]
        tg = detector_targets(tgts, device)
        with torch.autocast(device_type=ac_dev, enabled=use_amp):
            losses = model(imgs, tg)
            loss = losses["loss_classifier"] + losses["loss_box_reg"]  # RPN/backbone are frozen
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it} (seed {seed}, K={k})")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if it >= n_iter - 50:
            last.append(float(loss.detach()))

    pred_state = {n: v.detach().cpu() for n, v in model.state_dict().items()
                  if n.startswith("roi_heads.box_predictor.")}
    save_atomic({"predictor": pred_state, "seed": seed, "k": k, "split": split, "classes": VOC_CLASSES,
                 "model_cfg": {"min_size": bm["min_size"], "max_size": bm["max_size"],
                               "predictor": "cosine", "cos_scale": M["cos_scale"]},
                 "base_ckpt": str(base_ckpt)}, out_dir / "predictor.pth")
    info = {"iters": n_iter, "final_loss": sum(last) / max(len(last), 1), "seconds": time.time() - t0,
            "n_records": len(ds)}
    return model, info
