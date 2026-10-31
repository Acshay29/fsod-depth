"""scripts/plan_path.py — software-only robotics demo: detect a target object, locate it with the estimated depth map,
build an obstacle grid from the depth map and plan a collision-free path to it with A*.

  python scripts/plan_path.py --base-ckpt ~/Downloads/model_final.pth --target bottle --n 4
  python scripts/plan_path.py --selftest
"""
import argparse
import heapq
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def obstacle_grid(depth, box, cell, margin=0.12, near_is_high=True):
    d = depth if near_is_high else 1.0 - depth
    h, w = d.shape
    gh, gw = h // cell, w // cell
    D = cv2.resize(d, (gw, gh), interpolation=cv2.INTER_AREA)
    rowmed = np.median(D, axis=1, keepdims=True)
    stand_out = D - rowmed
    blocked = stand_out > margin
    x1, y1, x2, y2 = box
    c1, r1 = max(int(x1 // cell) - 1, 0), max(int(y1 // cell) - 1, 0)
    c2, r2 = min(int(x2 // cell) + 1, gw - 1), min(int(y2 // cell) + 1, gh - 1)
    blocked[r1:r2 + 1, c1:c2 + 1] = False
    cost = 1.0 + 3.0 * np.clip(stand_out, 0, 1)
    return blocked, cost, D


def astar(blocked, cost, start, goal):
    gh, gw = blocked.shape
    blocked = blocked.copy()
    for (r, c) in (start, goal):
        blocked[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = False
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    hfun = lambda p: math.hypot(p[0] - goal[0], p[1] - goal[1])
    open_ = [(hfun(start), 0.0, start)]
    came, g = {start: None}, {start: 0.0}
    while open_:
        _, gc, cur = heapq.heappop(open_)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        if gc > g.get(cur, 1e18):
            continue
        for dr, dc in moves:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < gh and 0 <= nc < gw) or blocked[nr, nc]:
                continue
            if dr and dc and (blocked[cur[0] + dr, cur[1]] or blocked[cur[0], cur[1] + dc]):
                continue
            ng = gc + math.hypot(dr, dc) * cost[nr, nc]
            if ng < g.get((nr, nc), 1e18):
                g[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(open_, (ng + hfun((nr, nc)), ng, (nr, nc)))
    return None


def plan(depth, box, near_is_high=True, margin=0.12):
    h, w = depth.shape
    cell = max(8, max(h, w) // 48)
    blocked, cost, D = obstacle_grid(depth, box, cell, margin, near_is_high)
    gh, gw = blocked.shape
    start = (gh - 1, gw // 2)
    x1, y1, x2, y2 = box
    goal = (min(int(y2 // cell), gh - 1), min(max(int((x1 + x2) / 2 // cell), 0), gw - 1))
    path = astar(blocked, cost, start, goal)
    info = {"cell": cell, "start": start, "goal": goal, "blocked_frac": float(blocked.mean()),
            "target_depth": float(np.median((depth if near_is_high else 1 - depth)[int(y1):int(y2), int(x1):int(x2)])),
            "centre": ((x1 + x2) / 2, (y1 + y2) / 2),
            "path_cells": None if path is None else len(path)}
    return path, blocked, info


def draw(img, box, label, path, blocked, info):
    vis = img.copy()
    cell = info["cell"]
    ov = vis.copy()
    gh, gw = blocked.shape
    for r in range(gh):
        for c in range(gw):
            if blocked[r, c]:
                cv2.rectangle(ov, (c * cell, r * cell), ((c + 1) * cell, (r + 1) * cell), (0, 0, 255), -1)
    vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(vis, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    if path:
        pts = np.array([[c * cell + cell // 2, r * cell + cell // 2] for r, c in path], np.int32)
        cv2.polylines(vis, [pts], False, (255, 255, 0), 3)
    sr, sc = info["start"]
    gr, gc = info["goal"]
    cv2.circle(vis, (sc * cell + cell // 2, sr * cell + cell // 2), 8, (255, 0, 0), -1)
    cv2.putText(vis, "START", (sc * cell - 10, sr * cell - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.circle(vis, (gc * cell + cell // 2, gr * cell + cell // 2), 8, (0, 255, 255), -1)
    status = f"path found ({info['path_cells']} cells)" if path else "NO PATH"
    cv2.putText(vis, f"A*: {status}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def selftest(out):
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    depth = (0.15 + 0.55 * (yy / h)).astype(np.float32)
    depth[250:420, 120:230] = 0.95
    depth[300:430, 380:470] = 0.95
    box = (300, 130, 360, 210)
    depth[130:210, 300:360] = 0.35
    img = cv2.applyColorMap((np.clip(depth, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    path, blocked, info = plan(depth, box)
    vis = draw(img, box, "target (synthetic)", path, blocked, info)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, vis)
    print("selftest:", "path found, length", info["path_cells"], "cells" if path else "NO PATH")
    assert path is not None, "selftest failed: no path"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base-ckpt")
    ap.add_argument("--config", default="configs/geo_split1.yaml")
    ap.add_argument("--target", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.12)
    ap.add_argument("--near-is-low", action="store_true")
    ap.add_argument("--out", default="demo_out/path_demo.png")
    a = ap.parse_args()
    if a.selftest:
        selftest("demo_out/path_selftest.png")
        return
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fsod_data.voc_fsod import TEST_SETS, load_records
    from preprocessing.depth_utils import depth_path, load_depth
    from training.geo_trainer import build_geo_model
    from training.utils import load_config

    cfg = load_config(a.config)
    P = cfg["paths"]
    ck = torch.load(a.base_ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    if a.target and a.target not in classes:
        sys.exit(f"--target must be one of the base classes: {classes}")
    model = build_geo_model(ck, "off").eval()
    recs = [r for r in load_records(P["voc_root"], TEST_SETS, cache=f'{P["cache"]}/voc_test.json') if r["objects"]]
    rows, done = [], 0
    for r in recs[::7]:
        if done >= a.n:
            break
        bgr = cv2.imread(str(Path(P["voc_root"]) / r["path"]))
        h, w = bgr.shape[:2]
        x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = model([x])[0]
        cand = [(float(s), b, int(l)) for b, s, l in zip(out["boxes"], out["scores"], out["labels"])
                if float(s) >= a.thr and (a.target is None or classes[int(l) - 1] == a.target)]
        if not cand:
            continue
        s, b, l = max(cand, key=lambda t: t[0])
        d = load_depth(depth_path(cfg["data"]["depth_dir"], r["path"]), expected_hw=(h, w))
        path, blocked, info = plan(d, tuple(float(v) for v in b), not a.near_is_low, a.margin)
        label = f"{classes[l - 1]} {s:.2f} depth {info['target_depth']:.2f}"
        vis = draw(bgr, b, label, path, blocked, info)
        print(f"{r['path']}: {label} centre=({info['centre'][0]:.0f},{info['centre'][1]:.0f}) path_cells={info['path_cells']}")
        rows.append(cv2.resize(vis, (800, int(800 * h / w))))
        done += 1
    if not rows:
        sys.exit("no image with a matching detection; try a different --target or lower --thr")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows))
    print("saved", a.out)


if __name__ == "__main__":
    main()
