"""scripts/env_probe.py — environment probe for Mac dev and Colab training."""
import argparse, importlib, platform, sys, time


def try_import(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"NOT AVAILABLE ({type(e).__name__}: {str(e)[:80]})"


def run(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"[{name}] FAIL {type(e).__name__}: {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", action="store_true",
                    help="also test Depth Anything V2-Small (downloads ~100MB)")
    args = ap.parse_args()

    import torch, torchvision
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print(f"python      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"torch       : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        print(f"cuda        : True [{p.name}, {p.total_memory / 2**30:.1f} GiB]")
    else:
        print("cuda        : False")
    print(f"mps         : {mps}")
    for pkg in ["detectron2", "transformers", "pycocotools", "cv2", "numpy"]:
        print(f"{pkg:<12}: {try_import(pkg)}")

    devices = ["cpu"] + (["cuda"] if cuda else [])

    def ops():
        from torchvision.ops import nms, roi_align
        for dev in devices:
            feat = torch.randn(1, 8, 32, 32, device=dev)
            rois = torch.tensor([[0, 2., 2., 20., 20.]], device=dev)
            out = roi_align(feat, rois, output_size=7, spatial_scale=1.0,
                            sampling_ratio=2, aligned=True)
            boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                                 dtype=torch.float32, device=dev)
            keep = nms(boxes, torch.tensor([0.9, 0.8, 0.7], device=dev), 0.5)
            print(f"[ops:{dev}] roi_align {tuple(out.shape)} nms keep={keep.tolist()} OK")

    def frcnn():
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        for dev in devices:
            m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                        num_classes=16, min_size=256, max_size=320).to(dev).train()
            imgs = [torch.rand(3, 256, 320, device=dev)]
            tgts = [{"boxes": torch.tensor([[20., 30., 120., 150.]], device=dev),
                     "labels": torch.tensor([3], device=dev)}]
            loss = sum(m(imgs, tgts).values())
            loss.backward()
            print(f"[frcnn:{dev}] train step loss={loss.item():.3f} OK")

    def depth():
        import numpy as np
        from PIL import Image
        from transformers import pipeline
        dev = 0 if cuda else ("mps" if mps else -1)
        pipe = pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
        img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype("uint8"))
        pipe(img)  # warm-up
        n, t = 5, time.time()
        for _ in range(n):
            out = pipe(img)
        d = out["predicted_depth"]
        print(f"[depth] out {tuple(d.shape)}, {(time.time() - t) / n * 1000:.0f} ms/img, device={dev}")

    run("ops", ops)
    run("frcnn", frcnn)
    if args.depth:
        run("depth", depth)


if __name__ == "__main__":
    main()