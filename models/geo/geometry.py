"""models/geo/geometry.py — region-level geometric descriptor computed ONLY from an estimated depth map
(plus the box itself). No learnable parameters, no depth-model internals.

For every RoI (x1,y1,x2,y2 in resized-image pixels) we compute GEO_DIM numbers:
  inside the box : mean, std, min, max of depth; mean gradient magnitude
  vs surroundings: mean depth minus image mean; mean depth minus the ring around the box (figure/ground
                   contrast); gradient inside vs ring; centre-minus-border depth (shape/occlusion cue)
  box geometry   : log(w/W), log(h/H), log(w/h), cy/H, cx/W
  image level    : depth std and mean gradient of the whole image (used by the reliability gate)
Depth convention: [0,1], bright = near (same as the cache)."""
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

GEO_DIM = 16
_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
_SOBEL_Y = _SOBEL_X.transpose(2, 3).contiguous()


def _grad_mag(d):
    """d: (1,1,h,w) -> log1p(100 * gradient magnitude), same shape."""
    kx, ky = _SOBEL_X.to(d), _SOBEL_Y.to(d)
    p = F.pad(d, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(p, kx), F.conv2d(p, ky)
    return torch.log1p(100.0 * torch.sqrt(gx * gx + gy * gy + 1e-12))


def _expand(boxes, factor, h, w):
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    bw, bh = (boxes[:, 2] - boxes[:, 0]) * factor, (boxes[:, 3] - boxes[:, 1]) * factor
    out = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)
    out[:, 0::2] = out[:, 0::2].clamp(0, w)
    out[:, 1::2] = out[:, 1::2].clamp(0, h)
    return out


def _border_mask(n, device):
    m = torch.ones(n, n, dtype=torch.bool, device=device)
    m[1:-1, 1:-1] = False
    return m


@torch.no_grad()
def region_descriptors(depth_maps, proposals, image_shapes):
    """depth_maps: list of (H0,W0) or (1,H0,W0) float tensors in [0,1], one per image (any resolution).
    proposals: list of (n_i,4) float boxes in the resized image frame. image_shapes: list of (h,w).
    Returns (sum n_i, GEO_DIM) float32, rows in the same order as the concatenated proposals."""
    out = []
    for dm, bx, (h, w) in zip(depth_maps, proposals, image_shapes):
        dev = bx.device
        if bx.shape[0] == 0:
            out.append(torch.zeros(0, GEO_DIM, device=dev))
            continue
        d = dm.to(dev).float().reshape(1, 1, *dm.shape[-2:])
        d = F.interpolate(d, size=(int(h), int(w)), mode="bilinear", align_corners=False).clamp(0, 1)
        g = _grad_mag(d)
        maps = torch.cat([d, g], dim=1)
        bx = bx.detach().float()
        win = roi_align(maps, [bx], output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True)
        ctx = roi_align(maps, [_expand(bx, 1.5, h, w)], output_size=7, spatial_scale=1.0, sampling_ratio=2,
                        aligned=True)
        din, gin, dctx, gctx = win[:, 0], win[:, 1], ctx[:, 0], ctx[:, 1]
        bm = _border_mask(7, dev)
        flat = din.flatten(1)
        mean_in, std_in = flat.mean(1), flat.std(1)
        ring_d, ring_g = dctx[:, bm].mean(1), gctx[:, bm].mean(1)
        centre = din[:, 2:5, 2:5].flatten(1).mean(1)
        border = din[:, bm].mean(1)
        bw = (bx[:, 2] - bx[:, 0]).clamp(min=1.0)
        bh = (bx[:, 3] - bx[:, 1]).clamp(min=1.0)
        img_mean, img_std, img_grad = d.mean(), d.std(), g.mean()
        n = bx.shape[0]
        feats = torch.stack([
            mean_in, std_in, flat.amin(1), flat.amax(1),
            mean_in - img_mean, mean_in - ring_d,
            gin.flatten(1).mean(1), ring_g, centre - border,
            torch.log((bw / w).clamp(min=1e-3)), torch.log((bh / h).clamp(min=1e-3)), torch.log(bw / bh),
            (bx[:, 1] + bx[:, 3]) / 2 / h, (bx[:, 0] + bx[:, 2]) / 2 / w,
            img_std.expand(n), img_grad.expand(n)], dim=1)
        out.append(torch.nan_to_num(feats))
    return torch.cat(out, 0)
