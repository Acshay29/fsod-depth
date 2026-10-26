"""models/geo/dga.py — Depth-conditioned Geometric Adaptation (DGA) wrapper for the torchvision Faster R-CNN.

Idea: leave the detector untouched; wrap two modules inside roi_heads:
  box_roi_pool -> also computes a geometric descriptor for every RoI from the depth map   (no parameters)
  box_head     -> f' = f + alpha * (gamma * f + beta)   FiLM on the 1024-d RoI feature
                  (gamma, beta) = Linear(embed(geo)), ZERO-initialised  => at start the model == the RGB baseline
                  alpha = sigmoid(gate(geo)) in (0,1) is a reliability gate (can switch the depth branch off)
Modes (for controls; same module, same parameters, different map fed to the descriptor):
  depth   estimated depth (the proposed model)
  shuffle depth map of ANOTHER image in the batch (geometry-free control: right statistics, wrong image)
  const   constant map (only box-geometry features carry information)
  gray    grayscale RGB used in place of depth (parameter-matched RGB-only control)"""
import torch
import torch.nn as nn

from models.geo.geometry import GEO_DIM, region_descriptors

MODES = ("depth", "shuffle", "const", "gray")


class GeoContext:
    def __init__(self, mode="depth"):
        assert mode in MODES, mode
        self.mode = mode
        self.maps = None      # per-image maps for the current forward
        self.geo = None       # (N, GEO_DIM) filled by GeoRoIPool
        self._prev = None     # last map seen (fallback for 'shuffle' with batch size 1)

    def prepare(self, images, depths):
        if self.mode == "gray":
            maps = [(0.299 * im[0] + 0.587 * im[1] + 0.114 * im[2]).detach() for im in images]
        elif self.mode == "const":
            maps = [torch.full(im.shape[-2:], 0.5, device=im.device) for im in images]
        else:
            if depths is None:
                raise ValueError("DGADetector needs depth maps (depths=[...]) in mode 'depth'/'shuffle'")
            maps = [d.to(images[0].device) for d in depths]
            if self.mode == "shuffle":
                if len(maps) > 1:
                    maps = maps[1:] + maps[:1]
                else:
                    prev, self._prev = self._prev, maps[0]
                    maps = [prev if prev is not None else maps[0].flip(-2)]
        self.maps, self.geo = maps, None

    def clear(self):
        self.maps, self.geo = None, None


class GeoRoIPool(nn.Module):
    def __init__(self, pool, ctx):
        super().__init__()
        self.pool, self.ctx = pool, ctx

    def forward(self, features, proposals, image_shapes):
        x = self.pool(features, proposals, image_shapes)
        self.ctx.geo = region_descriptors(self.ctx.maps, proposals, image_shapes)
        assert self.ctx.geo.shape[0] == x.shape[0], "RoI/descriptor row mismatch"
        return x


class DGAHead(nn.Module):
    """FiLM is BOUNDED so the depth branch can only nudge the feature, never blow it up:
    gamma = s * tanh(.),  beta = s * tanh(.) * mean|f|   with s = film_scale (default 0.5)."""

    def __init__(self, head, ctx, rep_dim=1024, emb=64, film_scale=0.5):
        super().__init__()
        self.head, self.ctx, self.film_scale = head, ctx, film_scale
        self.embed = nn.Sequential(nn.LayerNorm(GEO_DIM), nn.Linear(GEO_DIM, 128), nn.ReLU(inplace=True),
                                   nn.Linear(128, emb), nn.ReLU(inplace=True))
        self.film = nn.Linear(emb, 2 * rep_dim)
        self.gate = nn.Sequential(nn.Linear(GEO_DIM, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.last_alpha = None

    def forward(self, x):
        f = self.head(x)
        geo = self.ctx.geo
        assert geo is not None and geo.shape[0] == f.shape[0], "geometry not available for this forward"
        geo = geo.to(f.dtype)
        gamma, beta = self.film(self.embed(geo)).chunk(2, dim=1)
        scale = f.detach().abs().mean(dim=1, keepdim=True) + 1e-6
        gamma = self.film_scale * torch.tanh(gamma)
        beta = self.film_scale * torch.tanh(beta) * scale
        alpha = torch.sigmoid(self.gate(geo))
        self.last_alpha = alpha.detach()
        return f + alpha * (gamma * f + beta)


class DGADetector(nn.Module):
    """Wraps a torchvision FasterRCNN (from models.fsod.frcnn.build_detector)."""

    def __init__(self, det, mode="depth", emb=64, film_scale=0.5):
        super().__init__()
        self.det = det
        self.ctx = GeoContext(mode)
        rh = det.roi_heads
        rep = rh.box_head.fc7.out_features
        rh.box_roi_pool = GeoRoIPool(rh.box_roi_pool, self.ctx)
        rh.box_head = DGAHead(rh.box_head, self.ctx, rep_dim=rep, emb=emb, film_scale=film_scale)

    @property
    def mode(self):
        return self.ctx.mode

    def geo_parameters(self):
        h = self.det.roi_heads.box_head
        return list(h.embed.parameters()) + list(h.film.parameters()) + list(h.gate.parameters())

    def forward(self, images, targets=None, depths=None):
        self.ctx.prepare(images, depths)
        try:
            return self.det(images, targets)
        finally:
            self.ctx.clear()


def load_detector_state(dga, state):
    """Load a plain detector state dict (e.g. the A0 base checkpoint['model']) into the wrapped detector.
    Only the wrapped box_head keys change name ('roi_heads.box_head.X' -> 'roi_heads.box_head.head.X').
    The new geo modules keep their zero-init FiLM. Returns the list of missing keys (should be geo only)."""
    remap = {}
    for k, v in state.items():
        if k.startswith("roi_heads.box_head."):
            k = k.replace("roi_heads.box_head.", "roi_heads.box_head.head.", 1)
        remap[k] = v
    missing, unexpected = dga.det.load_state_dict(remap, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:3]}"
    return missing
