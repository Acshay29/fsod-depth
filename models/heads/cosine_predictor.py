"""models/heads/cosine_predictor.py — cosine-similarity box predictor (TFA 'w/cos') + base->all weight transfer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosinePredictor(nn.Module):
    """Drop-in replacement for torchvision's FastRCNNPredictor.
    num_classes INCLUDES background (index 0). Class scores = scale * cos(feature, class weight)."""

    def __init__(self, in_channels, num_classes, scale=20.0):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes, bias=False)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.scale = scale
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        xn = F.normalize(x, p=2, dim=1, eps=1e-5)
        wn = F.normalize(self.cls_score.weight, p=2, dim=1, eps=1e-5)
        return self.scale * (xn @ wn.t()), self.bbox_pred(x)


@torch.no_grad()
def transfer_base_predictor(new_pred, base_state, base_classes, all_classes):
    """Copy background + base-class rows from a trained linear predictor (state dict of the base model)
    into a new CosinePredictor. Novel-class rows keep their random initialisation."""
    w = base_state["roi_heads.box_predictor.cls_score.weight"]
    bw = base_state["roi_heads.box_predictor.bbox_pred.weight"]
    bb = base_state["roi_heads.box_predictor.bbox_pred.bias"]
    pairs = [(0, 0)] + [(i + 1, all_classes.index(c) + 1) for i, c in enumerate(base_classes)]
    for old, new in pairs:
        new_pred.cls_score.weight[new].copy_(w[old])
        new_pred.bbox_pred.weight[4 * new:4 * new + 4].copy_(bw[4 * old:4 * old + 4])
        new_pred.bbox_pred.bias[4 * new:4 * new + 4].copy_(bb[4 * old:4 * old + 4])
    return len(pairs) - 1
