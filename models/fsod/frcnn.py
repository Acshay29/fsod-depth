"""models/fsod/frcnn.py — Faster R-CNN (ResNet-50 FPN) builder for the TFA-style RGB-only baseline (A0)."""
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import misc as misc_nn_ops


def build_detector(num_classes, pretrained_backbone=True, min_size=(480, 512, 544, 576, 608, 640),
                   max_size=1000, trainable_layers=3, predictor="linear", cos_scale=20.0, **kw):
    """num_classes INCLUDES background (base training on 15 classes -> 16).
    min_size as a tuple = random scale choice in training; the LAST entry is used at test time.
    predictor: 'linear' (standard, used in base training) or 'cosine' (TFA w/cos, used in few-shot fine-tuning).
    Frozen BatchNorm is used in every case, so a model built with pretrained_backbone=False has exactly the
    same layers as a trained one and its checkpoint loads cleanly."""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    backbone = resnet50(weights=weights, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size, **kw)
    if predictor == "cosine":
        from models.heads.cosine_predictor import CosinePredictor
        in_ch = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = CosinePredictor(in_ch, num_classes, cos_scale)
    elif predictor != "linear":
        raise ValueError(f"unknown predictor '{predictor}'")
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
