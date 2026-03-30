import torch.nn as nn
import torch.nn.functional as F

from .Mamba_backbone import Backbone_VSSM
from .model_utils import _ACTLAYERS, _NORMLAYERS


def build_backbone(pretrained, **kwargs):
    return Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)


def resolve_decoder_components(kwargs):
    norm_layer = _NORMLAYERS.get(kwargs["norm_layer"].lower(), None)
    ssm_act_layer = _ACTLAYERS.get(kwargs["ssm_act_layer"].lower(), None)
    mlp_act_layer = _ACTLAYERS.get(kwargs["mlp_act_layer"].lower(), None)
    clean_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"norm_layer", "ssm_act_layer", "mlp_act_layer"}
    }
    return norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs


def build_head(out_channels, in_channels=128):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


def resize_to_input(logits, reference):
    return F.interpolate(logits, size=reference.shape[-2:], mode="bilinear")
