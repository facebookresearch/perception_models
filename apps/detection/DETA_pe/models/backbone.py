# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
from functools import partial
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.cuda.amp import autocast
from torchvision.models._utils import IntermediateLayerGetter
from util.misc import is_main_process, NestedTensor

from .position_encoding import build_position_encoding
from .swin import get_swinl
from .pev1 import get_pev1_and_fpn_backbone


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(
        self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer,
        )
        assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class SwinBackbone(nn.Module):
    def __init__(self):
        # we skip R50 FrozenBatchNorm2d, dilation, train l{2,3,4} only
        super().__init__()
        self.body = get_swinl()
        self.features = ["res3", "res4", "res5"]
        self.strides = [8, 16, 32]
        self.num_channels = [384, 768, 1536]

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        m = tensor_list.mask[None]
        assert m is not None
        out: Dict[str, NestedTensor] = {}
        for name in self.features:
            mask = F.interpolate(m.float(), size=xs[name].shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(xs[name], mask)
        return out


class PEv1Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.body = get_pev1_and_fpn_backbone(args)
        self.features = self.body._out_features

        self.bf16 = args.bf16
        self.fp16 = args.fp16

        _out_feature_strides = self.body._out_feature_strides
        _out_feature_channels = self.body._out_feature_channels
        self.strides = [_out_feature_strides[f] for f in _out_feature_strides.keys()]
        self.num_channels = [
            _out_feature_channels[f] for f in _out_feature_channels.keys()
        ]

    def forward(self, tensor_list: NestedTensor):
        # xs = self.body(tensor_list.tensors)
        # backbone
        if self.bf16:
            with autocast(dtype=torch.bfloat16):
                xs = self.body(tensor_list.tensors.to(torch.bfloat16))
            xs = {k: v.float() for k, v in xs.items()}
        elif self.fp16:
            with autocast(dtype=torch.float16):
                xs = self.body(tensor_list.tensors.half())
            xs = {k: v.float() for k, v in xs.items()}
        else:
            xs = self.body(tensor_list.tensors)

        m = tensor_list.mask[None]
        assert m is not None
        out: Dict[str, NestedTensor] = {}

        for name in self.features:
            mask = F.interpolate(m.float(), size=xs[name].shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(xs[name], mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if "swin" in args.backbone:
        backbone = SwinBackbone()
    elif "pev1" in args.backbone:
        backbone = PEv1Backbone(args)
    else:
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation
        )
    model = Joiner(backbone, position_embedding)
    return model
