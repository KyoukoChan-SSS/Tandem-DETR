"""
Tandem-DETR
Copyright (c) 2026 The Tandem-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register
from .hybrid_encoder import ConvNormLayer_fuse

__all__ = ["TandemDETR"]


@register()
class TandemDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        self.stem_proj = ConvNormLayer_fuse(
            ch_in=backbone.stem_out_channels,
            ch_out=encoder.hidden_dim,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x, targets=None):
        out = self.backbone(x)
        feats, stem = out["feats"], out["stem"]
        stem = self.stem_proj(stem)
        assert stem.shape[-2:] == feats[0].shape[-2:], (
            f"stem {tuple(stem.shape[-2:])} and P2 {tuple(feats[0].shape[-2:])} "
            f"must share spatial size"
        )
        feats = self.encoder([stem, *feats])
        return self.decoder(feats, targets)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
