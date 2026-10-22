"""
Tandem-DETR
Copyright (c) 2026 The Tandem-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .hybrid_encoder import (
    ConvNormLayer_fuse,
    RepNCSPELAN4,
    SCDown,
    TransformerEncoder,
    TransformerEncoderLayer,
)

__all__ = ["TandemEncoder", "TandemFusion"]


FEAT_STRIDES = [4, 4, 8, 16, 32]   # stem, P2, P3, P4, P5
NHEAD = 8
DROPOUT = 0.0
ENC_ACT = "gelu"
USE_ENCODER_IDX = [4]              # AIFI on the stride-32 level
NUM_ENCODER_LAYERS = 1
PE_TEMPERATURE = 10000
ACT = "silu"
FUSION_DEPTH = 3


class TandemFusion(nn.Module):

    def __init__(self, c, expansion=1.0, n=FUSION_DEPTH, act=ACT):
        super().__init__()
        c4 = max(int(round(expansion * c / 2)), 1)
        self.fusion = RepNCSPELAN4(2 * c, c, 2 * c, c4, n=n, act=act)
        self.short = ConvNormLayer_fuse(2 * c, c, 1, 1, act=act)
        self.head_stem = ConvNormLayer_fuse(c, c, 1, 1, act=act)
        self.head_p2 = ConvNormLayer_fuse(c, c, 1, 1, act=act)

    def forward(self, stem, p2):
        x = torch.cat([stem, p2], dim=1)
        fused = self.fusion(x)
        fused = fused + self.short(x)
        stem_out = stem + self.head_stem(fused)
        p2_out = p2 + self.head_p2(fused)
        return stem_out, p2_out


@register()
class TandemEncoder(nn.Module):
    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels=[256, 96, 384, 768, 1536],
        hidden_dim=256,
        dim_feedforward=1024,
        depth_mult=1.0,
        expansion=1.0,
        eval_spatial_size=None,
    ):
        super().__init__()
        assert len(in_channels) == len(FEAT_STRIDES), (
            f"TandemEncoder expects {len(FEAT_STRIDES)} input levels "
            f"(stem, P2..P5); got in_channels={in_channels}."
        )

        self.in_channels = in_channels
        self.feat_strides = FEAT_STRIDES
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = USE_ENCODER_IDX
        self.num_encoder_layers = NUM_ENCODER_LAYERS
        self.pe_temperature = PE_TEMPERATURE
        self.eval_spatial_size = eval_spatial_size
        self.num_levels = len(in_channels)
        self.out_channels = [hidden_dim for _ in range(self.num_levels)]
        self.out_strides = FEAT_STRIDES
        self.stride_ratios = [
            int(FEAT_STRIDES[i + 1] // FEAT_STRIDES[i])
            for i in range(self.num_levels - 1)
        ]

        # Per-level input projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_channel, hidden_dim, 1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )
            self.input_proj.append(proj)

        # Level embedding, zero-initialised
        self.level_embed = nn.Parameter(torch.zeros(self.num_levels, hidden_dim))

        # AIFI
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=NHEAD,
            dim_feedforward=dim_feedforward,
            dropout=DROPOUT,
            activation=ENC_ACT,
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), NUM_ENCODER_LAYERS)
                for _ in range(len(USE_ENCODER_IDX))
            ]
        )

        # FPN top-down for P5->P4->P3->P2; the stem<->P2 step is TandemFusion
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for k in range(self.num_levels - 2):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
            )

        self.tandem_fusion = TandemFusion(c=hidden_dim, expansion=expansion)

        # PAN bottom-up
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            r = self.stride_ratios[i]
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, r))
            )
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

    def forward(self, feats):
        assert len(feats) == self.num_levels
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        for i in range(self.num_levels):
            proj_feats[i] = proj_feats[i] + self.level_embed[i].view(1, -1, 1, 1)

        # AIFI
        for i, enc_ind in enumerate(self.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                ).to(src_flatten.device)
            else:
                pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(
                    src_flatten.device
                )
            memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
            proj_feats[enc_ind] = (
                memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
            )

        # FPN top-down; at idx == 1 the stem<->P2 step is TandemFusion
        inner_outs = [proj_feats[-1]]
        for idx in range(self.num_levels - 1, 0, -1):
            k = self.num_levels - 1 - idx
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            if idx == 1:
                new_stem, new_p2 = self.tandem_fusion(feat_low, feat_high)
                inner_outs[0] = new_p2
                inner_outs.insert(0, new_stem)
            else:
                feat_high = self.lateral_convs[k](feat_high)
                inner_outs[0] = feat_high
                upsample_feat = F.interpolate(
                    feat_high, size=feat_low.shape[-2:], mode="nearest"
                )
                inner_out = self.fpn_blocks[k](
                    torch.concat([upsample_feat, feat_low], dim=1)
                )
                inner_outs.insert(0, inner_out)

        # PAN bottom-up
        outs = [inner_outs[0]]
        for idx in range(self.num_levels - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        return outs
