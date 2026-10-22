"""
Tandem-DETR
Copyright (c) 2026 The Tandem-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import Dict, List

import torch

from ...core import register
from .hgnetv2 import HGNetv2

__all__ = ["HGNetv2_Stem"]


@register()
class HGNetv2_Stem(HGNetv2):
    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=(0, 1, 2, 3),
        freeze_stem_only=True,
        freeze_at=-1,
        freeze_norm=False,
        pretrained=True,
        local_model_dir="weight/hgnetv2/",
    ):
        super().__init__(
            name=name,
            use_lab=use_lab,
            return_idx=list(return_idx),
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )
        # stem_channels = [in, mid, out]; StemBlock produces `out`.
        self.stem_out_channels: int = self.arch_configs[name]["stem_channels"][2]

    def forward(self, x) -> Dict[str, object]:
        stem = self.stem(x)
        feats: List[torch.Tensor] = []
        h = stem
        for idx, stage in enumerate(self.stages):
            h = stage(h)
            if idx in self.return_idx:
                feats.append(h)
        return {"feats": feats, "stem": stem}
