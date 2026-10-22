"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import argparse

import torch
import torch.nn as nn
from calflops import calculate_flops

from src.core import YAMLConfig


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=None)
    for name in ("HGNetv2", "HGNetv2_Stem"):
        if name in cfg.yaml_cfg:
            cfg.yaml_cfg[name]["pretrained"] = False
    height, width = cfg.yaml_cfg["eval_spatial_size"]

    class Model_for_flops(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            outputs = self.model(images)
            return outputs

    model = Model_for_flops().eval()

    flops, macs, _ = calculate_flops(
        model=model, input_shape=(1, 3, height, width), output_as_string=True,
        output_precision=4, print_detailed=False
    )
    params = sum(p.numel() for p in model.parameters())
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", required=True, type=str
    )
    args = parser.parse_args()

    main(args)
