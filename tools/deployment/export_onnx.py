"""Export Tandem-DETR with its detection postprocessor.

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR).
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
Modified for Tandem-DETR: config-sized export and runtime verification.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn

from src.core import YAMLConfig


class DeploymentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def verify_outputs(expected, actual):
    """Match every detection; equal-score top-k outputs may change order."""
    from scipy.optimize import linear_sum_assignment

    for reference, result in zip(expected, actual):
        if reference.shape != result.shape or not np.isfinite(result).all():
            raise AssertionError("ONNX output shape or finiteness mismatch")
    for batch in range(expected[0].shape[0]):
        labels, boxes, scores = [value[batch] for value in expected]
        other_labels, other_boxes, other_scores = [value[batch] for value in actual]
        matches = labels[:, None] == other_labels[None, :]
        matches &= np.isclose(other_scores[None, :], scores[:, None], rtol=1e-4, atol=1e-4)
        matches &= np.isclose(other_boxes[None, :, :], boxes[:, None, :],
                              rtol=1e-4, atol=1e-2).all(axis=-1)
        rows, columns = linear_sum_assignment(~matches)
        if not matches[rows, columns].all():
            count = int((~matches[rows, columns]).sum())
            raise AssertionError(f"{count} unmatched detections in batch item {batch}")


@torch.inference_mode()
def main(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    model = DeploymentModel(cfg).eval().to(args.device)
    height, width = cfg.yaml_cfg["eval_spatial_size"]
    torch.manual_seed(0)
    data = torch.rand(args.batch_size, 3, height, width, device=args.device)
    # Postprocessor expects (width, height), unlike eval_spatial_size.
    sizes = torch.tensor([[width, height]], device=args.device).repeat(args.batch_size, 1)
    output = Path(args.output) if args.output else Path(args.resume).with_suffix(".onnx")
    if output.resolve() == Path(args.resume).resolve():
        raise ValueError("output must not overwrite the checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["images", "orig_target_sizes", "labels", "boxes", "scores"]
    dynamic_axes = {name: {0: "batch"} for name in names} if args.dynamic_batch else None
    torch.onnx.export(
        model, (data, sizes), str(output),
        input_names=names[:2], output_names=names[2:],
        dynamic_axes=dynamic_axes, opset_version=17,
        dynamo=False, do_constant_folding=True,
    )

    if args.simplify:
        import onnx
        import onnxsim

        simplified, valid = onnxsim.simplify(onnx.load(str(output)))
        if not valid:
            raise RuntimeError("ONNX simplification validation failed")
        onnx.save(simplified, str(output))

    if args.check:
        import onnx

        onnx.checker.check_model(str(output))

    if args.verify:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = torch.get_num_threads()
        session = ort.InferenceSession(str(output), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        batches = sorted({1, 2, args.batch_size}) if args.dynamic_batch else [args.batch_size]
        for batch in batches:
            images = torch.rand(batch, 3, height, width, device=args.device)
            original_sizes = torch.tensor([[1280, 720]], device=args.device).repeat(batch, 1)
            expected = model(images, original_sizes)
            actual = session.run(names[2:], {
                "images": images.cpu().numpy(),
                "orig_target_sizes": original_sizes.cpu().numpy(),
            })
            verify_outputs([value.cpu().numpy() for value in expected], actual)
            print(f"ONNX Runtime matches PyTorch: batch={batch}")
    print(f"Saved {output} ({height}x{width}, dynamic_batch={args.dynamic_batch})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--verify", action="store_true", help="compare ONNX Runtime with PyTorch")
    main(parser.parse_args())
