"""Configuration and preprocessing regressions for the release tools."""

import unittest

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from src.core import YAMLConfig
from src.zoo.tandem.tandem_decoder import TandemTransformer
from tools.inference.onnx_inf import prepare_inputs
from tools.deployment.export_onnx import verify_outputs


class ReleaseTests(unittest.TestCase):
    def test_export_comparison_accepts_only_matching_detection_sets(self):
        expected = [np.array([[1, 1]]),
                    np.array([[[0., 0., 10., 20.], [50., 60., 70., 80.]]]),
                    np.array([[0.5, 0.5]])]
        reordered = [value[:, ::-1].copy() for value in expected]
        verify_outputs(expected, reordered)
        for index in range(3):
            changed = [value.copy() for value in reordered]
            changed[index][0, 0] += 1
            with self.subTest(output=index), self.assertRaises(AssertionError):
                verify_outputs(expected, changed)
        duplicated = [np.repeat(value[:, :1], 2, axis=1) for value in expected]
        with self.assertRaises(AssertionError):
            verify_outputs(expected, duplicated)

    def test_traced_decoder_anchor_selection_accepts_larger_batch(self):
        class DecoderInput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = TandemTransformer(
                    num_classes=3, hidden_dim=8, num_queries=2,
                    feat_channels=[8], feat_strides=[4], num_levels=1,
                    nhead=2, num_layers=1, dim_feedforward=16,
                    num_denoising=0, eval_spatial_size=[8, 8],
                )

            def forward(self, memory):
                return self.decoder._get_decoder_input(memory, [(2, 2)])[:2]

        module = DecoderInput().eval()
        with torch.no_grad():
            traced = torch.jit.trace(module, torch.randn(1, 4, 8))
            for batch in (1, 2, 3):
                memory = torch.randn(batch, 4, 8)
                for actual, expected in zip(traced(memory), module(memory)):
                    torch.testing.assert_close(actual, expected)

    def test_checkpoint_skips_pretrained_backbone(self):
        for option in ("resume", "tuning"):
            for size in ("m", "l", "x"):
                with self.subTest(option=option, size=size):
                    cfg = YAMLConfig(f"configs/tandem/tandem_{size}.yml",
                                     **{option: "checkpoint.pth"})
                    self.assertFalse(cfg.yaml_cfg["HGNetv2_Stem"]["pretrained"])

    def test_configs_do_not_leak_between_loads(self):
        YAMLConfig("configs/tandem/tandem_m.yml", resume="checkpoint.pth")
        cfg = YAMLConfig("configs/tandem/tandem_l.yml")
        self.assertTrue(cfg.yaml_cfg["HGNetv2_Stem"]["pretrained"])
        self.assertNotIn("use_lab", cfg.yaml_cfg["HGNetv2_Stem"])
        self.assertFalse(cfg.yaml_cfg.get("resume"))

    def test_onnx_preprocessing_matches_torch_on_non_square_image(self):
        pixels = np.random.default_rng(0).integers(0, 256, (93, 157, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        inputs = prepare_inputs(image, (80, 96))
        expected = T.Compose([T.Resize((80, 96)), T.ToTensor()])(image)[None].numpy()
        np.testing.assert_array_equal(inputs["images"], expected)
        np.testing.assert_array_equal(inputs["orig_target_sizes"], [[157, 93]])
        self.assertEqual(inputs["orig_target_sizes"].dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
