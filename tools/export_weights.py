"""Strip a training checkpoint down to the weights that are evaluated.

A training checkpoint holds the raw model, the EMA model, the optimizer and the
schedulers. Evaluation uses the EMA weights, so the released file keeps only
those, stored under the ``model`` key as in the official D-FINE releases.

    python tools/export_weights.py best_stg2.pth tandem_detr_m_visdrone.pth
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="training checkpoint (e.g. best_stg2.pth)")
    parser.add_argument("dst", help="output weight file")
    args = parser.parse_args()

    state = torch.load(args.src, map_location="cpu")
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    torch.save({"model": weights}, args.dst)
    n = sum(v.numel() for v in weights.values())
    print(f"{args.src} -> {args.dst}: {len(weights)} tensors, {n:,} values")


if __name__ == "__main__":
    main()
