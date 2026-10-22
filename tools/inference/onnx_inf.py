"""Image/video inference for exported Tandem-DETR models.

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified for Tandem-DETR: match evaluation resize and original image coordinates.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def prepare_inputs(image, input_size):
    height, width = input_size
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return {
        "images": np.ascontiguousarray(pixels[None]),
        "orig_target_sizes": np.array([image.size], dtype=np.int64),
    }


def predict(session, image, input_size, threshold):
    labels, boxes, scores = session.run(
        ["labels", "boxes", "scores"], prepare_inputs(image, input_size)
    )
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        if score >= threshold:
            draw.rectangle(box.tolist(), outline="red", width=2)
            draw.text(tuple(box[:2]), f"{label} {score:.2f}", fill="blue")
    return result


def main(args):
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    shape = next(value.shape for value in session.get_inputs() if value.name == "images")
    if isinstance(shape[0], int) and shape[0] != 1:
        raise ValueError("This image/video tool requires batch 1 or a dynamic batch model")
    input_size = shape[2:]
    if len(input_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in input_size):
        raise ValueError("Export a model with fixed spatial dimensions")
    path = Path(args.input)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        with Image.open(path) as image:
            result = predict(session, image.convert("RGB"), input_size, args.threshold)
        output = args.output or "onnx_result.jpg"
        result.save(output)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        output = args.output or "onnx_result.mp4"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 cap.get(cv2.CAP_PROP_FPS), (width, height))
        try:
            if not writer.isOpened():
                raise ValueError(f"Cannot write video: {output}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = predict(session, image, input_size, args.threshold)
                writer.write(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR))
        finally:
            cap.release()
            writer.release()
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float, default=0.4)
    main(parser.parse_args())
