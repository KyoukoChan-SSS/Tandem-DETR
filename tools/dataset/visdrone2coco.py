"""Convert VisDrone2019-DET annotations to COCO format.

VisDrone stores one text file per image, one object per line:

    x, y, w, h, score, category, truncation, occlusion

Categories 1-10 are the ten object classes. Category 0 (ignored regions) and
category 11 (others) are not evaluated and are dropped. The category ids are
kept as they are, so the detector uses ``num_classes: 12`` with indices 0 and
11 unused.

    python tools/dataset/visdrone2coco.py --root dataset/VisDrone2019

reads   <root>/DET-{train,val}/{images,annotations}
writes  <root>/annotations_coco/VisDrone2019-DET_{train,val}_coco.json
"""

import argparse
import json
import os

from PIL import Image

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def convert(split_dir):
    image_dir = os.path.join(split_dir, "images")
    ann_dir = os.path.join(split_dir, "annotations")
    images, annotations = [], []
    for image_id, file_name in enumerate(sorted(os.listdir(image_dir))):
        stem, ext = os.path.splitext(file_name)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(os.path.join(image_dir, file_name)) as im:
            width, height = im.size
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        with open(os.path.join(ann_dir, stem + ".txt")) as f:
            for line in f:
                fields = line.strip().split(",")
                if len(fields) < 8:
                    continue
                x, y, w, h, _, category = (int(v) for v in fields[:6])
                if not 1 <= category <= 10:
                    continue
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                })
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="folder holding DET-train and DET-val")
    parser.add_argument("--out", default=None, help="output folder (default: <root>/annotations_coco)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.root, "annotations_coco")
    os.makedirs(out_dir, exist_ok=True)
    for split in args.splits:
        coco = convert(os.path.join(args.root, f"DET-{split}"))
        path = os.path.join(out_dir, f"VisDrone2019-DET_{split}_coco.json")
        with open(path, "w") as f:
            json.dump(coco, f)
        print(f"{path}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


if __name__ == "__main__":
    main()
