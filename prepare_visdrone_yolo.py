from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET annotations to YOLO format.")
    parser.add_argument("--source", type=Path, default=Path("archive"), help="Extracted archive root.")
    parser.add_argument("--output", type=Path, default=Path("data/visdrone_yolo"), help="Output YOLO dataset root.")
    parser.add_argument("--train-limit", type=int, default=0, help="Optional sampled train image count. 0 means all.")
    parser.add_argument("--val-limit", type=int, default=0, help="Optional sampled val image count. 0 means all.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def split_root(source: Path, split: str) -> Path:
    if split == "train":
        return source / "VisDrone2019-DET-train" / "VisDrone2019-DET-train"
    if split == "val":
        return source / "VisDrone2019-DET-val" / "VisDrone2019-DET-val"
    raise ValueError(f"Unsupported split: {split}")


def sample_images(images: list[Path], limit: int, seed: int) -> list[Path]:
    images = sorted(images)
    if limit <= 0 or limit >= len(images):
        return images
    rng = random.Random(seed)
    return sorted(rng.sample(images, limit))


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_annotation(ann_path: Path, image_size: tuple[int, int]) -> tuple[list[str], Counter]:
    width, height = image_size
    lines: list[str] = []
    class_counts: Counter = Counter()

    if not ann_path.exists():
        return lines, class_counts

    for raw in ann_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) < 6:
            continue
        try:
            x, y, w, h = (float(parts[i]) for i in range(4))
            category = int(float(parts[5]))
        except ValueError:
            continue

        if category < 1 or category > 10 or w <= 1 or h <= 1:
            continue

        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + w))
        y2 = max(0.0, min(float(height), y + h))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            continue

        cls = category - 1
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        nw = bw / width
        nh = bh / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        class_counts[VISDRONE_NAMES[cls]] += 1

    return lines, class_counts


def convert_split(source: Path, output: Path, split: str, limit: int, seed: int) -> dict:
    src_root = split_root(source, split)
    src_images = src_root / "images"
    src_annotations = src_root / "annotations"
    dst_images = output / "images" / split
    dst_labels = output / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    selected_images = sample_images(list(src_images.glob("*.jpg")), limit, seed)
    split_counts: Counter = Counter()
    empty_labels = 0

    for idx, image_path in enumerate(selected_images, start=1):
        with Image.open(image_path) as image:
            image_size = image.size
        label_lines, class_counts = convert_annotation(src_annotations / f"{image_path.stem}.txt", image_size)

        link_or_copy(image_path, dst_images / image_path.name)
        (dst_labels / f"{image_path.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        if not label_lines:
            empty_labels += 1
        split_counts.update(class_counts)

        if idx % 500 == 0:
            print(f"{split}: converted {idx}/{len(selected_images)} images")

    return {
        "images": len(selected_images),
        "empty_labels": empty_labels,
        "objects": int(sum(split_counts.values())),
        "class_counts": dict(split_counts),
    }


def write_yaml(output: Path) -> Path:
    yaml_path = output / "visdrone.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VISDRONE_NAMES))
    content = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 10\n"
        "names:\n"
        f"{names}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "class_names": VISDRONE_NAMES,
        "limits": {"train": args.train_limit, "val": args.val_limit},
        "splits": {},
    }
    summary["splits"]["train"] = convert_split(args.source, args.output, "train", args.train_limit, args.seed)
    summary["splits"]["val"] = convert_split(args.source, args.output, "val", args.val_limit, args.seed + 1)
    yaml_path = write_yaml(args.output)
    summary["yaml"] = str(yaml_path.resolve())

    summary_path = args.output / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
