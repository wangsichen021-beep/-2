from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def configure_environment() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor"
    if vendor.exists():
        sys.path.insert(0, str(vendor))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "Ultralytics"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO single-stage detector on VisDrone.")
    parser.add_argument("--data", type=Path, default=Path("data/visdrone_yolo/visdrone.yaml"))
    parser.add_argument("--model", type=Path, default=Path("archive/yolov9_finetuned.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/task2"))
    parser.add_argument("--name", default="train_visdrone_yolov9_custom")
    parser.add_argument("--export-name", default=None, help="Optional filename for exported best weight under models/.")
    parser.add_argument("--device", default=None, help="Use 0 for first CUDA GPU, cpu for CPU. Auto if omitted.")
    return parser.parse_args()


def main() -> None:
    configure_environment()
    args = parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        seed=42,
        deterministic=False,
        cache=False,
        plots=True,
        val=True,
        patience=max(2, args.epochs),
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    export_dir = Path("models")
    export_dir.mkdir(exist_ok=True)
    export_name = args.export_name or f"task2_visdrone_{args.name.replace('train_visdrone_', '').replace('_custom', '')}_best.pt"
    exported_best = export_dir / export_name
    if best.exists():
        shutil.copy2(best, exported_best)

    summary = {
        "data": str(args.data.resolve()),
        "base_model": str(args.model.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": str(device),
        "save_dir": str(save_dir.resolve()),
        "best": str(best.resolve()) if best.exists() else None,
        "last": str(last.resolve()) if last.exists() else None,
        "exported_best": str(exported_best.resolve()) if exported_best.exists() else None,
    }
    summary_path = save_dir / "task2_train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
