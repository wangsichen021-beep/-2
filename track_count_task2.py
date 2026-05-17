from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np


def configure_environment() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor"
    if vendor.exists():
        sys.path.insert(0, str(vendor))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "Ultralytics"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track objects, draw Tracking IDs, and count virtual-line crossings.")
    parser.add_argument("--model", type=Path, default=Path("models/task2_visdrone_yolo26n_best.pt"))
    parser.add_argument("--video", type=Path, default=Path("视频.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2"))
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--line-axis", choices=["x", "y"], default="y")
    parser.add_argument("--line-ratio", type=float, default=0.55)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional debug limit. 0 means full video.")
    return parser.parse_args()


def side_of_line(cx: float, cy: float, axis: str, line_pos: int) -> int:
    value = cx if axis == "x" else cy
    if value < line_pos:
        return -1
    if value > line_pos:
        return 1
    return 0


def color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973)
    color = rng.integers(70, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(frame, (x, y0), (min(frame.shape[1] - 1, x + tw + 6), y0 + th + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 3, y0 + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detections: list[dict],
    names: dict,
    axis: str,
    line_pos: int,
    count: int,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    if axis == "x":
        cv2.line(annotated, (line_pos, 0), (line_pos, height), (30, 220, 255), 3)
        cv2.putText(annotated, "counting line", (line_pos + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 220, 255), 2)
    else:
        cv2.line(annotated, (0, line_pos), (width, line_pos), (30, 220, 255), 3)
        cv2.putText(annotated, "counting line", (16, max(32, line_pos - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 220, 255), 2)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        track_id = int(det["track_id"])
        color = color_for_track(track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (int(det["cx"]), int(det["cy"])), 4, color, -1)
        class_name = names.get(int(det["cls"]), str(int(det["cls"])))
        draw_label(annotated, f"ID {track_id} {class_name} {det['conf']:.2f}", x1, max(18, y1), color)

    cv2.rectangle(annotated, (8, 8), (360, 72), (0, 0, 0), -1)
    cv2.putText(annotated, f"Crossing Count: {count}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Frame {frame_idx}  Time {frame_idx / max(fps, 1e-6):.2f}s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return annotated


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def pick_occlusion_window(frame_dets: dict[int, list[dict]], total_frames: int) -> dict:
    best = {"frame": 0, "score": -1.0, "reason": "fallback: highest detection density"}
    for frame_idx, dets in frame_dets.items():
        score = float(len(dets)) * 0.1
        reason = "highest detection density"
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                overlap = iou(dets[i]["xyxy"], dets[j]["xyxy"])
                dx = dets[i]["cx"] - dets[j]["cx"]
                dy = dets[i]["cy"] - dets[j]["cy"]
                dist = math.hypot(dx, dy)
                proximity = 1.0 / (1.0 + dist / 80.0)
                pair_score = overlap * 4.0 + proximity + len(dets) * 0.05
                if pair_score > score:
                    score = pair_score
                    if overlap > 0.05:
                        reason = f"bbox overlap IoU={overlap:.2f}"
                    else:
                        reason = f"dense crossing, center distance={dist:.1f}px"
        if score > best["score"]:
            best = {"frame": frame_idx, "score": score, "reason": reason}

    start = max(0, min(total_frames - 4, best["frame"] - 1))
    return {"start": start, "frames": [start + i for i in range(4)], "reason": best["reason"], "score": best["score"]}


def save_occlusion_frames(
    video_path: Path,
    output_dir: Path,
    frame_ids: list[int],
    frame_dets: dict[int, list[dict]],
    names: dict,
    axis: str,
    line_pos: int,
    per_frame_counts: dict[int, int],
    fps: float,
) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    saved: list[str] = []
    panels: list[np.ndarray] = []
    occlusion_dir = output_dir / "occlusion_frames"
    occlusion_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        annotated = draw_overlay(
            frame,
            frame_dets.get(frame_id, []),
            names,
            axis,
            line_pos,
            per_frame_counts.get(frame_id, 0),
            frame_id,
            fps,
        )
        path = occlusion_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(path), annotated)
        saved.append(str(path))
        panels.append(cv2.resize(annotated, (360, 640)))

    cap.release()
    if panels:
        panel = np.concatenate(panels, axis=1)
        panel_path = output_dir / "occlusion_panel.jpg"
        cv2.imwrite(str(panel_path), panel)
        saved.append(str(panel_path))
    return saved


def analyze_id_continuity(frame_ids: list[int], frame_dets: dict[int, list[dict]]) -> dict:
    ids_by_frame = {fid: sorted({int(det["track_id"]) for det in frame_dets.get(fid, [])}) for fid in frame_ids}
    all_ids = sorted(set().union(*(set(ids) for ids in ids_by_frame.values()))) if ids_by_frame else []
    persistent_ids = [tid for tid in all_ids if all(tid in ids_by_frame.get(fid, []) for fid in frame_ids)]
    intermittent_ids = [tid for tid in all_ids if tid not in persistent_ids]
    return {
        "ids_by_frame": ids_by_frame,
        "persistent_ids": persistent_ids,
        "intermittent_ids": intermittent_ids,
        "verdict": "ID maintained for persistent tracks" if persistent_ids else "ID loss or insufficient continuous detections in selected window",
    }


def main() -> None:
    configure_environment()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    line_pos = int((width if args.line_axis == "x" else height) * args.line_ratio)

    output_video = args.output_dir / "tracked_counted_video.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    last_side: dict[int, int] = {}
    counted_ids: set[int] = set()
    count_events: list[dict] = []
    rows: list[dict] = []
    frame_dets: dict[int, list[dict]] = defaultdict(list)
    per_frame_counts: dict[int, int] = {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        detections: list[dict] = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for box, track_id, cls, conf in zip(xyxy, ids, classes, confs):
                x1, y1, x2, y2 = [float(v) for v in box]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                side = side_of_line(cx, cy, args.line_axis, line_pos)
                if side != 0:
                    previous = last_side.get(int(track_id))
                    if previous is not None and previous != side and int(track_id) not in counted_ids:
                        counted_ids.add(int(track_id))
                        count_events.append(
                            {
                                "track_id": int(track_id),
                                "frame": frame_idx,
                                "time_sec": round(frame_idx / fps, 3),
                                "class_id": int(cls),
                                "class_name": model.names.get(int(cls), str(int(cls))),
                                "from_side": previous,
                                "to_side": side,
                            }
                        )
                    last_side[int(track_id)] = side

                det = {
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps,
                    "track_id": int(track_id),
                    "cls": int(cls),
                    "class_name": model.names.get(int(cls), str(int(cls))),
                    "conf": float(conf),
                    "xyxy": [x1, y1, x2, y2],
                    "cx": float(cx),
                    "cy": float(cy),
                    "side": int(side),
                }
                detections.append(det)
                rows.append(
                    {
                        "frame": frame_idx,
                        "time_sec": f"{frame_idx / fps:.3f}",
                        "track_id": int(track_id),
                        "class_id": int(cls),
                        "class_name": det["class_name"],
                        "confidence": f"{float(conf):.4f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                        "center_x": f"{cx:.2f}",
                        "center_y": f"{cy:.2f}",
                        "line_side": int(side),
                    }
                )

        frame_dets[frame_idx] = detections
        per_frame_counts[frame_idx] = len(counted_ids)
        annotated = draw_overlay(frame, detections, model.names, args.line_axis, line_pos, len(counted_ids), frame_idx, fps)
        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"processed {frame_idx}/{total_frames} frames, count={len(counted_ids)}")

    cap.release()
    writer.release()

    csv_path = args.output_dir / "tracking_log.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "time_sec",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "line_side",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    occlusion = pick_occlusion_window(frame_dets, frame_idx)
    occlusion_paths = save_occlusion_frames(
        args.video,
        args.output_dir,
        occlusion["frames"],
        frame_dets,
        model.names,
        args.line_axis,
        line_pos,
        per_frame_counts,
        fps,
    )
    id_analysis = analyze_id_continuity(occlusion["frames"], frame_dets)

    summary = {
        "model": str(args.model.resolve()),
        "video": str(args.video.resolve()),
        "output_video": str(output_video.resolve()),
        "tracking_log": str(csv_path.resolve()),
        "fps": fps,
        "frame_size": [width, height],
        "frames_processed": frame_idx,
        "duration_sec": round(frame_idx / fps, 3),
        "line": {"axis": args.line_axis, "position": line_pos, "ratio": args.line_ratio},
        "crossing_count": len(counted_ids),
        "counted_track_ids": sorted(counted_ids),
        "count_events": count_events,
        "total_logged_detections": len(rows),
        "unique_track_ids": sorted({int(row["track_id"]) for row in rows}),
        "occlusion": {
            **occlusion,
            "paths": occlusion_paths,
            "id_analysis": id_analysis,
        },
        "tracker": args.tracker,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
    }
    summary_path = args.output_dir / "tracking_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
