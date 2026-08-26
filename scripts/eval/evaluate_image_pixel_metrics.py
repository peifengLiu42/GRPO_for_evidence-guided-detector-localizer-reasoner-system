#!/usr/bin/env python3
"""Evaluate image labels and normalized [GROUNDING] masks against RealText masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realtext_grpo.rewards import parse_box_candidates, parse_label  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--gt_mask_root",
        type=Path,
        default=Path(os.environ.get("REALTEXT_GT_MASK_ROOT", "/path/to/RealTextV2/train/regen_mask")),
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument("--details_jsonl", type=Path, default=None)
    parser.add_argument("--coord_max", type=int, default=999)
    parser.add_argument("--target_width", type=int, default=1344)
    parser.add_argument("--target_height", type=int, default=894)
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def report_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("report"), str):
        return row["report"]
    if isinstance(row.get("response"), str):
        return row["response"]
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def stem_from_row(row: dict[str, Any]) -> str:
    for key in ("image_name", "image_path", "images"):
        value = row.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value:
            return Path(str(value)).stem
    raise ValueError(f"Cannot infer image stem from row keys: {sorted(row)}")


def find_gt_mask(root: Path, stem: str) -> Path:
    candidates = [
        root / f"{stem}_mask.png",
        root / f"{stem}.png",
        root / f"{stem}_mask.jpg",
        root / f"{stem}.jpg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.glob(f"**/{stem}*mask*.png"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing GT mask for {stem} under {root}")


def load_gt_mask(path: Path, target_size: tuple[int, int] | None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if target_size is not None:
        image = image.resize(target_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def normalized_box_to_pixels(
    box: tuple[int, int, int, int], width: int, height: int, coord_max: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = int(round(x1 / coord_max * width))
    x2 = int(round(x2 / coord_max * width))
    y1 = int(round(y1 / coord_max * height))
    y2 = int(round(y2 / coord_max * height))
    left, right = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    top, bottom = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def prediction_mask(report: str, size: tuple[int, int], coord_max: int) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    if parse_label(report) != "FORGED":
        return mask
    boxes, _invalid = parse_box_candidates(report)
    for box in boxes:
        pixel_box = normalized_box_to_pixels(box, width, height, coord_max)
        if pixel_box is not None:
            left, top, right, bottom = pixel_box
            mask[top:bottom, left:right] = True
    return mask


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_pixel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["pixel"]["tp"] for row in rows)
    fp = sum(row["pixel"]["fp"] for row in rows)
    fn = sum(row["pixel"]["fn"] for row in rows)
    tn = sum(row["pixel"]["tn"] for row in rows)
    precision_cm = tp / (tp + fp) if tp + fp else 0.0
    recall_cm = tp / (tp + fn) if tp + fn else 0.0
    f1_cm = 2 * precision_cm * recall_cm / (precision_cm + recall_cm) if precision_cm + recall_cm else 0.0
    iou_cm = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "precision_mean": mean([row["pixel"]["precision"] for row in rows]),
        "recall_mean": mean([row["pixel"]["recall"] for row in rows]),
        "f1_mean": mean([row["pixel"]["f1"] for row in rows]),
        "iou_mean": mean([row["pixel"]["iou"] for row in rows]),
        "precision_cm": precision_cm,
        "recall_cm": recall_cm,
        "f1_cm": f1_cm,
        "iou_cm": iou_cm,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def image_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["gt_label"] == "FORGED" and row["pred_label"] == "FORGED" for row in details)
    fp = sum(row["gt_label"] == "AUTHENTIC" and row["pred_label"] == "FORGED" for row in details)
    fn = sum(row["gt_label"] == "FORGED" and row["pred_label"] != "FORGED" for row in details)
    tn = sum(row["gt_label"] == "AUTHENTIC" and row["pred_label"] != "FORGED" for row in details)
    forged_recall = tp / (tp + fn) if tp + fn else 0.0
    authentic_recall = tn / (tn + fp) if tn + fp else 0.0
    balanced_acc = (forged_recall + authentic_recall) / 2
    accuracy = (tp + tn) / len(details) if details else 0.0

    forged_precision = tp / (tp + fp) if tp + fp else 0.0
    forged_f1 = (
        2 * forged_precision * forged_recall / (forged_precision + forged_recall)
        if forged_precision + forged_recall
        else 0.0
    )
    auth_precision = tn / (tn + fn) if tn + fn else 0.0
    auth_f1 = (
        2 * auth_precision * authentic_recall / (auth_precision + authentic_recall)
        if auth_precision + authentic_recall
        else 0.0
    )
    forged_support = tp + fn
    auth_support = tn + fp
    weighted_f1 = (
        (forged_f1 * forged_support + auth_f1 * auth_support) / (forged_support + auth_support)
        if forged_support + auth_support
        else 0.0
    )
    return {
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "level": {
            "balanced_acc": balanced_acc,
            "weighted_f1": weighted_f1,
            "accuracy": accuracy,
            "forged_f1": forged_f1,
        },
    }


def main() -> None:
    args = parse_args()
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    details = []
    status_counts: dict[str, int] = {}
    target_size = None

    for row in iter_jsonl(args.predictions):
        stem = stem_from_row(row)
        gt_path = find_gt_mask(args.gt_mask_root, stem)
        eval_size = (
            (args.target_width, args.target_height)
            if args.target_width > 0 and args.target_height > 0
            else None
        )
        gt = load_gt_mask(gt_path, eval_size)
        height, width = gt.shape
        target_size = [width, height]
        report = report_text(row)
        pred_label = parse_label(report) or "UNKNOWN"
        pred = prediction_mask(report, (width, height), args.coord_max)
        Image.fromarray((pred.astype(np.uint8) * 255), mode="L").save(
            args.mask_dir / f"{stem}_mask.png"
        )
        gt_label = "FORGED" if bool(gt.any()) else "AUTHENTIC"
        pixel = binary_metrics(pred, gt)
        details.append(
            {
                "stem": stem,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "gt_mask": str(gt_path),
                "pred_mask": str(args.mask_dir / f"{stem}_mask.png"),
                "pixel": pixel,
            }
        )
        status_counts["ok"] = status_counts.get("ok", 0) + 1

    image = image_metrics(details)
    tampered = [row for row in details if row["gt_label"] == "FORGED"]
    result = {
        "bbox_coord_mode": "normalized",
        "mask_dir": str(args.mask_dir),
        "num_samples": len(details),
        "status_counts": status_counts,
        "image_confusion": image["confusion"],
        "image_level": image["level"],
        "pixel_all_images": summarize_pixel(details),
        "pixel_tampered_only": summarize_pixel(tampered),
        "mask_policy": "AUTHENTIC conclusion => black; FORGED conclusion => draw [GROUNDING] boxes",
        "gt_mask_root": str(args.gt_mask_root),
        "target_size_for_pixel_eval": target_size,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.details_jsonl:
        args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.details_jsonl.open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
