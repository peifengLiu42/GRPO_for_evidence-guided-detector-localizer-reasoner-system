#!/usr/bin/env python3
"""Filter vLLM rollouts into a ShareGPT RFT dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realtext_grpo.rewards import format_reward, parse_box_candidates, parse_label  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data/realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.json"
DEFAULT_STATS = REPO_ROOT / "data/realtext_reference_rft_from_sft5_distmix_authdown_shortprompt.stats.json"
DEFAULT_DETAILS = REPO_ROOT / "outputs/rft_sft5_rollouts/filtered_details.jsonl"
DEFAULT_GT_MASK_ROOT = Path(os.environ.get("REALTEXT_GT_MASK_ROOT", "/path/to/RealTextV2/train/regen_mask"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_jsonl", type=Path, nargs="+")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats_output", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--details_jsonl", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--gt_mask_root", type=Path, default=DEFAULT_GT_MASK_ROOT)
    parser.add_argument("--target_width", type=int, default=1344)
    parser.add_argument("--target_height", type=int, default=894)
    parser.add_argument("--norm_base", type=int, default=999)
    parser.add_argument("--min_score", type=float, default=0.40)
    parser.add_argument("--min_forged_iou", type=float, default=0.02)
    parser.add_argument("--min_forged_precision", type=float, default=0.05)
    parser.add_argument("--max_forged_area_ratio", type=float, default=0.35)
    parser.add_argument("--max_authentic_boxes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def resolve_gt_mask_path(stem: str, gt_mask_root: Path) -> Path:
    matches = sorted(gt_mask_root.rglob(f"{stem}_mask.png"))
    if not matches:
        raise FileNotFoundError(f"Missing GT mask for {stem} under {gt_mask_root}")
    return matches[0]


def denorm_coord(value: int, size: int, norm_base: int) -> int:
    if value <= 0:
        return 0
    if value >= norm_base:
        return size
    return int(round((value + 0.5) / norm_base * size))


def convert_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    norm_base: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    px1 = denorm_coord(x1, width, norm_base)
    py1 = denorm_coord(y1, height, norm_base)
    px2 = denorm_coord(x2, width, norm_base)
    py2 = denorm_coord(y2, height, norm_base)
    px1, px2 = sorted((max(0, min(width, px1)), max(0, min(width, px2))))
    py1, py2 = sorted((max(0, min(height, py1)), max(0, min(height, py2))))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def boxes_to_mask(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    norm_base: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        converted = convert_box(box, width, height, norm_base)
        if converted is None:
            continue
        x1, y1, x2, y2 = converted
        mask[y1:y2, x1:x2] = 1
    return mask


def pixel_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    target_width: int,
    target_height: int,
) -> dict[str, float]:
    pred = cv2.resize(
        pred_mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    gt = cv2.resize(
        (gt_mask > 0).astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gt.any() else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not gt.any() else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    area_ratio = float(pred.sum() / pred.size)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pred_area_ratio": area_ratio,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def score_completion(
    text: str,
    row: dict[str, Any],
    gt_mask: np.ndarray,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = row["grpo_metadata"]
    gt_label = str(metadata["gt_label"])
    pred_label = parse_label(text) or "UNKNOWN"
    valid_boxes, invalid_boxes = parse_box_candidates(text)
    pred_mask = boxes_to_mask(valid_boxes if pred_label == "FORGED" else [], width, height, args.norm_base)
    pix = pixel_metrics(pred_mask, gt_mask, args.target_width, args.target_height)
    fmt = float(format_reward([text])[0])
    class_correct = pred_label == gt_label
    num_boxes = len(valid_boxes)
    hard_pass = fmt >= 0.75 and class_correct and not invalid_boxes
    if gt_label == "AUTHENTIC":
        hard_pass = hard_pass and num_boxes <= args.max_authentic_boxes and pred_label == "AUTHENTIC"
    else:
        hard_pass = (
            hard_pass
            and pred_label == "FORGED"
            and num_boxes > 0
            and pix["iou"] >= args.min_forged_iou
            and pix["precision"] >= args.min_forged_precision
            and pix["pred_area_ratio"] <= args.max_forged_area_ratio
        )

    if gt_label == "AUTHENTIC":
        pixel_quality = 1.0 if num_boxes == 0 and pred_label == "AUTHENTIC" else 0.0
        score = 0.20 * fmt + 0.30 * float(class_correct) + 0.50 * pixel_quality
    else:
        precision_bonus = pix["precision"]
        overbox_penalty = max(0.0, pix["pred_area_ratio"] - 0.10) * 2.0
        score = (
            0.20 * fmt
            + 0.20 * float(class_correct)
            + 0.25 * pix["f1"]
            + 0.25 * pix["iou"]
            + 0.10 * precision_bonus
            - overbox_penalty
        )
    if not math.isfinite(score):
        score = -1.0
    return {
        "score": float(score),
        "hard_pass": bool(hard_pass and score >= args.min_score),
        "format": fmt,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "class_correct": bool(class_correct),
        "num_valid_boxes": num_boxes,
        "num_invalid_boxes": len(invalid_boxes),
        "pixel": pix,
    }


def write_json_atomic(path: Path, value: Any, overwrite: bool) -> bytes:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return payload


def main() -> None:
    args = parse_args()
    rollouts = []
    for path in args.rollout_jsonl:
        rollouts.extend(read_jsonl(path))
    seen: set[str] = set()
    unique_rollouts = []
    for row in rollouts:
        stem = str(row.get("stem"))
        if stem in seen:
            raise ValueError(f"Duplicate rollout stem: {stem}")
        seen.add(stem)
        unique_rollouts.append(row)

    rft_records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    score_values: list[float] = []
    pixel_values: dict[str, list[float]] = {key: [] for key in ("precision", "recall", "f1", "iou")}

    for row in unique_rollouts:
        stem = str(row["stem"])
        image_path = Path(str(row["images"]))
        with Image.open(image_path) as image:
            width, height = image.size
        gt_mask = cv2.imread(str(resolve_gt_mask_path(stem, args.gt_mask_root)), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise RuntimeError(f"Failed to read GT mask for {stem}")
        if gt_mask.shape[:2] != (height, width):
            gt_mask = cv2.resize(gt_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        scored = []
        for completion_index, completion in enumerate(row.get("completions") or []):
            text = str(completion.get("text", ""))
            metrics = score_completion(text, row, gt_mask, width, height, args)
            scored.append(
                {
                    "completion_index": completion_index,
                    "text": text,
                    **metrics,
                }
            )
        if not scored:
            counters["no_completions"] += 1
            continue
        best = max(scored, key=lambda item: (item["hard_pass"], item["score"]))
        accepted = bool(best["hard_pass"])
        counters["accepted" if accepted else "rejected"] += 1
        counters[f"label_{row['grpo_metadata']['gt_label']}_{'accepted' if accepted else 'rejected'}"] += 1
        details.append(
            {
                "stem": stem,
                "accepted": accepted,
                "best_completion_index": best["completion_index"],
                "best_score": best["score"],
                "best_metrics": {key: value for key, value in best.items() if key != "text"},
                "all_scores": [
                    {key: value for key, value in item.items() if key != "text"}
                    for item in scored
                ],
                "grpo_metadata": row["grpo_metadata"],
            }
        )
        if not accepted:
            continue
        messages = [
            row["messages"][0],
            row["messages"][1],
            {"role": "assistant", "content": best["text"]},
        ]
        rft_records.append({"messages": messages, "images": row["images"]})
        score_values.append(float(best["score"]))
        for key in pixel_values:
            pixel_values[key].append(float(best["pixel"][key]))

    payload = write_json_atomic(args.output_json, rft_records, args.overwrite)
    stats = {
        "dataset_type": args.output_json.stem,
        "rollout_files": [str(path) for path in args.rollout_jsonl],
        "output": str(args.output_json),
        "output_sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "num_rollout_prompts": len(unique_rollouts),
        "num_rft_records": len(rft_records),
        "acceptance_rate": len(rft_records) / len(unique_rollouts) if unique_rollouts else 0.0,
        "counts": dict(counters),
        "score_mean": float(np.mean(score_values)) if score_values else 0.0,
        "selected_pixel_mean": {
            key: float(np.mean(values)) if values else 0.0
            for key, values in pixel_values.items()
        },
        "filters": {
            "min_score": args.min_score,
            "min_forged_iou": args.min_forged_iou,
            "min_forged_precision": args.min_forged_precision,
            "max_forged_area_ratio": args.max_forged_area_ratio,
            "max_authentic_boxes": args.max_authentic_boxes,
        },
        "metadata_removed_from_training_json": True,
    }
    write_json_atomic(args.stats_output, stats, args.overwrite)
    args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.details_jsonl.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
