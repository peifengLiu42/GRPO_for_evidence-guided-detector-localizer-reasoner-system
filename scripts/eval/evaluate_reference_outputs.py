#!/usr/bin/env python3
"""Evaluate reference-evidence report JSONL outputs against GT reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realtext_grpo.rewards import (  # noqa: E402
    format_reward,
    grounding_components,
    one_to_one_grounding_metrics,
    parse_box_candidates,
    parse_label,
    parse_risk,
)


DEFAULT_GT_JSON = REPO_ROOT / "data/realtext_refine_train_sft.json"


def iter_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list: {path}")
    return value


def image_stem(row: dict[str, Any]) -> str | None:
    for key in ("image_name", "image_path", "images"):
        value = row.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value:
            return Path(str(value)).stem
    metadata = row.get("grpo_metadata") or {}
    stem = metadata.get("stem")
    return str(stem) if stem else None


def assistant_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("report"), str):
        return row["report"]
    if isinstance(row.get("response"), str):
        return row["response"]
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def gt_boxes_from_answer(answer: str) -> list[tuple[int, int, int, int]]:
    boxes, _ = parse_box_candidates(answer)
    return boxes


def load_gt_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
    gt: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in iter_json_or_jsonl(path):
            stem = image_stem(row)
            if not stem:
                continue
            answer = assistant_text(row)
            label = parse_label(answer)
            if label is None:
                continue
            metadata = row.get("grpo_metadata") or {}
            gt[stem] = {
                "stem": stem,
                "answer": answer,
                "label": metadata.get("gt_label", label),
                "risk": parse_risk(answer),
                "boxes": gt_boxes_from_answer(answer),
                "difficulty_bucket": metadata.get("difficulty_bucket", "unknown"),
                "evidence_case": metadata.get("evidence_case", "unknown"),
            }
    if not gt:
        raise RuntimeError(f"No usable GT reports loaded from: {paths}")
    return gt


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"count": 0}
    cls_counts = Counter((row["gt_label"], row["pred_label"]) for row in rows)
    tp = cls_counts[("FORGED", "FORGED")]
    fn = cls_counts[("FORGED", "AUTHENTIC")] + sum(
        value for (gt, pred), value in cls_counts.items() if gt == "FORGED" and pred not in {"FORGED", "AUTHENTIC"}
    )
    fp = cls_counts[("AUTHENTIC", "FORGED")]
    tn = cls_counts[("AUTHENTIC", "AUTHENTIC")]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(row["classification"] for row in rows) / count
    return {
        "count": count,
        "format_mean": safe_mean([row["format_raw"] for row in rows]),
        "r_task_no_explanation_mean": safe_mean([row["r_task_no_explanation"] for row in rows]),
        "classification_accuracy": accuracy,
        "forged_precision": precision,
        "forged_recall": recall,
        "forged_f1": f1,
        "box_f1_mean": safe_mean([row["box_f1"] for row in rows]),
        "set_iou_mean": safe_mean([row["set_iou"] for row in rows]),
        "grounding_reward_mean": safe_mean([row["grounding_reward"] for row in rows]),
        "authentic_hallucinated_grounding_rate": safe_mean(
            [row["authentic_hallucinated_grounding"] for row in rows if row["gt_label"] == "AUTHENTIC"]
        ),
        "forged_missed_grounding_rate": safe_mean(
            [row["forged_missed_grounding"] for row in rows if row["gt_label"] == "FORGED"]
        ),
        "invalid_box_rate": safe_mean([row["has_invalid_box"] for row in rows]),
        "avg_pred_boxes": safe_mean([row["num_pred_boxes"] for row in rows]),
        "confusion_matrix": {
            "tp_forged": tp,
            "fp_forged": fp,
            "tn_authentic": tn,
            "fn_forged": fn,
        },
    }


def evaluate(prediction_path: Path, gt_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    details = []
    missing_gt = []
    for row in iter_json_or_jsonl(prediction_path):
        stem = image_stem(row)
        if not stem or stem not in gt_map:
            missing_gt.append(stem or "<missing_stem>")
            continue
        gt = gt_map[stem]
        report = assistant_text(row)
        pred_label = parse_label(report)
        pred_boxes, invalid_boxes = parse_box_candidates(report)
        cls, box_f1, set_iou, invalid_penalty = grounding_components(
            report,
            gt["boxes"],
            gt["label"],
        )
        quality = (cls + box_f1 + set_iou) / 3.0
        grounding_reward = 0.75 * max(-1.0, min(1.0, quality - invalid_penalty))
        fmt = format_reward([report])[0]
        details.append(
            {
                "stem": stem,
                "gt_label": gt["label"],
                "pred_label": pred_label or "UNKNOWN",
                "classification": cls,
                "format_raw": fmt,
                "box_f1": box_f1,
                "set_iou": set_iou,
                "grounding_reward": grounding_reward,
                "r_task_no_explanation": 0.15 * fmt + grounding_reward,
                "num_gt_boxes": len(gt["boxes"]),
                "num_pred_boxes": len(pred_boxes),
                "num_invalid_boxes": len(invalid_boxes),
                "has_invalid_box": 1.0 if invalid_boxes else 0.0,
                "authentic_hallucinated_grounding": 1.0
                if gt["label"] == "AUTHENTIC" and pred_boxes
                else 0.0,
                "forged_missed_grounding": 1.0
                if gt["label"] == "FORGED" and not pred_boxes
                else 0.0,
                "difficulty_bucket": gt["difficulty_bucket"],
                "evidence_case": gt["evidence_case"],
            }
        )

    by_label = {
        label: summarize_rows([row for row in details if row["gt_label"] == label])
        for label in ("FORGED", "AUTHENTIC")
    }
    by_difficulty = {
        key: summarize_rows([row for row in details if row["difficulty_bucket"] == key])
        for key in sorted({row["difficulty_bucket"] for row in details})
    }
    return {
        "prediction_path": str(prediction_path),
        "evaluated": len(details),
        "missing_gt": len(missing_gt),
        "missing_gt_examples": missing_gt[:20],
        "overall": summarize_rows(details),
        "by_label": by_label,
        "by_difficulty": by_difficulty,
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+", help="Prediction JSON/JSONL files.")
    parser.add_argument(
        "--gt_json",
        type=Path,
        action="append",
        default=None,
        help="GT SFT/GRPO JSON. Can be passed multiple times.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--details_jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_paths = args.gt_json or [DEFAULT_GT_JSON]
    gt_map = load_gt_map(gt_paths)
    reports = [evaluate(path, gt_map) for path in args.predictions]
    output = reports[0] if len(reports) == 1 else {"reports": reports}

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.details_jsonl:
        args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.details_jsonl.open("w", encoding="utf-8") as handle:
            for report in reports:
                for row in report["details"]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    compact = []
    for report in reports:
        compact.append(
            {
                "prediction_path": report["prediction_path"],
                "evaluated": report["evaluated"],
                "missing_gt": report["missing_gt"],
                **report["overall"],
            }
        )
    print(json.dumps(compact[0] if len(compact) == 1 else compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
