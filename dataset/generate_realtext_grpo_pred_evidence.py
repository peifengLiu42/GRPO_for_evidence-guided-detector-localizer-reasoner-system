#!/usr/bin/env python3
"""Build RealTextV2 GRPO data with predicted detector/localizer evidence.

The assistant answer remains the ground-truth forensic report.  Only the user
message's evidence is rebuilt from DINOv3 classification predictions and DTD
predicted masks.  The fixed in-domain validation stems are always excluded.

Default evidence rule matches the current fusion inference pipeline:

    prompt says FORGED with DTD boxes iff detector_forged and DTD_has_boxes
    otherwise use the existing authentic/no-box prompt

The prompt strings below are intentionally byte-for-byte identical to those in
``dataset/generate_explain_llm.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SFT = Path(os.environ.get("REALTEXT_SOURCE_SFT", PROJECT_ROOT / "data/realtext_train_sft.json"))
DEFAULT_OUTPUT = Path(os.environ.get("REALTEXT_GRPO_OUTPUT", PROJECT_ROOT / "data/realtext_grpo_dtd_dinov3_train.json"))
DEFAULT_IMAGE_ROOT = Path(os.environ.get("REALTEXT_IMAGE_ROOT", "/path/to/RealTextV2/train/image"))
DEFAULT_GT_MASK_ROOT = Path(os.environ.get("REALTEXT_GT_MASK_ROOT", "/path/to/RealTextV2/train/regen_mask"))
DEFAULT_EXCLUDE_LIST = Path(os.environ.get("REALTEXT_EXCLUDE_LIST", "/path/to/RealTextV2/train/test.txt"))
DEFAULT_DTD_MASK_DIR = Path(os.environ.get("REALTEXT_DTD_MASK_DIR", "/path/to/dtd_masks"))
DEFAULT_DETECTOR_JSON = Path(os.environ.get("REALTEXT_DETECTOR_JSON", "/path/to/detector_predictions.jsonl"))

SYSTEM_PROMPT = (
    "You are an image forensics assistant. Output only this report format:\n\n"
    "I. Overall Assessment\n"
    "[Conclusion]: FORGED or AUTHENTIC\n"
    "[RISK_SCORE]: 0-100 manipulation likelihood\n\n"
    "II. Detailed Anomaly Analysis\n"
    "For each anomaly:\n"
    "### ANOMALY_001\n"
    "[GROUNDING]: [xmin,ymin,xmax,ymax] normalized 0-999\n"
    "[REASON]: visual or semantic evidence\n"
    "If none, state no anomalies detected.\n\n"
    "III. Summary\n"
    "Brief synthesis."
)

CONCLUSION_RE = re.compile(
    r"\[?\s*Conclusion\s*\]?\s*[:：]\s*\**\s*(FORGED|AUTHENTIC)\b", re.IGNORECASE
)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GRPO JSON using DINOv3 detector predictions and DTD mask evidence."
    )
    parser.add_argument("--source_sft", type=Path, default=DEFAULT_SOURCE_SFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--gt_mask_root", type=Path, default=DEFAULT_GT_MASK_ROOT)
    parser.add_argument("--exclude_list", type=Path, default=DEFAULT_EXCLUDE_LIST)
    parser.add_argument("--dtd_mask_dir", type=Path, default=DEFAULT_DTD_MASK_DIR)
    parser.add_argument("--detector_json", type=Path, default=DEFAULT_DETECTOR_JSON)
    parser.add_argument("--detector_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=1,
        help="Ignore DTD connected components with fewer foreground pixels (the default DTD masks are already filtered).",
    )
    parser.add_argument(
        "--evidence_rule",
        choices=("detector_and_dtd", "dtd_only"),
        default="detector_and_dtd",
        help="detector_and_dtd reproduces the current inference gate; dtd_only is an ablation.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N eligible rows; 0 means all.")
    parser.add_argument(
        "--include_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store detector/DTD predictions and GT-mask difficulty metrics outside the messages.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on duplicate stems, missing predictions/masks/images, or validation leakage.",
    )
    return parser.parse_args()


def image_path_from_record(record: dict[str, Any]) -> Path:
    image_value = record.get("images")
    if isinstance(image_value, list):
        if len(image_value) != 1:
            raise ValueError(f"Expected exactly one image, got {image_value!r}")
        image_value = image_value[0]
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("Record has no usable 'images' path")
    return Path(image_value)


def canonical_stem(value: str | Path) -> str:
    name = Path(str(value).strip()).name
    suffix = Path(name).suffix.lower()
    if suffix in VALID_IMAGE_EXTENSIONS or suffix in {".txt", ".md"}:
        return Path(name).stem
    return name


def load_stem_list(path: Path) -> set[str]:
    stems = {canonical_stem(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not stems:
        raise ValueError(f"Exclusion list is empty: {path}")
    return stems


def load_detector_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            items = payload["predictions"]
        elif isinstance(payload, dict):
            items = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("image", key)
                    items.append(item)
        else:
            raise ValueError(f"Unsupported detector prediction structure: {path}")

    predictions: dict[str, dict[str, Any]] = {}
    for item in items:
        image_value = item.get("image") or item.get("image_path") or item.get("relative_path") or item.get("image_name")
        if not image_value:
            continue
        stem = canonical_stem(image_value)
        if stem in predictions:
            raise ValueError(f"Duplicate detector prediction for {stem}")
        predictions[stem] = item
    return predictions


def detector_probability(item: dict[str, Any]) -> float | None:
    for key in ("prob_forged", "score", "forged_probability"):
        if key in item and item[key] is not None:
            return float(item[key])
    return None


def detector_is_forged(item: dict[str, Any], threshold: float) -> bool:
    probability = detector_probability(item)
    if probability is not None:
        return probability >= threshold
    if "pred_label_id" in item:
        return int(item["pred_label_id"]) == 1
    label = str(item.get("pred_label", "")).lower()
    if label:
        return label in {"forged", "fake", "tampered", "1"}
    raise ValueError(f"Cannot determine detector decision from: {item}")


def normalize_to_999(value: int, max_dim: int) -> int:
    if max_dim <= 0:
        return 0
    return max(0, min(999, int(round(value / max_dim * 999))))


def read_binary_mask(path: Path, threshold: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")
    return mask > threshold


def mask_to_bboxes(mask: np.ndarray, min_component_area: int) -> list[list[int]]:
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes: list[list[int]] = []
    for contour in contours:
        component_area = int(cv2.contourArea(contour))
        # contourArea is zero for one-pixel/one-line components.  Count pixels
        # in its bounding rectangle only when filtering is requested.
        x, y, w, h = cv2.boundingRect(contour)
        if min_component_area > 1:
            foreground_pixels = int(mask[y : y + h, x : x + w].sum())
            if foreground_pixels < min_component_area:
                continue
        elif component_area < 0:  # pragma: no cover - documents intent
            continue
        x1 = normalize_to_999(x, width)
        y1 = normalize_to_999(y, height)
        x2 = normalize_to_999(x + w - 1, width)
        y2 = normalize_to_999(y + h - 1, height)
        # GRPO's grounding parser requires strictly positive width and height.
        if x2 <= x1:
            x1, x2 = (998, 999) if x1 >= 999 else (x1, x1 + 1)
        if y2 <= y1:
            y1, y2 = (998, 999) if y1 >= 999 else (y1, y1 + 1)
        boxes.append([x1, y1, x2, y2])
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, bool]:
    resized = False
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        resized = True
    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    pred_area = int(pred_mask.sum())
    gt_area = int(gt_mask.sum())
    union = pred_area + gt_area - intersection
    iou = intersection / union if union else 1.0
    denominator = pred_area + gt_area
    f1 = 2 * intersection / denominator if denominator else 1.0
    return float(iou), float(f1), resized


def difficulty_bucket(gt_label: str, has_boxes: bool, iou: float | None) -> str:
    if gt_label == "AUTHENTIC":
        return "authentic_dtd_fp" if has_boxes else "authentic_clean"
    if not has_boxes:
        return "forged_dtd_empty"
    if iou is None:
        return "forged_unknown_iou"
    if iou >= 0.7:
        return "forged_iou_ge_0.7"
    if iou >= 0.3:
        return "forged_iou_0.3_0.7"
    return "forged_iou_lt_0.3"


def parse_gt_label(messages: list[dict[str, Any]]) -> str:
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_messages:
        raise ValueError("Record has no assistant ground-truth report")
    text = str(assistant_messages[-1].get("content", ""))
    match = CONCLUSION_RE.search(text)
    if not match:
        raise ValueError("Cannot parse [Conclusion] from assistant ground-truth report")
    return match.group(1).upper()


def resolve_gt_mask_path(image_path: Path, image_root: Path, gt_mask_root: Path) -> Path:
    try:
        relative = image_path.resolve().relative_to(image_root.resolve())
        return gt_mask_root / relative.parent / f"{image_path.stem}_mask.png"
    except ValueError:
        matches = sorted(gt_mask_root.rglob(f"{image_path.stem}_mask.png"))
        return matches[0] if matches else gt_mask_root / f"{image_path.stem}_mask.png"


def process_mask_task(task: tuple[str, str, str, int, int]) -> dict[str, Any]:
    stem, dtd_path_string, gt_path_string, mask_threshold, min_component_area = task
    dtd_path = Path(dtd_path_string)
    gt_path = Path(gt_path_string)
    pred_mask = read_binary_mask(dtd_path, mask_threshold)
    boxes = mask_to_bboxes(pred_mask, min_component_area)
    result: dict[str, Any] = {
        "stem": stem,
        "dtd_mask_path": str(dtd_path),
        "dtd_positive_pixel_count": int(pred_mask.sum()),
        "dtd_boxes": boxes,
        "dtd_has_boxes": bool(boxes),
        "dtd_gt_iou": None,
        "dtd_gt_f1": None,
        "metric_mask_resized": False,
    }
    if gt_path.is_file():
        gt_mask = read_binary_mask(gt_path, mask_threshold)
        iou, f1, resized = compute_mask_metrics(pred_mask, gt_mask)
        result.update(
            {
                "gt_mask_path": str(gt_path),
                "gt_positive_pixel_count": int(gt_mask.sum()),
                "dtd_gt_iou": iou,
                "dtd_gt_f1": f1,
                "metric_mask_resized": resized,
            }
        )
    return result


def build_prompt(boxes: list[list[int]], final_forged: bool) -> str:
    """Return the unchanged mask-guided prompt used by the reasoner."""
    num_bboxes = len(boxes) if final_forged else 0
    if not final_forged:
        return (
            f"<image>Expert forgery detector has analyzed this image and detected {num_bboxes} bbox(s), "
            f"indicating it is an authentic image. Please verify this assessment and provide a "
            f"detailed analysis report strictly following the required forensic format."
        )
    bbox_str = ", ".join(f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in boxes)
    return (
        f"<image>Expert forgery detector has identified {num_bboxes} potential tampered region(s) at: {bbox_str}. "
        f"Please analyze these specific areas in detail, explain the visual artifacts and logical contradictions, "
        f"and provide a comprehensive forgery analysis report strictly following the required forensic format."
    )


def replace_prompt(record: dict[str, Any], user_prompt: str) -> dict[str, Any]:
    output = copy.deepcopy(record)
    messages = output.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Record 'messages' must be a list")

    user_indices = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    if len(user_indices) != 1:
        raise ValueError(f"Expected exactly one user message, found {len(user_indices)}")
    messages[user_indices[0]]["content"] = user_prompt

    if not any(message.get("role") == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    output["messages"] = messages
    return output


def evidence_case(gt_label: str, detector_forged: bool, dtd_has_boxes: bool) -> str:
    detector_label = "forged" if detector_forged else "authentic"
    dtd_label = "positive" if dtd_has_boxes else "empty"
    return f"gt_{gt_label.lower()}__detector_{detector_label}__dtd_{dtd_label}"


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_paths(args: argparse.Namespace) -> None:
    files = [args.source_sft, args.exclude_list, args.detector_json]
    directories = [args.image_root, args.gt_mask_root, args.dtd_mask_dir]
    missing = [str(path) for path in files if not path.is_file()]
    missing.extend(str(path) for path in directories if not path.is_dir())
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))
    if not 0 <= args.mask_threshold <= 255:
        raise ValueError("--mask_threshold must be in [0, 255]")
    if args.min_component_area < 1:
        raise ValueError("--min_component_area must be >= 1")


def main() -> None:
    args = parse_args()
    validate_paths(args)

    source = json.loads(args.source_sft.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("Source SFT JSON must contain a list")
    excluded_stems = load_stem_list(args.exclude_list)
    detector_map = load_detector_predictions(args.detector_json)

    eligible: list[tuple[dict[str, Any], Path, str, str]] = []
    source_stems: list[str] = []
    excluded_from_source = 0
    malformed = 0
    invalid_gt_report_stems: list[str] = []
    for record in source:
        try:
            image_path = image_path_from_record(record)
            stem = image_path.stem
        except (TypeError, ValueError):
            malformed += 1
            if args.strict:
                raise
            continue
        source_stems.append(stem)
        if stem in excluded_stems:
            excluded_from_source += 1
            continue
        try:
            gt_label = parse_gt_label(record.get("messages", []))
        except ValueError:
            # Such a row cannot receive the current GRPO label/grounding/report
            # rewards.  Skipping it is safer than fabricating a target label.
            invalid_gt_report_stems.append(stem)
            continue
        eligible.append((record, image_path, stem, gt_label))

    duplicate_source_stems = [stem for stem, count in Counter(source_stems).items() if count > 1]
    if duplicate_source_stems and args.strict:
        raise ValueError(f"Duplicate source stems, first examples: {duplicate_source_stems[:10]}")
    if args.limit > 0:
        eligible = eligible[: args.limit]

    missing_detector = [stem for _, _, stem, _ in eligible if stem not in detector_map]
    missing_dtd = [stem for _, _, stem, _ in eligible if not (args.dtd_mask_dir / f"{stem}.png").is_file()]
    missing_images = [str(path) for _, path, _, _ in eligible if not path.is_file()]
    if args.strict and (missing_detector or missing_dtd or missing_images):
        raise FileNotFoundError(
            "Incomplete inputs: "
            f"missing_detector={len(missing_detector)}, missing_dtd={len(missing_dtd)}, "
            f"missing_images={len(missing_images)}; examples="
            f"{(missing_detector + missing_dtd + missing_images)[:10]}"
        )

    usable = [
        row
        for row in eligible
        if row[2] in detector_map
        and (args.dtd_mask_dir / f"{row[2]}.png").is_file()
        and row[1].is_file()
    ]
    tasks = []
    for _, image_path, stem, _ in usable:
        gt_mask_path = resolve_gt_mask_path(image_path, args.image_root, args.gt_mask_root)
        tasks.append(
            (
                stem,
                str(args.dtd_mask_dir / f"{stem}.png"),
                str(gt_mask_path),
                args.mask_threshold,
                args.min_component_area,
            )
        )

    if args.workers <= 1:
        iterator: Iterable[dict[str, Any]] = map(process_mask_task, tasks)
        mask_results = list(tqdm(iterator, total=len(tasks), desc="Extract DTD evidence", unit="image"))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            iterator = executor.map(process_mask_task, tasks, chunksize=4)
            mask_results = list(tqdm(iterator, total=len(tasks), desc="Extract DTD evidence", unit="image"))
    mask_result_map = {item["stem"]: item for item in mask_results}

    output_records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    ious: list[float] = []
    f1s: list[float] = []
    for record, image_path, stem, gt_label in usable:
        mask_info = mask_result_map[stem]
        detector_item = detector_map[stem]
        detector_forged = detector_is_forged(detector_item, args.detector_threshold)
        dtd_has_boxes = bool(mask_info["dtd_has_boxes"])
        if args.evidence_rule == "detector_and_dtd":
            final_forged = detector_forged and dtd_has_boxes
        else:
            final_forged = dtd_has_boxes
        evidence_boxes = mask_info["dtd_boxes"] if final_forged else []
        user_prompt = build_prompt(evidence_boxes, final_forged)
        output_record = replace_prompt(record, user_prompt)

        bucket = difficulty_bucket(gt_label, dtd_has_boxes, mask_info["dtd_gt_iou"])
        case = evidence_case(gt_label, detector_forged, dtd_has_boxes)
        counters[f"gt_{gt_label.lower()}"] += 1
        counters[f"detector_{'forged' if detector_forged else 'authentic'}"] += 1
        counters[f"prompt_{'forged' if final_forged else 'authentic'}"] += 1
        counters[f"difficulty/{bucket}"] += 1
        counters[f"case/{case}"] += 1
        if mask_info["dtd_gt_iou"] is not None:
            ious.append(mask_info["dtd_gt_iou"])
            f1s.append(mask_info["dtd_gt_f1"])

        if args.include_metadata:
            output_record["grpo_metadata"] = {
                "stem": stem,
                "gt_label": gt_label,
                "detector_prob_forged": detector_probability(detector_item),
                "detector_pred_label": "FORGED" if detector_forged else "AUTHENTIC",
                "detector_correct": detector_forged == (gt_label == "FORGED"),
                "dtd_mask_path": mask_info["dtd_mask_path"],
                "dtd_positive_pixel_count": mask_info["dtd_positive_pixel_count"],
                "dtd_has_boxes": dtd_has_boxes,
                "dtd_boxes": mask_info["dtd_boxes"],
                "dtd_gt_iou": mask_info["dtd_gt_iou"],
                "dtd_gt_f1": mask_info["dtd_gt_f1"],
                "metric_mask_resized": mask_info["metric_mask_resized"],
                "difficulty_bucket": bucket,
                "evidence_rule": args.evidence_rule,
                "evidence_case": case,
                "prompt_label": "FORGED" if final_forged else "AUTHENTIC",
                "prompt_boxes": evidence_boxes,
            }
        output_records.append(output_record)

    leaked = sorted(
        image_path_from_record(record).stem
        for record in output_records
        if image_path_from_record(record).stem in excluded_stems
    )
    if leaked:
        raise RuntimeError(f"Validation leakage detected: {leaked[:10]}")
    if args.strict and args.limit <= 0 and excluded_from_source != len(excluded_stems):
        missing_exclusions = sorted(excluded_stems - set(source_stems))
        raise ValueError(
            f"Expected all {len(excluded_stems)} excluded stems in source, found {excluded_from_source}; "
            f"missing examples: {missing_exclusions[:10]}"
        )

    stats: dict[str, Any] = {
        "source_sft": str(args.source_sft.resolve()),
        "output": str(args.output.resolve()),
        "detector_json": str(args.detector_json.resolve()),
        "dtd_mask_dir": str(args.dtd_mask_dir.resolve()),
        "exclude_list": str(args.exclude_list.resolve()),
        "source_records": len(source),
        "source_unique_stems": len(set(source_stems)),
        "exclude_list_stems": len(excluded_stems),
        "excluded_from_source": excluded_from_source,
        "eligible_after_exclusion": len(eligible),
        "output_records": len(output_records),
        "malformed_source_records": malformed,
        "invalid_ground_truth_reports": len(invalid_gt_report_stems),
        "invalid_ground_truth_report_stems": invalid_gt_report_stems,
        "missing_detector": len(missing_detector),
        "missing_dtd_mask": len(missing_dtd),
        "missing_image": len(missing_images),
        "validation_leaks": len(leaked),
        "evidence_rule": args.evidence_rule,
        "detector_threshold": args.detector_threshold,
        "mask_threshold": args.mask_threshold,
        "min_component_area": args.min_component_area,
        "mean_dtd_gt_iou_all": sum(ious) / len(ious) if ious else None,
        "mean_dtd_gt_f1_all": sum(f1s) / len(f1s) if f1s else None,
        "counts": dict(sorted(counters.items())),
    }
    write_json_atomic(args.output, output_records)
    stats_path = args.output.with_suffix(".stats.json")
    write_json_atomic(stats_path, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote GRPO data: {args.output}")
    print(f"Wrote statistics: {stats_path}")


if __name__ == "__main__":
    main()
