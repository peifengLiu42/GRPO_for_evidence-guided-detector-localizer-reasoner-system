#!/usr/bin/env python3
"""Generate the leakage-free RealTextV2 reference-evidence GRPO candidate pool.

The model input always contains the raw detector decision/probability and every
DTD candidate box. GT-derived labels, localization metrics, difficulty buckets,
and sampling weights are stored only in ``grpo_metadata`` for rewards/sampling.
The assistant message remains the original GT forensic report so the same file
can also be used to select from-scratch SFT/RFT subsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
LEGACY_UTILS_DIR = REPO_ROOT / "dataset"
for import_dir in (SRC_DIR, LEGACY_UTILS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from generate_realtext_grpo_pred_evidence import (  # noqa: E402
    DEFAULT_DETECTOR_JSON,
    DEFAULT_DTD_MASK_DIR,
    DEFAULT_EXCLUDE_LIST,
    DEFAULT_GT_MASK_ROOT,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_SOURCE_SFT,
    detector_is_forged,
    detector_probability,
    evidence_case,
    image_path_from_record,
    load_detector_predictions,
    load_stem_list,
    parse_gt_label,
    process_mask_task,
    replace_prompt,
    resolve_gt_mask_path,
    write_json_atomic,
)
from realtext_grpo.prompts import SYSTEM_PROMPT, build_reference_evidence_prompt  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data/realtext_grpo_reference_evidence_train_shortprompt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full RealTextV2 reference-evidence data for from-scratch SFT/RFT and GRPO."
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
        help="DTD masks already use mp40 post-processing, so preserve every remaining component by default.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="Debug only: 0 generates the complete pool.")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on duplicates, missing inputs, missing probabilities, or validation leakage.",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    required_files = (args.source_sft, args.exclude_list, args.detector_json)
    required_dirs = (args.image_root, args.gt_mask_root, args.dtd_mask_dir)
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))
    if not 0 <= args.mask_threshold <= 255:
        raise ValueError("--mask_threshold must be in [0, 255]")
    if not 0.0 <= args.detector_threshold <= 1.0:
        raise ValueError("--detector_threshold must be in [0, 1]")
    if args.min_component_area < 1:
        raise ValueError("--min_component_area must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")


def require_detector_probability(item: dict[str, Any], stem: str) -> float:
    probability = detector_probability(item)
    if probability is None:
        raise ValueError(f"Detector prediction for {stem} has no prob_forged/score")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Detector probability for {stem} is outside [0, 1]: {probability}")
    return probability


def difficulty_bucket(gt_label: str, has_boxes: bool, iou: float | None) -> str:
    """Keep the plan's IoU buckets as the canonical sampler key."""
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


def f1_bucket(gt_label: str, has_boxes: bool, f1: float | None) -> str:
    """Add an explicit per-image F1 view without replacing the planned IoU key."""
    if gt_label == "AUTHENTIC":
        return "authentic_dtd_fp" if has_boxes else "authentic_clean"
    if not has_boxes:
        return "forged_dtd_empty"
    if f1 is None:
        return "forged_unknown_f1"
    if f1 >= 0.8:
        return "forged_f1_ge_0.8"
    if f1 >= 0.4:
        return "forged_f1_0.4_0.8"
    return "forged_f1_lt_0.4"


def recommended_sampling_weight(
    difficulty: str,
    detector_correct: bool,
    detector_forged: bool,
    dtd_has_boxes: bool,
) -> tuple[int, list[str]]:
    """Implement the max-not-product weighting rule from the plan."""
    hard_buckets = {
        "forged_iou_lt_0.3",
        "forged_dtd_empty",
        "authentic_dtd_fp",
        "forged_unknown_iou",
    }
    medium_buckets = {"forged_iou_0.3_0.7"}
    weight = 1
    reasons: list[str] = []
    if difficulty in medium_buckets:
        weight = max(weight, 2)
        reasons.append("medium_localization")
    if difficulty in hard_buckets:
        weight = max(weight, 4)
        reasons.append("hard_localization_or_false_positive")
    if not detector_correct:
        weight = max(weight, 4)
        reasons.append("detector_error")
    if detector_forged != dtd_has_boxes:
        weight = max(weight, 4)
        reasons.append("detector_dtd_conflict")
    if not reasons:
        reasons.append("standard")
    return weight, reasons


def prompt_sha256() -> str:
    example = build_reference_evidence_prompt("FORGED", 0.5, [[1, 2, 3, 4]])
    return hashlib.sha256(example.encode("utf-8")).hexdigest()


def set_system_prompt(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Record 'messages' must be a list")
    system_indices = [index for index, message in enumerate(messages) if message.get("role") == "system"]
    if not system_indices:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return
    for index in system_indices:
        messages[index]["content"] = SYSTEM_PROMPT


def main() -> None:
    args = parse_args()
    validate_paths(args)

    source = json.loads(args.source_sft.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("Source SFT JSON must contain a list")
    excluded_stems = load_stem_list(args.exclude_list)
    detector_map = load_detector_predictions(args.detector_json)

    source_stems: list[str] = []
    eligible: list[tuple[dict[str, Any], Path, str, str]] = []
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
            invalid_gt_report_stems.append(stem)
            continue
        eligible.append((record, image_path, stem, gt_label))

    duplicates = sorted(stem for stem, count in Counter(source_stems).items() if count > 1)
    if duplicates and args.strict:
        raise ValueError(f"Duplicate source stems, first 10: {duplicates[:10]}")
    if args.limit > 0:
        eligible = eligible[: args.limit]

    missing_detector = [stem for _, _, stem, _ in eligible if stem not in detector_map]
    missing_dtd = [stem for _, _, stem, _ in eligible if not (args.dtd_mask_dir / f"{stem}.png").is_file()]
    missing_images = [str(path) for _, path, _, _ in eligible if not path.is_file()]
    if args.strict and (missing_detector or missing_dtd or missing_images):
        raise FileNotFoundError(
            f"Incomplete inputs: detector={len(missing_detector)}, DTD={len(missing_dtd)}, "
            f"image={len(missing_images)}; first 10: "
            f"{(missing_detector + missing_dtd + missing_images)[:10]}"
        )

    usable = [
        row
        for row in eligible
        if row[2] in detector_map
        and (args.dtd_mask_dir / f"{row[2]}.png").is_file()
        and row[1].is_file()
    ]
    tasks = [
        (
            stem,
            str(args.dtd_mask_dir / f"{stem}.png"),
            str(resolve_gt_mask_path(image_path, args.image_root, args.gt_mask_root)),
            args.mask_threshold,
            args.min_component_area,
        )
        for _, image_path, stem, _ in usable
    ]
    if args.workers == 1:
        iterator: Iterable[dict[str, Any]] = map(process_mask_task, tasks)
        mask_results = list(tqdm(iterator, total=len(tasks), desc="Extract DTD evidence", unit="image"))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            iterator = executor.map(process_mask_task, tasks, chunksize=8)
            mask_results = list(
                tqdm(iterator, total=len(tasks), desc="Extract DTD evidence", unit="image")
            )
    mask_map = {item["stem"]: item for item in mask_results}
    missing_gt_metrics = sorted(
        item["stem"]
        for item in mask_results
        if item["dtd_gt_iou"] is None or item["dtd_gt_f1"] is None
    )
    if missing_gt_metrics and args.strict:
        raise FileNotFoundError(
            f"Missing GT masks/metrics for {len(missing_gt_metrics)} samples; "
            f"first 10: {missing_gt_metrics[:10]}"
        )

    output_records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    ious: list[float] = []
    f1s: list[float] = []
    probabilities: list[float] = []
    for record, _, stem, gt_label in usable:
        detector_item = detector_map[stem]
        probability = require_detector_probability(detector_item, stem)
        detector_forged = detector_is_forged(detector_item, args.detector_threshold)
        detector_label = "FORGED" if detector_forged else "AUTHENTIC"
        mask_info = mask_map[stem]
        dtd_boxes = mask_info["dtd_boxes"]
        dtd_has_boxes = bool(dtd_boxes)

        # Critical reference-evidence rule: never hide boxes using the detector.
        user_prompt = build_reference_evidence_prompt(detector_label, probability, dtd_boxes)
        output_record = replace_prompt(record, user_prompt)
        set_system_prompt(output_record)

        detector_correct = detector_forged == (gt_label == "FORGED")
        difficulty = difficulty_bucket(gt_label, dtd_has_boxes, mask_info["dtd_gt_iou"])
        localization_f1_bucket = f1_bucket(gt_label, dtd_has_boxes, mask_info["dtd_gt_f1"])
        case = evidence_case(gt_label, detector_forged, dtd_has_boxes)
        sample_weight, weight_reasons = recommended_sampling_weight(
            difficulty, detector_correct, detector_forged, dtd_has_boxes
        )
        output_record["grpo_metadata"] = {
            "stem": stem,
            "gt_label": gt_label,
            "detector_prob_forged": probability,
            "detector_pred_label": detector_label,
            "detector_correct": detector_correct,
            "dtd_mask_path": mask_info["dtd_mask_path"],
            "dtd_positive_pixel_count": mask_info["dtd_positive_pixel_count"],
            "dtd_has_boxes": dtd_has_boxes,
            "dtd_boxes": dtd_boxes,
            "dtd_gt_iou": mask_info["dtd_gt_iou"],
            "dtd_gt_f1": mask_info["dtd_gt_f1"],
            "metric_mask_resized": mask_info["metric_mask_resized"],
            "difficulty_bucket": difficulty,
            "localization_f1_bucket": localization_f1_bucket,
            "evidence_case": case,
            "prompt_mode": "reference_evidence",
            "prompt_boxes": dtd_boxes,
            "recommended_sampling_weight": sample_weight,
            "sampling_weight_reasons": weight_reasons,
        }
        output_records.append(output_record)

        counters[f"gt/{gt_label.lower()}"] += 1
        counters[f"detector/{detector_label.lower()}"] += 1
        counters[f"detector_correct/{str(detector_correct).lower()}"] += 1
        counters[f"dtd/{'positive' if dtd_has_boxes else 'empty'}"] += 1
        counters[f"difficulty/{difficulty}"] += 1
        counters[f"f1_bucket/{localization_f1_bucket}"] += 1
        counters[f"case/{case}"] += 1
        counters[f"sampling_weight/{sample_weight}"] += 1
        for reason in weight_reasons:
            counters[f"sampling_reason/{reason}"] += 1
        probabilities.append(probability)
        if mask_info["dtd_gt_iou"] is not None:
            ious.append(float(mask_info["dtd_gt_iou"]))
            f1s.append(float(mask_info["dtd_gt_f1"]))

    output_stems = {image_path_from_record(record).stem for record in output_records}
    leaked = sorted(output_stems & excluded_stems)
    if leaked:
        raise RuntimeError(f"Validation leakage detected: {leaked[:10]}")
    if args.strict and args.limit == 0:
        if excluded_from_source != len(excluded_stems):
            missing_exclusions = sorted(excluded_stems - set(source_stems))
            raise ValueError(
                f"Expected {len(excluded_stems)} excluded stems, found {excluded_from_source}; "
                f"missing first 10: {missing_exclusions[:10]}"
            )
        if len(output_records) != 12_148:
            raise ValueError(f"Expected 12,148 complete records, got {len(output_records)}")

    stats: dict[str, Any] = {
        "dataset_type": "realtext_reference_evidence_grpo_candidate_pool",
        "prompt_mode": "reference_evidence",
        "prompt_template_sha256": prompt_sha256(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "source_sft": str(args.source_sft.resolve()),
        "output": str(args.output.resolve()),
        "detector_json": str(args.detector_json.resolve()),
        "dtd_mask_dir": str(args.dtd_mask_dir.resolve()),
        "exclude_list": str(args.exclude_list.resolve()),
        "source_records": len(source),
        "source_unique_stems": len(set(source_stems)),
        "exclude_list_stems": len(excluded_stems),
        "excluded_from_source": excluded_from_source,
        "invalid_ground_truth_reports": len(invalid_gt_report_stems),
        "invalid_ground_truth_report_stems": invalid_gt_report_stems,
        "eligible_records": len(eligible),
        "output_records": len(output_records),
        "malformed_source_records": malformed,
        "missing_detector": len(missing_detector),
        "missing_dtd_mask": len(missing_dtd),
        "missing_image": len(missing_images),
        "missing_gt_mask_or_metrics": len(missing_gt_metrics),
        "validation_leaks": len(leaked),
        "detector_threshold": args.detector_threshold,
        "mask_threshold": args.mask_threshold,
        "min_component_area": args.min_component_area,
        "mean_dtd_gt_iou_all": sum(ious) / len(ious) if ious else None,
        "mean_dtd_gt_f1_all": sum(f1s) / len(f1s) if f1s else None,
        "detector_probability_min": min(probabilities) if probabilities else None,
        "detector_probability_max": max(probabilities) if probabilities else None,
        "counts": dict(sorted(counters.items())),
    }
    write_json_atomic(args.output, output_records)
    stats_path = args.output.with_suffix(".stats.json")
    write_json_atomic(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote reference-evidence data: {args.output}")
    print(f"Wrote statistics: {stats_path}")


if __name__ == "__main__":
    main()
