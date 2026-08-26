#!/usr/bin/env python3
"""Build a balanced from-scratch SFT subset with source-like evidence quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/realtext_grpo_reference_evidence_train_shortprompt.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.json"
DEFAULT_STATS = REPO_ROOT / "data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.stats.json"
DEFAULT_EXCLUDE_LIST = Path(os.environ.get("REALTEXT_EXCLUDE_LIST", "/path/to/RealTextV2/train/test.txt"))
LABELS = ("FORGED", "AUTHENTIC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats_output", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--exclude_list", type=Path, default=DEFAULT_EXCLUDE_LIST)
    parser.add_argument("--num_samples", type=int, default=7851)
    parser.add_argument("--forged_fraction", type=float, default=0.586931600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_stem(record: dict[str, Any]) -> str:
    metadata = record.get("grpo_metadata") or {}
    stem = metadata.get("stem")
    if stem:
        return str(stem)
    image = record.get("images")
    if isinstance(image, list):
        image = image[0] if len(image) == 1 else None
    if not image:
        raise ValueError("Record has neither grpo_metadata.stem nor one image path")
    return Path(str(image)).stem


def detector_dtd_conflict(metadata: dict[str, Any]) -> bool:
    detector_forged = metadata.get("detector_pred_label") == "FORGED"
    return detector_forged != bool(metadata.get("dtd_has_boxes"))


def must_include(record: dict[str, Any]) -> bool:
    metadata = record["grpo_metadata"]
    return not bool(metadata["detector_correct"]) or detector_dtd_conflict(metadata)


def validate_source(records: list[dict[str, Any]], excluded: set[str]) -> None:
    seen: set[str] = set()
    leaked: list[str] = []
    for index, record in enumerate(records):
        metadata = record.get("grpo_metadata")
        messages = record.get("messages")
        if not isinstance(metadata, dict) or metadata.get("prompt_mode") != "reference_evidence":
            raise ValueError(f"Record {index} is not reference-evidence data")
        if metadata.get("gt_label") not in LABELS:
            raise ValueError(f"Record {index} has an invalid GT label")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError(f"Record {index} must contain system/user/assistant messages")
        image = record.get("images")
        if not isinstance(image, str) or not Path(image).is_file():
            raise FileNotFoundError(f"Record {index} has a missing/non-scalar image: {image}")
        bucket = metadata.get("difficulty_bucket")
        if not bucket:
            raise ValueError(f"Record {index} has no difficulty bucket")
        stem = record_stem(record)
        if stem in seen:
            raise ValueError(f"Duplicate stem in source: {stem}")
        seen.add(stem)
        if stem in excluded:
            leaked.append(stem)
    if leaked:
        raise ValueError(f"Validation leakage in source, first 10: {leaked[:10]}")


def weighted_sample_without_replacement(
    records: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if count < 0 or count > len(records):
        raise ValueError(f"Cannot sample {count} rows from {len(records)} candidates")
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for record in records:
        weight = float(record["grpo_metadata"].get("recommended_sampling_weight", 1.0))
        uniform = max(rng.random(), 1e-300)
        priority = -math.log(uniform) / weight
        ranked.append((priority, record_stem(record), record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:count]]


def largest_remainder_targets(
    counts: Counter[str],
    total: int,
    minimums: Counter[str],
) -> dict[str, int]:
    raw = {bucket: counts[bucket] * total / sum(counts.values()) for bucket in counts}
    targets = {bucket: int(math.floor(value)) for bucket, value in raw.items()}
    remaining = total - sum(targets.values())
    order = sorted(counts, key=lambda bucket: (raw[bucket] - targets[bucket], counts[bucket], bucket), reverse=True)
    for bucket in order[:remaining]:
        targets[bucket] += 1

    for bucket, minimum in minimums.items():
        targets[bucket] = max(targets.get(bucket, 0), minimum)

    surplus = sum(targets.values()) - total
    while surplus > 0:
        candidates = [
            bucket for bucket in targets if targets[bucket] > minimums.get(bucket, 0)
        ]
        if not candidates:
            raise ValueError("Minimum required rows exceed the requested target")
        bucket = max(
            candidates,
            key=lambda item: (targets[item] - raw.get(item, 0.0), targets[item], item),
        )
        targets[bucket] -= 1
        surplus -= 1

    deficit = total - sum(targets.values())
    while deficit > 0:
        candidates = [bucket for bucket in counts if targets.get(bucket, 0) < counts[bucket]]
        if not candidates:
            raise ValueError("Not enough rows to satisfy requested target")
        bucket = max(
            candidates,
            key=lambda item: (raw.get(item, 0.0) - targets.get(item, 0), counts[item], item),
        )
        targets[bucket] = targets.get(bucket, 0) + 1
        deficit -= 1

    return dict(sorted(targets.items()))


def select_records(
    records: list[dict[str, Any]],
    num_samples: int,
    forged_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if num_samples < 1 or num_samples > len(records):
        raise ValueError("--num_samples must be within the source dataset size")
    if not 0.0 <= forged_fraction <= 1.0:
        raise ValueError("--forged_fraction must be in [0, 1]")

    class_targets = {
        "FORGED": int(round(num_samples * forged_fraction)),
        "AUTHENTIC": num_samples - int(round(num_samples * forged_fraction)),
    }
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    plan: dict[str, Any] = {"class_targets": class_targets, "bucket_targets": {}}

    for label in LABELS:
        label_records = [record for record in records if record["grpo_metadata"]["gt_label"] == label]
        bucket_counts = Counter(record["grpo_metadata"]["difficulty_bucket"] for record in label_records)
        mandatory = [record for record in label_records if must_include(record)]
        mandatory_counts = Counter(record["grpo_metadata"]["difficulty_bucket"] for record in mandatory)
        bucket_targets = largest_remainder_targets(
            bucket_counts,
            class_targets[label],
            mandatory_counts,
        )
        plan["bucket_targets"][label] = bucket_targets

        mandatory_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rest_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        mandatory_stems = {record_stem(record) for record in mandatory}
        for record in label_records:
            bucket = record["grpo_metadata"]["difficulty_bucket"]
            if record_stem(record) in mandatory_stems:
                mandatory_by_bucket[bucket].append(record)
            else:
                rest_by_bucket[bucket].append(record)

        for bucket, target in bucket_targets.items():
            bucket_mandatory = sorted(mandatory_by_bucket[bucket], key=record_stem)
            if len(bucket_mandatory) > target:
                raise ValueError(
                    f"Mandatory rows for {label}/{bucket} exceed target: "
                    f"{len(bucket_mandatory)} > {target}"
                )
            selected.extend(bucket_mandatory)
            fill_count = target - len(bucket_mandatory)
            selected.extend(weighted_sample_without_replacement(rest_by_bucket[bucket], fill_count, rng))

    rng.shuffle(selected)
    if len(selected) != num_samples or len({record_stem(row) for row in selected}) != num_samples:
        raise RuntimeError("SFT selection did not produce the requested unique row count")
    mandatory_all = {record_stem(record) for record in records if must_include(record)}
    selected_stems = {record_stem(row) for row in selected}
    if not mandatory_all.issubset(selected_stems):
        missing = sorted(mandatory_all - selected_stems)
        raise RuntimeError(f"Mandatory detector-error/conflict rows were lost: {missing[:10]}")
    return selected, plan


def counter_for(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(record["grpo_metadata"].get(field)) for record in records)
    return dict(sorted(counts.items()))


def iou_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0000001)]
    counts: Counter[str] = Counter()
    for record in records:
        value = record["grpo_metadata"].get("dtd_gt_iou")
        if value is None:
            counts["none"] += 1
            continue
        iou = float(value)
        for lower, upper in bins:
            if lower <= iou < upper:
                right = "1.0" if upper > 1.0 else f"{upper:.1f}"
                counts[f"[{lower:.1f},{right})"] += 1
                break
    return dict(counts)


def write_json_atomic(path: Path, value: Any, overwrite: bool) -> bytes:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return payload


def main() -> None:
    args = parse_args()
    if not args.input.is_file() or not args.exclude_list.is_file():
        raise FileNotFoundError(f"Missing input or exclude list: {args.input}, {args.exclude_list}")
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list")
    excluded = {
        Path(line.strip().split()[0]).stem
        for line in args.exclude_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    validate_source(records, excluded)
    selected, plan = select_records(records, args.num_samples, args.forged_fraction, args.seed)
    selected_stems = {record_stem(record) for record in selected}
    if selected_stems & excluded:
        raise RuntimeError("Validation overlap appeared after SFT selection")

    sft_records = [{"messages": record["messages"], "images": record["images"]} for record in selected]
    output_payload = write_json_atomic(args.output, sft_records, args.overwrite)
    stats = {
        "dataset_type": args.output.stem,
        "source": str(args.input.resolve()),
        "source_sha256": file_sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(output_payload).hexdigest(),
        "exclude_list": str(args.exclude_list.resolve()),
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "forged_fraction": args.forged_fraction,
        "source_records": len(records),
        "output_records": len(sft_records),
        "validation_overlap": 0,
        "selection_policy": (
            "Class targets are determined by num_samples and forged_fraction. "
            "Within each class, difficulty buckets follow the source distribution "
            "by largest remainder, with all detector-error or detector/DTD-conflict "
            "rows forced in."
        ),
        "metadata_removed_from_training_json": True,
        "plan": plan,
        "counts": {
            "gt_label": counter_for(selected, "gt_label"),
            "difficulty_bucket": counter_for(selected, "difficulty_bucket"),
            "localization_f1_bucket": counter_for(selected, "localization_f1_bucket"),
            "evidence_case": counter_for(selected, "evidence_case"),
            "recommended_sampling_weight": counter_for(selected, "recommended_sampling_weight"),
            "detector_correct": counter_for(selected, "detector_correct"),
        },
        "forged_iou_histogram": iou_histogram(
            [record for record in selected if record["grpo_metadata"]["gt_label"] == "FORGED"]
        ),
    }
    write_json_atomic(args.stats_output, stats, args.overwrite)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
