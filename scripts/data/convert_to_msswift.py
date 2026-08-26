#!/usr/bin/env python3
"""Convert reference-evidence GRPO JSON to ms-swift JSONL.

The ms-swift GRPO trainer consumes prompt-only messages and passes any
additional dataset columns to the custom reward plugin.  We therefore remove
the assistant answer from ``messages`` and keep it as ``reference_answer`` for
reward-side explanation matching/debugging.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realtext_grpo.dataset import parse_gt_boxes, parse_label, parse_risk  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "data/realtext_grpo_reference_evidence_train_shortprompt.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/realtext_grpo_reference_evidence_train_shortprompt_msswift_ultrahardmix3k_seed42.jsonl"

HARDMIX_BUCKET_FRACTIONS = {
    "forged_iou_lt_0.3": 0.24,
    "forged_iou_0.3_0.7": 0.23,
    "forged_iou_ge_0.7": 0.13,
    "forged_dtd_empty": 0.01,
    "authentic_dtd_fp": 0.24,
    "authentic_clean": 0.15,
}

LEAN_HARDMIX_BUCKET_FRACTIONS = {
    "forged_iou_lt_0.3": 0.20,
    "forged_iou_0.3_0.7": 0.25,
    "forged_iou_ge_0.7": 0.15,
    "forged_dtd_empty": 0.005,
    "authentic_dtd_fp": 0.20,
    "authentic_clean": 0.195,
}

ULTRA_HARDMIX_BUCKET_FRACTIONS = {
    "forged_iou_lt_0.3": 0.33,
    "forged_iou_0.3_0.7": 0.28,
    "forged_iou_ge_0.7": 0.05,
    "forged_dtd_empty": 0.02,
    "authentic_dtd_fp": 0.25,
    "authentic_clean": 0.07,
}


def normalize_image_path(value: Any) -> list[str] | None:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value
    return None


def load_candidates(path: Path, strict: bool = True) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Input must be a JSON list: {path}")

    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for record in records:
        messages = record.get("messages") or []
        if len(messages) < 3 or messages[-1].get("role") != "assistant":
            skipped["invalid_messages"] += 1
            continue
        metadata = record.get("grpo_metadata") or {}
        if metadata.get("prompt_mode") != "reference_evidence":
            skipped["wrong_prompt_mode"] += 1
            continue
        answer = str(messages[-1].get("content", ""))
        gt_label = parse_label(answer)
        if gt_label is None or gt_label != metadata.get("gt_label"):
            skipped["invalid_gt_label"] += 1
            continue
        images = normalize_image_path(record.get("images"))
        if images is None or not Path(images[0]).is_file():
            skipped["missing_image"] += 1
            continue
        sampling_weight = float(metadata.get("recommended_sampling_weight", 1.0))
        if sampling_weight <= 0:
            skipped["invalid_sampling_weight"] += 1
            continue

        candidates.append(
            {
                "messages": messages[:-1],
                "images": images,
                "gt_label": gt_label,
                "gt_risk": parse_risk(answer),
                "gt_boxes": parse_gt_boxes(answer),
                "reference_answer": answer,
                "stem": metadata.get("stem", Path(images[0]).stem),
                "difficulty_bucket": metadata.get("difficulty_bucket", "unknown"),
                "evidence_case": metadata.get("evidence_case", "unknown"),
                "sampling_weight": sampling_weight,
            }
        )

    if strict and skipped:
        raise ValueError(f"Invalid records encountered: {dict(skipped)}")
    if not candidates:
        raise RuntimeError(f"No usable records found in {path}")
    return candidates


def weighted_sample(
    rows: list[dict[str, Any]], num_records: int, forged_fraction: float, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label = {
        "FORGED": [row for row in rows if row["gt_label"] == "FORGED"],
        "AUTHENTIC": [row for row in rows if row["gt_label"] == "AUTHENTIC"],
    }
    if not by_label["FORGED"] or not by_label["AUTHENTIC"]:
        raise RuntimeError("Weighted sampling expects both FORGED and AUTHENTIC rows")
    num_forged = int(round(num_records * forged_fraction))
    num_authentic = num_records - num_forged
    selected: list[dict[str, Any]] = []
    for label, count in (("FORGED", num_forged), ("AUTHENTIC", num_authentic)):
        population = by_label[label]
        weights = [float(row["sampling_weight"]) for row in population]
        selected.extend(rng.choices(population, weights=weights, k=count))
    rng.shuffle(selected)
    return selected


def _counts_from_fractions(num_records: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {key: num_records * value for key, value in fractions.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = num_records - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def bucket_fraction_sample(
    rows: list[dict[str, Any]],
    num_records: int,
    seed: int,
    bucket_fractions: dict[str, float],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault(str(row["difficulty_bucket"]), []).append(row)

    counts = _counts_from_fractions(num_records, bucket_fractions)
    selected: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    for bucket, count in counts.items():
        population = by_bucket.get(bucket, [])
        if not population:
            missing[bucket] = count
            continue
        weights = [float(row["sampling_weight"]) for row in population]
        selected.extend(rng.choices(population, weights=weights, k=count))

    if missing:
        fallback = [row for row in rows if row["gt_label"] == "FORGED"]
        fallback_weights = [float(row["sampling_weight"]) for row in fallback]
        selected.extend(rng.choices(fallback, weights=fallback_weights, k=sum(missing.values())))

    rng.shuffle(selected)
    return selected


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    weight_sums = Counter()
    stems_by_bucket: dict[str, set[str]] = {}
    for row in rows:
        counts[f"label/{row['gt_label'].lower()}"] += 1
        counts[f"difficulty/{row['difficulty_bucket']}"] += 1
        counts[f"evidence_case/{row['evidence_case']}"] += 1
        counts[f"weight/{row['sampling_weight']:g}"] += 1
        weight_sums[row["gt_label"]] += float(row["sampling_weight"])
        stems_by_bucket.setdefault(str(row["difficulty_bucket"]), set()).add(str(row["stem"]))
    return {
        "records": len(rows),
        "counts": dict(sorted(counts.items())),
        "class_weight_sums": dict(sorted(weight_sums.items())),
        "unique_stems_by_bucket": {
            key: len(value) for key, value in sorted(stems_by_bucket.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--sampling_mode",
        choices=("natural", "weighted", "hardmix", "lean_hardmix", "ultra_hardmix"),
        default="ultra_hardmix",
    )
    parser.add_argument("--num_records", type=int, default=3000)
    parser.add_argument("--forged_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    candidates = load_candidates(input_path, strict=not args.no_strict)
    if args.sampling_mode == "weighted":
        rows = weighted_sample(candidates, args.num_records, args.forged_fraction, args.seed)
    elif args.sampling_mode == "hardmix":
        rows = bucket_fraction_sample(
            candidates, args.num_records, args.seed, HARDMIX_BUCKET_FRACTIONS
        )
    elif args.sampling_mode == "lean_hardmix":
        rows = bucket_fraction_sample(
            candidates, args.num_records, args.seed, LEAN_HARDMIX_BUCKET_FRACTIONS
        )
    elif args.sampling_mode == "ultra_hardmix":
        rows = bucket_fraction_sample(
            candidates, args.num_records, args.seed, ULTRA_HARDMIX_BUCKET_FRACTIONS
        )
    else:
        rows = list(candidates)
        if args.num_records > 0:
            rows = rows[: args.num_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(input_path),
        "output": str(output_path),
        "sampling_mode": args.sampling_mode,
        "seed": args.seed,
        "forged_fraction": args.forged_fraction,
        "hardmix_bucket_fractions": {
            "hardmix": HARDMIX_BUCKET_FRACTIONS,
            "lean_hardmix": LEAN_HARDMIX_BUCKET_FRACTIONS,
            "ultra_hardmix": ULTRA_HARDMIX_BUCKET_FRACTIONS,
        }.get(args.sampling_mode),
        "source_stats": stats(candidates),
        "output_stats": stats(rows),
    }
    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
