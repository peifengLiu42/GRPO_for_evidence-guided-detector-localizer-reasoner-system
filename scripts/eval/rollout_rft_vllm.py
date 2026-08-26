#!/usr/bin/env python3
"""Generate multi-sample RFT rollouts with vLLM for reference-evidence prompts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]

Image.MAX_IMAGE_PIXELS = None


DEFAULT_SOURCE = REPO_ROOT / "data/realtext_grpo_reference_evidence_train_shortprompt.json"
DEFAULT_MODEL = (
    REPO_ROOT
    / "outputs/qwen3vl4b_reference_sft5_final_visionlora_projector_merged_bf16"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/rft_sft5_rollouts/rollouts.part0.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_json", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip_jsonl",
        type=Path,
        action="append",
        default=[],
        help="Existing rollout JSONL files whose stems should be skipped.",
    )
    parser.add_argument("--model_name_or_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max_prompts", type=int, default=4096)
    parser.add_argument("--forged_fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resize", type=int, default=1280)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def record_stem(record: dict[str, Any]) -> str:
    metadata = record.get("grpo_metadata") or {}
    if metadata.get("stem"):
        return str(metadata["stem"])
    return Path(str(record["images"])).stem


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {path}")
    for index, record in enumerate(records):
        messages = record.get("messages")
        image = record.get("images")
        metadata = record.get("grpo_metadata")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Record {index} must have 3 sharegpt messages")
        if not isinstance(image, str) or not Path(image).is_file():
            raise FileNotFoundError(f"Record {index} missing image: {image}")
        if not isinstance(metadata, dict) or metadata.get("gt_label") not in {"FORGED", "AUTHENTIC"}:
            raise ValueError(f"Record {index} missing valid grpo_metadata")
    return records


def weighted_sample_without_replacement(
    records: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"Cannot sample {count} from {len(records)} rows")
    ranked = []
    for record in records:
        weight = float((record.get("grpo_metadata") or {}).get("recommended_sampling_weight", 1.0))
        priority = -1.0 * weight * rng.random()
        ranked.append((priority, record_stem(record), record))
    ranked.sort()
    return [item[2] for item in ranked[:count]]


def select_records(
    records: list[dict[str, Any]],
    max_prompts: int,
    forged_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if max_prompts <= 0 or max_prompts > len(records):
        raise ValueError("--max_prompts must be in [1, len(source)]")
    if not 0.0 <= forged_fraction <= 1.0:
        raise ValueError("--forged_fraction must be in [0, 1]")
    rng = random.Random(seed)
    forged_target = int(round(max_prompts * forged_fraction))
    authentic_target = max_prompts - forged_target
    by_label = {
        "FORGED": [row for row in records if row["grpo_metadata"]["gt_label"] == "FORGED"],
        "AUTHENTIC": [row for row in records if row["grpo_metadata"]["gt_label"] == "AUTHENTIC"],
    }
    selected = (
        weighted_sample_without_replacement(by_label["FORGED"], forged_target, rng)
        + weighted_sample_without_replacement(by_label["AUTHENTIC"], authentic_target, rng)
    )
    selected.sort(key=record_stem)
    return selected


def load_done_stems(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("stem"):
                done.add(str(value["stem"]))
    return done


def load_many_done_stems(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        done.update(load_done_stems(path))
    return done


def load_image(path: Path, resize: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if resize > 0:
        width, height = image.size
        scale = resize / max(width, height)
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def build_messages(record: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
    system_text = str(record["messages"][0]["content"])
    user_text = str(record["messages"][1]["content"])
    if user_text.startswith("<image>"):
        user_text = user_text[len("<image>") :].lstrip()
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, --num_shards)")
    records = select_records(
        load_records(args.source_json),
        max_prompts=args.max_prompts,
        forged_fraction=args.forged_fraction,
        seed=args.seed,
    )
    records = records[args.shard_index :: args.num_shards]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output_jsonl.write_text("", encoding="utf-8")
        done_stems: set[str] = set()
    else:
        done_stems = load_done_stems(args.output_jsonl)
    done_stems.update(load_many_done_stems(args.skip_jsonl))
    records = [record for record in records if record_stem(record) not in done_stems]
    if not records:
        print("No prompts left to process.")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model_name_or_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": args.max_pixels},
        seed=args.seed + args.shard_index,
    )
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )

    with args.output_jsonl.open("a", encoding="utf-8") as output_handle, tqdm(
        total=len(records),
        desc=f"RFT rollout shard {args.shard_index + 1}/{args.num_shards}",
        unit="prompt",
    ) as progress:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            images = [load_image(Path(record["images"]), args.resize) for record in batch]
            messages = [
                build_messages(record, image) for record, image in zip(batch, images)
            ]
            outputs = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts")
            for source_index, record, request_output in zip(
                range(start, start + len(batch)), batch, outputs
            ):
                completions = [
                    {
                        "text": output.text.strip(),
                        "finish_reason": str(output.finish_reason),
                        "stop_reason": str(output.stop_reason),
                    }
                    for output in request_output.outputs
                ]
                row = {
                    "source_index_in_shard": source_index,
                    "stem": record_stem(record),
                    "images": record["images"],
                    "messages": record["messages"],
                    "grpo_metadata": record["grpo_metadata"],
                    "completions": completions,
                    "rollout_config": {
                        "n": args.n,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "model_name_or_path": str(args.model_name_or_path),
                    },
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                progress.update(1)
            for image in images:
                image.close()

    print(f"Done: wrote {len(records)} rollout prompt(s) to {args.output_jsonl}")


if __name__ == "__main__":
    main()
