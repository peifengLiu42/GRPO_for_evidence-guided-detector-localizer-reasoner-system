#!/usr/bin/env python3
"""Batch reference-evidence inference with vLLM and one LoRA checkpoint.

This script consumes a prebuilt reference-evidence JSONL, where each row already
contains the image path and detector/DTD evidence prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realtext_grpo.prompts import SYSTEM_PROMPT  # noqa: E402


DEFAULT_EVIDENCE = REPO_ROOT / "data/realtext_indomain_reference_evidence.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/realtext_indomain_reference_vllm.jsonl"
DEFAULT_BASE_MODEL = Path(os.environ.get("QWEN3_VL_MODEL", "/path/to/Qwen3-VL-4B-Instruct"))
DEFAULT_ADAPTER = (
    REPO_ROOT / "outputs/qwen3vl4b_reference_rft_from_sft5_distmix_authdown_shortprompt_visionlora_projector/checkpoint-SELECTED"
)
REPORT_HEADER_RE = re.compile(
    r"(#\s*FORGERY\s+ANALYSIS\s+REPORT\s*\n+).*?(\*\*Overall Assessment:\*\*)",
    re.IGNORECASE | re.DOTALL,
)
THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.IGNORECASE | re.DOTALL)
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_jsonl", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_name_or_path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_checkpoint", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--merged_model",
        action="store_true",
        help=(
            "Load model_name_or_path as an already merged full model. This disables "
            "vLLM LoRA hot-loading and is required when the adapter trained vision or "
            "multimodal-projector weights."
        ),
    )
    parser.add_argument("--test_num", type=int, default=0, help="0 uses every evidence record.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resize", type=int, default=1280, help="Resize image long edge; 0 disables.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def clean_report(report: str) -> str:
    report = THINK_RE.sub("", report)
    report = report.replace("<think>\n\n</think>\n\n", "")
    report = report.replace("<think>\n</think>\n", "")
    return REPORT_HEADER_RE.sub(r"\1\2", report, count=1).lstrip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if record.get("prompt_mode") != "reference_evidence":
                raise ValueError(f"Non-reference prompt at {path}:{line_number}")
            image_path = Path(str(record.get("image_path", "")))
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image at {path}:{line_number}: {image_path}")
            if not record.get("prompt"):
                raise ValueError(f"Missing prompt at {path}:{line_number}")
            records.append(record)
    return records


def load_done_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                name = json.loads(line).get("image_name")
            except json.JSONDecodeError:
                continue
            if name:
                done.add(str(name))
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
    user_text = str(record["prompt"])
    if user_text.startswith("<image>"):
        user_text = user_text[len("<image>") :].lstrip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def validate_args(args: argparse.Namespace) -> None:
    required_paths = [
        (args.evidence_jsonl, "evidence JSONL"),
        (args.model_name_or_path, "model"),
    ]
    if not args.merged_model:
        required_paths.append((args.adapter_checkpoint, "LoRA adapter"))
    for path, label in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_model_len < 1:
        raise ValueError("Batch/length arguments must be positive")
    if args.data_parallel_size < 1:
        raise ValueError("--data_parallel_size must be positive")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, --num_shards)")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu_memory_utilization must be in (0, 1)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    records = load_jsonl(args.evidence_jsonl)
    if args.test_num > 0:
        records = records[: args.test_num]
    records = records[args.shard_index :: args.num_shards]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output_jsonl.write_text("", encoding="utf-8")
        done_names: set[str] = set()
    else:
        done_names = load_done_names(args.output_jsonl)
    records = [record for record in records if record["image_name"] not in done_names]
    if not records:
        print("No evidence records left to process.")
        return

    # Import only after validating inputs so basic data checks remain cheap.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs = dict(
        model=str(args.model_name_or_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        data_parallel_size=args.data_parallel_size,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": args.max_pixels},
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    if not args.merged_model:
        llm_kwargs.update(enable_lora=True, max_lora_rank=32, max_loras=1)
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )
    lora_request = None
    if not args.merged_model:
        lora_request = LoRARequest(
            "reference_sft_rft_adapter", 1, str(args.adapter_checkpoint)
        )

    with args.output_jsonl.open("a", encoding="utf-8") as output_handle, tqdm(
        total=len(records),
        desc=f"vLLM shard {args.shard_index + 1}/{args.num_shards}",
        unit="img",
    ) as progress:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            images = [load_image(Path(record["image_path"]), args.resize) for record in batch]
            messages = [
                build_messages(record, image) for record, image in zip(batch, images)
            ]
            chat_kwargs = {
                "sampling_params": sampling_params,
                "use_tqdm": False,
            }
            if lora_request is not None:
                chat_kwargs["lora_request"] = lora_request
            outputs = llm.chat(messages, **chat_kwargs)
            if len(outputs) != len(batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} inputs")
            for record, request_output in zip(batch, outputs):
                report = clean_report(request_output.outputs[0].text.strip())
                result = {
                    **record,
                    "report": report,
                    "inference_backend": "vllm",
                    "vllm_model": str(args.model_name_or_path),
                    "vllm_adapter": (
                        None if args.merged_model else str(args.adapter_checkpoint)
                    ),
                    "vllm_merged_model": bool(args.merged_model),
                }
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                progress.update(1)
            for image in images:
                image.close()

    print(f"Done: wrote {len(records)} report(s) to {args.output_jsonl}")


if __name__ == "__main__":
    main()
