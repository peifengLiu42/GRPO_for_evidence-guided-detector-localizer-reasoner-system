#!/usr/bin/env python3
"""Merge the selected SFT/RFT LoRA into a BF16 GRPO reference backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = Path(os.environ.get("QWEN3_VL_MODEL", "/path/to/Qwen3-VL-4B-Instruct"))
DEFAULT_ADAPTER = Path(
    os.environ.get(
        "REFERENCE_ADAPTER",
        REPO_ROOT / "outputs/qwen3vl4b_reference_rft_from_sft5_distmix_authdown_shortprompt_visionlora_projector/checkpoint-SELECTED",
    )
)
DEFAULT_OUTPUT = Path(
    os.environ.get(
        "REFERENCE_MERGED_MODEL",
        REPO_ROOT / "outputs/qwen3vl4b_reference_sft5_rft1_merged_bf16",
    )
)
MANIFEST_NAME = "reference_merge_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--foundation_adapter",
        type=Path,
        default=None,
        help="Optional earlier adapter to merge before the selected SFT/RFT adapter.",
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index.")
    parser.add_argument("--max_shard_size", default="5GB")
    args = parser.parse_args()

    required = [(args.base_model, "base model"), (args.adapter, "adapter")]
    if args.foundation_adapter is not None:
        required.append((args.foundation_adapter, "foundation adapter"))
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    adapter_config = args.adapter / "adapter_config.json"
    adapter_weights = args.adapter / "adapter_model.safetensors"
    for path in (adapter_config, adapter_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.base_model),
        dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    foundation_manifest = None
    if args.foundation_adapter is not None:
        foundation_config = args.foundation_adapter / "adapter_config.json"
        foundation_weights = args.foundation_adapter / "adapter_model.safetensors"
        for path in (foundation_config, foundation_weights):
            if not path.is_file():
                raise FileNotFoundError(path)
        model = PeftModel.from_pretrained(
            model, str(args.foundation_adapter), is_trainable=False
        )
        model = model.merge_and_unload(safe_merge=True)
        foundation_manifest = {
            "path": str(args.foundation_adapter.resolve()),
            "adapter_config_sha256": sha256(foundation_config),
            "adapter_model_sha256": sha256(foundation_weights),
        }
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = False
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor = AutoProcessor.from_pretrained(str(args.base_model), trust_remote_code=True)
    processor.save_pretrained(args.output_dir)

    reference_policy = "qwen3vl4b_base_plus_sft_rft_adapter_merged"
    if foundation_manifest is not None:
        reference_policy = "qwen3vl4b_base_plus_foundation_plus_sft_rft_adapter_merged"
    manifest = {
        "format_version": 2,
        "reference_policy": reference_policy,
        "base_model": str(args.base_model.resolve()),
        "foundation_adapter": foundation_manifest,
        "adapter": str(args.adapter.resolve()),
        "dtype": "bfloat16",
        "adapter_config_sha256": sha256(adapter_config),
        "adapter_model_sha256": sha256(adapter_weights),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved merged reference model: {args.output_dir}")


if __name__ == "__main__":
    main()
