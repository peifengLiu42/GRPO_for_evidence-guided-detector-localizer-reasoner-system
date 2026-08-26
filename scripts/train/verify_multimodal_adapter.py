#!/usr/bin/env python3
"""Verify that a PEFT checkpoint contains multimodal trainables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.adapter / "adapter_config.json"
    weights = sorted(args.adapter.glob("*.safetensors"))
    if not config_path.is_file() or not weights:
        raise FileNotFoundError(f"Incomplete PEFT checkpoint: {args.adapter}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_modules = [str(value) for value in config.get("target_modules") or []]
    modules_to_save = [str(value) for value in config.get("modules_to_save") or []]
    keys: list[str] = []
    for path in weights:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())

    categories = {
        "language_lora": [
            key for key in keys if "language_model" in key and "lora_" in key
        ],
        "vision_block_lora": [
            key for key in keys if "visual.blocks" in key and "lora_" in key
        ],
        "deepstack_merger_lora": [
            key
            for key in keys
            if "visual.deepstack_merger_list" in key and "lora_" in key
        ],
        "projector_full_weights": [
            key
            for key in keys
            # PEFT strips the runtime ``modules_to_save.default`` segment when
            # serializing, so saved full projector tensors have ordinary names.
            if "visual.merger" in key and "lora_" not in key
        ],
    }
    report = {
        "adapter": str(args.adapter.resolve()),
        "total_weight_keys": len(keys),
        "target_module_count": len(target_modules),
        "modules_to_save": modules_to_save,
        "category_key_counts": {name: len(values) for name, values in categories.items()},
        "category_examples": {name: values[:2] for name, values in categories.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    missing = [name for name, values in categories.items() if not values]
    if "visual.merger" not in modules_to_save:
        missing.append("modules_to_save=visual.merger")
    if missing:
        raise RuntimeError("Missing multimodal trainables: " + ", ".join(missing))
    print("MULTIMODAL_ADAPTER=PASS")


if __name__ == "__main__":
    main()
