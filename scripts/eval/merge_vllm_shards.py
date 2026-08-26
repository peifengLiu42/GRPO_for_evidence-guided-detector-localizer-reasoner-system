#!/usr/bin/env python3
"""Merge sharded vLLM JSONL outputs in the original evidence-file order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("parts", type=Path, nargs="+")
    args = parser.parse_args()

    evidence = read_jsonl(args.evidence_jsonl)
    by_name: dict[str, dict] = {}
    for part in args.parts:
        for record in read_jsonl(part):
            name = str(record.get("image_name", ""))
            if not name:
                raise ValueError(f"Record without image_name in {part}")
            if name in by_name:
                raise ValueError(f"Duplicate image_name across shards: {name}")
            by_name[name] = record

    expected_names = [str(record["image_name"]) for record in evidence]
    missing = [name for name in expected_names if name not in by_name]
    extra = sorted(set(by_name).difference(expected_names))
    if missing or extra:
        raise ValueError(f"Shard mismatch: missing={len(missing)}, extra={len(extra)}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for name in expected_names:
            handle.write(json.dumps(by_name[name], ensure_ascii=False) + "\n")
    print(f"Merged {len(expected_names)} unique records into {args.output_jsonl}")


if __name__ == "__main__":
    main()
