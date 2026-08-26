"""Reference-evidence GRPO dataset and class-balanced difficulty sampler."""

from __future__ import annotations

import copy
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sized

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


CONCLUSION_RE = re.compile(
    r"\[?\s*Conclusion\s*\]?\s*[:：]\s*\**\s*(FORGED|AUTHENTIC)\b", re.IGNORECASE
)
RISK_RE = re.compile(
    r"\[?\s*RISK_SCORE\s*\]?\s*[:：]\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
GROUNDING_RE = re.compile(
    r"\[?\s*GROUNDING\s*\]?\s*[:：]\s*"
    r"(?:\**\s*)?"
    r"\[\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*\]",
    re.IGNORECASE,
)


def parse_label(text: str) -> str | None:
    match = CONCLUSION_RE.search(text)
    if match:
        return match.group(1).upper()
    labels = {
        match.group(1).upper()
        for match in re.finditer(r"\b(FORGED|AUTHENTIC)\b", text, re.IGNORECASE)
    }
    return labels.pop() if len(labels) == 1 else None


def parse_risk(text: str) -> float | None:
    match = RISK_RE.search(text)
    return max(0.0, min(100.0, float(match.group(1)))) if match else None


def parse_gt_boxes(text: str) -> list[tuple[int, int, int, int]]:
    boxes = []
    for match in GROUNDING_RE.finditer(text):
        box = tuple(map(int, match.groups()))
        x1, y1, x2, y2 = box
        if all(0 <= value <= 999 for value in box) and x2 > x1 and y2 > y1:
            boxes.append(box)
    return boxes


def clean_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the textual image placeholder; the PIL image is supplied separately."""
    prompt = copy.deepcopy(messages)
    for message in prompt:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = message["content"].replace("<image>", "", 1).strip()
            break
    return prompt


def weighted_sample_without_replacement(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if count >= len(rows):
        selected = list(rows)
        rng.shuffle(selected)
        return selected
    # Efraimidis-Spirakis weighted reservoir keys. Larger weights are more
    # likely to produce keys close to zero and therefore enter the top-k set.
    keyed = []
    for row in rows:
        weight = max(float(row["sampling_weight"]), 1e-12)
        key = math.log(max(rng.random(), 1e-12)) / weight
        keyed.append((key, row))
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in keyed[:count]]


class ReferenceEvidenceGRPODataset(Dataset):
    """Load prompts for rollout and keep GT fields as hidden reward columns."""

    def __init__(
        self,
        data_path: str | Path,
        max_samples: int | None = None,
        image_max_side: int | None = 0,
        image_max_pixels: int | None = 1024 * 1024,
        seed: int = 42,
        forged_fraction: float = 0.5,
        strict: bool = True,
    ) -> None:
        records = json.loads(Path(data_path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"GRPO data must be a JSON list: {data_path}")
        self.image_max_side = image_max_side
        self.image_max_pixels = image_max_pixels
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
            image_path = record.get("images")
            if isinstance(image_path, list):
                image_path = image_path[0] if len(image_path) == 1 else None
            if not image_path or not Path(image_path).is_file():
                skipped["missing_image"] += 1
                continue
            sampling_weight = float(metadata.get("recommended_sampling_weight", 1.0))
            if sampling_weight <= 0:
                skipped["invalid_sampling_weight"] += 1
                continue
            candidates.append(
                {
                    "prompt": clean_prompt(messages[:-1]),
                    "image_path": str(image_path),
                    "gt_label": gt_label,
                    "gt_risk": parse_risk(answer),
                    "gt_boxes": parse_gt_boxes(answer),
                    "reference_answer": answer,
                    "sampling_weight": sampling_weight,
                    "difficulty_bucket": metadata.get("difficulty_bucket", "unknown"),
                    "evidence_case": metadata.get("evidence_case", "unknown"),
                    "stem": metadata.get("stem", Path(image_path).stem),
                }
            )

        if strict and skipped:
            raise ValueError(f"Invalid GRPO records encountered: {dict(skipped)}")
        if not candidates:
            raise RuntimeError(f"No usable reference-evidence samples found in {data_path}")
        self.rows = self._select_subset(
            candidates, max_samples=max_samples, seed=seed, forged_fraction=forged_fraction
        )
        self.stats = self._stats(self.rows)

    @staticmethod
    def _select_subset(
        rows: list[dict[str, Any]],
        max_samples: int | None,
        seed: int,
        forged_fraction: float,
    ) -> list[dict[str, Any]]:
        if max_samples is None or max_samples <= 0 or max_samples >= len(rows):
            return rows
        rng = random.Random(seed)
        forged_fraction = min(1.0, max(0.0, forged_fraction))
        forged = [row for row in rows if row["gt_label"] == "FORGED"]
        authentic = [row for row in rows if row["gt_label"] == "AUTHENTIC"]
        num_forged = min(len(forged), int(round(max_samples * forged_fraction)))
        num_authentic = min(len(authentic), max_samples - num_forged)
        selected = weighted_sample_without_replacement(forged, num_forged, rng)
        selected += weighted_sample_without_replacement(authentic, num_authentic, rng)
        remaining = max_samples - len(selected)
        if remaining > 0:
            selected_ids = {id(row) for row in selected}
            leftovers = [row for row in rows if id(row) not in selected_ids]
            selected += weighted_sample_without_replacement(leftovers, remaining, rng)
        rng.shuffle(selected)
        return selected

    @staticmethod
    def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter()
        weight_sums = Counter()
        for row in rows:
            label = row["gt_label"]
            counts[f"label/{label.lower()}"] += 1
            counts[f"weight/{row['sampling_weight']:g}"] += 1
            counts[f"difficulty/{row['difficulty_bucket']}"] += 1
            weight_sums[label] += row["sampling_weight"]
        return {
            "total": len(rows),
            "counts": dict(sorted(counts.items())),
            "class_weight_sums": dict(sorted(weight_sums.items())),
        }

    def sampling_probabilities(self, forged_fraction: float = 0.5) -> torch.Tensor:
        """Return class-balanced, within-class difficulty-weighted probabilities."""
        forged_fraction = min(1.0, max(0.0, forged_fraction))
        class_targets = {"FORGED": forged_fraction, "AUTHENTIC": 1.0 - forged_fraction}
        class_sums = Counter()
        for row in self.rows:
            class_sums[row["gt_label"]] += row["sampling_weight"]
        probabilities = []
        for row in self.rows:
            label = row["gt_label"]
            denominator = class_sums[label]
            probability = class_targets[label] * row["sampling_weight"] / denominator
            probabilities.append(probability)
        tensor = torch.tensor(probabilities, dtype=torch.double)
        if not torch.isclose(tensor.sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-10):
            raise RuntimeError(f"Sampling probabilities do not sum to one: {tensor.sum().item()}")
        return tensor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        width, height = image.size
        scale = 1.0
        if self.image_max_pixels and self.image_max_pixels > 0 and width * height > self.image_max_pixels:
            scale = min(scale, math.sqrt(self.image_max_pixels / float(width * height)))
        if self.image_max_side and self.image_max_side > 0 and max(width, height) > self.image_max_side:
            scale = min(scale, self.image_max_side / float(max(width, height)))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        item = dict(row)
        item["images"] = [image]
        return item


class WeightedRepeatSampler(Sampler[int]):
    """Weighted repeat sampler for grouped GRPO generations."""

    def __init__(
        self,
        data_source: Sized,
        probabilities: torch.Tensor,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int = 1,
        seed: int = 42,
    ) -> None:
        if len(probabilities) != len(data_source):
            raise ValueError("One probability is required for every dataset row")
        if mini_repeat_count < 1 or batch_size < 1 or repeat_count < 1:
            raise ValueError("Sampler repeat counts and batch size must be positive")
        self.data_source = data_source
        self.probabilities = probabilities.cpu().double()
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        draw_count = (self.num_samples // self.batch_size) * self.batch_size
        indices = torch.multinomial(
            self.probabilities,
            num_samples=draw_count,
            replacement=True,
            generator=self.generator,
        ).tolist()
        for start in range(0, draw_count, self.batch_size):
            chunk = indices[start : start + self.batch_size]
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        draw_count = (self.num_samples // self.batch_size) * self.batch_size
        return draw_count * self.mini_repeat_count * self.repeat_count
