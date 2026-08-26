"""GT-only rewards for structured document-forensics GRPO."""

from __future__ import annotations

import math
import os
import re
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment


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
BOX_RE = re.compile(
    r"\[\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*\]"
)
DEBUG_REWARD_CALLS = 0


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


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


def parse_box_candidates(
    text: str,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    valid, invalid = [], []
    for match in GROUNDING_RE.finditer(text):
        box = tuple(map(int, match.groups()))
        x1, y1, x2, y2 = box
        if all(0 <= value <= 999 for value in box) and x2 > x1 and y2 > y1:
            valid.append(box)
        else:
            invalid.append(box)
    return valid, invalid


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def one_to_one_grounding_metrics(
    target_boxes: list,
    pred_boxes: list[tuple[int, int, int, int]],
    match_iou_threshold: float = 0.5,
) -> tuple[float, float, list[float]]:
    """Return box-F1 and set-IoU after maximum-IoU one-to-one matching.

    ``set_iou`` divides the sum of matched IoUs by ``max(num_gt, num_pred)``.
    Unmatched GT and predicted boxes therefore contribute zero and lower the
    score. ``box_f1`` treats a matched pair as a TP only above the configured
    IoU threshold, so low-quality matches become one FP plus one FN.
    """
    num_gt, num_pred = len(target_boxes), len(pred_boxes)
    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, []
    if num_gt == 0 or num_pred == 0:
        return 0.0, 0.0, []

    iou_matrix = [
        [box_iou(tuple(target), prediction) for prediction in pred_boxes]
        for target in target_boxes
    ]
    row_indices, column_indices = linear_sum_assignment(iou_matrix, maximize=True)
    matched_ious = [
        float(iou_matrix[row][column]) for row, column in zip(row_indices, column_indices)
    ]
    true_positives = sum(iou >= match_iou_threshold for iou in matched_ious)
    false_positives = num_pred - true_positives
    false_negatives = num_gt - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    box_f1 = 2 * true_positives / denominator if denominator else 1.0
    set_iou = sum(matched_ious) / max(num_gt, num_pred)
    return float(box_f1), float(set_iou), matched_ious


def extract_explanation_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    kept = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if any(key in upper for key in ("REPORT ID", "DATE OF EXAMINATION", "CASE TYPE")):
            continue
        if "FORGERY ANALYSIS" in upper or "OVERALL ASSESSMENT" in upper:
            continue
        if CONCLUSION_RE.search(line) or RISK_RE.search(line) or GROUNDING_RE.search(line):
            continue
        if BOX_RE.fullmatch(line) or set(line) <= {"-", "*", "_", " "}:
            continue
        line = re.sub(r"\[?\s*REASON\s*\]?\s*[:：]\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^#+\s*", "", line)
        kept.append(line)
    return "\n".join(kept).strip()


class ExplanationRewarder:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(vectors, dim=0)

    def score(self, generations: list[str], references: list[str]) -> list[float]:
        generated = [extract_explanation_text(text) for text in generations]
        reference = [extract_explanation_text(text) for text in references]
        valid = [bool(generated[i]) and bool(reference[i]) for i in range(len(generated))]
        texts = [text if text else " " for pair in zip(generated, reference) for text in pair]
        embeddings = self.encode(texts)
        similarities = (embeddings[0::2] * embeddings[1::2]).sum(dim=1).tolist()
        return [max(0.0, float(score)) if ok else 0.0 for score, ok in zip(similarities, valid)]


def format_reward(completions, **kwargs) -> list[float]:
    scores = []
    for completion in completions:
        text = completion_text(completion)
        score = 0.0
        score += 0.25 if CONCLUSION_RE.search(text) else 0.0
        score += 0.25 if RISK_RE.search(text) else 0.0
        score += 0.25 if "SUMMARY" in text.upper() else 0.0
        score += 0.25 if ("GROUNDING" in text.upper() or "NO ANOMAL" in text.upper()) else 0.0
        scores.append(score)
    return scores


def grounding_components(
    text: str,
    target_boxes: list,
    label: str,
    match_iou_threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    pred_label = parse_label(text)
    pred_boxes, invalid_boxes = parse_box_candidates(text)
    classification = 1.0 if pred_label == label else 0.0
    box_f1, set_iou, _ = one_to_one_grounding_metrics(
        target_boxes, pred_boxes, match_iou_threshold=match_iou_threshold
    )
    invalid_penalty = min(1.0, 0.25 * len(invalid_boxes))
    return classification, box_f1, set_iou, invalid_penalty


def make_format_reward(weight: float):
    def weighted_format_reward(completions, **kwargs):
        return [weight * score for score in format_reward(completions, **kwargs)]

    weighted_format_reward.__name__ = "format_reward"
    return weighted_format_reward


def make_grounding_reward(
    total_weight: float,
    cls_weight: float,
    num_weight: float,
    iou_weight: float,
    match_iou_threshold: float = 0.5,
):
    component_weight_sum = cls_weight + num_weight + iou_weight
    if component_weight_sum <= 0:
        raise ValueError("Grounding component weights must sum to a positive value")

    def grounding_reward(completions, gt_boxes, gt_label, **kwargs):
        rewards = []
        for completion, target_boxes, label in zip(completions, gt_boxes, gt_label):
            cls, box_f1, set_iou, penalty = grounding_components(
                completion_text(completion),
                target_boxes,
                label,
                match_iou_threshold=match_iou_threshold,
            )
            quality = (
                cls_weight * cls + num_weight * box_f1 + iou_weight * set_iou
            ) / component_weight_sum
            rewards.append(total_weight * max(-1.0, min(1.0, quality - penalty)))
        return rewards

    grounding_reward.__name__ = "grounding_reward"
    return grounding_reward


def make_explanation_reward(weight: float, rewarder: ExplanationRewarder):
    def explanation_reward(completions, reference_answer, **kwargs):
        generated = [completion_text(completion) for completion in completions]
        return [weight * score for score in rewarder.score(generated, reference_answer)]

    explanation_reward.__name__ = "explanation_reward"
    return explanation_reward


def debug_print_reward(
    completions,
    gt_label,
    gt_risk,
    gt_boxes,
    image_path=None,
    difficulty_bucket=None,
    evidence_case=None,
    **kwargs,
) -> list[float]:
    global DEBUG_REWARD_CALLS
    DEBUG_REWARD_CALLS += 1
    every = int(os.environ.get("DEBUG_PRINT_EVERY", "10"))
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and every > 0 and DEBUG_REWARD_CALLS % every == 0:
        index = 0
        text = completion_text(completions[index])
        components = grounding_components(text, gt_boxes[index], gt_label[index])
        print(
            "\n[reference_grpo_debug]"
            f"\nimage={image_path[index] if image_path else ''}"
            f"\nevidence_case={evidence_case[index] if evidence_case else ''}"
            f"\ndifficulty={difficulty_bucket[index] if difficulty_bucket else ''}"
            f"\ngt_label={gt_label[index]} pred_label={parse_label(text)}"
            f"\ngt_risk={gt_risk[index]} pred_risk={parse_risk(text)}"
            f"\ngt_boxes={gt_boxes[index]} pred_boxes={parse_box_candidates(text)[0]}"
            f"\ngrounding_components={components}"
            f"\ntext={text[:1200]}\n[/reference_grpo_debug]\n",
            flush=True,
        )
    return [0.0 for _ in completions]
