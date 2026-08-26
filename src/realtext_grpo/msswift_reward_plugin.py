"""ms-swift reward plugin for RealText reference-evidence GRPO.

Register name:

    realtext_reference_evidence

The plugin intentionally starts with format + grounding rewards.  Explanation
embedding reward is left out for the first ms-swift probe to keep distributed
reward execution simple and comparable to the localization objective.
"""

from __future__ import annotations

import json
import os
from typing import Any

from realtext_grpo.rewards import (
    completion_text,
    format_reward,
    one_to_one_grounding_metrics,
    parse_box_candidates,
    parse_label,
)


from swift.rewards import ORM, orms


def _as_list(value: Any, length: int) -> list[Any]:
    if isinstance(value, list) and len(value) == length:
        return value
    return [value for _ in range(length)]


def _parse_boxes(value: Any) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return value


def _normalize_box(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = (
        max(0, min(999, x1)),
        max(0, min(999, y1)),
        max(0, min(999, x2)),
        max(0, min(999, y2)),
    )
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _normalize_boxes(values: list[Any]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for value in values:
        box = _normalize_box(value)
        if box is not None:
            boxes.append(box)
    return boxes


def _covered_area(
    boxes: list[tuple[int, int, int, int]],
    other_boxes: list[tuple[int, int, int, int]] | None = None,
) -> float:
    all_boxes = boxes if other_boxes is None else boxes + other_boxes
    if not all_boxes:
        return 0.0
    xs = sorted({coord for box in all_boxes for coord in (box[0], box[2])})
    ys = sorted({coord for box in all_boxes for coord in (box[1], box[3])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        x_mid = (left + right) / 2.0
        for top, bottom in zip(ys, ys[1:]):
            if bottom <= top:
                continue
            y_mid = (top + bottom) / 2.0
            covered = any(
                x1 <= x_mid < x2 and y1 <= y_mid < y2 for x1, y1, x2, y2 in boxes
            )
            if other_boxes is not None:
                covered = covered and any(
                    x1 <= x_mid < x2 and y1 <= y_mid < y2
                    for x1, y1, x2, y2 in other_boxes
                )
            if covered:
                area += float((right - left) * (bottom - top))
    return area


def _union_pixel_metrics(
    target_boxes: list[tuple[int, int, int, int]],
    pred_boxes: list[tuple[int, int, int, int]],
) -> dict[str, float]:
    if not target_boxes and not pred_boxes:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "gt_area": 0.0,
            "pred_area": 0.0,
        }
    gt_area = _covered_area(target_boxes)
    pred_area = _covered_area(pred_boxes)
    intersection = _covered_area(target_boxes, pred_boxes)
    union = gt_area + pred_area - intersection
    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = intersection / union if union > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
    }


class RealTextReferenceEvidenceORM(ORM):
    def __init__(self, args: Any = None, **kwargs: Any) -> None:
        super().__init__(args=args, **kwargs)
        self.lambda_format = float(os.environ.get("REALTEXT_GRPO_LAMBDA_FORMAT", "0.05"))
        self.lambda_grounding = float(os.environ.get("REALTEXT_GRPO_LAMBDA_GROUNDING", "0.95"))
        self.cls_weight = float(os.environ.get("REALTEXT_GRPO_GROUNDING_CLS_WEIGHT", "0.25"))
        self.num_weight = float(os.environ.get("REALTEXT_GRPO_GROUNDING_NUM_WEIGHT", "0.75"))
        self.iou_weight = float(os.environ.get("REALTEXT_GRPO_GROUNDING_IOU_WEIGHT", "1.75"))
        self.high_iou_bonus_weight = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_HIGH_IOU_BONUS_WEIGHT", "1.0")
        )
        self.pixel_precision_weight = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_PIXEL_PRECISION_WEIGHT", "1.25")
        )
        self.pixel_recall_weight = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_PIXEL_RECALL_WEIGHT", "0.60")
        )
        self.union_iou_weight = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_UNION_IOU_WEIGHT", "1.75")
        )
        self.match_iou_threshold = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_MATCH_IOU_THRESHOLD", "0.3")
        )
        self.high_iou_bonus_threshold = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_HIGH_IOU_BONUS_THRESHOLD", "0.5")
        )
        if not 0.0 <= self.high_iou_bonus_threshold < 1.0:
            raise ValueError("REALTEXT_GRPO_GROUNDING_HIGH_IOU_BONUS_THRESHOLD must be in [0, 1)")
        self.overbox_penalty_weight = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_OVERBOX_PENALTY_WEIGHT", "0.60")
        )
        self.overbox_ratio_start = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_OVERBOX_RATIO_START", "2.0")
        )
        self.authentic_fp_penalty_weight = float(
            os.environ.get("REALTEXT_GRPO_GROUNDING_AUTHENTIC_FP_PENALTY_WEIGHT", "0.80")
        )
        weight_sum = (
            self.cls_weight
            + self.num_weight
            + self.iou_weight
            + self.high_iou_bonus_weight
            + self.pixel_precision_weight
            + self.pixel_recall_weight
            + self.union_iou_weight
        )
        if weight_sum <= 0:
            raise ValueError("Grounding component weights must sum to a positive value")
        self.component_weight_sum = weight_sum

    def _grounding_score(self, text: str, target_boxes_raw: Any, label: str) -> float:
        target_boxes = _normalize_boxes(_parse_boxes(target_boxes_raw))
        pred_boxes, invalid_boxes = parse_box_candidates(text)
        pred_label = parse_label(text)

        classification = 1.0 if pred_label == label else 0.0
        box_f1, set_iou, _ = one_to_one_grounding_metrics(
            target_boxes, pred_boxes, match_iou_threshold=self.match_iou_threshold
        )
        pixel = _union_pixel_metrics(target_boxes, pred_boxes)
        high_iou_bonus = max(
            0.0,
            (set_iou - self.high_iou_bonus_threshold)
            / (1.0 - self.high_iou_bonus_threshold),
        )
        grounding_quality = (
            self.cls_weight * classification
            + self.num_weight * box_f1
            + self.iou_weight * set_iou
            + self.high_iou_bonus_weight * high_iou_bonus
            + self.pixel_precision_weight * pixel["precision"]
            + self.pixel_recall_weight * pixel["recall"]
            + self.union_iou_weight * pixel["iou"]
        ) / self.component_weight_sum

        invalid_penalty = min(1.0, 0.25 * len(invalid_boxes))
        penalty = invalid_penalty
        if label == "AUTHENTIC" and pred_boxes:
            penalty += self.authentic_fp_penalty_weight
        if label == "FORGED" and target_boxes and pred_boxes:
            area_ratio = pixel["pred_area"] / max(pixel["gt_area"], 1.0)
            if area_ratio > self.overbox_ratio_start:
                penalty += self.overbox_penalty_weight * min(
                    1.0, (area_ratio - self.overbox_ratio_start) / self.overbox_ratio_start
                )
        return max(-1.0, min(1.0, grounding_quality - penalty))

    def __call__(
        self,
        completions: list[Any],
        gt_boxes: Any = None,
        gt_label: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        labels = _as_list(gt_label, len(completions))
        boxes_list = _as_list(gt_boxes, len(completions))
        fmt_scores = format_reward(completions)
        rewards: list[float] = []
        for completion, label, target_boxes, fmt in zip(
            completions, labels, boxes_list, fmt_scores
        ):
            grounding_score = self._grounding_score(
                completion_text(completion), target_boxes, str(label).upper()
            )
            rewards.append(
                self.lambda_format * float(fmt)
                + self.lambda_grounding * float(grounding_score)
            )
        return rewards


orms["realtext_reference_evidence"] = RealTextReferenceEvidenceORM
