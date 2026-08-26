"""Prompt builders shared by reference-evidence inference and future GRPO data code."""

from __future__ import annotations


PROMPT_MODES = ("legacy", "reference_evidence")

SYSTEM_PROMPT = (
    "You are an image forensics assistant. Output only this report format:\n\n"
    "I. Overall Assessment\n"
    "[Conclusion]: FORGED or AUTHENTIC\n"
    "[RISK_SCORE]: 0-100 manipulation likelihood\n\n"
    "II. Detailed Anomaly Analysis\n"
    "For each anomaly:\n"
    "### ANOMALY_001\n"
    "[GROUNDING]: [xmin,ymin,xmax,ymax] normalized 0-999\n"
    "[REASON]: visual or semantic evidence\n"
    "If none, state no anomalies detected.\n\n"
    "III. Summary\n"
    "Brief synthesis."
)


def format_bboxes(bboxes: list[list[int]]) -> str:
    """Format normalized 0-999 boxes without changing their coordinates."""
    return ", ".join(f"[{x1},{y1},{x2},{y2}]" for x1, y1, x2, y2 in bboxes)


def build_legacy_prompt(bboxes: list[list[int]], final_forged: bool) -> str:
    """Reproduce the prompt used by the current hard-gated fusion pipeline."""
    num_bboxes = len(bboxes)
    if not final_forged:
        return (
            f"<image>Expert forgery detector has analyzed this image and detected {num_bboxes} bbox(s), "
            f"indicating it is an authentic image. Please verify this assessment and provide a "
            f"detailed analysis report strictly following the required forensic format."
        )

    return (
        f"<image>Expert forgery detector has identified {num_bboxes} potential tampered region(s) at: "
        f"{format_bboxes(bboxes)}. Please analyze these specific areas in detail, explain the visual "
        f"artifacts and logical contradictions, and provide a comprehensive forgery analysis report "
        f"strictly following the required forensic format."
    )


def build_reference_evidence_prompt(
    detector_label: str,
    prob_forged: float,
    bboxes: list[list[int]],
) -> str:
    """Build the non-gating prompt in which predictions are fallible references."""
    boxes_text = format_bboxes(bboxes) if bboxes else "none"
    return (
        f"<image>Detector: {detector_label}, prob_forged={prob_forged:.6f}. "
        f"Localizer boxes ({len(bboxes)}): {boxes_text}.\n"
        "Predictions may be wrong. Verify with the image; keep, fix, add, or reject boxes "
        "as evidence supports. Produce the required forensic report."
    )


def build_user_prompt(
    prompt_mode: str,
    detector_label: str,
    prob_forged: float,
    bboxes: list[list[int]],
    legacy_final_forged: bool,
) -> str:
    if prompt_mode == "legacy":
        return build_legacy_prompt(bboxes=bboxes, final_forged=legacy_final_forged)
    if prompt_mode == "reference_evidence":
        return build_reference_evidence_prompt(
            detector_label=detector_label,
            prob_forged=prob_forged,
            bboxes=bboxes,
        )
    raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}; expected one of {PROMPT_MODES}")
