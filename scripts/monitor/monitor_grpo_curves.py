#!/usr/bin/env python3
"""Continuously parse GRPO logs and refresh training-curve artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = Path(os.environ.get("GRPO_TRAIN_LOG", REPO_ROOT / "outputs/grpo_train.log"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("GRPO_CURVE_DIR", REPO_ROOT / "training_curves/grpo"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="Training log path. Supports old Trainer text logs and ms-swift logging.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--every-steps", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render the latest available metrics once and exit.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Only atomically update the latest PNG instead of keeping step snapshots.",
    )
    args = parser.parse_args()
    if args.every_steps <= 0 or args.poll_seconds <= 0 or args.rolling_window <= 0:
        parser.error("--every-steps, --poll-seconds, and --rolling-window must be positive")
    return args


def parse_step_and_max_steps(value: Any) -> tuple[int | None, int | None]:
    if isinstance(value, str) and "/" in value:
        left, right = value.split("/", 1)
        try:
            step = int(left.strip())
        except ValueError:
            step = None
        try:
            max_steps = int(right.strip())
        except ValueError:
            max_steps = None
        return step, max_steps
    return None, None


def parse_line_record(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None

    if stripped.startswith("{"):
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            record = None
        if isinstance(record, dict):
            return record

    start = stripped.find("{'loss':")
    if start < 0:
        return None
    try:
        record = ast.literal_eval(stripped[start:])
    except (SyntaxError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def parse_records(log_path: Path) -> list[dict[str, Any]]:
    """Extract flat metric dictionaries from text logs or ms-swift logging.jsonl."""
    if not log_path.is_file():
        return []

    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = parse_line_record(line.replace("\r", "\n"))
            if not isinstance(record, dict) or "reward" not in record:
                continue

            step, max_steps = parse_step_and_max_steps(record.get("global_step/max_steps"))
            clean: dict[str, Any] = {"step": float(step or (len(records) + 1))}
            if max_steps is not None:
                clean["max_steps"] = float(max_steps)

            for key, value in record.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    clean[str(key)] = float(value)
            records.append(clean)

    # Resumed ms-swift runs may append repeated steps after a restart. Keep the latest
    # metric row for each global step so the curve reflects the current training trace.
    by_step: dict[int, dict[str, Any]] = {}
    for record in records:
        step = int(record["step"])
        by_step[step] = record
    return [by_step[step] for step in sorted(by_step)]


def values(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([record.get(key, np.nan) for record in records], dtype=float)


def rolling_mean(array: np.ndarray, window: int) -> np.ndarray:
    result = np.full(array.shape, np.nan, dtype=float)
    for index in range(len(array)):
        chunk = array[max(0, index - window + 1) : index + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            result[index] = float(finite.mean())
    return result


def configure_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlabel("Training step")


def plot_line(
    axis: plt.Axes,
    steps: np.ndarray,
    data: np.ndarray,
    label: str,
    color: str,
    *,
    alpha: float = 1.0,
    linewidth: float = 1.7,
) -> None:
    if np.isfinite(data).any():
        axis.plot(steps, data, label=label, color=color, alpha=alpha, linewidth=linewidth)


def combined_legend(axis: plt.Axes, second: plt.Axes) -> None:
    lines = [
        line
        for line in axis.get_lines() + second.get_lines()
        if not line.get_label().startswith("_")
    ]
    axis.legend(lines, [line.get_label() for line in lines], fontsize=8, frameon=False)


def render_dashboard(
    records: list[dict[str, Any]],
    output_path: Path,
    rolling_window: int,
    max_steps: int,
) -> None:
    steps = values(records, "step")
    reward = values(records, "reward")
    reward_std = values(records, "reward_std")
    loss = values(records, "loss")
    kl = values(records, "kl")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    fig.patch.set_facecolor("#fffdf8")
    for axis in axes.flat:
        axis.set_facecolor("#fffdf8")
        configure_axis(axis)

    # 1. Overall reward and its local trend/dispersion.
    axis = axes[0, 0]
    plot_line(axis, steps, reward, "Reward", "#4c78d0", alpha=0.38, linewidth=1.2)
    plot_line(
        axis,
        steps,
        rolling_mean(reward, rolling_window),
        f"Reward MA({rolling_window})",
        "#174ea6",
        linewidth=2.4,
    )
    if np.isfinite(reward_std).any():
        lower = reward - reward_std
        upper = reward + reward_std
        axis.fill_between(steps, lower, upper, color="#4c78d0", alpha=0.10, label="± reward std")
    axis.set_title("1. Reward / Score", loc="left", fontweight="bold")
    axis.set_ylabel("Reward")
    axis.legend(fontsize=8, frameon=False)

    # 2. Reward decomposition (the task-grounding signal is the key task proxy).
    axis = axes[0, 1]
    for key, label, color in (
        ("rewards/RealTextReferenceEvidenceORM/mean", "RealText reward", "#4e79a7"),
        ("rewards/format_reward/mean", "Format", "#59a14f"),
        ("rewards/grounding_reward/mean", "Grounding", "#f28e2b"),
        ("rewards/explanation_reward/mean", "Explanation", "#af7aa1"),
    ):
        plot_line(axis, steps, values(records, key), label, color)
    axis.set_title("2. Reward Components", loc="left", fontweight="bold")
    axis.set_ylabel("Component reward")
    axis.legend(fontsize=8, frameon=False)

    # 3. KL divergence from the frozen SFT/RFT reference.
    axis = axes[0, 2]
    plot_line(axis, steps, kl, "KL", "#8e5cc7", alpha=0.45, linewidth=1.2)
    plot_line(
        axis,
        steps,
        rolling_mean(kl, rolling_window),
        f"KL MA({rolling_window})",
        "#5b2c98",
        linewidth=2.3,
    )
    axis.set_title("3. KL Divergence", loc="left", fontweight="bold")
    axis.set_ylabel("KL")
    axis.legend(fontsize=8, frameon=False)

    # 4. Optimizer behavior. Symlog preserves ordinary values and visible spikes.
    axis = axes[1, 0]
    plot_line(axis, steps, loss, "GRPO loss", "#d62728", linewidth=1.8)
    axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    axis.set_ylabel("Loss")
    second = axis.twinx()
    grad_norm = values(records, "grad_norm")
    plot_line(second, steps, grad_norm, "Grad norm", "#ff9d00", alpha=0.65, linewidth=1.4)
    second.set_yscale("symlog", linthresh=1.0)
    second.set_ylabel("Gradient norm (symlog)")
    axis.set_title("4. Loss & Gradient Norm", loc="left", fontweight="bold")
    combined_legend(axis, second)

    # 5. Generation length and truncation frequency.
    axis = axes[1, 1]
    mean_length = values(records, "completions/mean_length")
    plot_line(axis, steps, mean_length, "Mean completion length", "#00798c", linewidth=1.8)
    axis.set_ylabel("Tokens")
    second = axis.twinx()
    clipped = values(records, "completions/clipped_ratio")
    plot_line(second, steps, clipped, "Clipped ratio", "#e45756", linewidth=1.6)
    second.set_ylim(bottom=0.0)
    second.set_ylabel("Clipped ratio")
    axis.set_title("5. Generation Length & Truncation", loc="left", fontweight="bold")
    combined_legend(axis, second)

    # 6. Exploration/degeneracy indicators plus the LR schedule.
    axis = axes[1, 2]
    plot_line(axis, steps, reward_std, "Reward std", "#2ca02c", linewidth=1.8)
    plot_line(
        axis,
        steps,
        values(records, "frac_reward_zero_std"),
        "Zero-std group fraction",
        "#7f7f7f",
        linewidth=1.5,
    )
    plot_line(
        axis,
        steps,
        values(records, "clip_ratio/region_mean"),
        "Policy clip ratio",
        "#bc5090",
        linewidth=1.5,
    )
    axis.set_ylabel("Ratio / dispersion")
    second = axis.twinx()
    learning_rate = values(records, "learning_rate")
    plot_line(second, steps, learning_rate, "Learning rate", "#003f5c", linewidth=1.6)
    second.set_ylabel("Learning rate")
    axis.set_title("6. Exploration & Policy Update", loc="left", fontweight="bold")
    combined_legend(axis, second)

    latest = records[-1]
    summary = (
        f"Step {int(latest['step'])}/{max_steps}   "
        f"reward={latest.get('reward', float('nan')):.4f}   "
        f"KL={latest.get('kl', float('nan')):.4f}   "
        f"loss={latest.get('loss', float('nan')):.4f}   "
        f"reward_std={latest.get('reward_std', float('nan')):.4f}"
    )
    fig.suptitle(
        "GRPO Training Convergence Dashboard\n" + summary,
        fontsize=18,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    fig.savefig(temporary, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    os.replace(temporary, output_path)


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fields = ["step"] + sorted({key for record in records for key in record if key != "step"})
    temporary = output_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, output_path)


def read_rendered_step(state_path: Path) -> int:
    try:
        return int(json.loads(state_path.read_text(encoding="utf-8"))["rendered_step"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def write_state(
    state_path: Path,
    log_path: Path,
    latest_step: int,
    rendered_step: int,
    every_steps: int,
) -> None:
    state = {
        "source_log": str(log_path.resolve()),
        "latest_parsed_step": latest_step,
        "rendered_step": rendered_step,
        "update_every_steps": every_steps,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = state_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, state_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = args.output_dir / "history"
    latest_png = args.output_dir / "grpo_training_curves_latest.png"
    csv_path = args.output_dir / "training_metrics.csv"
    state_path = args.output_dir / "monitor_state.json"
    rendered_step = read_rendered_step(state_path)

    while True:
        records = parse_records(args.log)
        latest_step = len(records)
        if latest_step:
            if args.once:
                target_step = latest_step
            elif rendered_step == 0:
                # Give a newly launched monitor an immediate, useful first image.
                target_step = latest_step
            else:
                target_step = (latest_step // args.every_steps) * args.every_steps

            if target_step > rendered_step:
                selected = records[:target_step]
                render_dashboard(selected, latest_png, args.rolling_window, args.max_steps)
                write_csv(records, csv_path)
                if not args.no_snapshots:
                    history_dir.mkdir(parents=True, exist_ok=True)
                    snapshot = history_dir / f"grpo_training_curves_step_{target_step:06d}.png"
                    shutil.copy2(latest_png, snapshot)
                rendered_step = target_step
                write_state(
                    state_path,
                    args.log,
                    latest_step,
                    rendered_step,
                    args.every_steps,
                )
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"rendered step {rendered_step}: {latest_png}",
                    flush=True,
                )

            if latest_step >= args.max_steps:
                if rendered_step < latest_step:
                    render_dashboard(records, latest_png, args.rolling_window, args.max_steps)
                    write_csv(records, csv_path)
                    write_state(state_path, args.log, latest_step, latest_step, args.every_steps)
                print(f"Reached max step {args.max_steps}; monitor exiting.", flush=True)
                return

        if args.once:
            if not records:
                raise SystemExit(f"No GRPO metric records found in {args.log}")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
