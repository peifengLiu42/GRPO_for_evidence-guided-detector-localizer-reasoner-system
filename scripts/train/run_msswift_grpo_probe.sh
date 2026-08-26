#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV:-doc_qwen3vl_msswift}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29671}"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/outputs/qwen3vl4b_reference_sft5_rft1_distmix_authdown_shortprompt_merged_bf16}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
SOURCE_DATA="${SOURCE_DATA:-${PROJECT_ROOT}/data/realtext_grpo_reference_evidence_train_shortprompt.json}"
DATA="${DATA:-${PROJECT_ROOT}/data/realtext_grpo_reference_evidence_train_shortprompt_msswift_ultrahardmix3k_seed42.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen3vl4b_reference_grpo_msswift_probe_steps500_gen4_lora32}"
PLUGIN="${PLUGIN:-${PROJECT_ROOT}/src/realtext_grpo/msswift_reward_plugin.py}"

MAX_STEPS="${MAX_STEPS:-500}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
BETA="${BETA:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
NUM_RECORDS="${NUM_RECORDS:-3000}"
FORGED_FRACTION="${FORGED_FRACTION:-0.5}"
SEED="${SEED:-42}"
USE_VLLM="${USE_VLLM:-false}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ADD_VERSION="${ADD_VERSION:-0}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-false}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-false}"
DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE:-false}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}"
LOSS_TYPE="${LOSS_TYPE:-grpo}"
SCALE_REWARDS="${SCALE_REWARDS:-group}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SWIFT_SINGLE_DEVICE_MODE="${SWIFT_SINGLE_DEVICE_MODE:-1}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
export MASTER_PORT="${MASTER_PORT}"
export SWIFT_SINGLE_DEVICE_MODE="${SWIFT_SINGLE_DEVICE_MODE}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export REALTEXT_GRPO_LAMBDA_FORMAT="${REALTEXT_GRPO_LAMBDA_FORMAT:-0.05}"
export REALTEXT_GRPO_LAMBDA_GROUNDING="${REALTEXT_GRPO_LAMBDA_GROUNDING:-0.95}"
export REALTEXT_GRPO_GROUNDING_MATCH_IOU_THRESHOLD="${REALTEXT_GRPO_GROUNDING_MATCH_IOU_THRESHOLD:-0.3}"
export REALTEXT_GRPO_GROUNDING_CLS_WEIGHT="${REALTEXT_GRPO_GROUNDING_CLS_WEIGHT:-0.25}"
export REALTEXT_GRPO_GROUNDING_NUM_WEIGHT="${REALTEXT_GRPO_GROUNDING_NUM_WEIGHT:-0.75}"
export REALTEXT_GRPO_GROUNDING_IOU_WEIGHT="${REALTEXT_GRPO_GROUNDING_IOU_WEIGHT:-1.75}"
export REALTEXT_GRPO_GROUNDING_HIGH_IOU_BONUS_WEIGHT="${REALTEXT_GRPO_GROUNDING_HIGH_IOU_BONUS_WEIGHT:-1.0}"
export REALTEXT_GRPO_GROUNDING_HIGH_IOU_BONUS_THRESHOLD="${REALTEXT_GRPO_GROUNDING_HIGH_IOU_BONUS_THRESHOLD:-0.5}"
export REALTEXT_GRPO_GROUNDING_PIXEL_PRECISION_WEIGHT="${REALTEXT_GRPO_GROUNDING_PIXEL_PRECISION_WEIGHT:-1.25}"
export REALTEXT_GRPO_GROUNDING_PIXEL_RECALL_WEIGHT="${REALTEXT_GRPO_GROUNDING_PIXEL_RECALL_WEIGHT:-0.60}"
export REALTEXT_GRPO_GROUNDING_UNION_IOU_WEIGHT="${REALTEXT_GRPO_GROUNDING_UNION_IOU_WEIGHT:-1.75}"
export REALTEXT_GRPO_GROUNDING_OVERBOX_PENALTY_WEIGHT="${REALTEXT_GRPO_GROUNDING_OVERBOX_PENALTY_WEIGHT:-0.60}"
export REALTEXT_GRPO_GROUNDING_OVERBOX_RATIO_START="${REALTEXT_GRPO_GROUNDING_OVERBOX_RATIO_START:-2.0}"
export REALTEXT_GRPO_GROUNDING_AUTHENTIC_FP_PENALTY_WEIGHT="${REALTEXT_GRPO_GROUNDING_AUTHENTIC_FP_PENALTY_WEIGHT:-0.80}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

if [[ ! -s "${DATA}" ]]; then
  python scripts/data/convert_to_msswift.py \
    --input "${SOURCE_DATA}" \
    --output "${DATA}" \
    --sampling_mode ultra_hardmix \
    --num_records "${NUM_RECORDS}" \
    --forged_fraction "${FORGED_FRACTION}" \
    --seed "${SEED}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  DATA_PATH="${DATA}" python - <<'PY'
import json
import os
from pathlib import Path
data = Path(os.environ["DATA_PATH"])
first = json.loads(data.read_text(encoding="utf-8").splitlines()[0])
print(json.dumps({
    "messages_roles": [m["role"] for m in first["messages"]],
    "images": first["images"],
    "gt_label": first["gt_label"],
    "num_gt_boxes": len(first["gt_boxes"]),
    "difficulty_bucket": first["difficulty_bucket"],
}, ensure_ascii=False, indent=2))
from realtext_grpo.msswift_reward_plugin import RealTextReferenceEvidenceORM
reward = RealTextReferenceEvidenceORM()
print("plugin_reward_smoke:", reward([first["reference_answer"]], gt_boxes=[first["gt_boxes"]], gt_label=[first["gt_label"]]))
PY
  swift rlhf --help | sed -n '1,80p'
  exit 0
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${BASE_MODEL}" \
  --model_type "${MODEL_TYPE}" \
  --dataset "${DATA}" \
  --external_plugins "${PLUGIN}" \
  --reward_funcs realtext_reference_evidence \
  --tuner_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --target_modules all-linear \
  --no_freeze_vit \
  --no_freeze_aligner \
  --torch_dtype bfloat16 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_generations "${NUM_GENERATIONS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --loss_type "${LOSS_TYPE}" \
  --scale_rewards "${SCALE_REWARDS}" \
  --beta "${BETA}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --max_length "${MAX_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --save_steps "${SAVE_STEPS}" \
  --save_strategy steps \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR}" \
  --gradient_checkpointing true \
  --use_vllm "${USE_VLLM}" \
  --dynamic_sample "${DYNAMIC_SAMPLE}" \
  --max_resample_times "${MAX_RESAMPLE_TIMES}" \
  --log_completions "${LOG_COMPLETIONS}" \
  --ignore_data_skip "${IGNORE_DATA_SKIP}" \
  --seed "${SEED}" \
  $(if [[ "${ADD_VERSION}" == "1" ]]; then printf '%s' "--add_version"; else printf '%s' "--no_add_version"; fi) \
  $(if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then printf '%q %q' "--resume_from_checkpoint" "${RESUME_FROM_CHECKPOINT}"; fi)
