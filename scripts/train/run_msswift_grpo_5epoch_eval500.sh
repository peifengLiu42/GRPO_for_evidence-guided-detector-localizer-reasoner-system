#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKDIR}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
NUM_RECORDS="${NUM_RECORDS:-3000}"

BASE_MODEL="${BASE_MODEL:-${WORKDIR}/outputs/qwen3vl4b_reference_sft5_rft1_distmix_authdown_shortprompt_merged_bf16}"
SOURCE_DATA="${SOURCE_DATA:-${WORKDIR}/data/realtext_grpo_reference_evidence_train_shortprompt.json}"
DATA="${DATA:-${WORKDIR}/data/realtext_grpo_reference_evidence_train_shortprompt_msswift_ultrahardmix3k_seed42.jsonl}"
DYNAMIC_DATA="${DYNAMIC_DATA:-1}"
DYNAMIC_DATA_DIR="${DYNAMIC_DATA_DIR:-${WORKDIR}/data/dynamic_ultrahardmix3k_eval500_gen8}"
DYNAMIC_SAMPLING_MODE="${DYNAMIC_SAMPLING_MODE:-ultra_hardmix}"
DATA_SEED_BASE="${DATA_SEED_BASE:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/outputs/qwen3vl4b_reference_grpo_msswift_from_sft5_rft1_run2_5epoch_eval500_gen8_ultrahard_lora32}"
EVAL_ROOT="${EVAL_ROOT:-${WORKDIR}/outputs/msswift_grpo_run2_5epoch_eval500_gen8_ultrahard}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/outputs/logs}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/msswift_grpo_run2_5epoch_eval500_gen8_train.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/msswift_grpo_run2_5epoch_eval500_gen8_eval.log}"
ENABLE_CURVE_MONITOR="${ENABLE_CURVE_MONITOR:-1}"
CURVE_LOG="${CURVE_LOG:-${OUTPUT_DIR}/logging.jsonl}"
CURVE_DIR="${CURVE_DIR:-${WORKDIR}/training_curves/$(basename "${OUTPUT_DIR}")}"
CURVE_MONITOR_LOG="${CURVE_MONITOR_LOG:-${LOG_DIR}/msswift_grpo_run2_5epoch_eval500_gen8_curves.log}"
CURVE_EVERY_STEPS="${CURVE_EVERY_STEPS:-10}"
CURVE_POLL_SECONDS="${CURVE_POLL_SECONDS:-30}"
CURVE_ROLLING_WINDOW="${CURVE_ROLLING_WINDOW:-10}"
CURVE_NO_SNAPSHOTS="${CURVE_NO_SNAPSHOTS:-0}"
CURVE_MONITOR_CONDA_ENV="${CURVE_MONITOR_CONDA_ENV:-doc_qwen3vl_msswift}"

mkdir -p "${OUTPUT_DIR}" "${EVAL_ROOT}" "${LOG_DIR}" "${DYNAMIC_DATA_DIR}"

IFS=',' read -r -a train_gpu_array <<< "${TRAIN_GPUS}"
TRAIN_NPROC="${TRAIN_NPROC:-${#train_gpu_array[@]}}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
BETA="${BETA:-0.05}"
DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE:-true}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}"
LOSS_TYPE="${LOSS_TYPE:-grpo}"
SCALE_REWARDS="${SCALE_REWARDS:-group}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GLOBAL_COMPLETION_BATCH=$((TRAIN_NPROC * PER_DEVICE_TRAIN_BATCH_SIZE))
if [[ $((GLOBAL_COMPLETION_BATCH % NUM_GENERATIONS)) -ne 0 ]]; then
  echo "[stage] invalid GRPO batch: TRAIN_NPROC * PER_DEVICE_TRAIN_BATCH_SIZE must be divisible by NUM_GENERATIONS" >&2
  echo "[stage] got TRAIN_NPROC=${TRAIN_NPROC}, PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}, NUM_GENERATIONS=${NUM_GENERATIONS}" >&2
  exit 1
fi
PROMPTS_PER_STEP=$((TRAIN_NPROC * PER_DEVICE_TRAIN_BATCH_SIZE / NUM_GENERATIONS))
if [[ "${PROMPTS_PER_STEP}" -lt 1 ]]; then
  echo "[stage] invalid prompt batch: TRAIN_NPROC=${TRAIN_NPROC}, PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}, NUM_GENERATIONS=${NUM_GENERATIONS}" >&2
  exit 1
fi
STEPS_PER_EPOCH=$(((NUM_RECORDS + PROMPTS_PER_STEP - 1) / PROMPTS_PER_STEP))
TOTAL_STEPS="${TOTAL_STEPS:-$((STEPS_PER_EPOCH * TOTAL_EPOCHS))}"
CURVE_MONITOR_PID=""

start_curve_monitor() {
  if [[ "${ENABLE_CURVE_MONITOR}" != "1" ]]; then
    return
  fi

  mkdir -p "${CURVE_DIR}"
  local snapshot_args=()
  if [[ "${CURVE_NO_SNAPSHOTS}" == "1" ]]; then
    snapshot_args+=(--no-snapshots)
  fi

  echo "[stage] start curve monitor: ${CURVE_DIR}"
  (
    if command -v conda >/dev/null 2>&1; then
      conda run -n "${CURVE_MONITOR_CONDA_ENV}" python \
        scripts/monitor/monitor_grpo_curves.py \
        --log "${CURVE_LOG}" \
        --output-dir "${CURVE_DIR}" \
        --every-steps "${CURVE_EVERY_STEPS}" \
        --poll-seconds "${CURVE_POLL_SECONDS}" \
        --rolling-window "${CURVE_ROLLING_WINDOW}" \
        --max-steps "${TOTAL_STEPS}" \
        "${snapshot_args[@]}"
    else
      python scripts/monitor/monitor_grpo_curves.py \
        --log "${CURVE_LOG}" \
        --output-dir "${CURVE_DIR}" \
        --every-steps "${CURVE_EVERY_STEPS}" \
        --poll-seconds "${CURVE_POLL_SECONDS}" \
        --rolling-window "${CURVE_ROLLING_WINDOW}" \
        --max-steps "${TOTAL_STEPS}" \
        "${snapshot_args[@]}"
    fi
  ) >> "${CURVE_MONITOR_LOG}" 2>&1 &
  CURVE_MONITOR_PID="$!"
}

stop_curve_monitor() {
  if [[ -n "${CURVE_MONITOR_PID}" ]] && kill -0 "${CURVE_MONITOR_PID}" 2>/dev/null; then
    kill "${CURVE_MONITOR_PID}" 2>/dev/null || true
    wait "${CURVE_MONITOR_PID}" 2>/dev/null || true
  fi
}

trap stop_curve_monitor EXIT

latest_checkpoint() {
  find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' \
    | sed -E 's/.*checkpoint-([0-9]+)$/\1 &/' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

stage_index_for_step() {
  local target_step="$1"
  echo $(((target_step - 1) / EVAL_INTERVAL))
}

stage_data_path() {
  local stage_index="$1"
  local seed="$2"
  printf '%s/realtext_grpo_reference_evidence_train_shortprompt_msswift_%s_%s_stage%04d_seed%d.jsonl' \
    "${DYNAMIC_DATA_DIR}" "${DYNAMIC_SAMPLING_MODE}" "${NUM_RECORDS}" "${stage_index}" "${seed}"
}

prepare_stage_data() {
  local target_step="$1"
  if [[ "${DYNAMIC_DATA}" != "1" ]]; then
    printf '%s\n' "${DATA}"
    return
  fi

  local stage_index
  stage_index="$(stage_index_for_step "${target_step}")"
  local seed=$((DATA_SEED_BASE + stage_index))
  local stage_data
  stage_data="$(stage_data_path "${stage_index}" "${seed}")"

  if [[ ! -s "${stage_data}" ]]; then
    echo "[stage] sample data stage=${stage_index} seed=${seed}: ${stage_data}" >&2
    python scripts/data/convert_to_msswift.py \
      --input "${SOURCE_DATA}" \
      --output "${stage_data}" \
      --sampling_mode "${DYNAMIC_SAMPLING_MODE}" \
      --num_records "${NUM_RECORDS}" \
      --seed "${seed}" \
      >&2
  else
    echo "[stage] reuse data stage=${stage_index} seed=${seed}: ${stage_data}" >&2
  fi

  printf '%s\n' "${stage_data}"
}

run_train_until() {
  local target_step="$1"
  local ckpt="${OUTPUT_DIR}/checkpoint-${target_step}"
  if [[ -d "${ckpt}" ]]; then
    echo "[stage] checkpoint exists, skip training: ${ckpt}"
    return
  fi

  local resume
  resume="$(latest_checkpoint || true)"
  local train_data
  train_data="$(prepare_stage_data "${target_step}")"
  local ignore_data_skip="false"
  if [[ "${DYNAMIC_DATA}" == "1" ]]; then
    ignore_data_skip="true"
  fi
  echo "[stage] train until step ${target_step}; resume=${resume:-<none>}; data=${train_data}"
  env \
    GPUS="${TRAIN_GPUS}" \
    NPROC_PER_NODE="${TRAIN_NPROC}" \
    NUM_GENERATIONS="${NUM_GENERATIONS}" \
    NUM_TRAIN_EPOCHS="${TOTAL_EPOCHS}" \
    MAX_STEPS="${target_step}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    BETA="${BETA}" \
    DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE}" \
    MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES}" \
    LOSS_TYPE="${LOSS_TYPE}" \
    SCALE_REWARDS="${SCALE_REWARDS}" \
    TEMPERATURE="${TEMPERATURE}" \
    TOP_P="${TOP_P}" \
    TOP_K="${TOP_K}" \
    REPETITION_PENALTY="${REPETITION_PENALTY}" \
    SAVE_STEPS="${EVAL_INTERVAL}" \
    MAX_COMPLETION_LENGTH=2048 \
    MAX_LENGTH=4096 \
    DATA="${train_data}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    RESUME_FROM_CHECKPOINT="${resume}" \
    IGNORE_DATA_SKIP="${ignore_data_skip}" \
    ADD_VERSION=0 \
    bash scripts/train/run_msswift_grpo_probe.sh \
    2>&1 | tee -a "${TRAIN_LOG}"

  if [[ ! -d "${ckpt}" ]]; then
    echo "[stage] expected checkpoint was not created: ${ckpt}" >&2
    exit 1
  fi
}

run_eval_for() {
  local step="$1"
  local ckpt="${OUTPUT_DIR}/checkpoint-${step}"
  local eval_dir="${EVAL_ROOT}/checkpoint-${step}"
  local merged="${eval_dir}/merged_bf16"
  local pred="${eval_dir}/realtext_indomain_vllm.jsonl"
  local metrics="${eval_dir}/image_pixel_metrics.json"
  local masks="${eval_dir}/masks"
  local details="${eval_dir}/details.jsonl"

  if [[ -s "${metrics}" ]]; then
    echo "[stage] metrics exists, skip eval: ${metrics}"
    return
  fi
  mkdir -p "${eval_dir}"

  if [[ ! -d "${merged}" ]]; then
    echo "[stage] merge checkpoint-${step}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS%%,*}" conda run -n doc_qwen3vl_msswift python \
      scripts/train/merge_reference_model.py \
      --base_model "${BASE_MODEL}" \
      --adapter "${ckpt}" \
      --output_dir "${merged}" \
      --device 0 \
      2>&1 | tee -a "${EVAL_LOG}"
  fi

  echo "[stage] vLLM eval checkpoint-${step}"
  IFS=',' read -r -a eval_gpu_array <<< "${EVAL_GPUS}"
  local num_shards="${#eval_gpu_array[@]}"
  local parts=()
  local pids=()
  for shard_index in "${!eval_gpu_array[@]}"; do
    local gpu="${eval_gpu_array[${shard_index}]}"
    local part="${eval_dir}/realtext_indomain_vllm.part${shard_index}.jsonl"
    parts+=("${part}")
    (
      CUDA_VISIBLE_DEVICES="${gpu}" VLLM_USE_FLASHINFER_SAMPLER=0 conda run -n doc_livr_vllm python \
        scripts/eval/infer_vllm.py \
        --model_name_or_path "${merged}" \
        --merged_model \
        --output_jsonl "${part}" \
        --batch_size 16 \
        --resize 1280 \
        --max_new_tokens 2048 \
        --max_model_len 8192 \
        --max_pixels 1048576 \
        --gpu_memory_utilization 0.85 \
        --data_parallel_size 1 \
        --num_shards "${num_shards}" \
        --shard_index "${shard_index}" \
        --overwrite
    ) 2>&1 | tee -a "${EVAL_LOG}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  conda run -n doc_livr_vllm python \
    scripts/eval/merge_vllm_shards.py \
    --evidence_jsonl data/realtext_indomain_reference_evidence.jsonl \
    --output_jsonl "${pred}" \
    "${parts[@]}" \
    2>&1 | tee -a "${EVAL_LOG}"

  conda run -n doc_livr_vllm python \
    scripts/eval/evaluate_image_pixel_metrics.py \
    "${pred}" \
    --output_json "${metrics}" \
    --mask_dir "${masks}" \
    --details_jsonl "${details}" \
    2>&1 | tee -a "${EVAL_LOG}"
}

target="${EVAL_INTERVAL}"
start_curve_monitor
while [[ "${target}" -le "${TOTAL_STEPS}" ]]; do
  run_train_until "${target}"
  run_eval_for "${target}"
  target=$((target + EVAL_INTERVAL))
done

if [[ $((TOTAL_STEPS % EVAL_INTERVAL)) -ne 0 ]]; then
  run_train_until "${TOTAL_STEPS}"
  run_eval_for "${TOTAL_STEPS}"
fi

echo "[stage] done: ${OUTPUT_DIR}"
