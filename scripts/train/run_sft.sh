#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-llama-factory}"
GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29651}"
CONFIG="${CONFIG:-configs/qwen3vl4b_reference_sft_from_scratch_distmix_authdown_shortprompt.yaml}"
LLAMA_FACTORY_CLI="${LLAMA_FACTORY_CLI:-llamafactory-cli}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing SFT config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.json ]]; then
  echo "Generate the from-scratch SFT subset first:" >&2
  echo "  python scripts/data/generate_from_scratch_sft_data.py --output data/realtext_reference_sft_from_scratch_distmix_authdown_shortprompt.json" >&2
  exit 1
fi

"${LLAMA_FACTORY_CLI}" train "${CONFIG}"
