#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALVATRON_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${GALVATRON_DIR}/.." && pwd)"
GPT_DIR="${GALVATRON_DIR}/models/gpt"
INVOKE_DIR="$(pwd)"

cd "${GPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f "${SCRIPT_DIR}/deterministic_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/deterministic_env.sh"
fi

export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MODE="${MODE:-test}"
VARIANT="${ALIGN_VARIANT:-llama}"
ALIGN_CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    --config)
      ALIGN_CONFIG="${2:-}"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done
if [[ "${MODE}" != "baseline" && "${MODE}" != "test" ]]; then
  echo "[ERROR] MODE must be 'baseline' or 'test', got: ${MODE}"
  exit 1
fi
if [[ "${VARIANT}" != "llama" && "${VARIANT}" != "moe" ]]; then
  echo "[ERROR] VARIANT must be 'llama' or 'moe', got: ${VARIANT}"
  exit 1
fi

VARIANT_DIR="${SCRIPT_DIR}/${VARIANT}"
if [[ -z "${ALIGN_CONFIG}" ]]; then
  ALIGN_CONFIG="${VARIANT_DIR}/train_dist.yaml"
fi

if [[ "${ALIGN_CONFIG}" != /* ]]; then
  if [[ -f "${INVOKE_DIR}/${ALIGN_CONFIG}" ]]; then
    ALIGN_CONFIG="${INVOKE_DIR}/${ALIGN_CONFIG}"
  elif [[ -f "${GALVATRON_DIR}/${ALIGN_CONFIG}" ]]; then
    ALIGN_CONFIG="${GALVATRON_DIR}/${ALIGN_CONFIG}"
  elif [[ -f "${SCRIPT_DIR}/${ALIGN_CONFIG}" ]]; then
    ALIGN_CONFIG="${SCRIPT_DIR}/${ALIGN_CONFIG}"
  elif [[ -f "${VARIANT_DIR}/${ALIGN_CONFIG}" ]]; then
    ALIGN_CONFIG="${VARIANT_DIR}/${ALIGN_CONFIG}"
  fi
fi

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-0.0.0.0}"
MASTER_PORT="${MASTER_PORT:-12345}"
CONFIG_SUFFIX=""
if [[ "$(basename "${ALIGN_CONFIG}")" == *hf* ]]; then
  CONFIG_SUFFIX="_hf"
  if [[ "${VARIANT}" == "moe" ]]; then
    echo "[ERROR] MoE + HuggingFace config is not available yet."
    echo "  Use: bash scripts/accuracy_alignment/record_loss.sh --variant moe --mode ${MODE}"
    echo "  HF placeholder: ${VARIANT_DIR}/train_dist_hf.yaml (to be added)"
    exit 1
  fi
fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${MODE}" == "baseline" ]]; then
  CURVE_STORE="${CURVE_STORE:-${VARIANT_DIR}/${VARIANT}_loss_baseline${CONFIG_SUFFIX}.csv}"
else
  CURVE_STORE="${CURVE_STORE:-${VARIANT_DIR}/${VARIANT}_loss_test${CONFIG_SUFFIX}.csv}"
fi
WANDB_PROJECT="${WANDB_PROJECT:-galvatron}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${VARIANT}_${MODE}${CONFIG_SUFFIX}_${RUN_TAG}}"
export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT

if [[ ! -f "${ALIGN_CONFIG}" ]]; then
  echo "[ERROR] Align config not found: ${ALIGN_CONFIG}"
  exit 1
fi

TMP_DIR="$(mktemp -d "/tmp/accuracy_alignment_${VARIANT}_${MODE}_XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT
mkdir -p "$(dirname "${CURVE_STORE}")"
LOG_PATH="${TMP_DIR}/train.log"

if [[ "${NNODES}" == "1" ]]; then
  additional_args="--standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  ${additional_args} \
  train_dist.py "${ALIGN_CONFIG}" "$@" 2>&1 | tee "${LOG_PATH}"

python "${SCRIPT_DIR}/upload_curves_to_wandb.py" \
  --mode "${MODE}" \
  --project "${WANDB_PROJECT}" \
  --entity "${WANDB_ENTITY}" \
  --run-name "${WANDB_RUN_NAME}" \
  --log "${LOG_PATH}" \
  --curve-csv "${CURVE_STORE}"

echo "Loss recording completed."
echo "  variant: ${VARIANT}"
echo "  mode: ${MODE}"
echo "  config: ${ALIGN_CONFIG}"
echo "  stored curve: ${CURVE_STORE}"
