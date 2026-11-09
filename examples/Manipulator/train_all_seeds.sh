#!/usr/bin/env bash
set -uo pipefail

# Train PoSafeNet-Hard and PoSafeNet-Mixture for seeds 0..5 sequentially
# (the 6 seeds used by eval_posafenet.py). Final test evaluation is skipped;
# validation still runs after every epoch. A failed run is logged and the
# script continues with the next run.

cd "$(dirname "${BASH_SOURCE[0]}")"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_posafenet.py}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

MODELS=(
  "posafenet_hard"
  "posafenet_mixture"
)

SEEDS=(0 1 2 3 4 5)

mkdir -p batch_logs

TOTAL=$((${#MODELS[@]} * ${#SEEDS[@]}))
RUN_INDEX=0
FAILED=0

echo "============================================================"
echo "Batch training"
echo "Train script: ${TRAIN_SCRIPT}"
echo "GPU: ${GPU_ID}"
echo "Models: ${#MODELS[@]}"
echo "Seeds: ${#SEEDS[@]}"
echo "Total runs: ${TOTAL}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Final test evaluation: skipped"
echo "Start: $(date)"
echo "============================================================"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    LOG_FILE="batch_logs/${model}_seed_${seed}.log"

    echo
    echo "============================================================"
    echo "[${RUN_INDEX}/${TOTAL}] model=${model}, seed=${seed}"
    echo "Log: ${LOG_FILE}"
    echo "Start: $(date)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      python "${TRAIN_SCRIPT}" \
        --model "${model}" \
        --seed "${seed}" \
        --exp "${seed}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --skip-test \
        >"${LOG_FILE}" 2>&1

    STATUS=$?

    if [[ ${STATUS} -eq 0 ]]; then
      echo "[DONE] model=${model}, seed=${seed}"
    else
      FAILED=$((FAILED + 1))
      echo "[FAILED] model=${model}, seed=${seed}, exit=${STATUS}"
      echo "See ${LOG_FILE}"
    fi

    echo "End: $(date)"
  done
done

echo
echo "============================================================"
echo "Batch training finished"
echo "Successful runs: $((TOTAL - FAILED))/${TOTAL}"
echo "Failed runs: ${FAILED}"
echo "End: $(date)"
echo "============================================================"

if [[ ${FAILED} -ne 0 ]]; then
  exit 1
fi