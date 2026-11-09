#!/bin/bash

# Train all four PoSafeNet driving variants for seeds 0, 1, 2.
# If one run fails, continue with the remaining runs.
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "$LOG_DIR"

success_count=0
failed_count=0
total_count=0

run_training() {
    local method_name="$1"
    local script_name="$2"
    local seed="$3"

    local script_path="${SCRIPT_DIR}/${script_name}"
    local log_file="${LOG_DIR}/${method_name}_seed_${seed}.log"

    total_count=$((total_count + 1))

    echo "============================================================"
    echo "Starting: ${method_name}, seed=${seed}"
    echo "Script: ${script_path}"
    echo "Log: ${log_file}"
    echo "Start time: $(date)"
    echo "============================================================"

    if [[ ! -f "$script_path" ]]; then
        echo "FAILED: script not found: ${script_path}"
        failed_count=$((failed_count + 1))
        echo "End time: $(date)"
        echo
        return
    fi

    if SEED="$seed" bash "$script_path" >"$log_file" 2>&1; then
        echo "SUCCESS: ${method_name}, seed=${seed}"
        success_count=$((success_count + 1))
    else
        local exit_code=$?
        echo "FAILED: ${method_name}, seed=${seed}, exit_code=${exit_code}"
        echo "Check log: ${log_file}"
        failed_count=$((failed_count + 1))
    fi

    echo "End time: $(date)"
    echo
}

# ============================================================
# posafenet
# ============================================================
for seed in 0 1 2; do
    run_training "posafenet" "train_posafenet.sh" "$seed"
done

# ============================================================
# posafenet_hard
# ============================================================
for seed in 0 1 2; do
    run_training "posafenet_hard" "train_posafenet_hard.sh" "$seed"
done

# ============================================================
# posafenet_big
# ============================================================
for seed in 0 1 2; do
    run_training "posafenet_big" "train_posafenet_big.sh" "$seed"
done

# ============================================================
# posafenet_big_hard
# ============================================================
for seed in 0 1 2; do
    run_training "posafenet_big_hard" "train_posafenet_big_hard.sh" "$seed"
done


echo "============================================================"
echo "All training jobs finished."
echo "Total:   ${total_count}"
echo "Success: ${success_count}"
echo "Failed:  ${failed_count}"
echo "Final time: $(date)"
echo "Logs: ${LOG_DIR}"
echo "============================================================"

# exit non-zero if any run failed
if [[ "$failed_count" -gt 0 ]]; then
    exit 1
fi

exit 0