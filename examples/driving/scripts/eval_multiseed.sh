#!/bin/bash
set -euo pipefail

export PYTHONNOUSERSITE=True
export PYOPENGL_PLATFORM=egl

# ============================================================
# Closed-loop VISTA evaluation of the PoSafeNet driving models
# (4 variants x 3 training seeds, 10 episodes each, fixed scenario).
#
# Environment variables:
#   VISTA_DATA_ROOT : folder containing traces/ and carpack01/   (default: $HOME/data/vista)
#   RESULT_ROOT     : checkpoint root  (default: <driving>/results, as written by scripts/train_*.sh)
#   OUTPUT_ROOT     : per-run eval outputs (default: <driving>/eval_out/multiseed)
#   PKL_ROOT        : collected result pickles (default: <driving>/eval_out/pkl_multiseed)
#   EXPERIMENTS_FILTER : optional regex to run only matching experiments, e.g. '^posafenet$'
#   SEEDS_OVERRIDE  : optional space-separated seeds, e.g. "0"
#   N_EPISODES      : episodes per checkpoint (default: 10, as used for the paper)
# ============================================================
DRIVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DRIVING_DIR"

DATA_ROOT="${VISTA_DATA_ROOT:-$HOME/data/vista}"
TRACE_ROOT="$DATA_ROOT/traces"
MESH_DIR="$DATA_ROOT/carpack01"

RESULT_ROOT="${RESULT_ROOT:-./results}"
SCENARIO_NPZ=./scenarios/fixed_scenario_seed0.npz
OUTPUT_ROOT="${OUTPUT_ROOT:-./eval_out/multiseed}"
PKL_ROOT="${PKL_ROOT:-./eval_out/pkl_multiseed}"
N_EPISODES="${N_EPISODES:-10}"

TRACE_PATH="$TRACE_ROOT/20220113-131922_lexus_devens_outerloop_reverse"

CKPT_NAME="${CKPT_NAME:-epoch=0-step=1948.ckpt}"  # checkpoint evaluated in the paper (first half-epoch)

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$PKL_ROOT"

# ============================================================
# Experiment definitions
# ============================================================
# EXP_PREFIX:MODEL_MODULE:CKPT_PREFIX
EXPERIMENTS=(
    "posafenet:posafenet:posafenet"
    "posafenet_hard:posafenet_hard:posafenet_hard"
    "posafenet_big:posafenet_big:posafenet_big"
    "posafenet_big_hard:posafenet_big_hard:posafenet_big_hard"
)

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
    read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
    SEEDS=(0 1 2)
fi

# ============================================================
# Run benchmark
# ============================================================
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r EXP_PREFIX MODEL_MODULE CKPT_PREFIX <<< "$experiment"
    if [[ -n "${EXPERIMENTS_FILTER:-}" && ! "$EXP_PREFIX" =~ $EXPERIMENTS_FILTER ]]; then
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="${EXP_PREFIX}_seed_${SEED}"

        # checkpoint folder of this experiment/seed
        CKPT_EXP_NAME="${CKPT_PREFIX}_seed_${SEED}"
        EXP_DIR="$RESULT_ROOT/$CKPT_EXP_NAME"

        OUT_DIR="$OUTPUT_ROOT/$EXP_NAME"
        FINAL_PKL="$PKL_ROOT/results_${EXP_NAME}.pkl"

        echo
        echo "============================================================"
        echo "Benchmarking: $EXP_NAME"
        echo "Model module: $MODEL_MODULE"
        echo "Experiment dir: $EXP_DIR"
        echo "Output dir: $OUT_DIR"
        echo "Final pkl: $FINAL_PKL"
        echo "============================================================"

        if [[ ! -d "$EXP_DIR" ]]; then
            echo "ERROR: Experiment directory not found:"
            echo "  $EXP_DIR"
            exit 1
        fi

        mapfile -t CKPT_CANDIDATES < <(
            find "$EXP_DIR" \
                -type f \
                -path "*/version_*/checkpoints/$CKPT_NAME" \
                | sort -V
        )

        if [[ ${#CKPT_CANDIDATES[@]} -eq 0 ]]; then
            echo "ERROR: Checkpoint not found for $EXP_NAME:"
            echo "  */version_*/checkpoints/$CKPT_NAME"
            exit 1
        fi

        CKPT="${CKPT_CANDIDATES[$((${#CKPT_CANDIDATES[@]} - 1))]}"

        if [[ ${#CKPT_CANDIDATES[@]} -gt 1 ]]; then
            echo "WARNING: Multiple matching checkpoints found:"
            printf '  %s\n' "${CKPT_CANDIDATES[@]}"
            echo "Using highest version:"
        fi

        echo "Checkpoint: $CKPT"

        # start from a clean output folder
        rm -rf "$OUT_DIR"
        mkdir -p "$OUT_DIR"

        python eval.py \
            --model-module "$MODEL_MODULE" \
            --ckpt "$CKPT" \
            --trace-paths "$TRACE_PATH" \
            --mesh-dir "$MESH_DIR" \
            --n-agents 3 \
            --reset-mode segment_start \
            --n-episodes "$N_EPISODES" \
            --scenario-npz "$SCENARIO_NPZ" \
            --use-curvilinear-dynamics \
            --out-dir "$OUT_DIR" \
            --max-step 100 \
            --save-traj-npz \
            --no-cuda \
            --use-display

        # locate the pickle produced in OUT_DIR
        mapfile -t GENERATED_PKLS < <(
            find "$OUT_DIR" -type f -name "*.pkl" | sort
        )

        if [[ ${#GENERATED_PKLS[@]} -eq 0 ]]; then
            echo "ERROR: No pkl generated for $EXP_NAME under:"
            echo "  $OUT_DIR"
            exit 1
        fi

        if [[ ${#GENERATED_PKLS[@]} -gt 1 ]]; then
            echo "ERROR: Multiple pkl files generated for $EXP_NAME:"
            printf '  %s\n' "${GENERATED_PKLS[@]}"
            exit 1
        fi

        cp "${GENERATED_PKLS[0]}" "$FINAL_PKL"

        echo "Completed: $EXP_NAME"
        echo "Saved pkl: $FINAL_PKL"
    done
done

echo
echo "============================================================"
echo "All requested evaluations completed."
echo "Evaluation outputs:"
echo "  $OUTPUT_ROOT"
echo "Named PKL results:"
echo "  $PKL_ROOT"
echo "============================================================"