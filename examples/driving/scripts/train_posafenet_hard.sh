#!/bin/bash

set -e

export PYTHONNOUSERSITE=True  # for Python
export PYOPENGL_PLATFORM=egl  # for VISTA

# Usage (from anywhere):
#   SEED=0 bash scripts/train_posafenet_hard.sh
#   SEED=1 bash scripts/train_posafenet_hard.sh
#   SEED=2 bash scripts/train_posafenet_hard.sh
# SEED defaults to 0. Extra arguments are appended to the run.py call.
#
# POSAFENET_DRIVING_DATA : training dataset root (default: <driving>/data)
# POSAFENET_RESULTS_DIR  : output root for logs/checkpoints (default: <driving>/results)
DRIVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_SRC="${POSAFENET_DRIVING_DATA:-$DRIVING_DIR/data}"
RESULTS_DIR="${POSAFENET_RESULTS_DIR:-$DRIVING_DIR/results}"
cd "$DRIVING_DIR"

SEED="${SEED:-0}"

# one experiment name per seed
EXP_NAME="posafenet_hard_seed_${SEED}"

args=(
    # =========================================================
    # Trainer
    # =========================================================
    --default_root_dir "$RESULTS_DIR"
    --exp_name "$EXP_NAME"
    --seed "$SEED"

    --gpus "0,"
    --max_epochs 10
    --benchmark
    --detect_anomaly

    # =========================================================
    # Logging
    # =========================================================
    --val_check_interval 0.5
    --log_every_n_steps 50
    --enable_checkpointing
    --enable_model_summary

    # =========================================================
    # Data
    # =========================================================
    --data-module sequence_data_module
    --data-src "$DATA_SRC"
    --batch-size 48
    --num-workers 16
    --sequence-size 32

    # =========================================================
    # Model
    # =========================================================
    --model-module posafenet_hard

    --lr 0.001
    --output-mode "v" "delta" "a" "omega"

    --loss-coef-v 0.25
    --loss-coef-delta 0.25
    --loss-coef-a 0.25
    --loss-coef-omega 0.25

    --cnn-dropout 0.3
    --lstm-size 64
    --q-mlp-dropout 0.3
    --abnet-heads 2

    --use-cf
    --use-lane-keeping-CBFs
    --learn-mixture-weight
)

# allow appending/overriding arguments from the command line
args+=("$@")

python run.py "${args[@]}"