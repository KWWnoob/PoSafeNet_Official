#!/bin/bash
# Regenerate the fixed evaluation scenario (initial placements of ego / NPC cars).
# The scenario used for the reported results is shipped as
#   scenarios/fixed_scenario_seed0.npz
# This script writes to SCENARIO_OUT (default: eval_out/fixed_scenario_seed0_regenerated.npz)
# so the shipped file is never overwritten.
set -e

export PYTHONNOUSERSITE=True
export PYOPENGL_PLATFORM=egl

DRIVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DRIVING_DIR"

DATA_ROOT="${VISTA_DATA_ROOT:-$HOME/data/vista}"
TRACE_ROOT=$DATA_ROOT/traces
MESH_DIR=$DATA_ROOT/carpack01

SCENARIO_OUT="${SCENARIO_OUT:-./eval_out/fixed_scenario_seed0_regenerated.npz}"
mkdir -p "$(dirname "$SCENARIO_OUT")"

python eval.py \
  --make-scenario \
  --seed 0 \
  --n-episodes 20 \
  --scenario-out "$SCENARIO_OUT" \
  --trace-paths \
    $TRACE_ROOT/20220113-131922_lexus_devens_outerloop_reverse \
  --mesh-dir $MESH_DIR \
  --n-agents 3 \
  --reset-mode segment_start \
  --road-width 4.0 \
  --use-curvilinear-dynamics
