#!/usr/bin/env bash
# Run from IsaacLab_RS, after activating its Python environment.
# Usage: bash /path/to/submission/evaluate_matrix.sh B Isaac-Ant-Rough-v0 checkpoint.pt output_dir [terrain_seed ...]
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 MODEL TASK CHECKPOINT OUTPUT_DIR [TERRAIN_SEED ...]" >&2
  exit 2
fi
model="$1"
task="$2"
checkpoint="$3"
output_dir="$4"
shift 4
if [ ! -f ./isaaclab.sh ] || [ ! -f "$checkpoint" ]; then
  echo "Run in IsaacLab_RS with an existing checkpoint" >&2
  exit 2
fi
if [ "$#" -eq 0 ]; then
  set -- 2001 2002
fi
mkdir -p "$output_dir"
for terrain_seed in "$@"; do
  for friction in default low high; do
    case "$friction" in
      default) static=1.0; dynamic=1.0 ;;
      low) static=0.4; dynamic=0.3 ;;
      high) static=1.5; dynamic=1.2 ;;
    esac
    output="$output_dir/${model}_${friction}_terrain${terrain_seed}_seed24.json"
    if [ -e "$output" ] || [ -e "${output%.json}.log" ]; then
      echo "Existing result/log: $output; choose a new output directory" >&2
      exit 2
    fi
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate_ant.py \
      --task "$task" --checkpoint "$checkpoint" --headless --num_envs 100 --seed 24 \
      --output "$output" \
      "env.scene.terrain.terrain_generator.seed=$terrain_seed" \
      "env.scene.terrain.physics_material.static_friction=$static" \
      "env.scene.terrain.physics_material.dynamic_friction=$dynamic" \
      2>&1 | tee "${output%.json}.log"
    # Some Isaac Sim failures exit with status zero. Require the completed JSON.
    test -s "$output"
  done
done
