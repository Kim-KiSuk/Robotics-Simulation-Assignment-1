#!/usr/bin/env bash
# Run only inference; checkpoint and distributed terrain definitions are fixed.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the IsaacLab_RS checkout with this overlay installed}"
ISAACLAB_ROOT=$(cd -- "$ISAACLAB_ROOT" && pwd)
RECORD=false
if [[ $# -eq 1 && $1 == --video ]]; then
  RECORD=true
elif [[ $# -ne 0 ]]; then
  echo 'Usage: bash scripts/evaluate_final.sh [--video]' >&2
  exit 2
fi
export TEAM_ANT_EVAL_V21_ROOT="$ROOT/evaluation/team_v21"
CHECKPOINT="$ROOT/artifacts/checkpoints/final/ant_final.pt"
EXPECTED=c0784d03e5d02882d0c57aa1d95bca94adfed85aaf88b8284d03839d6f70a382
test "$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)" = "$EXPECTED"
(
  cd "$TEAM_ANT_EVAL_V21_ROOT"
  sha256sum -c SHA256SUMS.txt
)
mkdir -p "$ROOT/local_runs"
OUT=$(mktemp -d "$ROOT/local_runs/team_v21_XXXXXXXX")
trap 'echo "평가 기록 위치: $OUT"' EXIT
sha256sum "$CHECKPOINT" > "$OUT/checkpoint_sha256.txt"
cd "$ISAACLAB_ROOT"
for LEVEL in SeenControl UnseenEasy UnseenMedium UnseenHard; do
  VIDEO_ARGS=()
  if $RECORD; then
    VIDEO_ARGS=(--video --video_length 960 --video_folder "$OUT/videos/$LEVEL")
  fi
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
    --task "Isaac-Ant-TeamEvalV21-Balance-${LEVEL}-v0" \
    --checkpoint "$CHECKPOINT" --seed 24 --num_envs 100 --headless \
    --diagnostics --results_file "$OUT/$LEVEL.json" "${VIDEO_ARGS[@]}" \
    2>&1 | tee "$OUT/$LEVEL.log"
  if rg -ni 'Scene state is corrupted|failed to allocate|simulation will miss interactions|CUDA error|out of memory|PhysX error:' "$OUT/$LEVEL.log"; then
    echo 'Physics/CUDA error; stop and inspect logs.' >&2
    exit 1
  fi
  rg -q 'Completed first episodes: 100/100' "$OUT/$LEVEL.log"
  if $RECORD; then
    test -s "$OUT/videos/$LEVEL/rl-video-step-0.mp4"
  fi
done
./isaaclab.sh -p "$ROOT/scripts/collect_results.py" "$OUT"
echo "평가 완료. 결과 저장 위치: $OUT"
