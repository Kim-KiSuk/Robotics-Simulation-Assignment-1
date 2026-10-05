#!/usr/bin/env bash
# User-run final evaluation: no training, no checkpoint selection by score.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MANIFEST="$ROOT/docs/ant_team_eval_v21/checkpoint_manifest.json"
CHECKPOINT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$MANIFEST")
EXPECTED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$MANIFEST")
test -f "$CHECKPOINT"
ACTUAL=$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)
test "$ACTUAL" = "$EXPECTED"
OUT=$(mktemp -d "$ROOT/logs/team_eval_v21_final_XXXXXXXX")
trap 'echo "평가 기록 위치: $OUT"' EXIT
mkdir -p "$OUT/smoke" "$OUT/evaluation_sources" "$OUT/training_params"
cp "$MANIFEST" "$OUT/checkpoint_manifest.json"
cp docs/ant_team_eval_v21/GPT_EVALUATION_PROMPT_KO.md "$OUT/"
cp -R /home/kisuk/team_ant_eval_environment_only_v2_1 "$OUT/distribution_v2_1"
cp "$(dirname "$CHECKPOINT")/params/env.yaml" "$(dirname "$CHECKPOINT")/params/agent.yaml" "$OUT/training_params/"
sha256sum "$CHECKPOINT" | tee "$OUT/checkpoint_sha256.txt"
git status --short > "$OUT/git_status.txt"
ANT=source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant
for FILE in "$ANT/ant_team_eval_v21_env_cfg.py" "$ANT/team_eval_v21_distribution.py" \
            "$ANT/agents/ant_team_eval_v21_ppo_cfg.py" "$ANT/__init__.py" \
            scripts/environments/check_ant_team_eval_v21.py \
            scripts/environments/run_ant_team_eval_v21.sh \
            scripts/environments/summarize_ant_team_eval_v21.py; do
  mkdir -p "$OUT/evaluation_sources/$(dirname "$FILE")"
  cp "$FILE" "$OUT/evaluation_sources/$FILE"
done
(
  cd /home/kisuk/team_ant_eval_environment_only_v2_1
  sha256sum -c SHA256SUMS.txt
) | tee "$OUT/distribution_checksums.log"

check_log() {
  if rg -ni 'Scene state is corrupted|failed to allocate|simulation will miss interactions|CUDA error|out of memory|PhysX error:' "$1"; then
    echo "물리/CUDA 오류가 발견되어 중단합니다. 결과 폴더를 공유해 주세요."
    exit 1
  fi
}

./isaaclab.sh -p scripts/environments/list_envs.py > "$OUT/registered_tasks.log" 2>&1
for LEVEL in SeenControl UnseenEasy UnseenMedium UnseenHard; do
  rg -q "Isaac-Ant-TeamEvalV21-Balance-${LEVEL}-v0" "$OUT/registered_tasks.log"
  ./isaaclab.sh -p scripts/environments/check_ant_team_eval_v21.py \
    --task "Isaac-Ant-TeamEvalV21-Balance-${LEVEL}-v0" \
    --checkpoint "$CHECKPOINT" --headless --seed 24 --num_envs 2 \
    --output "$OUT/smoke/$LEVEL.json" 2>&1 | tee "$OUT/smoke/$LEVEL.log"
  check_log "$OUT/smoke/$LEVEL.log"
done

# Only after ALL four smoke tests pass, evaluate each first episode once.
for LEVEL in SeenControl UnseenEasy UnseenMedium UnseenHard; do
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
    --task "Isaac-Ant-TeamEvalV21-Balance-${LEVEL}-v0" \
    --checkpoint "$CHECKPOINT" --seed 24 --num_envs 100 --headless \
    --diagnostics --results_file "$OUT/$LEVEL.json" \
    2>&1 | tee "$OUT/$LEVEL.log"
  check_log "$OUT/$LEVEL.log"
  rg -q 'Completed first episodes: 100/100' "$OUT/$LEVEL.log"
  python - "$OUT/$LEVEL.json" "$EXPECTED" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d['completed'] != 100 or d['checkpoint_sha256'] != sys.argv[2]:
    raise SystemExit('Incomplete result or changed checkpoint; stop before next environment.')
PY
done

./isaaclab.sh -p scripts/environments/summarize_ant_team_eval_v21.py "$OUT"
echo "평가 완료. 결과 저장 위치: $OUT"
