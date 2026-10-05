# 설치·학습·평가 재현

## 1. 전제

실제 실행 환경은 Python 3.11의 `lerobot-arena`, NVIDIA RTX 2080 8GB, 수업 IsaacLab_RS commit `e83a5d2f11ca1b5f03b690e1978479e620c500e2`다. Isaac Sim과 해당 버전에 맞는 Isaac Lab 의존성을 먼저 설치해야 한다. 이 저장소는 설치된 시뮬레이터 전체를 대체하지 않는다. 로컬 패키지 버전은 [실행 환경 기록](../configs/software_versions.json)에 있다.

## 2. 별도 원본 checkout에 overlay 적용

현재 학습 폴더의 작업을 덮어쓰지 않도록 별도 checkout을 권장한다. 아래 `PUBLICATION_ROOT`는 **이 제출 저장소를 clone한 절대경로**다.

```bash
conda activate lerobot-arena
export PUBLICATION_ROOT=/absolute/path/to/Robotics-Simulation-Assignment-1

git clone https://github.com/cailab-hy/IsaacLab_RS.git ~/IsaacLab_RS_final
git -C ~/IsaacLab_RS_final checkout e83a5d2f11ca1b5f03b690e1978479e620c500e2
export ISAACLAB_ROOT="$HOME/IsaacLab_RS_final"

python "$PUBLICATION_ROOT/apply_overlay.py" --target "$ISAACLAB_ROOT" --check
python "$PUBLICATION_ROOT/apply_overlay.py" --target "$ISAACLAB_ROOT"
cd "$ISAACLAB_ROOT"
./isaaclab.sh -i rsl_rl
```

`apply_overlay.py`는 파일 hash와 원본 commit을 검사하고 예상하지 못한 로컬 변경이 있으면 중단한다. 기존 설치가 다른 checkout의 editable package를 가리키면 이 checkout으로 설치 경로를 맞춰야 한다.

## 3. 최종 네 환경 평가 + 영상

```bash
cd "$PUBLICATION_ROOT"
bash scripts/evaluate_final.sh --video
```

이 스크립트는 저장소에 포함한 **고정 최종 모델**을 로드한다. 원본 배포 파일 SHA-256을 검사하고 네 Task를 순차 실행한다. GUI 창은 열지 않지만 카메라 렌더링을 켜 16초 영상을 저장한다. 결과는 `local_runs/team_v21_XXXXXXXX/`, 영상은 `videos/<환경>/rl-video-step-0.mp4`다. 녹화가 필요 없으면 `--video`만 생략한다.

| 환경 | Task |
| --- | --- |
| Seen Control | `Isaac-Ant-TeamEvalV21-Balance-SeenControl-v0` |
| Unseen Easy | `Isaac-Ant-TeamEvalV21-Balance-UnseenEasy-v0` |
| Unseen Medium | `Isaac-Ant-TeamEvalV21-Balance-UnseenMedium-v0` |
| Unseen Hard | `Isaac-Ant-TeamEvalV21-Balance-UnseenHard-v0` |

한 환경을 GUI로 확인하려면:

```bash
export TEAM_ANT_EVAL_V21_ROOT="$PUBLICATION_ROOT/evaluation/team_v21"
cd "$ISAACLAB_ROOT"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-TeamEvalV21-Balance-UnseenHard-v0 \
  --checkpoint "$PUBLICATION_ROOT/artifacts/checkpoints/final_lift/model_5999.pt" \
  --seed 24 --num_envs 100 --real-time --diagnostics \
  --video --video_length 960 --video_folder "$PUBLICATION_ROOT/local_runs/hard_preview"
```

최초 제출 평가와 녹화 재실행은 이미 [results/team_v21](../results/team_v21)에 있다. 재실행은 새 측정이며 기존 제출값을 덮어쓰거나 유리한 결과를 고르는 용도로 사용하지 않는다. 공개용 경로 wrapper는 CPU 검증했으며, 제공된 실제 영상은 원래 작업공간의 동등한 명령에서 생성했다.

## 4. 실제 최종 학습 명령 — 기록용

아래는 최종 모델을 만들 때 사용한 조건이다. **기존 제출 모델 평가에는 재학습이 필요 없다.** 현재 최종 평가 이후의 추가 학습은 같은 unseen 비교에 포함하지 않는다.

```bash
conda activate lerobot-arena
cd "$ISAACLAB_ROOT"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-WaveRange-Diverse-Balance-v0 \
  --headless --num_envs 1024 --max_iterations 6000 --seed 42 \
  --run_name balance_failure2_lift_scratch_seed42 \
  agent.experiment_name=ant_wave_diverse_balance_failure2_lift_scratch \
  agent.resume=false \
  agent.algorithm.learning_rate=0.0005 \
  agent.algorithm.schedule=adaptive \
  env.rewards.failure.weight=-2.0 \
  'env.events.reset_base.params.pose_range.z=[0.15,0.15]' \
  env.scene.terrain.terrain_generator.seed=1201 \
  env.events.robot_material.params.material_seed=42017
```

최종 모델은 6000회 scratch 학습이며 `--warm_start`나 `--resume`을 사용하지 않았다. 1024×32×6000=196,608,000 transition이다. GPU 메모리 부족이나 PhysX 파손이 발생한 실행은 정상 학습으로 사용하지 않는다.

## 5. 図・動画の再生成と資料検証

```bash
cd "$PUBLICATION_ROOT"
python scripts/verify_publication.py
python scripts/build_figures.py
python scripts/build_training_curve.py
python scripts/build_previews.py
python -m tensorboard.main --logdir artifacts/tensorboard
```

검증은 표준 라이브러리, 그림은 matplotlib, 학습 곡선은 tensorboard, GIF는 OpenCV/Pillow를 사용한다. 학습 곡선은 학습 당시 shaped reward이며 원본 7항으로 측정한 최종 평가값과 구분한다.

[初期Six段階の命令](REPRODUCE_SIX_STAGE.md)
