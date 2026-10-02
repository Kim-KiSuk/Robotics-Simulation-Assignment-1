# 설치·학습·플레이·평가

## 설치

이 저장소는 수업 원본에 적용하는 코드 묶음입니다. Isaac Sim 5.1 / 수업용 IsaacLab 2.3 계열과 Python 3.11 실행 환경을 먼저 준비합니다.

```bash
git clone https://github.com/cailab-hy/IsaacLab_RS.git IsaacLab_RS
cd IsaacLab_RS
git checkout e83a5d2f11ca1b5f03b690e1978479e620c500e2
cd ..
git clone https://github.com/Kim-KiSuk/Robotics-Simulation-Assignment-1.git
python3 Robotics-Simulation-Assignment-1/apply_overlay.py --target IsaacLab_RS --check
python3 Robotics-Simulation-Assignment-1/apply_overlay.py --target IsaacLab_RS
```

두 저장소를 위와 같이 나란히 놓았다고 가정합니다. 설치기는 원본 commit과 SHA-256을 확인하고 로컬 변경이 있으면 쓰기 전에 중단합니다. 기존 실험 중인 checkout에 덮어쓰지 말고 별도의 원본 checkout에 설치하세요. 여러 checkout이 있으면 활성 conda 환경의 editable package도 실행할 checkout을 가리켜야 합니다. 이번 자료를 이미 구현한 사용자의 작업 폴더에는 재설치할 필요가 없습니다.

원본 `ant_env_cfg.py`, `rsl_rl_ppo_cfg.py`, `train.py`, 공용 `terrains/config/rough.py`는 유지합니다. 별도 환경·센서·지형 파일, Task 등록, 첫 에피소드 평가의 선택적 진단 기능을 추가합니다. Stability Task와 warm-start 기능은 이 공개 시점에 포함하지 않습니다.

## 공개 모델과 입력 호환성

| 이름 | 파일 | 입력 | E3 평가 Task |
|---|---|---:|---|
| 평지 A | `artifacts/checkpoints/A/model_999.pt` | 60 | `Isaac-Ant-Six-Eval-Blocks-v0` |
| 초기 B/C | `artifacts/checkpoints/B/model_999.pt`, `C/model_999.pt` | 60 | `Isaac-Ant-Six-Eval-Blocks-v0` |
| SixMix | `artifacts/checkpoints/SixMix/model_3999.pt` | 60 | `Isaac-Ant-Six-Eval-Blocks-v0` |
| HeightScan | `artifacts/checkpoints/HeightScan/model_3999.pt` | 123 | `Isaac-Ant-Six-Eval-Blocks-HeightScan-v0` |

60/123차원 E3 Task는 같은 지형·마찰·로봇·원본 보상·시간·종료 기준을 사용하며 높이 스캔 입력 유무가 다릅니다. 체크포인트와 관측 설정을 맞춰야 합니다. R/H/HR의 H(240차원 이력)는 HeightScan(123차원)과 다릅니다.

## 공개 모델 평가

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

# 높이 관측 모델: E3, seed 24, 첫 에피소드 100개
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-HeightScan-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/HeightScan/model_3999.pt \
  --headless --seed 24 --num_envs 100 \
  --results_file logs/ant_e3_public_eval/height_scan_seed24.json

# 평지 Baseline: 같은 E3의 60차원 관측 Task
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/A/model_999.pt \
  --headless --seed 24 --num_envs 100 \
  --results_file logs/ant_e3_public_eval/baseline_seed24.json

# 관측 추가의 효과를 비교할 때 사용할 60차원 SixMix
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/SixMix/model_3999.pt \
  --headless --seed 24 --num_envs 100 \
  --results_file logs/ant_e3_public_eval/six_mix_seed24.json
```

`--results_file`은 진단을 켜고 원본 Ant의 7개 보상을 적용합니다. 100/100 완료 후 에피소드별 결과, checkpoint hash, 실행 조건 JSON과 환경·agent YAML을 저장합니다. 기존 결과 파일은 덮어쓰지 않으므로 재실행할 때 새로운 파일명을 사용하세요.

`Target progress`는 목표점까지 거리의 감소량입니다. `Forward displacement`는 world X 순변위이며 지그재그를 포함한 경로 길이가 아닙니다. 위치는 자동 reset 직전에 기록합니다. 시간 제한과 실패가 동시에 발생하면 실패로 집계합니다. Baseline에는 높이 스캔이 없으므로 scan 누락 0이 센서 정상 작동의 증거는 아닙니다.

E3의 사용자 제공 수치에는 정확한 명령과 모델 hash가 연결되어 있지 않습니다. 위 명령은 공개 체크포인트를 재현 가능하게 평가하는 방법이며, 사용자 제공 수치와 일치한다고 보장하는 명령이 아닙니다. [기록의 범위](../results/SIX_RESULTS.md)를 참고하세요.

E1 또는 E2에서 60차원 모델을 평가하려면 Task를 각각 `Isaac-Ant-Six-Eval-v0`, `Isaac-Ant-Six-Eval-Continuous-v0`로 바꿉니다. 현재 E1/E2에는 123차원 Task가 등록되어 있지 않습니다. 기존 `evaluate_ant.py`/`evaluate_matrix.sh`는 초기 Rough·Varied 실험용이고 Six 평가에는 위의 `play_one_episode.py`를 사용합니다.

## 화면과 영상으로 확인

정량 평가와 같은 100개 환경으로 녹화합니다. 체크포인트 파일이 들어 있는 공개 폴더에 영상이 섞이지 않도록 출력 폴더를 지정합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-HeightScan-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/HeightScan/model_3999.pt \
  --seed 24 --num_envs 100 --real-time --diagnostics \
  --video --video_length 960 --video_folder logs/ant_e3_public_video/height_scan
```

`--headless`를 사용하지 않아 GUI가 열립니다. 환경 0을 추적하는 카메라의 영상 하나로 100개 전체의 성공률을 판단하지 않습니다. 환경 수를 줄이면 격자 배치와 경험하는 지형도 바뀝니다.

## 처음부터 학습

높이 관측 모델은 60차원 모델을 이어받는 방식이 아닙니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Six-TrainMix-HeightScan-v0 \
  --headless --num_envs 1024 --max_iterations 4000 --seed 42 \
  --run_name six_mix_height_seed42
```

결과는 `logs/rsl_rl/ant_six_height/<실행시각>_six_mix_height_seed42/model_3999.pt`입니다. 60차원 비교 모델은 Task를 `Isaac-Ant-Six-TrainMix-v0`, run name을 `six_mix_1024_seed42`로 바꾸며 로그는 `ant_six`에 저장됩니다. 두 모델 모두 131,072,000 transitions입니다. 환경 수를 줄이거나 학습 횟수를 바꾸면 그 차이도 보고서에 기록합니다.

```bash
tensorboard --logdir ../Robotics-Simulation-Assignment-1/artifacts/tensorboard
# 자신의 새 학습 곡선
tensorboard --logdir logs/rsl_rl/ant_six_height --port 6007
```

TensorBoard는 학습 중에도 볼 수 있습니다. 학습 보상 곡선과 미사용 환경의 첫 에피소드 평가는 다른 측정입니다.

## 검증과 자료 범위

```bash
python scripts/environments/test_ant_height_scan_math.py
python scripts/environments/test_ant_friction.py
./isaaclab.sh -p scripts/environments/check_ant_height_scan.py --help
```

생성기·PhysX·센서·짧은 PPO 실행에 대한 기존 검증과 이번 패키지 검사 범위는 [업로드 검증 기록](../results/UPLOAD_VALIDATION.md)에 구분했습니다. GPU가 부족했던 4096개 혼합 지형 학습의 체크포인트와 짧은 검증용 checkpoint는 공개 최종 모델에 넣지 않았습니다. 중간 checkpoint 전체, 캐시, Isaac Sim 자산, 강의 PDF도 포함하지 않습니다.

[상세 환경 제작 기록](ant_six_envs/README.md) · [초기 실험 실행 기록](EARLY_REPRODUCE.md) · [출처](../THIRD_PARTY_NOTICES.md)
