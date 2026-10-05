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

원본 `ant_env_cfg.py`, `rsl_rl_ppo_cfg.py`, 공용 `terrains/config/rough.py`는 유지합니다. 별도 환경·센서·지형 파일, Task 등록, 평가 진단 기능을 추가했습니다. `train.py`에는 Stability 추가 학습을 위한 `--warm_start` 옵션을 추가했습니다.

## 공개 모델과 입력 호환성

| 이름 | 파일 | 입력 | E3 평가 Task |
|---|---|---:|---|
| 평지 A | `artifacts/checkpoints/baseline/ant_baseline.pt` | 60 | `Isaac-Ant-Six-Eval-Blocks-v0` |
| 초기 B/C | `artifacts/checkpoints/rough/ant_rough.pt`, `artifacts/checkpoints/friction_dr/ant_friction_dr.pt` | 60 | `Isaac-Ant-Six-Eval-Blocks-v0` |
| SixMix | `artifacts/checkpoints/terrain_mix/ant_terrain_mix.pt` | 60 | `Isaac-Ant-Six-Eval-Blocks-v0` |
| HeightScan | `artifacts/checkpoints/height_scan/ant_height_scan.pt` | 123 | `Isaac-Ant-Six-Eval-Blocks-HeightScan-v0` |
| Stability | `artifacts/checkpoints/stability/ant_stability.pt` | 123 | `Isaac-Ant-Six-Eval-Blocks-HeightScan-v0` |

60/123차원 E3 Task는 같은 지형·마찰·로봇·원본 보상·시간·종료 기준을 사용하며 높이 스캔 입력 유무가 다릅니다. 체크포인트와 관측 설정을 맞춰야 합니다. R/H/HR의 H(240차원 이력)는 HeightScan(123차원)과 다릅니다.

## 공개 모델 평가

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

# Stability: 최신 추가 학습 모델, 평가에는 원본 보상 사용
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-HeightScan-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/stability/ant_stability.pt \
  --headless --seed 24 --num_envs 100 \
  --results_file logs/ant_e3_public_eval/stability_seed24.json

# 높이 관측 부모 모델: 같은 조건에서 비교
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-HeightScan-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/height_scan/ant_height_scan.pt \
  --headless --seed 24 --num_envs 100 \
  --results_file logs/ant_e3_public_eval/height_scan_seed24.json

# 평지 Baseline: 같은 E3의 60차원 관측 Task
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/baseline/ant_baseline.pt \
  --headless --seed 24 --num_envs 100 \
  --results_file logs/ant_e3_public_eval/baseline_seed24.json

# 관측 추가의 효과를 비교할 때 사용할 60차원 SixMix
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/terrain_mix/ant_terrain_mix.pt \
  --headless --seed 24 --num_envs 100 \
  --results_file logs/ant_e3_public_eval/six_mix_seed24.json
```

`--results_file`은 진단을 켜고 원본 Ant의 7개 보상을 적용합니다. 100/100 완료 후 에피소드별 결과, checkpoint hash, 실행 조건 JSON과 환경·agent YAML을 저장합니다. 기존 결과 파일은 덮어쓰지 않으므로 재실행할 때 새로운 파일명을 사용하세요.

`Target progress`는 목표점까지 거리의 감소량입니다. `Forward displacement`는 world X 순변위이며 지그재그를 포함한 경로 길이가 아닙니다. 위치는 자동 reset 직전에 기록합니다. 시간 제한과 실패가 동시에 발생하면 실패로 집계합니다. Baseline에는 높이 스캔이 없으므로 scan 누락 0이 센서 정상 작동의 증거는 아닙니다.

초기 E3 보고값에는 명령과 모델 hash가 연결되어 있지 않습니다. 최신 Stability 출력에는 명령과 모델 경로가 있지만 에피소드별 JSON은 없습니다. 부모 모델의 별도 재평가 JSON도 함께 공개합니다. 위 명령으로 실행 모드를 통일하고 결과 JSON을 남겨 비교하세요. 과거의 모든 출력값과 일치한다고 보장하는 명령은 아닙니다. [기록의 범위](../results/SIX_RESULTS.md)를 참고하세요.

E1 또는 E2에서 60차원 모델을 평가하려면 Task를 각각 `Isaac-Ant-Six-Eval-v0`, `Isaac-Ant-Six-Eval-Continuous-v0`로 바꿉니다. 현재 E1/E2에는 123차원 Task가 등록되어 있지 않습니다. 기존 `evaluate_ant.py`/`evaluate_matrix.sh`는 초기 Rough·Varied 실험용이고 Six 평가에는 위의 `play_one_episode.py`를 사용합니다.

## 화면과 영상으로 확인

정량 평가와 같은 100개 환경으로 녹화합니다. 체크포인트 파일이 들어 있는 공개 폴더에 영상이 섞이지 않도록 출력 폴더를 지정합니다. 아래는 HeightScan입니다. Stability를 보려면 checkpoint를 `../Robotics-Simulation-Assignment-1/artifacts/checkpoints/stability/ant_stability.pt`로, video_folder를 `logs/ant_e3_public_video/stability`로 바꾸고 같은 Task를 사용합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-HeightScan-v0 \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/height_scan/ant_height_scan.pt \
  --seed 24 --num_envs 100 --real-time --diagnostics \
  --video --video_length 960 --video_folder logs/ant_e3_public_video/height_scan
```

`--headless`를 사용하지 않아 GUI가 열립니다. 환경 0을 추적하는 카메라의 영상 하나로 100개 전체의 성공률을 판단하지 않습니다. 환경 수를 줄이면 격자 배치와 경험하는 지형도 바뀝니다.

## 학습 명령

SixMix와 HeightScan은 각각 처음부터 학습했습니다. 아래 설정은 저장된 학습 run과 대응합니다. Stability의 부모 경로는 공개된 동일 SHA-256 파일로 바꿔 다른 컴퓨터에서도 실행할 수 있게 적었습니다.

### SixMix: 다섯 구성 혼합

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Six-TrainMix-v0 \
  --headless --num_envs 1024 --max_iterations 4000 --seed 42 \
  --run_name six_mix_1024_seed42
```

### HeightScan: 주변 높이 관측

60차원 SixMix 가중치를 그대로 이어받지 않습니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Six-TrainMix-HeightScan-v0 \
  --headless --num_envs 1024 --max_iterations 4000 --seed 42 \
  --run_name six_mix_height_seed42
```

결과는 `logs/rsl_rl/ant_six_height/<실행시각>_six_mix_height_seed42/model_3999.pt`입니다. 60차원 비교 모델은 Task를 `Isaac-Ant-Six-TrainMix-v0`, run name을 `six_mix_1024_seed42`로 바꾸며 로그는 `ant_six`에 저장됩니다. 두 모델 모두 131,072,000 transitions입니다. 환경 수를 줄이거나 학습 횟수를 바꾸면 그 차이도 보고서에 기록합니다.

### Stability: HeightScan에서 추가 학습

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Six-TrainMix-HeightScan-Stability-v0 \
  --headless --num_envs 1024 --max_iterations 500 --seed 42 \
  --run_name height_stability_seed42 \
  --warm_start ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/height_scan/ant_height_scan.pt \
  --warm_start_min_std 0.02
```

결과는 `logs/rsl_rl/ant_six_height_stability/<실행시각>_height_stability_seed42/model_499.pt`입니다. 16,384,000 transition을 추가합니다. actor·critic·정책 noise를 읽고 optimizer와 반복 카운터는 초기화합니다. `--resume`과 함께 사용하지 않습니다. 0.02는 시작 시 noise의 하한이며 학습 내내 강제로 유지하는 값이 아닙니다.

실제 실행에 사용한 부모 경로는 `logs/rsl_rl/ant_six_height/2026-10-02_05-21-39_six_mix_height_seed42/model_3999.pt`였으며 [warm_start.json](../artifacts/checkpoints/stability/warm_start.json)에 SHA와 초기화 기록이 있습니다.

평가는 학습 Task 대신 **`Isaac-Ant-Six-Eval-Blocks-HeightScan-v0`**를 사용합니다. `Isaac-Ant-Six-Eval-Blocks-v0`는 60차원이라 이 모델을 불러올 수 없습니다.

### TensorBoard

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
python scripts/environments/test_ant_stability.py \
  --checkpoint ../Robotics-Simulation-Assignment-1/artifacts/checkpoints/height_scan/ant_height_scan.pt
./isaaclab.sh -p scripts/environments/check_ant_height_scan.py --help
```

생성기·PhysX·센서·짧은 PPO 실행에 대한 기존 검증과 이번 패키지 검사 범위는 [업로드 검증 기록](../results/UPLOAD_VALIDATION.md)에 구분했습니다. GPU가 부족했던 4096개 혼합 지형 학습의 체크포인트와 짧은 검증용 checkpoint는 공개 최종 모델에 넣지 않았습니다. 중간 checkpoint 전체, 캐시, Isaac Sim 자산, 강의 PDF도 포함하지 않습니다.

[상세 환경 제작 기록](ant_six_envs/README.md) · [초기 실험 실행 기록](EARLY_REPRODUCE.md) · [출처](../THIRD_PARTY_NOTICES.md)
