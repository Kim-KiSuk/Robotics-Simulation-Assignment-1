# Ant 험지 환경 실행 안내

`Isaac-Ant-Rough-v0`는 원본 `Isaac-Ant-v0`와 별개의 실험 환경입니다.
`ant_env_cfg.py`, 공용 `terrains/config/rough.py`, 기존 PPO 설정 및 baseline
체크포인트는 수정하지 않습니다. 새 Task 등록과 전용 설정만 추가합니다.

## 파일과 현재 실험 범위

| 파일 | 역할 |
| --- | --- |
| `ant_rough_env_cfg.py` | 험지, 높이 센서, 관측/종료 설정 |
| `terrain_mdp.py` | 지면 기준 높이 관측 및 낙상 판정 |
| `terrain.py` | 원본 XY 배치 보존, 시작 높이 보정, 충돌 메시 분할 |
| `agents/ant_rough_ppo_cfg.py` | 원본 PPO 상속, 로그 이름만 `ant_rough`로 분리 |
| `__init__.py` | `Isaac-Ant-Rough-v0` 등록 |

첫 버전은 평지 20%, 요철 40%, 파도 20%, 경사/역경사 각각 10%의 생성 비율을
사용합니다. 전체 지형은 528×528m, 12m 타일 44×44개, 수평 해상도 0.4m입니다.
비율은 샘플링 확률이며 실제 지도 내 개수와 로봇의 경험 비율은 다를 수 있습니다.
원본 4096개/5m 격자의 외곽부터 지도 가장자리까지 약 106.5m가 남습니다.
지도는 유한하므로 빠른 정책이나 큰 환경 수에서는 경계 이탈을 따로 점검해야 합니다.

마찰은 정적/동적 모두 1.0으로 고정됩니다. 마찰 무작위화와 curriculum은 아직
포함하지 않습니다. 관절 제어, 보상, PPO, 16초 제한은 원본을 유지합니다.
높이 관측은 월드 Z 대신 지면 상대 높이이며, 0.31m 미만에서 종료합니다.
지면 ray가 지도 밖으로 나가면 관측을 0으로 만들고 실패로 종료합니다.
관측 차원이 같더라도 높이 의미는 달라지므로 평가에서 반드시 명시합니다.

## 실행 준비

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS
```

여러 IsaacLab checkout이 있다면 실제 Python import가 이 저장소를 가리키는지도
확인합니다. 다른 checkout에서 설치한 editable package를 사용하면 새 Task가 없습니다.

```bash
python -c 'import importlib.util; print(importlib.util.find_spec("isaaclab_tasks").origin)'
```

## 물리/센서 점검 (정책 성능 평가는 아님)

```bash
./isaaclab.sh -p scripts/environments/check_ant_rough.py --headless
```

16개 로봇으로 전체 기본 지형을 만들고, 원본 설정 독립성, 초기 높이, 센서 수,
부분 초기화, 관측/보상 유한성과 120스텝 실행을 확인합니다.

## 짧은 PPO 실행

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Rough-v0 --num_envs 16 --max_iterations 2 \
  --seed 42 --run_name smoke_test
```

화면이 필요 없으면 `--headless`를 추가합니다. 테스트 체크포인트는 충분히 학습된
모델이 아니므로 제대로 걷지 않아도 이상하지 않습니다.

## 본 학습 (직접 시작할 때)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Rough-v0 --headless --seed 42 \
  --num_envs 4096 --max_iterations 1000 --run_name rough_seed42 \
  env.scene.terrain.terrain_generator.seed=42
```

로그/모델은 `logs/rsl_rl/ant_rough/<날짜_실행이름>/`에 저장됩니다. 기존 baseline은
`logs/rsl_rl/ant/`에 남습니다. 4096개 장시간 실행의 메모리는 소규모 점검과 다르므로
실제 장비에서 확인해야 합니다. OOM으로 환경 수를 줄이면 `환경 수 × 32 × iterations`
학습량 변화도 기록하고 비교 실험의 학습량을 맞춥니다.

## 새 지도에서 평가

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Rough-v0 --headless --num_envs 100 --seed 24 \
  --checkpoint logs/rsl_rl/ant_rough/<RUN_FOLDER>/<CHECKPOINT>.pt \
  env.scene.terrain.terrain_generator.seed=2001
```

`--seed`는 초기 상태 등의 seed이고 지형 seed는 별도 설정입니다. **새 지도를 만들려면
`env.scene.terrain.terrain_generator.seed`도 바꿔야 합니다.** 같은 생성 분포의 새 seed는
새 지도의 평가이지, 학습 범위를 벗어난 지형 분포 전체의 검증은 아닙니다.
평가 마찰은 명령 끝에 아래처럼 지정합니다.

```text
env.scene.terrain.physics_material.static_friction=0.4
env.scene.terrain.physics_material.dynamic_friction=0.3
```

기본 모델도 같은 험지 Task에 baseline 체크포인트를 지정해 비교할 수 있습니다.
그 경우 baseline에도 지면 상대 높이 관측과 험지 종료 기준을 적용했다는 점을 기록합니다.
100개 환경의 첫 에피소드 보상 평균/표준편차는 기존 `play_one_episode.py`를 사용합니다.
동일 Task, 지형 seed, 초기 상태 seed, 환경 수와 마찰 조건으로 비교하세요.

## 구현 범위와 라이선스

지면 기준 높이, 격자 배치 보존 및 충돌 메시 분할을 사용합니다.
별도 Task와 전용 generator 하위 클래스로 구성하며 공용 지형 코드는 변경하지 않습니다.
각 소스 파일의 원 저작권 고지와 BSD-3-Clause 라이선스를 유지합니다.
