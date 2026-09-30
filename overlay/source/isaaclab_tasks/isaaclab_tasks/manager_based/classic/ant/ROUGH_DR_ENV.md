# 실험 C: 험지 + 바닥 마찰 다양화

`Isaac-Ant-Rough-DR-v0`는 B의 `Isaac-Ant-Rough-v0`를 상속하는 독립 Task입니다.
기존 A/B 환경 설정, 지형 생성 코드, 보상, PPO 설정, 모델은 유지합니다.
원본 등록 파일에는 C 항목만 추가합니다.

## 달라지는 조건

| 항목 | B | C |
| --- | --- | --- |
| 지형 메시/시작 위치/센서 | 기존 험지 | B와 동일 |
| 관측/행동/보상/종료/시간 | 기존 설정 | B와 동일 |
| PPO 구조/학습량 | 기존 PPO | B와 동일 |
| 로봇 발 재질 | 기본 재질 | B와 동일 |
| 바닥 마찰 | 모든 구역 1.0/1.0 | 구역별로 다른 정적/동적 마찰 |
| 로그 폴더 | `logs/rsl_rl/ant_rough` | `logs/rsl_rl/ant_rough_dr` |

`terrain_dr.py`는 B의 충돌 메시를 그대로 만든 다음, 각 충돌 구역의 바닥 재질만
바꿉니다. 기본 528m 지도에는 132×132m 구역 16개가 있습니다. 추가 collider를
만들지 않으므로 B와 동일한 분할을 유지합니다.

- 정적 마찰: `[0.4, 1.5]` 범위를 구역 수만큼 나눠 각 구간에서 하나씩 샘플링하고 공간 배치를 섞음.
- 동적 마찰: 해당 정적 마찰 × `[0.75, 1.0]`에서 뽑은 비율. 항상 정적 마찰 이하.
- 마찰 seed: 기본 `43`. 별도 NumPy Generator를 사용해 지형/초기 상태 난수를 소비하지 않음.
- 반발계수와 마찰 결합 방식은 B의 `0.0`, `average`를 유지.

이는 **공간별 마찰 다양화**입니다. 에피소드가 끝날 때 새 마찰을 뽑거나 로봇마다
개별 재질을 적용하는 방식이 아닙니다. 마찰 지도는 한 실행 동안 고정되며, 구역을
이동하는 로봇은 다른 마찰을 만날 수 있습니다. 소수 환경이나 짧은 이동에서는 전체
16구역을 경험하지 않을 수 있고, 기본 격자와 같은 위치에서 매번 시작합니다.
보고서에는 이 적용 단위와 한계를 명시합니다. 계수는 바닥에 설정한 값이며, 실제
발-지면 접촉의 유효 마찰은 발 재질과 결합 규칙에도 영향을 받습니다.

## 수정 위치

| 파일 | 역할 |
| --- | --- |
| `ant_rough_dr_env_cfg.py` | B 설정을 복사해 C 전용 지형 importer 연결 |
| `terrain_dr.py` | 마찰 범위/seed, 구역별 재질 생성 및 바인딩 |
| `friction_utils.py` | 재현 가능한 마찰 샘플링, 입력 범위 검증 |
| `agents/ant_rough_dr_ppo_cfg.py` | PPO를 상속하고 출력 폴더만 분리 |

`params/env.yaml`에 범위와 seed가 저장됩니다. 실행 시 `[FRICTION_MAP]` 로그에는
충돌 구역별 재질 경로와 실제 샘플링 계수가 출력됩니다.

## 점검

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

# CPU 테스트: 시뮬레이터를 실행하지 않음
python scripts/environments/test_ant_friction.py

# 다른 GPU 학습/평가를 마친 뒤 실행
./isaaclab.sh -p scripts/environments/check_ant_rough_dr.py --headless
```

시뮬레이터 검사는 A/B 설정 독립성, 원본과 같은 지형 설정, PPO 동등성,
16개 collider에 실제 연결된 physics material의 계수, 반발계수/결합 규칙,
60차원 관측/8차원 행동, 부분 reset과 120스텝을 확인합니다.

화면으로 짧게 실행하려면:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Rough-DR-v0 --num_envs 16 --max_iterations 10 \
  --seed 42 --run_name dr_visual_check
```

## 본 학습

B와 같은 지형 seed, 초기화/정책 seed, 환경 수, 업데이트 횟수를 사용합니다.
아래는 새 정책을 처음부터 학습하는 명령이며 B에서 추가 학습하는 명령이 아닙니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Rough-DR-v0 --headless \
  --num_envs 4096 --max_iterations 1000 --seed 42 \
  --run_name rough_dr_seed42 \
  env.scene.terrain.terrain_generator.seed=42 \
  env.scene.terrain.friction_seed=43
```

기본 마찰 범위 변경 예시(명령 끝에 추가):

```text
env.scene.terrain.static_friction_range=[0.4,1.5]
env.scene.terrain.dynamic_friction_ratio_range=[0.75,1.0]
```

DR 활성 상태에서는 `physics_material.static_friction`만 바꿔도 각 구역의 샘플링
범위가 바뀌지 않습니다. 위 두 범위 설정을 사용합니다. 균일한 재질을 사용하려면
`env.scene.terrain.randomize_friction=false`와 `physics_material` 계수를 지정하거나,
다음처럼 B Task로 평가합니다.

## B/C를 같은 평가 환경에서 비교

**학습 Task가 C라고 평가 Task도 C로 할 필요는 없습니다.** 관측/행동/정책 구조가
B와 같으므로 C 체크포인트를 B의 고정 마찰 환경에서 평가할 수 있습니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Rough-v0 --headless --num_envs 100 --seed 24 \
  --checkpoint logs/rsl_rl/ant_rough_dr/<RUN_FOLDER>/<CHECKPOINT>.pt \
  env.scene.terrain.terrain_generator.seed=2001 \
  env.scene.terrain.physics_material.static_friction=0.4 \
  env.scene.terrain.physics_material.dynamic_friction=0.3
```

위 명령에서 체크포인트만 B 모델 경로로 바꾸어 비교합니다. 보상 함수, 관측 의미,
지형 seed, 초기 상태 seed, 환경 수, 종료 조건과 마찰 조건을 동일하게 유지합니다.
기본/높은 마찰도 같은 방법으로 비교합니다. B를 균일한 마찰 지도, C를 무작위 마찰
지도에서 각각 평가해 얻은 수치를 직접 비교하지 않습니다.

평가 시 마찰 지도 자체도 바꾸려면 C Task의 `friction_seed`를 바꾸되, 비교하는 모든
모델에 같은 지도와 seed를 사용합니다. 지형 seed와 마찰 seed는 독립적입니다.

## 참고

지형/관측 기반은 [B 구현 안내](ROUGH_ENV.md)를 따릅니다. C의 마찰 샘플링과 Task는
별도로 추가했습니다. 재질은 Isaac Lab의
[physics material API](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sim.utils.html#isaaclab.sim.utils.bind_physics_material)
를 통해 지면 collider에 적용합니다. 이 단계는 성능 향상을 입증한 결과가 아니라
비교 실험을 위한 구현이며, 실제 학습·평가 결과로 효과를 확인해야 합니다.
