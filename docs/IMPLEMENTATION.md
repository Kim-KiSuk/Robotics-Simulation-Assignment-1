# 원본 대비 구현과 보완

비교 기준은 수업 원본 `e83a5d2f11ca1b5f03b690e1978479e620c500e2`다. [overlay](../overlay)에 같은 상대 경로의 파일을 제공한다. 아래 Ant 파일의 공통 경로는 `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/`이다.

## 학습 지형과 마찰

초기 환경은 T1~T5 + E1으로 구성했다. Mix는 다섯 구성에서 타일을 각각 20% 확률로 선택한다. 실제 경험 시간 비율이 정확히 20%라는 뜻은 아니다.

| 학습 구성 | 기본 수치 | 지면 정적/동적 마찰 |
| --- | --- | --- |
| T1 | 높이 샘플 ±2cm / ±6cm, 각 50%, 간격 0.8m | 1.0 / 1.0 |
| T2 | amplitude 2~10cm, 최초 2/4 waves → 최종 2/4/6/8 동일 비율 | 0.8 / 0.6 |
| T3 | 경사·역경사 3~12°, platform 1m | 1.2 / 1.0 |
| T4 | 블록 높이 변동 2~10cm, 최종 폭 0.55/0.70/0.90m | 0.6 / 0.45 |
| T5 | 계단 상하·grid·경사 상하 각 20%; 계단 3~8cm, grid 2~8cm | 1.4 / 1.1 |

T5 최종 계단 폭은 0.45/0.60/0.80m, grid 폭은 0.55/0.65/0.85m다. 12m 타일 44×44개로 528×528m를 만든다. 블록 높이는 양·음으로 뽑으므로 이웃 높이차는 변동 크기의 두 배에 이를 수 있다. amplitude는 실제 봉우리~골 높이차와 같은 말이 아니며 noise 보간은 샘플 범위를 넘을 수 있다.

최종 terrain seed **1201**, 정책 seed **42**, robot material seed **42017**을 분리했다. 지형 메시를 episode마다 다시 만들지 않는다. `AntTerrainImporter`는 격자 XY와 시작점 주변 5×5 ray의 최대 지면 높이를 사용한다. 렌더링·ray용 전체 메시와 공간 분할 충돌 메시를 구분한다.

**Friction DR:** `ant_friction_train_env_cfg.py`의 startup event가 `friction_train_mdp.py:randomize_training_material_per_env`를 호출한다. 로봇 정적 마찰 0.2~1.8, 동적 0.15~1.5, 반발 0, 64 bucket이며 같은 로봇의 shape들에 같은 재질을 준다. dynamic≤static을 유지한다. 지면과 로봇 마찰은 **average** 규칙으로 결합하므로 robot 범위를 접촉 마찰 범위라고 부르지 않는다. 실행 중 재추첨하지 않는다.

## 관측·행동·정책

원본 60차원 상태의 몸 높이를 world Z 대신 지면 상대 높이로 바꾸고 주변 9×7 높이 63개를 더해 **123차원**으로 학습했다. 스캔은 torso yaw 기준 전방 1.8m·후방 0.6m·좌우 0.9m를 0.3m 간격으로 읽는다. RGB-D 카메라가 아니라 이상적인 RayCaster다.

`height_scan_sensor.py`는 drift=0인 센서 reset에서 불필요한 난수 소비를 피한다. `terrain_mdp.py`는 ray 미검출 시 상대 높이를 0으로, `height_scan_mdp.py`는 주변 스캔 미검출을 −1로 대체한다. 최종 평가에서도 이 입력 처리를 유지했다. 값의 유한성과 실제 지면 검출은 구분한다.

행동은 원본 **8개 effort, scale 7.5**다. 액션 smoothing은 토크 필터 추가가 아니라 이전 명령과의 차이에 비용을 주는 방식이다. 로봇 하드웨어·USD·관절 수·구동기 설정은 유지했다. actor/critic hidden 400–200–100, ELU다.

## 학습 보상

원본 7항(progress, alive, upright, move_to_target, action_l2, energy, joint_pos_limits)에 다음을 추가한다.

| 항목 | 최종 비용 | 의도 |
| --- | --- | --- |
| action_change | `−0.002 × Σ(a_t − a_(t−1))²` | 급격한 명령 변화 감소 |
| body_rotation | `−0.01 × (ω_x² + ω_y²)` | 몸체 roll/pitch 회전 감소 |
| failure | 실제 종료 사건마다 **−2** | 정상 timeout과 실패 구분 |

앞의 두 항은 RewardManager가 step_dt를 곱하는 rate 항이다. failure 함수는 dt로 나누어 **사건당 −2**가 된다. 정상 timeout에는 없고 실패·timeout이 동시에 발생하면 실패로 처리한다. 평가에서는 추가 세 항을 제외한다.

코드는 `ant_diverse_train_env_cfg.py`, `robust_mdp.py`다. Balance의 failure 기본값 0을 CLI `env.rewards.failure.weight=-2.0`으로 덮어쓴다.

## 초기 소환 보완

일부 블록 위 첫 소환에서 발이 지형 내부에 보였고 영 행동 진단에서 초기 위쪽 속도와 뒤집힘이 관찰됐다. root reset을 +0.15m 높인 진단에서 낙하·착지가 정상화되어 학습에도 적용했다. 명령은 `env.events.reset_base.params.pose_range.z=[0.15,0.15]`다. 초기 관절 난수 분포는 유지했다.

기존 모델을 이어 학습하지 않고 새로 6000회 학습했다. +0.15m는 검토한 조건의 고정 여유이며 모든 지형에서 충돌을 완전히 방지한다는 증명은 아니다.

## 팀 v2.1 평가

`ant_team_eval_v21_env_cfg.py`는 observation/action/sensor를 유지하면서 배포본 지형·마찰을 연결한다. 원본 reward와 **world Z < 0.31m 종료**, 원본 reset을 사용한다. 개발 E3의 지면 상대 종료 및 SpawnLift reset과 다르다.

| 환경 | terrain seed | 로봇 정적 / 동적 마찰 |
| --- | ---: | --- |
| Seen Control | 51004 | 0.50~1.50 / 0.40~1.20 |
| Unseen Easy | 53001 | 1.0 / 1.0 |
| Unseen Medium | 53012 | 0.65~1.35 / 0.55~1.15 |
| Unseen Hard | 53003 | 0.35~1.60 / 0.25~1.30 |

지면 1/1, 결합 multiply. 8m 타일 40×20개, border 20m, `max_init_terrain_level=4`와 native origins다. generator의 `curriculum=True`는 고정 지형 생성 설정이고, 학습 중 진행하는 curriculum은 꺼져 있다. [배포 파일](../evaluation/team_v21)은 원본 바이트 그대로 복사했고 해시를 확인한다.

## 파일·상속

| 파일 | 역할 |
| --- | --- |
| `ant_env_cfg.py` | 원본 robot/action/reward/reset |
| `ant_rough_env_cfg.py`, `terrain.py`, `terrain_mdp.py` | 상대 높이, 시작 높이, 충돌 메시 |
| `six_terrain_spec.py`, `six_terrain.py`, `ant_six_env_cfg.py` | 다섯 지형과 E1, 혼합 재질 |
| `ant_height_scan_env_cfg.py`, `height_scan_mdp.py`, `height_scan_sensor.py` | 63개 높이 입력 |
| `wave_range_spec.py`, `diverse_training_spec.py` | 파도 수·grid/계단 폭 변형 |
| `ant_friction_train_env_cfg.py`, `friction_train_mdp.py` | 로봇 마찰 DR |
| `ant_diverse_train_env_cfg.py`, `robust_mdp.py` | 최종 보상·reset |
| `agents/ant_diverse_train_ppo_cfg.py` | PPO runner |
| `ant_team_eval_v21_env_cfg.py`, `team_eval_v21_distribution.py` | 공통 평가·고정 배포본 로드 |

상속은 `AntDiverseBalanceEnvCfg → AntDiverseActionSmoothEnvCfg → AntDiverseOriginalRewardEnvCfg → AntDiverseTrainEnvCfg → AntForwardRewardEnvCfg → AntWaveRangeEnvCfg → AntHeightScanTrainEnvCfg → AntRoughEnvCfg → AntEnvCfg`다. 중간 ForwardReward를 상속하지만 OriginalReward에서 원본 보상으로 대체한 뒤 Balance 항을 추가한다. 마찰 이벤트는 `AntFrictionTrainEventsCfg`를 별도로 구성해 연결한다.

공개 overlay는 배포본 경로만 `TEAM_ANT_EVAL_V21_ROOT` 환경변수로 지정할 수 있도록 바꿨다. 실제 평가 당시 파일은 [평가 코드 스냅샷](../results/team_v21/final/evaluation_sources)에 보존했다.
