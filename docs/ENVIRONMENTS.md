# 환경 구성: 학습용 5개와 평가용 1개에서 시작

지형이 달라도 한 정책으로 전진하도록, 서로 다른 보행 대응을 요구하는 다섯 구성을 학습용으로 정했습니다. 첫 평가용 E1은 같은 생성 분포에서 배치와 마찰 조합을 바꿨습니다. 이후 E2·E3와 높이 관측·안정성 추가 학습으로 확장했습니다.

## 최초 여섯 환경

| 이름 | 생성 설정 | 지면 정적 / 동적 마찰 | terrain seed | 선정 기준 |
|---|---|---:|---:|---|
| T1 | 요철 샘플 ±2cm / ±6cm 각 50%, 샘플 간격 0.8m | 1.0 / 1.0 | 1101 | 작은 불규칙 지면 |
| T2 | 파도 amplitude 2~10cm, 2파 / 4파 각 50% | 0.8 / 0.6 | 1102 | 반복 오르내림 |
| T3 | 경사 / 역경사 3~12° 각 50%, 중앙 플랫폼 1m | 1.2 / 1.0 | 1103 | 연속 기울기 변화 |
| T4 | 사각 블록 폭 0.7m, 높이 변동 크기 2~10cm, 플랫폼 0.8m | 0.6 / 0.45 | 1104 | 과제 예시의 “높이가 다른 사각 블록” |
| T5 | 계단 상하·random grid·경사 상하 각 20% | 1.4 / 1.1 | 1105 | IsaacLab의 다섯 지형 생성기 경험 |
| E1 | T1~T5 생성 구성 각 20%, 새 배치 | 0.9 / 0.75 | 9117 | 학습에 쓰지 않은 배치·마찰 조합 |

T5의 계단 단 높이는 3~8cm, 단 폭 0.6m, 플랫폼 1.2m입니다. grid 폭은 0.65m, 높이 변동 크기 2~8cm, 플랫폼 0.8m이며 경사는 T3과 같습니다. 타일은 12m, 지도는 44×44타일(528m), 높이필드 수평 해상도 0.4m·수직 단위 0.005m입니다.

평지 전용 타일과 구멍은 넣지 않았습니다. 중앙 플랫폼과 국소 수평면은 있습니다. 블록 꼭대기는 변동 크기 `a`에 대해 `[-a, +a]`에서 뽑으므로 인접 높이차는 최대 `2a`입니다. 파도 amplitude와 요철 샘플 범위는 최종 메시의 봉우리~골 높이차와 같지 않습니다.

## 한 정책으로 혼합 학습

`Isaac-Ant-Six-TrainMix-v0`는 T1~T5의 생성 구성을 각 20%로 조합하고 seed **1100**으로 새 혼합 지도를 만듭니다. 개별 지도를 순서대로 로드하거나 다섯 모델을 합치는 방식은 아닙니다. 20%는 생성 확률이며 시작 위치나 경험 시간의 정확한 비율은 아닙니다.

마찰 값은 종류별 타일의 충돌 재질에 연결되고 실행 중 고정됩니다. 지형 seed를 바꾸면 배치와 높이가 달라집니다. 매 에피소드에 지형이나 마찰을 다시 뽑지 않습니다. 지형과 마찰의 영향을 분리하려면 동일한 평가 지형에서 마찰만 바꿔야 합니다.

## 이후 평가맵과 추가 학습

| 구성 | 목적과 변경 | terrain seed |
|---|---|---:|
| E2 | 요철 40%, 파도 30%, 경사 상하 30%; 중앙 플랫폼 제거 | 9217 |
| E3 | 블록 35%, 요철 25%, 파도 20%, 경사 상하 20%; 블록 대응도 평가 | 9317 |
| HeightScan 학습 | TrainMix 지형·마찰을 유지하고 주변 높이 관측 63개 추가 | 1100 |
| Stability 학습 | 기존 생성 범위 70% + 확대 범위 30%; 부모 정책 추가 학습 | 1200 |

E2/E3는 8m 타일 40×40개, 수평 해상도 0.25m·수직 단위 0.005m, 지면 마찰 **0.9 / 0.75**입니다. 요철 샘플 ±6cm(간격 0.5m), 파도 amplitude 8~18cm(3파·5파), 경사 상하 8~16°를 사용합니다. E3 블록은 폭 0.8m, 높이 변동 크기 3~12cm입니다. 평지 전용 타일·중앙 플랫폼·평탄 테두리는 없지만 블록 윗면과 양자화된 높이필드에는 수평면이 있습니다.

Stability의 확대 범위는 요철 샘플 ±4/±8cm, 파도 amplitude 4~16cm(3/5파), 경사 4~16°, 블록 높이 변동 크기 3~12cm, 계단 높이 4~10cm입니다. 새 seed로 전체 지도를 다시 생성하므로 70%가 이전 타일 좌표를 그대로 보존한다는 뜻은 아닙니다. 마찰 값은 기존 종류별 값을 유지합니다.

학습 실행 seed는 **42**, 평가 실행 seed는 **24**입니다. CLI `--seed`와 지형의 `terrain_generator.seed`는 다릅니다. E1의 9117은 정책 성능을 보기 전에 초기 100개 로봇이 다섯 구성을 모두 포함하는지 확인해 선택했습니다. 반복 튜닝에 사용한 E1~E3는 검증용이며, 별도 seed **9417**은 아직 정책 점수를 확인하지 않은 후보로 남겨두었습니다. seed 하나만 바꾼 평가는 미사용 배치에 대한 시험이며 새 지형 종류 전체의 강건성을 입증하지 않습니다.

## 공통 조건과 관측

- 로봇 형상·구동기·물성, 8개 effort 행동(scale 7.5), 원본 7개 평가 보상을 유지합니다.
- 제한 시간 16초, 물리 120Hz / 정책 60Hz, 최대 960 step입니다. XY 시작 위치는 5m 격자입니다.
- 험지 공통 입력은 몸통 world Z 대신 지면 대비 높이를 사용합니다. 낙상 기준도 상대 높이 0.31m입니다. 이는 평지 Baseline의 높이 입력 의미와 다릅니다.
- SixMix는 60D, HeightScan/Stability는 123D입니다. 추가 63개는 몸통 yaw에 맞춘 9×7 높이 샘플이며 카메라 영상이 아닙니다.
- Stability 학습에만 실패 사건 −5와 낮은 몸통 높이의 위험 감점을 추가합니다. 평가는 원본 보상 Task로 전환합니다.

## 코드와 상속 경로

아래 파일명 링크는 공개 코드입니다. 적용 후 Ant 파일의 공통 경로는 `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/`입니다.

| 역할 | 파일 | 부모 설정 |
|---|---|---|
| T1~T5, Mix, E1 | [ant_six_env_cfg.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_six_env_cfg.py) | `ant_rough_env_cfg.py: AntRoughEnvCfg` |
| E2 | [ant_continuous_eval_env_cfg.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_continuous_eval_env_cfg.py) | `ant_rough_env_cfg.py: AntRoughEnvCfg` |
| E3 60D | [ant_blocks_eval_env_cfg.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_blocks_eval_env_cfg.py) | `ant_rough_env_cfg.py: AntRoughEnvCfg` |
| HeightScan 학습 / E3 123D | [ant_height_scan_env_cfg.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_height_scan_env_cfg.py) | 학습: `AntRoughEnvCfg`; 평가: 같은 파일의 `AntHeightScanTrainEnvCfg` |
| Stability 학습 | [ant_height_stability_env_cfg.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_height_stability_env_cfg.py) | `ant_height_scan_env_cfg.py: AntHeightScanTrainEnvCfg` |
| 공통 험지 부모 | [ant_rough_env_cfg.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_rough_env_cfg.py) | [ant_env_cfg.py: AntEnvCfg](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py) |

별도 terrain 파일:

- [six_terrain_spec.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/six_terrain_spec.py): T1~T5/E1/Mix 수치·seed·**마찰 값 정의**.
- [six_terrain.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/six_terrain.py): 난수 격리와 **타일별 마찰 재질 적용**. 현재 Six의 friction randomization 관련 구현은 이 파일과 위 spec입니다.
- [continuous_eval_spec.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/continuous_eval_spec.py), [continuous_eval_surface.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/continuous_eval_surface.py): E2 구성·연속 표면.
- [blocks_eval_spec.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/blocks_eval_spec.py), [block_surface.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/block_surface.py): E3 구성·블록 표면.
- [stability_terrain_spec.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/stability_terrain_spec.py): Stability의 70/30 비율과 확대 범위.
- [terrain.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/terrain.py), [terrain_mdp.py](../overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/terrain_mdp.py): 공통 충돌 메시·시작 높이·지면 높이 처리.

공용 `source/isaaclab/isaaclab/terrains/config/rough.py`와 원본 Ant 설정은 수정하지 않았습니다. Six 전용 추가 파일에서 생성기들을 구성합니다. 실행에는 spec 파일만이 아니라 overlay 전체와 고정한 수업 원본이 필요합니다.

## Task 목록

| 용도 | Task |
|---|---|
| T1~T5 개별 확인 | `Isaac-Ant-Six-Train1-v0` ~ `Isaac-Ant-Six-Train5-v0` |
| SixMix 학습 | `Isaac-Ant-Six-TrainMix-v0` |
| HeightScan 학습 | `Isaac-Ant-Six-TrainMix-HeightScan-v0` |
| Stability 추가 학습 | `Isaac-Ant-Six-TrainMix-HeightScan-Stability-v0` |
| E1 / E2 평가, 60D | `Isaac-Ant-Six-Eval-v0` / `Isaac-Ant-Six-Eval-Continuous-v0` |
| E3 평가, 60D | `Isaac-Ant-Six-Eval-Blocks-v0` |
| E3 평가, 123D | `Isaac-Ant-Six-Eval-Blocks-HeightScan-v0` |

최신 학습 Task는 Stability이며 평가 Task는 E3 HeightScan입니다. 최신 모델이라고 해서 우수 모델로 확정한 것은 아닙니다. [직접 실행할 명령](REPRODUCE.md) · [평가 결과](../results/SIX_RESULTS.md)
