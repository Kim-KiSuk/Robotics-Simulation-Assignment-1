# 자체 평가맵: 무엇을 만들고 왜 평가했는가?

우리가 만든 E1~E3와 팀 공통 v2.1, 조교의 공식 unseen 평가는 서로 다르다. **자체 맵은 학습용 메시 생성에 넣지 않았지만**, 그 결과를 보고 방법을 고친 이후에는 개발 검증 환경으로 해석한다. 팀 공통 v2.1은 최종 모델을 고정한 뒤 평가했다. 조교의 공식 unseen 결과는 아직 포함하지 않았다.

## 1. 환경의 발전 과정

| 맵 | 만들거나 변경한 이유 | 구성 | 지형 seed |
| --- | --- | --- | ---: |
| E1 | 다섯 학습 구성의 새로운 배치·마찰 조합 확인 | T1~T5 각 20%, 12m 타일, 44×44 | 9117 |
| E2 | 연속적인 높낮이 변화가 드러나는 지형 확인 | 요철 40%, 파도 30%, 경사 상하 각 15%, 8m 타일, 40×40 | 9217 |
| E3 | 연속 지형과 높이가 다른 블록을 함께 평가 | 블록 35%, 요철 25%, 파도 20%, 경사 상하 각 10%, 8m 타일, 40×40 | 9317 |
| E3-SpawnLift | 시작 시 발·블록 겹침의 영향을 줄여 비교 | **E3와 같은 메시·마찰**, reset Z만 +0.15m | 9317 |

E1~E3 지면 마찰은 정적 **0.9**, 동적 **0.75**다. 평가 정책 난수 `--seed 24`와 terrain seed는 별개다. 로봇 형상·관절·effort 행동·정책 입력은 비교 대상 사이에서 공통으로 유지한다.

E3-SpawnLift는 새로운 지형을 하나 더 만든 것이 아니라 E3의 **초기화 조건을 별도로 구분한 Task**다. 두 모델 비교에서 같은 +0.15m를 적용한다. 원래 E3의 점수와 직접 합치지 않는다.

## 2. E3의 실제 지형

![E3: 사각 블록·요철·파도·경사](ant_six_envs/assets/blocks_eval/preview.png)

실제 생성 함수를 사용한 CPU 메시 표본이다. 높낮이를 읽기 쉽게 세로를 3배 확대했으며 시뮬레이터 영상과 구분한다.

| 요소 | 설정 | 시험하려는 문제 |
| --- | --- | --- |
| 사각 블록 | 칸 폭 0.8m, 높이 변동 3~12cm, 비율 35% | 불연속적인 발 디딤 높이 |
| 요철 | 샘플 −6~+6cm, 양자화 1cm, 샘플 간격 0.5m | 불규칙한 작은 접촉 변화 |
| 파도 | amplitude 설정 8~18cm, 타일당 3/5 waves, 각 10% | 학습의 2/4/6/8 waves와 다른 주기 |
| 경사·역경사 | 8~16°, 각각 10% | 학습 경사 범위 3~12°와 일부 중첩·확장 |

전체는 320×320m다. 평지 전용 타일·중앙 평탄 플랫폼·평탄 테두리는 제외했다. 블록 윗면이나 높이필드의 작은 수평면까지 없다는 의미는 아니다. 이웃 블록의 높이차는 최대 약 24cm일 수 있다. gap처럼 바닥이 뚫린 지형은 아니다.

지형 종류가 완전히 새로운 것만을 unseen으로 간주하지 않았다. 학습과 비슷한 계열도 다른 seed·배치·범위에서 확인했고, 새로운 geometry family는 팀 v2.1의 rails/gap/cylinders에서 별도로 확인했다.

## 3. 자체 평가 결과와 실제 영상

[![최종 정책의 E3-SpawnLift 보행](../artifacts/media/self_eval/E3_SpawnLift_preview.gif)](../artifacts/media/self_eval/E3_SpawnLift.mp4)

앞 6초 미리보기이며 클릭하면 실제 16초 영상을 볼 수 있다. 아래 수치는 화면에 보이는 한 마리의 결과가 아니라 100개 환경의 첫 episode다.

| 지표 | 기존 Failure2 | 최종 Failure2Lift |
| --- | ---: | ---: |
| 원본 보상 mean ± std | 73.75 ± 21.31 | **81.90 ± 17.05** |
| Episode steps mean ± std | 890.92 ± 166.37 | **934.19 ± 95.38** |
| 960-step 생존율 | 82% | **88%** |
| +x 거리 mean ± std | 78.66 ± 22.74m | **82.79 ± 17.96m** |

두 모델 모두 같은 E3-SpawnLift Task, seed 24, 100개 환경, 원본 7항 보상으로 평가했다. 모두 100/100 완료했다. 최종 정책은 실제 종료율이 12%로 남아 있으며, 지형 전체에서 실패가 사라진 것은 아니다. 한 학습 seed의 비교이므로 일반적인 개선이라고 확정하지 않는다.

[기존 모델 원자료](../results/development/failure2_eval_lift/results.json) · [최종 원자료](../results/development/final_lift/results.json) · [최종 터미널 로그](../results/development/final_lift/evaluation.log)

## 4. 코드와 평가 프로토콜

공통 코드 경로: `overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/`.

- E1: `six_terrain_spec.py`, `six_terrain.py`, `ant_six_env_cfg.py`.
- E2: `continuous_eval_spec.py`, `continuous_eval_surface.py`, `ant_continuous_eval_env_cfg.py`.
- E3: `blocks_eval_spec.py`, `block_surface.py`, `ant_blocks_eval_env_cfg.py`.
- 123D 센서 및 SpawnLift: `ant_height_scan_env_cfg.py`.

자체 E3 계열은 지면 상대 높이로 낙상을 판정한다. 팀 v2.1은 원본 world-Z 종료·원본 reset을 강제한다. 이 차이를 평가 설정과 결과 해석에 명시했다. 과제의 기본 보상 요구에 맞춰 **두 경우 모두 원본 Ant 7개 항과 가중치**로 점수를 측정한다.

[제출용 명령어](../submission/evaluation_command.txt) · [팀 공통 결과](RESULTS.md)
