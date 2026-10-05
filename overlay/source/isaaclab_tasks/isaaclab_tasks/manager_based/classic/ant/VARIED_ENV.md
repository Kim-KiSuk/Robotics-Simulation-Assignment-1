# 지형 v2: 평지 타일 없이 다양한 요철·파도·경사

현재의 완만한 험지에서 학습한 정책이 더 큰 지형 변화에 대응하는지 평가하고,
필요하면 같은 새 지형에서 다시 학습할 수 있도록 별도 Task를 추가했다.
기존 `Isaac-Ant-Rough-*` 환경과 모델은 유지된다.

## 변경한 지형

| 종류 | 생성 확률 | 설정 |
|---|---:|---|
| 작은 요철 | 15% | 높이 샘플 −4~+4cm, 샘플 간격 0.8m |
| 중간 요철 | 15% | 높이 샘플 −8~+8cm, 샘플 간격 0.8m |
| 큰 요철 | 10% | 높이 샘플 −12~+12cm, 샘플 간격 1.2m |
| 긴 파도 | 15% | amplitude 0.06~0.18m, 타일당 파도 2개 |
| 짧은 파도 | 15% | amplitude 0.06~0.18m, 타일당 파도 4개 |
| 경사 | 15% | rise/run = tan(6°)~tan(18°) |
| 역경사 | 15% | 같은 범위, 방향 반전 |

평지 타일은 **0%**이고 경사 중앙의 평평한 플랫폼 설정도 0m로 줄였다.
단, 높이 양자화와 생성기의 타일 경계 처리에 따른 국소 평탄 구간은 존재한다.
확률은 타일 생성 확률이며 로봇이 실제로 경험하는 시간 비율과 같지는 않다.

요철은 세 단계 중에서 선택되고, 파도 높이와 경사 파라미터는 타일마다 뽑힌
difficulty에 따라 범위 내에서 달라진다. 요철의 spline 보간은 샘플 범위를
넘는 높이를 만들 수 있다. CPU 검사(seed 42, 중간 difficulty)의 큰 요철은
실제 메시 높이가 약 −20.5~+16.5cm였다. 파도 amplitude는 봉우리~골 높이차가
아니며, 경사 각도는 파라미터 환산값으로 메시의 모든 면이 동일 각도라는 뜻이 아니다.

지도 크기 528×528m, 타일 12×12m, 수평 해상도 0.4m, 로봇 배치 5m,
행동·관측 의미·물리·초기화·낙상 기준·episode 길이는 기존 B와 같다.
기본 마찰은 1.0이고 바닥의 시각 재질만 회색으로 바꿨다. 카메라는 정책 입력이 아니다.
현재 열어둔 공용 `terrains/config/rough.py` 대신 Ant 전용 설정을 사용한다.

## Task와 모델 구분

| Task | 용도 | 관측 | 새 학습 로그 |
|---|---|---:|---|
| `Isaac-Ant-Varied-v0` | 기존 A/B/C/R 재평가, 새 지형 기본 PPO 학습 | 60 | `ant_varied` |
| `Isaac-Ant-Varied-Reward-v0` | 새 지형에서 R 보상으로 학습 | 60 | `ant_varied_reward` |
| `Isaac-Ant-Varied-History-v0` | 기존 H/HR 재평가, 새 지형에서 H 학습 | 240 | `ant_varied_history` |
| `Isaac-Ant-Varied-History-Reward-v0` | 새 지형에서 HR 학습 | 240 | `ant_varied_history_reward` |

새 환경에서 기존 모델을 평가하는 데 재학습은 필요 없다. H/HR에는 반드시
240차원 관측 Task를 사용한다. A를 새 지형에서 평가할 때 높이는 공통 지면 상대
높이로 입력되므로, A의 원래 평지 높이 관측과 의미가 다르다는 점을 결과에 명시한다.

## 검증 상태와 실행 전 확인

CPU 검사에서 실제 IsaacLab 높이필드 생성 함수를 사용해 7종류 × 3개 난이도의
21개 메시를 생성했다. 유한한 좌표·0이 아닌 높낮이·정상 삼각형 면적·seed 재현성을
확인했다. 같은 크기의 전체 지도는 기존과 동일한 3,484,800개 삼각형이다.
seed 42/4001/4002의 타일 선택에서 평지 타입이 없는 것도 확인했다.

작성 시점에 기존 H가 GPU 약 6.6GB를 사용해 학습 중이므로 **새 Task의 PhysX 실행과
모델 재평가는 아직 수행하지 않았다.** H 학습과 GUI를 종료한 뒤 아래 순서로 실행한다.
CPU 검사가 시뮬레이션의 보행 성능을 검증하는 것은 아니다.

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

# CPU 메시 검사: 시뮬레이션/GPU 실행 없음
python scripts/environments/inspect_ant_varied.py

# GPU가 비어 있을 때: 새 task의 로봇 배치·센서·관측·reset 검사
./isaaclab.sh -p scripts/environments/check_ant_varied.py --headless
./isaaclab.sh -p scripts/environments/check_ant_varied.py \
  --headless --task Isaac-Ant-Varied-History-v0
```

두 물리 검사에서 마지막 `[CHECK] PASS`가 확인되면 모델 평가를 진행한다.

## 기존 모델을 새 맵에서 다시 평가

새 지형 seed 4001/4002 × 마찰 default/low/high × 각 100개 환경을 사용한다.
환경별 첫 에피소드 전체를 원본 7개 공통 보상으로 평가하며 생존율·이동 거리도 기록한다.
마찰은 각각 정적/동적 `(1.0,1.0)`, `(0.4,0.3)`, `(1.5,1.2)`이다.
아래 경로는 현재 작업 중인 `IsaacLab_RS` 기준이다.

```bash
B_CKPT="ant_submission/artifacts/checkpoints/rough/ant_rough.pt"
R_CKPT="logs/rsl_rl/ant_rough_reward/2026-10-01_01-34-45_reward_seed42/model_999.pt"

bash ant_submission/evaluate_matrix.sh B Isaac-Ant-Varied-v0 \
  "$B_CKPT" logs/ant_varied_eval 4001 4002

bash ant_submission/evaluate_matrix.sh R Isaac-Ant-Varied-v0 \
  "$R_CKPT" logs/ant_varied_eval 4001 4002

# 기존 H 학습이 끝나고 model_999.pt가 생성된 후
H_CKPT="logs/rsl_rl/ant_rough_history/2026-10-01_13-15-55_history_seed42/model_999.pt"
bash ant_submission/evaluate_matrix.sh H Isaac-Ant-Varied-History-v0 \
  "$H_CKPT" logs/ant_varied_eval 4001 4002

rg 'Completed first episodes:|\[RESULT\]' logs/ant_varied_eval
```

기존 결과와 혼합하지 않도록 별도 폴더를 사용한다. 이미 실행한 결과/로그가 있으면
덮어쓰지 않고 중단하므로 재실행은 새로운 출력 폴더를 지정한다.
A/C도 해당 모델 체크포인트와 이름을 넣고 `Isaac-Ant-Varied-v0`로 동일하게 평가할 수 있다.

## 새 지형에서 다시 학습

먼저 기존 모델의 새 지형 성능을 기록하고, 새 지형으로 학습한 모델과 비교하면
학습 분포를 넓힌 효과를 볼 수 있다. 아래 명령은 B2, 즉 새 지형의 기본 PPO다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Varied-v0 --headless \
  --num_envs 4096 --max_iterations 1000 --seed 42 --run_name varied_seed42 \
  env.scene.terrain.terrain_generator.seed=42
```

R2/H2/HR2는 위 표의 학습 Task와 구별되는 `--run_name`을 사용한다.
`--resume` 없이 같은 예산으로 처음부터 학습한다. 평가할 때는 B2/R2에
`Isaac-Ant-Varied-v0`, H2/HR2에 `Isaac-Ant-Varied-History-v0`를 사용한다.
기존 모델과 구분되도록 평가 이름도 B2/R2/H2/HR2로 적는다.

4001/4002는 후속 방법 선택을 위한 validation 조건으로 사용한다. 최종 후보를
정한 후 별도의 seed 7001/7002 등과 새로운 지형 형상에서 검증한다. 새 지형으로
재학습한 뒤에는 이 지형 분포 자체를 unseen이라고 부르지 않는다.
