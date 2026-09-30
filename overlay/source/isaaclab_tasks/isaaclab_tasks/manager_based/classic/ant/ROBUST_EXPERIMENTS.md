# 보상·관측 개선 실험

현재 확보한 B(험지)와 C(험지+마찰 무작위화)의 결과에서는 B가 기본/고마찰
조건에서 더 좋다. 따라서 다음 실험은 B에서 시작하며, 기존 A/B/C 체크포인트와
설정은 그대로 보존한다. 아래 변경은 **개선 가설**이고, 학습 후 성능 검증이 필요하다.

| 실험 | Task | 관측 | 학습 보상 | 로그 폴더 |
|---|---|---|---|---|
| B | `Isaac-Ant-Rough-v0` | 60 | 원본 7개 항목 | `ant_rough` |
| R | `Isaac-Ant-Rough-Reward-v0` | 60 | 원본 + 추가 3개 항목 | `ant_rough_reward` |
| H | `Isaac-Ant-Rough-History-v0` | 240 | 원본 | `ant_rough_history` |
| HR | `Isaac-Ant-Rough-History-Reward-v0` | 240 | 원본 + 추가 3개 항목 | `ant_rough_history_reward` |

PPO 설정, 제어 주기, 행동 8차원, 로봇 물리, 지형, 마찰 1.0, 초기화, 종료 조건은
B와 같다. H/HR은 입력 차원이 늘어나므로 첫 네트워크 층의 파라미터 수는 증가한다.

## 보상 설계

- **실패 -5**: 지면 기준 낙상/지도 이탈로 종료하면 한 번 부과한다. 시간 제한만
  도달한 경우에는 부과하지 않는다. 마지막 스텝의 낙상도 실패로 센다.
  RewardManager가 `dt`를 곱하므로 함수에서 `1/dt`로 보정한다.
- **낮은 몸통 높이**: 지면 상대 높이 0.40m 이상에서는 0, 0.31m 이하에서는
  최대 1인 제곱 위험도에 가중치 -0.5를 적용한다. 매 스텝 `dt`가 곱해진다.
  너무 큰 가중치는 낮은 자세의 유효한 보행을 억제할 수 있다.
- **동작 변화량**: `sum((action_t - action_{t-1})**2)`에 -0.002를 적용한다.
  매 스텝 `dt`가 곱해진다. 급격한 힘 명령을 줄이는 가설이며, 너무 키우면
  지형 대응과 전진 속도를 떨어뜨릴 수 있다.

기존 progress/alive/upright/heading/effort/energy/joint-limit 보상을 유지한다.
속도 상한을 새로 넣지 않는다. 가중치는 초기 실험값이지 최적값이 아니다.

## 관측 설계

기존 60차원 상태의 현재 시점과 이전 3개 시점을 사용한다. 60Hz 제어에서
현재부터 약 0.05초 전까지의 정보이며 총 240차원이다. 지형/마찰 정답값을
추가하지 않고, 속도·접촉 반응의 변화를 정책이 활용하도록 한다.

IsaacLab의 history는 **관측 항목마다 시간축을 쌓은 후 항목을 연결**한다.
`[전체 상태 t-3, 전체 상태 t-2, ...]` 순서라고 가정해 직접 구현하면 안 된다.
환경별 reset 때 history가 초기화된다. H/HR을 제출한다면 평가 측에도 이
관측 전처리/버퍼가 필요하다. 60차원만 받는 고정 평가 인터페이스에는 호환되지 않는다.
평가 측의 코드 적용 범위가 불확실하면 R 또는 B를 제출 후보로 유지한다.

## 실행 순서

현재 `IsaacLab_RS`의 활성 Python 환경에서 실행한다.

```bash
./isaaclab.sh -p scripts/environments/check_ant_robust.py --headless

# R: 보상만 변경. 기존 결과와 동일한 학습량으로 새 학습을 시작한다.
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Rough-Reward-v0 --headless \
  --num_envs 4096 --max_iterations 1000 --seed 42 --run_name reward_seed42 \
  env.scene.terrain.terrain_generator.seed=42

# H: 관측 이력만 변경
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Rough-History-v0 --headless \
  --num_envs 4096 --max_iterations 1000 --seed 42 --run_name history_seed42 \
  env.scene.terrain.terrain_generator.seed=42

# HR: 각각의 결과를 확인한 뒤 결합 실험
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Rough-History-Reward-v0 --headless \
  --num_envs 4096 --max_iterations 1000 --seed 42 --run_name history_reward_seed42 \
  env.scene.terrain.terrain_generator.seed=42
```

위 명령은 순차 실행한다. H/HR에 B/C 체크포인트를 `--resume`으로 넣으면 입력
차원이 맞지 않는다. R도 공정한 학습량 비교를 위해 우선 처음부터 학습한다.
미세 조정 실험을 따로 한다면 B의 추가 학습 대조군을 함께 둔다.

## 공통 기준으로 평가

`evaluate_ant.py`는 학습 보상과 관계없이 **원본 7개 보상으로 평가**한다.
reward, episode steps, 시간 제한까지 실패 없이 생존한 비율, 목표 방향 순이동
거리/속도, 지도 ray 누락 횟수와 환경별 원자료를 JSON으로 저장한다.
거리 계산은 자동 reset 이전 위치를 사용한다. 기존 JSON은 덮어쓰지 않는다.

```bash
# 실제 학습 폴더 경로를 넣는다. B/C/R은 모두 아래 60차원 평가 Task 사용 가능.
CKPT='logs/rsl_rl/ant_rough_reward/실제_학습폴더/model_999.pt'
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate_ant.py \
  --task Isaac-Ant-Rough-v0 --checkpoint "$CKPT" \
  --headless --num_envs 100 --seed 24 \
  --output logs/ant_robust_eval/R_default_terrain2003_seed24.json \
  env.scene.terrain.terrain_generator.seed=2003 \
  env.scene.terrain.physics_material.static_friction=1.0 \
  env.scene.terrain.physics_material.dynamic_friction=1.0
```

H/HR은 평가 Task를 `Isaac-Ant-Rough-History-v0`로 설정한다. 출력 이름도
모델/마찰/seed에 맞게 바꾼다. low는 `(0.4, 0.3)`, default는 `(1.0, 1.0)`,
high는 `(1.5, 1.2)`이다. 이는 **지면 재질 계수**이며 로봇 발 재질과 average
방식으로 합성되므로 그대로 접촉면의 최종 마찰계수라고 해석하지 않는다.

기존 지형 seed 2001/2002는 이미 결과를 보고 방법 선택에 사용했으므로 validation으로
취급한다. 2003/2004/2005 등은 방법을 확정한 뒤 별도 held-out 평가에 사용한다.
새 seed만으로 학습 분포 밖 지형 검증이 되지는 않는다. 더 큰 요철/경사와
학습에 없던 지형 형상은 별도 조건으로 추가하고 A/B/C/R/H/HR에 똑같이 적용한다.
최종적으로 여러 학습 seed를 반복해야 학습 운에 따른 차이도 구분할 수 있다.

학습 보상이 다른 R/HR의 TensorBoard reward를 B/C와 직접 비교하지 않는다.
생존율만 높고 전진하지 않는 경우도 있으므로 공통 reward, 이동, 생존을 함께 본다.
낙상률에는 지도 이탈도 포함될 수 있으므로 `missing_ground_episodes`를 확인한다.

## 현재 권장 우선순위

1. A를 B/C와 **같은 험지**에서도 평가하여 평지 수치와 험지 수치의 혼합 비교를 피한다.
2. R과 H를 각각 학습하고 validation 조건에서 B와 비교한다.
3. 효과가 있는 방법만 결합하고 새로운 held-out 지형 seed/형상에서 평가한다.
4. C는 저마찰에서 B와 거의 같고 고마찰에서 더 낮았다. 이 결과만으로 무작위화가
   무효라고 단정하지 않는다. C의 넓은 마찰 범위와 고정 공간 분포가 학습 난도를
   높였는지, 좁은 범위/더 긴 학습/다른 friction seed를 후속 실험으로 검증한다.
