# 검증 기록 (2026-10-01)

본 기록은 코드의 동작 검증이다. R/H/HR의 보행 성능이 B/C보다 향상되었다는
증거로 사용하지 않는다. R/H/HR의 1000-iteration 본 학습은 아직 수행하지 않았다.

## 통과한 검사

1. `check_ant_robust.py --headless`: A/B/C 설정 보존, 새 Task 3개 등록,
   PPO/물리/종료/초기화 동일성 확인. 4개 환경, 4×4 타일의 작은 지도에서
   240차원 관측과 유한한 보상, 강제 낙상, 자동 reset 전 위치 기록,
   reset된 환경의 관측 이력 초기화, 시간 제한과 실패 페널티 구분을 확인했다.
   [로그](validation/robust_check.log)
2. HR의 `train.py --num_envs 16 --max_iterations 2 --seed 42` 실행:
   4×4 타일에서 1024 transitions, PPO 업데이트 2회와 checkpoint 저장을 확인했다.
   [로그](validation/hr_train_smoke.log)
   이 작은 검증용 checkpoint는 성능 제출 모델에 포함하지 않았다.
3. 새 평가기로 기존 B checkpoint를 **원래의 전체 지도**에서 평가:
   terrain seed 2001, reset seed 24, default friction, 100 environments,
   원본 16초 episode. 기존 보상과 steps의 평균/표준편차가 1e-6 이내로 일치했다.
   [JSON 원자료](common_evaluation/B_default_terrain2001_seed24.json)
   및 같은 폴더의 resolved YAML/실행 로그를 포함한다.
4. HR 검증용 checkpoint를 새 평가기로 실행: 240차원 관측으로 4/4 episodes 완료.
   보상은 원본 7개 항목으로 변경됨을 확인했다. **0.2초/12스텝의 연결 검사**이므로
   이 실행의 생존율 100%는 성능 지표가 아니다.
   [JSON](validation/HR_interface_smoke.json), [로그](validation/hr_eval_smoke.log)
5. 기존 지면 마찰 샘플러의 CPU 테스트 4개 통과.
6. 제출 Python 전체 문법 검사, 평가 matrix Bash 문법 검사 통과.
7. 임시 원본 checkout에서 overlay dry-run/실제 적용/재실행을 검사했다.
   모든 payload SHA-256이 일치했고 로컬 변경을 발견하면 덮어쓰지 않고 중단했다.
8. 원본 평가 로그 15개가 모두 100/100 완료임을 확인하고 CSV 통계를 재생성했다.

## B 재평가에서 추가로 기록한 값

해당 **한 조건**의 결과이며 B의 모든 지형·마찰 조건 평균이 아니다.

| 지표 | 결과 |
|---|---:|
| Reward 평균 ± 표준편차 | 98.739225 ± 27.950061 |
| Episode steps 평균 ± 표준편차 | 900.730000 ± 207.360838 |
| 960스텝까지 실패 없이 생존 | 92/100 |
| 실패 종료 | 8/100 |
| 지면 ray 누락 episode | 0/100 |
| 목표까지 거리 감소 평균 | 95.488048m |
| episode별 목표 방향 평균 속도의 평균 | 6.053169m/s |

거리에는 종료 스텝을 포함하고 자동 reset 뒤 시작점으로 돌아간 위치는 포함하지 않는다.
속도는 환경별 `목표 거리 감소 / 실제 episode 시간`을 평균낸 값이다.

## 실행 환경과 한계

- 로컬 수업 환경: Isaac Sim 5.1, IsaacLab 2.3 계열, Python 3.11, RTX 2080 8GB.
- 사용자 영상 녹화와 최초 GPU 검증이 겹쳤을 때 실제 CUDA OOM이 발생했다.
  녹화를 종료하지 않았으며, 이후 GPU 여유가 생긴 상태에서 위 검사를 완료했다.
- 체크포인트 3개와 로그를 보존했다. 새 방법의 본 학습, 여러 학습 seed 반복,
  새로운 지형 형상/학습 범위 밖 지형 평가는 후속 작업이다.
- H/HR 제출 시 평가 측에 240차원 history 전처리를 적용할 수 있는지 확인해야 한다.
  R은 B와 같은 60차원 입력을 유지한다.
