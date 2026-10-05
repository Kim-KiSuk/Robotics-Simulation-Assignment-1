# Ant 팀 공통 평가 환경 v2.1

이 ZIP은 **평가 환경만** 배포한다. 학습 코드, checkpoint, 특정 조원의 관측·행동 구성과 PPO
설정은 포함하지 않는다. 각 조원은 자기 checkpoint와 호환되는 기존 환경 config를 유지하면서
지형·마찰·원본 평가 보상·원본 종료 조건만 평가 wrapper에 연결한다.

> v1과 v2.0은 사용하지 않는다. v1의 경계 문제를 고쳤고, v2.0의 블록형 Medium은 조원 학습
> 지형과 유사해 제외했다. v2.1은 Medium을 세 조원 모두 쓰지 않은 narrow gap으로 교체했다.

## 포함 파일

- `environment/team_eval_terrain_cfg_v2.py`: 고정된 네 지형, seed, 마찰 범위
- `environment/team_eval_material_event.py`: 환경별로 균일한 마찰을 주는 이벤트
- `GPT_EVALUATION_PROMPT_KO.md`: 각자 GPT에 그대로 전달할 작업 지시문
- `result_template.csv`: 결과 취합 형식
- `VALIDATION_REPORT.md`: v2 검증 근거
- `SHA256SUMS.txt`: 배포 파일 무결성 확인용

## 네 환경

| 이름 | 지형 | terrain config seed | 정마찰 / 동마찰 |
| --- | --- | ---: | --- |
| unseen_easy | 높이 4 cm의 낮은 레일 | 53001 | 1.0 / 1.0 |
| unseen_medium | 실제 폭 8.5 cm의 좁은 사각 gap | 53012 | 0.65–1.35 / 0.55–1.15 |
| unseen_hard | 반복 원통, 높이·반경·기울기 변화 | 53003 | 0.35–1.60 / 0.25–1.30 |
| seen_control | 평면·rough·grid·오르막 각 25% | 51004 | 0.50–1.50 / 0.40–1.20 |

`easy/medium/hard`는 지형·마찰 파라미터로 정한 설계 난이도다. 정책마다 특정 장애물에 대한
강점이 달라 중간과 어려움의 reward 순서가 뒤집힐 수 있으며, 이는 오류가 아니라 분석할 결과다.

재현성에는 terrain config seed와 CLI의 `--seed 24`가 모두 필요하다. 일부 Isaac Lab mesh
generator의 세부 배치는 환경 전역 난수도 사용하므로 CLI seed를 바꾸면 같은 config라도 배치가
달라질 수 있다. 배포 검증은 과제 지정값 `--seed 24`에서 두 번 생성해 geometry hash가 일치했다.

모든 지형은 8 m × 8 m patch, 40행 × 20열, 20 m border다. 평가 wrapper는 반드시
`max_init_terrain_level=4`로 제한한다. 런타임 검증에서 origin x는 -156~-124 m였고 +x 외곽은
+180 m이므로 최소 전진 여유는 304 m다.

## 공통 평가 규약

- 명령 인자: `--seed 24 --num_envs 100 --headless`
- 첫 episode만 집계
- episode: 16초, 최대 960 policy steps
- 종료: time-out 또는 원본 world-frame torso 높이 0.31 m 미만
- 평가 보상: 원본 `Isaac-Ant-v0`의 7개 항
  - progress `1.0`
  - alive `0.5`
  - upright `0.1`
  - move_to_target `0.5`
  - action_l2 `-0.005`
  - energy `-0.05`
  - joint_pos_limits `-0.1`
- 필수값: episode reward total mean/std
- 진단값: steps mean/std, 960-step survival, final +x displacement mean/std
- 평가 중 curriculum 비활성화

평가 wrapper의 보상은 원본 `ant_env_cfg.RewardsCfg`, 종료는 원본 `TerminationsCfg`로 교체한다.
반면 observation 순서·차원/history, action, robot/actuator/sensor, policy architecture와 runner는
checkpoint 학습 당시 설정을 그대로 유지한다.

## 실행 형식

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task <조원별_평가_TASK_ID> \
  --seed 24 \
  --num_envs 100 \
  --checkpoint <CHECKPOINT_절대경로> \
  --headless
```

네 환경 모두 `Completed first episodes: 100/100`인지 확인한다. 실행 전 checkpoint SHA-256을
동결하고, terminal log, 결과 CSV, 실제 task ID, observation/action 차원과 활성 reward 목록을
함께 공유한다.

## 공정성

세 조원의 최종 checkpoint와 SHA-256을 먼저 고정한 뒤 v2를 배포한다. 결과를 본 뒤 checkpoint를
고르거나 추가 학습하면 이 세 환경은 최종 unseen test가 아니라 validation set이 된다. v2 평가 뒤
학습한 모델은 기존 결과와 같은 표에서 공정한 unseen 비교로 표시하지 않는다.
