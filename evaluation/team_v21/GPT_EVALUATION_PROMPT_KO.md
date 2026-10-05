# 아래 내용을 각자 GPT에 그대로 전달하세요

나는 Isaac Lab의 `Isaac-Ant-v0` 기반 RSL-RL checkpoint를 가지고 있다. 첨부된 v2.1 배포본의
`team_eval_terrain_cfg_v2.py`와 `team_eval_material_event.py`를 수정하지 말고, 내 checkpoint 전용
평가 task 네 개를 만들어 과제 지정 스크립트로 평가해 줘.

내 정보:

- IsaacLab 프로젝트: `<ISAACLAB_RS_절대경로>`
- 기존 학습 task ID: `<기존_TASK_ID>`
- 기존 env config 파일/class: `<ENV_CFG_파일과_CLASS>`
- RSL-RL runner config 파일/class: `<RUNNER_CFG_파일과_CLASS>`
- checkpoint: `<CHECKPOINT_절대경로>`

반드시 다음 규칙을 지켜 줘.

1. 먼저 checkpoint SHA-256, Git 상태, observation/action 차원과 순서, episode 길이, termination,
   reward, robot/actuator/sensor 설정을 읽기 전용으로 확인해. 학습이나 fine-tuning은 실행하지 마.
2. 배포본의 `SHA256SUMS.txt`를 검증하고 v2.1인지 확인해. v1/v2.0 파일·seed·task를 사용하지 마.
3. 기존 env config를 상속하는 평가 전용 wrapper를 새 파일로 작성하고 기존 학습 config를 직접
   덮어쓰지 마.
4. 지형은 배포본의 네 `TEAM_V2_*_CFG`를 쓰고 `max_init_terrain_level=4`로 고정해. 이 값이 없거나
   `None`이면 맵 끝과 가까운 행에 스폰될 수 있으므로 전체 평가를 시작하지 마.
5. wrapper에서 reward는 원본 `ant_env_cfg.RewardsCfg`, termination은 원본 `TerminationsCfg`,
   episode는 16초, training curriculum은 비활성화해.
6. checkpoint 호환을 위해 학습 당시 observation 차원·순서/history, action 차원·scale,
   robot/actuator/sensor, policy architecture와 runner config는 그대로 유지해. 원본 60차원 관측으로
   임의 변경하지 마.
7. ground material은 static/dynamic 1.0, combine mode `multiply`, restitution 0.0으로 둬. 배포된
   이벤트로 한 env 안의 모든 robot collision shape가 같은 material pair를 쓰게 해.
8. 충돌하지 않는 새 Gym task ID 네 개를 등록하고 `list_envs.py`에서 확인해.
9. 각 task를 `--num_envs 2`로 reset + 1 step smoke test해. 관측 shape mismatch, checkpoint key
   오류, NaN, terrain 생성 오류가 있으면 100환경 평가 전에 멈추고 수정해.
10. smoke test 뒤 네 task를 각각 아래 형식으로 한 번 평가해. task/checkpoint 외에는
    `--seed 24 --num_envs 100 --headless`를 고정해.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task <평가_TASK_ID> \
  --seed 24 \
  --num_envs 100 \
  --checkpoint <CHECKPOINT_절대경로> \
  --headless
```

11. `play_one_episode.py`는 각 env의 첫 episode만 집계하고 원본 7항 reward 누적값을 출력해야 해.
    `Completed first episodes: 100/100`, reward mean/std, steps mean/std, 960-step survival,
    가능하면 final +x mean/std를 보존해.
12. 결과를 `result_template.csv`에 적고 terminal log, 변경 파일, 실제 task ID, observation/action
    차원, 활성 reward 7개, checkpoint hash를 함께 정리해.
13. 평가 중 seed·난이도·reward를 바꾸거나 checkpoint를 학습하지 마. Git commit/push도 하지 마.

완료 보고에서 단순 Isaac Sim 경고와 평가 무효화 오류를 구분하고, 경계 검증을 위해 실제
`env_origins[:,0]` 최댓값이 -124 m 이하인지도 확인해 줘.
