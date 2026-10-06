# 과제 조건 대조와 LMS 제출물

기준은 사용자가 제공한 과제 공지와 추가 평가 안내다. GitHub 자료를 준비한 상태이며 **LMS 업로드 완료를 뜻하지 않는다**.

| 요구사항 | 확인한 내용 | 근거/제출 파일 |
| --- | --- | --- |
| Isaac-Ant-v0 기반, 미지 지형 보행 | 원본 Ant에서 지형·마찰·관측·학습 보상을 설계 | [연구 문제·가설](METHOD.md), [구현](IMPLEMENTATION.md) |
| 자체 새로운 환경 구성·평가 | E1~E3를 학습 메시와 분리; E3-SpawnLift에서 공통 비교 | [자체 평가맵](SELF_EVALUATION.md) |
| 학습 코드·환경 코드·가중치 포함 프로젝트 | 고정 원본 + overlay + 최종 가중치가 들어 있는 프로젝트 압축본 | [IsaacLab_RS_final.tar.gz](../artifacts/project/IsaacLab_RS_final.tar.gz), [구성·해시](../artifacts/project/manifest.json) |
| 실행 명령어 별도 제출 | PDF 7쪽 형식의 `play_one_episode.py` 직접 실행 | [evaluation_command.txt](../submission/evaluation_command.txt) |
| seed 24, num_envs 100 | 자체 최종 평가와 팀 공통 평가 모두 사용 | [자체 JSON](../results/development/final_lift/results.json), [팀 CSV](../results/team_v21/final/result_template.csv) |
| 100개 첫 episode mean/std | 원본 누적 보상·steps 평균/표준편차를 표와 PPT에 기재 | [결과](RESULTS.md) |
| 학습 보상 수정 시 원본 보상으로 평가 | 평가 reward는 7항과 원본 가중치; 학습 안정성 비용 제외 | 평가 env.yaml 및 `RewardsCfg` |
| 하드웨어 구조 변경 금지 | 원본 robot·관절·actuator 설정 유지, USD 수정 없음 | 원본 `ant_env_cfg.py` 해시와 overlay 동일성 검사 |
| 센서 추가 허용 | Python RayCaster로 63개 높이 추가; USD 변경 없음 | `ant_height_scan_env_cfg.py` |
| 연구 문제 → 가설 → 실험 → 분석 | 지형 다양화·관측·안정성·초기 접촉을 구분해 설명 | [Method](METHOD.md), [Results](RESULTS.md) |
| 독창성 | 통통 튐 관찰에서 출발, 움직임 억제 실패와 소환 겹침 진단을 구분 | 자체 관찰·원인 가설·동일 조건 비교를 PPT에 제시; 점수 보장 아님 |
| 팀 v2.1은 자체 공통 비교이며 조교 평가 수치가 아님 | README 및 PPT 명시 |

