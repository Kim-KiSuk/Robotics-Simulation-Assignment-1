# 설치와 제출용 평가

## 프로젝트 준비

[IsaacLab_RS_final.tar.gz](../artifacts/project/IsaacLab_RS_final.tar.gz)는 수업 원본의 실행 소스와 과제 overlay, 최종 가중치, 팀 v2.1 배포본을 포함한 프로젝트 폴더다. [manifest](../artifacts/project/manifest.json)에 원본 commit과 파일별 SHA-256을 기록했다. Isaac Sim과 다운로드되는 로봇 USD, conda 의존성은 별도 설치한다.

1. 압축을 홈 디렉터리에 풀어 `~/IsaacLab_RS_final/isaaclab.sh`가 있는 상태로 만든다.
2. 기존 `lerobot-arena` 환경을 활성화한다.
3. 프로젝트 루트에서 `./isaaclab.sh -i rsl_rl`로 이 checkout의 editable package 경로를 설치한다.
4. 아래 제출용 명령을 실행한다. 이미 다른 IsaacLab checkout이 설치되어 있으면 해당 경로를 확인한다.

실제 실행 환경은 Python 3.11, NVIDIA RTX 2080 8GB, 수업 원본 commit `e83a5d2f11ca1b5f03b690e1978479e620c500e2`다. [로컬 패키지 버전](../configs/software_versions.json)을 함께 제공한다.

## 제출용 평가 명령 — 자체 E3-SpawnLift

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS_final

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-HeightScan-SpawnLift-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint artifacts/checkpoints/final_lift/model_5999.pt \
  --headless --diagnostics --video --video_length 960 \
  --video_folder logs/final_self_eval/video
```

이 명령은 원본 7항 보상으로 첫 episode를 집계한다. 터미널의 **Completed first episodes: 100/100**, 누적 reward mean/std, steps mean/std를 확인한다. 녹화 파일은 `logs/final_self_eval/video/rl-video-step-0.mp4`에 저장된다. 카메라 입력을 정책에 새로 추가하는 것이 아니라 화면 녹화만 켠다.

[명령어 TXT](../submission/evaluation_command.txt) · [평가맵 설정](SELF_EVALUATION.md) · [기존 측정 결과](../results/development/final_lift/results.json)

## 기존 checkout에 적용할 경우

프로젝트 압축본 대신 고정된 원본 checkout을 준비하고 루트의 `apply_overlay.py`를 사용해도 된다. `--target`은 대상 checkout, `--check`는 쓰기 전 검사다. 예상하지 못한 로컬 변경은 덮어쓰지 않는다. 이 경우 최종 체크포인트 위치를 공개 저장소의 `artifacts/checkpoints/final_lift/model_5999.pt` 절대경로로 맞춘다.

## 팀 공통 v2.1 재현

팀 공통 네 환경은 `scripts/evaluate_final.sh`에 묶어 두었다. 별도 checkout을 가리키는 `ISAACLAB_ROOT`를 설정하고 공개 저장소 루트에서 해당 스크립트를 실행한다. `--video` 옵션은 영상 저장을 켠다. 최초 측정과 영상 반복은 이미 [results/team_v21](../results/team_v21)에 보존되어 있다. 이 자료는 조교의 공식 unseen 평가 결과가 아니다.

## 실행 기록과 자료 검증

학습 방법은 [Method](METHOD.md), 실제 설정은 [env.yaml](../configs/final/env.yaml)과 [agent.yaml](../configs/final/agent.yaml), 당시 학습 명령 원문은 [training_command.txt](../configs/final/training_command.txt)에 보존했다. 제출된 정책을 평가하기 위해 재학습할 필요는 없다.

`python scripts/verify_publication.py`로 해시·집계·문서 링크를 확인할 수 있다. 그림은 `build_figures.py`, 영상 미리보기는 `build_previews.py`, 발표 파일은 `build_presentation.py`로 재생성한다. `build_training_curve.py`와 TensorBoard는 원본 학습 기록 재검토용이며 학습 reward를 평가 reward와 혼동하지 않는다.

프로젝트 패키지와 공개 경로 wrapper는 파일·CPU·shell 수준에서 검사했다. 새 시뮬레이션을 실행하지 않았으며 실제 보행 근거는 저장된 사용자 평가 로그·영상이다.
