# 설치와 제출용 평가

## 프로젝트 준비

[IsaacLab_RS_final.tar.gz](../artifacts/project/IsaacLab_RS_final.tar.gz)는 수업 원본의 실행 소스와 과제 overlay, 최종 가중치, 팀 v2.1 배포본을 포함한 프로젝트 폴더다. [manifest](../artifacts/project/manifest.json)에 원본 commit과 파일별 SHA-256을 기록했다. Isaac Sim과 다운로드되는 로봇 USD, conda 의존성은 별도 설치한다.

저장소를 내려받는 것과 내부 실행 프로젝트를 압축 해제하는 것은 **별도 단계**다. GitHub Download ZIP을 풀면 배포 저장소만 생기며, 그 안의 `artifacts/project/IsaacLab_RS_final.tar.gz`도 풀어야 한다.

처음 다운로드부터 평가까지의 LMS 제출용 명령은 [evaluation_command.txt](../submission/evaluation_command.txt)에 있다. 이 TXT의 `"<평가 환경>"`은 조교가 제공하는 실제 평가 Task ID로 대체해야 한다. 해당 Task는 프로젝트에 별도 등록되어 있고 최종 모델의 123D 관측·8D 행동·센서 구성과 호환되어야 한다. 비공개 평가 환경은 배포본에 포함되어 있지 않다. 이미 clone하거나 Download ZIP을 푼 경우에는 다음 준비 명령을 사용한다.

```bash
conda activate lerobot-arena &&
(
  set -e
  # 아래 경로를 README.md와 artifacts 폴더가 있는 실제 저장소 위치로 바꾼다.
  cd "/절대경로/Robotics-Simulation-Assignment-1"
  python scripts/prepare_project.py --destination "$HOME/IsaacLab_RS_final"
  cd "$HOME/IsaacLab_RS_final"
  ./isaaclab.sh -i rsl_rl
)
```

`prepare_project.py`는 내부 tar.gz와 파일별 SHA-256을 검사한 뒤 압축을 푼다. 같은 파일이 이미 있으면 재사용하고, 다른 파일이 있으면 덮어쓰지 않고 중단한다. 이때 다른 목적지를 선택하고 아래 평가 명령의 `cd`도 같은 위치로 바꾼다. 학습이나 평가, 패키지 설치를 자동 실행하지는 않는다.

`./isaaclab.sh -i rsl_rl`은 해당 conda 환경의 IsaacLab editable package 경로를 이 프로젝트로 연결한다. 다른 checkout의 Python 코드를 잘못 불러오는 것을 막기 위해 최초 준비 시 실행한다. 기존 수업 환경에서 수행하며, 새 conda 환경에 Isaac Sim까지 설치해 주는 명령은 아니다. 설치 중 오류가 나면 평가로 넘어가지 말고 먼저 해결한다.

저장소 대신 **내부 tar.gz만 직접 다운로드**한 경우에는 다음처럼 해제한다. 빈 목적지에서 실행하며 기존 폴더는 덮어쓰지 않는다.

```bash
(
  set -e
  test ! -e "$HOME/IsaacLab_RS_final"
  tar -xzf "$HOME/Downloads/IsaacLab_RS_final.tar.gz" -C "$HOME"
)
```

그 다음 같은 conda 환경에서 `cd ~/IsaacLab_RS_final`, `./isaaclab.sh -i rsl_rl`을 실행한다. 아래 평가 명령은 **이 준비가 끝난 뒤** 실행한다.

실제 실행 환경은 Python 3.11, NVIDIA RTX 2080 8GB, 수업 원본 commit `e83a5d2f11ca1b5f03b690e1978479e620c500e2`다. [로컬 패키지 버전](../configs/software_versions.json)을 함께 제공한다.

## 자체 평가 재현 명령 — E3-SpawnLift

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS_final

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Six-Eval-Blocks-HeightScan-SpawnLift-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint artifacts/checkpoints/final/ant_final.pt \
  --headless --diagnostics --video --video_length 960 \
  --video_folder logs/final_self_eval/video
```

이 명령은 원본 7항 보상으로 첫 episode를 집계한다. 터미널의 **Completed first episodes: 100/100**, 누적 reward mean/std, steps mean/std를 확인한다. 녹화 파일은 `logs/final_self_eval/video/rl-video-step-0.mp4`에 저장된다. 카메라 입력을 정책에 새로 추가하는 것이 아니라 화면 녹화만 켠다.

[LMS 제출용 준비·평가 TXT](../submission/evaluation_command.txt) · [준비 완료 후 자체 E3 평가 TXT](../submission/evaluation_only.txt) · [평가맵 설정](SELF_EVALUATION.md) · [기존 측정 결과](../results/development/final_lift/results.json)

## 기존 checkout에 적용할 경우

프로젝트 압축본 대신 고정된 원본 checkout을 준비하고 루트의 `apply_overlay.py`를 사용해도 된다. `--target`은 대상 checkout, `--check`는 쓰기 전 검사다. 예상하지 못한 로컬 변경은 덮어쓰지 않는다. 이 경우 최종 체크포인트 위치를 공개 저장소의 `artifacts/checkpoints/final/ant_final.pt` 절대경로로 맞춘다.

## 팀 공통 v2.1 재현

팀 공통 네 환경은 `scripts/evaluate_final.sh`에 묶어 두었다. 별도 checkout을 가리키는 `ISAACLAB_ROOT`를 설정하고 공개 저장소 루트에서 해당 스크립트를 실행한다. `--video` 옵션은 영상 저장을 켠다. 최초 측정과 영상 반복은 이미 [results/team_v21](../results/team_v21)에 보존되어 있다. 이 자료는 조교의 공식 unseen 평가 결과가 아니다.

## 실행 기록과 자료 검증

학습 방법은 [Method](METHOD.md), 실제 설정은 [env.yaml](../configs/final/env.yaml)과 [agent.yaml](../configs/final/agent.yaml), 당시 학습 명령 원문은 [training_command.txt](../configs/final/training_command.txt)에 보존했다. 제출된 정책을 평가하기 위해 재학습할 필요는 없다.

`python scripts/verify_publication.py`로 해시·집계·문서 링크를 확인할 수 있다. 그림은 `build_figures.py`, 영상 미리보기는 `build_previews.py`, 발표 파일은 `build_presentation.py`로 재생성한다. `build_training_curve.py`와 TensorBoard는 원본 학습 기록 재검토용이며 학습 reward를 평가 reward와 혼동하지 않는다.

프로젝트 패키지와 공개 경로 wrapper는 파일·CPU·shell 수준에서 검사했다. 새 시뮬레이션을 실행하지 않았으며 실제 보행 근거는 저장된 사용자 평가 로그·영상이다.
