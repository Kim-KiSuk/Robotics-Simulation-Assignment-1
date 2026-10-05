# 최종 평가 영상

## 직접 만든 E3-SpawnLift

[16초 자체 평가 영상](../artifacts/media/self_eval/E3_SpawnLift.mp4) · [6초 미리보기](../artifacts/media/self_eval/E3_SpawnLift_preview.gif) · [맵 구성과 수치](SELF_EVALUATION.md)

아래 팀 공통 v2.1과 별개의 개발 검증 환경이다. 학습·평가의 초기 높이를 맞춘 최종 모델의 기록을 보존했다.

## 팀 공통 v2.1

다음 네 영상은 최종 checkpoint를 고정한 상태에서 **v2.1, seed 24, num_envs 100**으로 녹화한 실제 시뮬레이터 화면이다. 원본 1280×720, 60fps, 960 frames, 각 16초이며 재생 속도를 바꾸지 않았다.

| 환경 | 전체 영상 | 미리보기 | 프레임 |
| --- | --- | --- | --- |
| Seen Control | [MP4](../artifacts/media/team_v21/SeenControl.mp4) | [6초 GIF](../artifacts/media/team_v21/SeenControl_preview.gif) | [2초 지점](../artifacts/media/team_v21/SeenControl.png) |
| Unseen Easy | [MP4](../artifacts/media/team_v21/UnseenEasy.mp4) | [6초 GIF](../artifacts/media/team_v21/UnseenEasy_preview.gif) | [2초 지점](../artifacts/media/team_v21/UnseenEasy.png) |
| Unseen Medium | [MP4](../artifacts/media/team_v21/UnseenMedium.mp4) | [6초 GIF](../artifacts/media/team_v21/UnseenMedium_preview.gif) | [2초 지점](../artifacts/media/team_v21/UnseenMedium.png) |
| Unseen Hard | [MP4](../artifacts/media/team_v21/UnseenHard.mp4) | [6초 GIF](../artifacts/media/team_v21/UnseenHard_preview.gif) | [2초 지점](../artifacts/media/team_v21/UnseenHard.png) |

GIF는 0~6초를 10fps·480×270으로 줄인 미리보기다. 각 프레임 간격을 0.1초로 유지해 원본 속도로 반복한다. 전체 MP4는 변경 없이 복사했고 [해시 목록](../artifacts/media/manifest.json)에 기록했다. GitHub 파일 페이지에서 재생되지 않으면 Raw/Download로 열 수 있다.

카메라는 전체 100개 로봇을 동시에 보여주지 않는다. 먼저 끝난 로봇은 자동 reset되어 후속 episode가 영상에 보일 수도 있지만, 점수 집계는 각 로봇의 첫 episode까지만 한다. 장면 하나로 전체 생존율이나 성공률을 판단하지 않는다.

[영상 실행의 로그·JSON](../results/team_v21/video_repeat) · [제출용 최초 평가](../results/team_v21/final) · [미리보기 생성 코드](../scripts/build_previews.py)
