# 공개 자료와 검증 범위

## 포함한 자료

- 원본 IsaacLab_RS에 적용할 overlay, 파일별 SHA-256, 원본 commit.
- 최종 모델과 개발 비교용 Balance6000/Failure2 모델, 실제 학습 설정과 최종 TensorBoard.
- 최종 v2.1 최초 평가의 네 JSON·CSV·로그·환경/runner 설정·smoke 검사 기록.
- 같은 모델로 진행한 영상 재실행 결과와 16초 MP4 네 개, 6초 GIF 미리보기, 생성 코드.
- 팀 v2.1 배포 파일의 변경 없는 사본과 실제 평가 코드 스냅샷.
- 이전 SixMix·HeightScan·Stability 자료와 후속 개발 실험의 결과.

Isaac Sim 설치 파일, 로봇 USD, 강의 PDF, 계정 인증정보는 포함하지 않는다. 원자료의 `/home/kisuk/...` 경로는 실행 당시 출처 식별용이며 공개 재현에는 저장소 상대경로를 사용한다.

## 검증

[verify_publication.py](../scripts/verify_publication.py)는 모델·배포 파일·overlay·영상 해시, 100개 episode의 mean/std와 생존율, 최초 평가와 녹화 재실행의 episode별 일치, 주요 문서 링크를 검사한다. 표준편차는 ddof=0으로 재계산한다. 검증 로그는 [publication_validation.txt](../results/publication_validation.txt)에 있다.

네 MP4는 실제로 디코딩하여 1280×720, 60fps, 960 frames를 확인했다. GIF는 해당 영상 앞 6초의 원래 시간 간격을 유지한다. 보행을 합성한 영상이 아니다. 그래프는 공개 CSV/JSON에서 생성한다.

overlay 설치 검사는 별도 임시 checkout에서 수행하고 Python 문법을 검사한다. 배포 경로를 환경변수로 받은 코드와 공개 실행 wrapper는 CPU·shell 수준에서 검사했다. **공개 경로 wrapper로 시뮬레이션을 새로 실행하지는 않았다.** 실제 GPU 실행 근거는 사용자가 완료한 원본 평가/녹화 로그다.

## 기록의 구분

`results/team_v21/final`이 제출용 최초 평가이고 `video_repeat`는 녹화용 반복이다. 서로 독립적인 800개 episode로 합산하지 않는다. 개발 E3 결과와 팀 v2.1은 지형·종료·reset이 달라 별도 해석한다.

`configs/final/source_manifest.json`과 원본 결과 폴더의 manifest는 평가 전 준비 스냅샷이어서 pending 문구가 남을 수 있다. 현재 완료 상태와 공개 파일 경로는 루트 [manifest.json](../manifest.json)을 기준으로 한다.

출처는 [라이선스 고지](../THIRD_PARTY_NOTICES.md)에 정리했다. 설명의 형식을 참고했으며 다른 팀의 실험 수치·학습 방법을 본 프로젝트의 실적으로 기재하지 않았다.
