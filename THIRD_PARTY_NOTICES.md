# 출처와 라이선스

- 문서 구성 참고(2026-10-06):
  [baeminseongp/Robotics_Simulation_8](https://github.com/baeminseongp/Robotics_Simulation_8)의
  연구 문제·가설·설정·검증 한계 제시 방식과
  [Stick-0/isaac-ant-rough-terrain](https://github.com/Stick-0/isaac-ant-rough-terrain)의
  단계별 개선·영상 미리보기 중심 구성을 참고했다. 두 프로젝트의 학습 모델·영상·수치를
  복사하지 않았으며 우리 기록을 근거로 독립적인 설명을 작성했다.
- `evaluation/team_v21`는 사용자가 평가용으로 제공한 팀 공통 v2.1 배포본이다.
  지형·재질 Python 파일의 BSD-3-Clause 헤더와 원본 파일 해시를 유지했다.

- 기반 코드: [IsaacLab_RS](https://github.com/cailab-hy/IsaacLab_RS),
  commit `e83a5d2f11ca1b5f03b690e1978479e620c500e2`.
- 원 프로젝트: [Isaac Lab](https://github.com/isaac-sim/IsaacLab), BSD-3-Clause.
  배포 코드의 기존 copyright/SPDX 헤더 및 저장소의 LICENSE를 유지했다.
- 험지 구현 설계 참고:
  [Stick-0/isaac-ant-rough-terrain](https://github.com/Stick-0/isaac-ant-rough-terrain),
  검토 commit `4c50a77`. 지면 기준 높이와 충돌 메시 분할 접근을 참고했다.
  본 저장소는 별도 Task 등록, 기본 XY 배치, 더 작은 지형, 마찰 실험,
  분리된 보상/관측 실험을 사용한다. 참고 저장소의 성능 수치를 본 실험 결과로
  사용하지 않는다.
- Isaac Sim/로봇 USD 자산은 포함하지 않는다. 기반 설치에서 제공되는 자산과
  해당 이용 조건을 사용한다. 강의 PDF도 재배포하지 않는다.
