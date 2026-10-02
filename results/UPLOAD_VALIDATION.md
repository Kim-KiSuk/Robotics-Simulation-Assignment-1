# 업로드 묶음 검증 — 2026-10-02

## 이번에 확인한 범위

이번 작업은 HeightScan/E3까지의 결과를 공개 저장소로 정리한 작업이다. 새 정책 학습이나 추가 성능 평가를 실행하지 않았다. 이후의 Stability 코드·학습 보상·warm-start·pilot 결과는 포함하지 않았다. 사용자의 현재 `IsaacLab_RS` 작업 코드와 학습 로그는 수정하지 않았다.

- **코드 묶음 59개 파일:** `overlay_manifest.json`에 원본 및 공개 파일 SHA-256을 기록했다. 원본 `train.py`를 유지하고 새 Stability Task 등록을 제외했다.
- **이전 검증 시점과 일치:** E3/높이 관측 생성 시 기록한 14개 source SHA-256이 공개 사본과 모두 일치했다. Task registry와 평가 스크립트를 포함한다. 이전 [검증 요약](../docs/ant_six_envs/assets/blocks_eval/validation_summary.json)의 `source_sha256`으로 확인할 수 있다.
- **적용 검사:** 원본 commit을 가리키는 별도 임시 Git checkout에 필요한 원본 파일을 꺼내서 설치 preflight → 적용 → 재검사를 통과했다. 50개 파일 적용 후 모든 payload hash가 일치했고, 재검사에서는 변경할 파일이 0개였다. 9개는 원본과 동일한 지원 파일이다.
- **로컬 수정 보존:** 임시 대상 파일을 수정한 뒤 설치기가 `Local changes found`로 중단하는 것을 확인했다.
- **CPU 테스트 8개 통과:** 높이의 상대값·평행 이동·누락값 처리, 종료 직전 위치 고정·timeout/실패 구분·미완료 평가 거부 4개, 마찰 표본 범위·재현성·난수 격리 등의 4개다.
- **Python 문법 검사:** 공개 Python 파일을 AST로 파싱했다. 문서의 로컬 파일 링크와 overlay payload hash도 검사했다.
- **최종 checkpoint 5개:** A/B/C/SixMix/HeightScan의 SHA-256, policy tensor 유한성, iteration, 입력 차원을 확인했다. SixMix는 60차원, HeightScan은 123차원이다. [검사 결과](checkpoint_audit.json)
- **완료된 학습 로그:** SixMix/HeightScan 각각 21개 scalar tag의 4000개 표본이 모두 유한하고 iteration 축이 3999까지 기록되었다. `/time` 항목의 가로축은 iteration이 아닌 경과 시간이다.

체크포인트와 로그 검사는 완료 파일의 무결성과 유한성을 확인한 것이다. 전체 학습 중 물리 오류가 없었다는 별도 증거나 unseen 성능 보장은 아니다. 기능 검증용 `model_59.pt`와 오류가 있던 4096개 혼합 지형 run은 최종 모델에 포함하지 않았다.

## 기존에 수행한 시뮬레이션 검증

이전의 실제 생성·PhysX·센서 검증 기록을 코드와 함께 보존했다. 이번 업로드에서 GPU 검사를 새로 반복하지 않았다.

| 범위 | 기존 확인 내용 | 기록 |
|---|---|---|
| Six 학습 환경 | 타일 생성·지형 배치·작은 PhysX 실행·1024개 60-iteration PPO | [제작 문서](../docs/ant_six_envs/README.md) |
| E2 | 연속 지형 생성, 100개 환경의 센서·충돌 지지 | [PhysX 검사](../docs/ant_six_envs/assets/continuous_eval/physics_check.json) |
| E3 | 18개 지형/난이도 조합, 전체 타일, 60/123차원 각각 100개 환경, 지면 충돌 | [검증 요약](../docs/ant_six_envs/assets/blocks_eval/validation_summary.json) |
| HeightScan | yaw 정렬, 센서 reset의 난수 불변, 60/123D 초기 상태 일치, 1024개 60-iteration PPO | [같은 검증 기록](../docs/ant_six_envs/assets/blocks_eval/validation_summary.json) |
| 첫 에피소드 평가 | 진단 켜기/끄기의 기존 reward·steps 6자리 일치, JSON 저장 | [같은 검증 기록](../docs/ant_six_envs/assets/blocks_eval/validation_summary.json) |

검증 기록의 `/tmp/...` 콘솔 경로는 당시 로컬 경로로, 해당 원본 콘솔 파일 모두를 공개 저장소에 포함한 것은 아니다. JSON의 검사 요약과 코드 hash는 보존했다.

## 결과 출처

E1/E2/E3 표는 사용자 제공 출력의 전사이며 이번 검증의 성능 결과가 아니다. 특히 E3의 정확한 실행 명령과 checkpoint hash 연결은 미확인이다. [평가 결과의 해석 범위](SIX_RESULTS.md)를 참고한다. 초기 평가 기록과 새 표를 서로 다른 맵의 성능 순위로 합치지 않는다.
