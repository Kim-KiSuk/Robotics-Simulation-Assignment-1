# 업로드 검증 — Six 환경 설계부터 Stability까지

2026-10-02에 기존 공개 저장소를 정리하고 Stability 코드·최종 체크포인트·학습 로그·기존 평가 기록을 추가했다. 사용자의 `IsaacLab_RS` 작업 코드와 학습 로그는 수정하지 않았다. **이번 작업에서는 새 학습이나 GPU 성능 평가를 실행하지 않았다.**

## 이번에 실제 확인한 내용

| 검사 | 결과 |
|---|---|
| 공개 코드 65개 | manifest SHA-256 일치, 현재 작업 코드와 byte 단위 일치 |
| Python 문법 | 공개 코드와 설치기 총 59개 파일 AST 파싱 통과 |
| 고정 원본에 적용 | 별도 임시 checkout에서 preflight → 57개 파일 설치 → 재검사 시 변경 0개 |
| 기존 수정 보호 | 대상 파일을 수정하면 쓰기 전에 `Local changes found`로 중단 |
| 관측·평가 수식 검사 | 높이 상대값·누락 처리·첫 에피소드 진단 등 4개 통과 |
| 마찰 검사 | 표본 범위·재현성·난수 격리 등 기존 CPU 테스트 4개 통과 |
| Stability 지형 | 24개 구성 × 3개 난이도, 총 72개 실제 생성 검사 통과 |
| Stability 보상 | 실패 사건 −5의 dt 불변성, 몸 높이 위험 감점 경계 검사 통과 |
| Warm-start | 실제 RSL-RL에 공개 HeightScan checkpoint를 로드; 결정론적 행동·가중치 보존, optimizer·반복 카운터 초기화, 초기 noise 하한 확인 |
| 최종 체크포인트 6개 | SHA, iteration, 입력 차원, 정책 tensor 17개 각각의 유한성 확인 |
| 학습 로그 | SixMix/HeightScan 각각 21개 scalar tag × 4000개, Stability 23개 × 500개 표본 모두 유한 |
| 주요 문서 링크 | README·환경 구성·실행 명령·결과·pilot 안내의 상대 파일 링크 확인 |

임시 checkout은 고정한 원본 commit `e83a5d2f11ca1b5f03b690e1978479e620c500e2`에서 필요한 원본 파일을 꺼낸 뒤 overlay를 적용한 검사 대상이다. 전체 Isaac Sim 설치를 새로 수행한 검사는 아니다. CPU 지형 검사는 원본 생성 함수를 사용했다.

검사 기록: [적용·파일 검사](publication_checks.json), [체크포인트](checkpoint_audit.json), [TensorBoard](training_log_audit.json), [Stability CPU 검사](stability/publication_cpu_check.json).

TensorBoard의 `/time` tag는 경과 시간을 가로축으로 쓰므로 iteration으로 해석하지 않았다. 그 외 학습 tag는 SixMix/HeightScan에서 3999, Stability에서 499까지 확인했다. 최종 파일과 로그의 유한성은 전체 학습 중 물리 오류가 없었다는 보장이나 보행 개선의 증거가 아니다.

## 기존 검사와 평가 기록

이전 Six/E2/E3/HeightScan의 생성·PhysX·짧은 학습 검사는 [상세 제작 기록](../docs/ant_six_envs/README.md)에 보존했다. Stability 본학습 전 CPU·PhysX·60회 pilot 검사는 [별도 폴더](validation/stability_pretraining/README.md)에 넣었다. JSON의 로컬 경로·source hash·`ready_for_500_iteration_experiment` 상태는 **검사 당시의 기록**이다. 이들을 현재의 새 GPU 검사 결과로 표시하지 않는다.

이전 공개 시점에는 Stability를 제외했으나 이번 공개에는 포함한다. 이전 검증 설명은 [이전 업로드 기록](UPLOAD_VALIDATION_PRE_STABILITY.md)으로 보존했다. 당시 registry/train.py hash가 현재 파일과 달라지는 것은 이 변경 때문이다.

최신 Stability 점수는 사용자 제공 명령과 출력이며, HeightScan 부모 재평가는 기존 에피소드 JSON이다. 둘의 실행 모드와 초기 E3 보고값의 미해결 차이를 [결과 문서](SIX_RESULTS.md)에 적었다. 성능 향상을 확인한 것으로 주장하지 않는다.
