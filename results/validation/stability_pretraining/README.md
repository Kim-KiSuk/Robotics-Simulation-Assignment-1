# Stability 본학습 전 검사 기록

이 폴더는 500회 본학습 **이전**의 CPU·소규모 PhysX·60회 PPO 검사 및 pilot 평가 기록이다.
`validation_summary.json`의 `ready_for_500_iteration_experiment`는 당시 상태이며 현재 완료 여부를 나타내지 않는다.

- `parent_control.json`: HeightScan model_3999의 E3 원본 보상 평가. 에피소드 100개와 실행 설정 포함.
- `conservative_pilot_eval.json`: 채택 설정의 짧은 60회 pilot 평가. 최종 500회 모델이 아니다.
- `rejected_fast_update_pilot.json`: 더 큰 학습률/noise를 사용한 미채택 pilot.
- `user_e3_results.json`: 이전 사용자 제공 결과. 최신 Stability 결과와 다르다.
- 나머지 JSON: 당시 검사·부모 학습 요약. 파일에 남은 로컬 임시 경로와 hash는 당시 기록이다.

최신 500회 모델 결과는 [결과 설명](../../SIX_RESULTS.md), 이번 업로드 검사는 [업로드 검증](../../UPLOAD_VALIDATION.md)에 별도로 정리한다.
