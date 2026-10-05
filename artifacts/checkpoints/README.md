# 모델 파일 안내

파일 이름은 학습 횟수 대신 역할을 나타낸다. **최종 제출·평가 모델은 `final/ant_final.pt` 하나**다. 나머지는 개발 비교용으로 보존했다.

| 모델 | 파일 | 역할 |
| --- | --- | --- |
| Baseline | [ant_baseline.pt](baseline/ant_baseline.pt) | 원본 평지 기준 모델 |
| Rough | [ant_rough.pt](rough/ant_rough.pt) | 초기 험지 모델 |
| FrictionDR | [ant_friction_dr.pt](friction_dr/ant_friction_dr.pt) | 초기 마찰 다양화 모델 |
| TerrainMix | [ant_terrain_mix.pt](terrain_mix/ant_terrain_mix.pt) | 다섯 지형 구성 혼합 |
| HeightScan | [ant_height_scan.pt](height_scan/ant_height_scan.pt) | 주변 지면 높이 관측 추가 |
| Stability | [ant_stability.pt](stability/ant_stability.pt) | 초기 안정성 보상 실험 |
| Balance | [ant_balance.pt](balance/ant_balance.pt) | 액션 변화·각속도 비용 |
| FailurePenalty | [ant_failure_penalty.pt](failure_penalty/ant_failure_penalty.pt) | 실패 사건 비용 추가 |
| **Final** | [ant_final.pt](final/ant_final.pt) | 학습 초기 높이를 보완한 최종 모델 |

이름만 바꿨으며 가중치 파일의 바이트와 SHA-256은 모두 동일하다. [초기 모델 해시](manifest.json) · [후속 모델 해시](final_models_manifest.json)

기존 실행 로그·JSON·CSV의 체크포인트 이름은 측정 당시 기록이므로 그대로 보존했다. 같은 SHA-256으로 공개 파일과 대응할 수 있다. 새 평가 명령과 프로젝트 압축본은 새 이름을 사용한다.
