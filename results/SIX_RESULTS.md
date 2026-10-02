# SixMix · HeightScan 평가 기록

## 출처와 비교 범위

이 페이지의 E1/E2/E3 수치는 사용자가 대화에 붙여 넣은 터미널 출력이다. 이번 업로드 작업에서 새로 평가한 결과가 아니다. [JSON](six_reported_results.json)과 [CSV](six_reported_results.csv)는 그 출력을 옮긴 것이며, 에피소드별 원자료를 복원한 파일이 아니다. 표준편차는 출력된 값을 그대로 보존했다.

E1/E2는 실행 명령도 제공되었으며 `seed 24`, `num_envs 100`, `real-time`, 960-step 영상 옵션과 checkpoint 경로를 확인할 수 있었다. E3는 모델별 출력만 제공되었다. 대화 맥락상 블록 평가맵의 결과로 기록하되 **정확한 실행 명령, checkpoint 해시, CLI override가 연결된 재현 결과로 확정하지 않는다.** 공개한 최종 checkpoint와 이 E3 수치가 일치한다고 보장하지 않는다. 다음 평가는 `--results_file`로 원자료와 모델 해시를 함께 남긴다.

## 공식 보상과 에피소드 길이

| 맵 | 모델 | Reward mean | Reward std | Steps mean | Steps std |
|---|---|---:|---:|---:|---:|
| E1 | SixMix | 72.007249 | 24.702503 | 854.810000 | 254.199083 |
| E2 | 평지 A | 14.283684 | 12.772225 | 409.940000 | 339.345423 |
| E2 | SixMix | 54.428906 | 29.085658 | 733.490000 | 328.546846 |
| E3 | 평지 A | 9.344116 | 8.764984 | 386.830000 | 332.840444 |
| E3 | HeightScan | 53.930177 | 23.518258 | 775.280000 | 297.930330 |

모든 출력에서 첫 에피소드 100/100 완료를 확인했다. E1/E2/E3는 서로 다른 맵이다. 보상 비율을 이동 거리 비율로 바꾸어 해석하지 않으며 평균 steps에서 생존율을 계산하지 않는다.

## E3 추가 진단

| 지표 | 평지 A | HeightScan |
|---|---:|---:|
| Target progress mean ± std (m) | 9.585305 ± 9.210645 | 52.982837 ± 23.187748 |
| Forward displacement mean ± std (m) | 9.604842 ± 9.244595 | 52.992913 ± 23.189865 |
| Timeout without failure | 0.17 | 0.61 |
| Termination | 0.83 | 0.39 |
| Missing ground / scan episodes | 0 / 0 | 0 / 0 |

Target progress는 고정 목표점 `(1000, 0)`까지 평면 거리의 감소량이고, forward displacement는 world X 순변위다. 자동 reset 직전 위치를 사용한다. 60차원 Baseline에는 주변 높이 스캔이 없으므로 scan 누락 0은 그 센서의 정상 작동을 입증하지 않는다.

출력상 전진과 생존이 함께 개선되었지만, 학습 지형과 관측이 동시에 달라졌다. **높이 관측만의 효과**를 검증하려면 원본 보상·동일 E3에서 SixMix(60차원)와 HeightScan(123차원)을 비교해야 한다. 그 비교의 최종 수치는 이번 묶음에 없다.

A는 절대 몸 높이 입력으로 학습했지만 험지 평가 Task에서는 지면 기준 높이를 입력한다. 관측 의미 차이를 포함한 전이 평가다. 39% 종료의 원인이 블록, 마찰, 속도 중 무엇인지는 집계 수치만으로 단정하지 않는다. 에피소드별 종료 위치·지형과 영상을 함께 확인해야 한다.

E3를 보고 다음 방법을 결정하면 E3는 검증용이다. 최종 보고용 미사용 seed/조건은 별도로 두고 모든 비교 모델에 동일하게 적용한다. 하나의 학습 seed와 지도에서 나온 결과가 모든 unseen 조건의 강건성을 입증하지는 않는다.

## 보존한 학습 결과와 과거 실험

[Checkpoint manifest](../artifacts/checkpoints/manifest.json)는 공개 파일의 실제 SHA-256, 원래 run 이름과 학습량을 기록한다. SixMix와 HeightScan은 각각 1024환경 × 32step × 4000iterations이며, [TensorBoard 로그](../artifacts/tensorboard)와 실행 당시 환경/PPO YAML도 포함한다. 학습 곡선의 평균 reward는 공식 평가의 첫 에피소드 누적 보상과 집계 방식이 다르다.

초기 B/R의 Varied 평가 12조건은 [기존 JSON·로그·실행 설정](../artifacts/evaluation_logs/varied)에 보존했다. 해당 기록에는 개별 에피소드와 checkpoint hash가 있다. R checkpoint 파일 자체는 현재 작업 폴더에 없어 이번 업로드에 포함하지 않았다. 초기 B/C 비교는 [당시 설명](../docs/EARLY_EXPERIMENTS.md)과 [합산 표](pooled.csv)를 참고한다.
