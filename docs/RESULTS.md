# 실험 결과와 해석

수치는 저장된 JSON/CSV에서 가져왔다. 아래 비교는 **개발용 E3, 초기화가 다른 SpawnLift E3, 최종 팀 v2.1**로 구분한다. 모두 원본 7개 reward로 평가했지만 지형·초기화·종료 조건이 같지 않으므로 서로 다른 표를 단일 순위로 합치지 않는다.

## 1. 개발용 E3: 실패를 줄이는 보상

seed 24, 100개 환경, 지형 seed 9317, 지면 마찰 0.9/0.75, 지면 상대 몸높이 0.31m 종료, 최대 960 steps다.

| 모델 | 보상 mean ± std | 생존율 | 평균 +x 거리 |
| --- | ---: | ---: | ---: |
| Balance | 68.62 ± 24.65 | 81% | 71.77m |
| FailurePenalty | 72.80 ± 25.00 | 86% | 77.50m |

[Balance](../results/development/balance6000/results.json) · [FailurePenalty](../results/development/failure2/results.json)

Balance→FailurePenalty는 동일 학습 조건에서 실패 사건 비용 −2를 넣은 비교다. 보상 +4.18, 생존 +5%p, 거리 +5.73m가 관찰됐다. 이는 해당 seed의 결과이며 다른 seed에서도 효과가 유지된다는 증명은 아니다.

## 2. 잘되지 않은 접근도 남긴다

| 접근 | 보상 mean ± std | 생존율 | 해석 |
| --- | ---: | ---: | --- |
| 긴 rollout | 58.31 ± 22.56 | 81% | rollout/GAE/배치가 함께 바뀐 실험; 반복 수만 늘린 비교가 아님 |
| 자세 위험 비용 | 48.58 ± 25.60 | 54% | 자세 비용 강화가 일반적인 개선으로 이어지지 않음 |
| 수직 움직임 비용 | 55.35 ± 28.67 | 68% | 통통 튐 억제만으로 전진·생존이 개선되지 않음 |
| 초기 XY 다양화 | 44.75 ± 26.02 | 55% | 시작 위치를 늘린 것만으로 해결되지 않음 |

[긴 rollout](../results/development/long_rollout/results.json) · [자세](../results/development/posture/results.json) · [수직](../results/development/vertical/results.json) · [초기 XY](../results/development/random_spawn/results.json)

이 기록은 자연스러워 보이는 보행과 실제 평가 성능이 같지 않다는 점을 보여준다. 원인을 각각 완전히 분리한 실험은 아니므로 특정 페널티가 항상 해롭다고 일반화하지 않는다.

## 3. 초기 소환 보완: 같은 SpawnLift E3에서 비교

발과 블록의 겹침을 관찰한 뒤 reset Z +0.15m를 별도 평가 Task에 적용했다. 기존 FailurePenalty의 원래 E3 점수와 바로 이어 붙이지 않고, **양쪽 모두 같은 SpawnLift 평가**로 비교한다.

| 모델 | 학습 reset 추가 높이 | 보상 mean ± std | 생존율 | 평균 steps | 평균 +x 거리 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 FailurePenalty | 0m | 73.75 ± 21.31 | 82% | 890.92 | 78.66m |
| 최종 Final | +0.15m | 81.90 ± 17.05 | 88% | 934.19 | 82.79m |

[동일 조건의 기존 모델](../results/development/failure2_eval_lift/results.json) · [최종 모델](../results/development/final_lift/results.json)

학습 높이 변경 후 보상 +8.15, 생존 +6%p, 거리 +4.13m를 관찰했다. 두 모델은 같은 초기 가중치 파일을 공유한 추가 학습이 아니라 각자 처음부터 학습했다. 새 정책이 모든 에피소드에서 더 빠르다는 뜻은 아니며, 길어진 생존 시간이 누적 전진에 영향을 준다.

## 4. 팀 공통 자체 평가 v2.1: 고정한 정책의 새로운 지형 평가


평가한 체크포인트는 `final/ant_final.pt`이며 SHA-256은 다음과 같다.

```text
c0784d03e5d02882d0c57aa1d95bca94adfed85aaf88b8284d03839d6f70a382
```

### 평가 조건

- 네 환경마다 seed 24, num_envs 100, 첫 episode 1회, 최대 16초/960 steps.
- 원본 Ant 7개 reward: progress 1, alive 0.5, upright 0.1, move_to_target 0.5, action_l2 −0.005, energy −0.05, joint_pos_limits −0.1.
- 원본 world-Z 종료와 원본 reset. 학습 +0.15m lift는 미적용.
- observation 123D, action 8D, 학습 당시 센서 유지.
- native terrain origins, initial levels 0~4. 기록된 초기 x 범위 −156~−124m.
- policy update 없음. 4개 smoke check 통과 후 전체 평가 진행.
- 생존은 **steps=960이고 timeout이며 실패 종료가 아닌 episode**. std는 ddof=0.

### 결과

| 환경 | 보상 mean ± std | Steps mean ± std | 생존율 | +x 거리 mean ± std |
| --- | ---: | ---: | ---: | ---: |
| Seen Control | 94.77 ± 32.68 | 916.61 ± 146.35 | 89% | 96.75 ± 33.45m |
| Unseen Easy | 91.24 ± 40.22 | 811.33 ± 315.09 | 79% | 93.18 ± 41.00m |
| Unseen Medium | 69.91 ± 35.04 | 784.42 ± 330.97 | 74% | 72.41 ± 36.04m |
| Unseen Hard | 88.64 ± 45.20 | 774.51 ± 333.17 | 70% | 90.84 ± 46.14m |

[제출 CSV](../results/team_v21/final/result_template.csv) · [로그·설정·episode 원자료](../results/team_v21/final)

### 무엇이 확인됐는가?

Easy의 평균 전진은 Seen과 비슷하지만 생존율은 10%p 낮다. Hard는 평균 보상이 Medium보다 높으면서 생존율은 더 낮다. 분포의 평균과 실패 빈도가 서로 다른 측면을 나타낸다. Easy/Medium/Hard는 배포된 조건의 이름이지 모든 정책에 같은 성능 순서를 보장하는 이름은 아니다.

보상 90점 목표는 Seen과 Easy에서 달성했으나 생존율 90%는 모든 환경에서 미달했다. 처음 보는 세 지형에서 일정 수준의 전진은 유지됐지만 안정성 한계가 남았다. v2.1에는 다른 모델을 같은 조건으로 재평가한 대조군이 없으므로 각 개발 변경의 최종 기여를 여기서 따로 계산할 수 없다.

### Medium의 센서 미검출

Medium에서 몸 아래 지면 ray 미검출을 한 번 이상 겪은 episode는 **86/100**, 주변 스캔 미검출은 **100/100**이었다. 나머지 세 환경은 모두 0이다. 이 수치는 누락 프레임 비율이나 100개 정책이 항상 관측을 잃었다는 뜻이 아니다.

배포된 `MeshGapTerrainCfg`는 중앙 플랫폼과 외곽 사이에 실제 빈 틈을 만든다. 아래 방향 ray가 틈을 지나면 지면을 만나지 않을 수 있다. 기존 센서의 0/−1 대체 처리는 유지했다. 누락이 성능 저하의 주원인인지, 각 누락이 모두 틈 때문인지는 현재 기록만으로 단정하지 않는다.

## 5. 녹화용 재실행

최초 평가 `team_eval_v21_final_10LwhZeS`와 영상 재실행 `team_eval_v21_video_hdtwZBvG`를 각각 보존했다. 동일 checkpoint로 네 환경을 다시 실행했으며 **CSV 집계값과 episode별 기록이 일치하는지 공개 검증 스크립트로 확인**한다. 영상은 60fps, 960 frames, 16초다.

재실행을 독립 표본 400개로 더하거나 더 좋은 결과를 선택하지 않는다. 제출 CSV는 최초 평가 파일이다. 결과를 본 뒤 모델을 바꾸거나 추가 학습하지 않았다.

[영상·해석 범위](MEDIA.md) · [검증 기록](PUBLICATION.md)
