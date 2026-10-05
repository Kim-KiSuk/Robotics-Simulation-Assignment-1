# v2.1 환경 검증 보고서

검증일: 2026-10-04  
검증 장비: NVIDIA RTX 6000 Ada, Isaac Sim 5.1, `lerobot-arena`

## 수정 이력

- v1 폐기: 일부 origin이 x=76 m이고 +x 외곽이 x=100 m여서 전진 여유가 24 m뿐이었다.
- v2.0 폐기: Medium의 discrete block 지형이 조원 학습 자료의 random-grid/block 계열과 유사했다.
- v2.1: 40×20 맵과 제한 스폰을 유지하고 Medium을 새 seed의 narrow gap 계열로 교체했다.

## 세 조원 학습 지형과의 중복 검사

제공받은 `ant_environments.zip`, `ant_training_seeds.md`, `env.yaml`과 현재 로컬 학습 config를
대조했다. 조원 학습 계열은 평면, rough/noise, wave, slope/inverted slope, random grid/square
blocks, stairs/inverted stairs였다.

v2.1 unseen 계열은 다음과 같으며 동일 generator 계열이 학습 목록에 없다.

- Easy: `MeshRailsTerrainCfg`
- Medium: `MeshGapTerrainCfg`
- Hard: `MeshRepeatedCylindersTerrainCfg`

정책 평가만 수행하며 세 지형으로 추가 학습하거나 checkpoint를 선택하지 않는다.

## 실제 geometry 검사

네 지형을 evaluation seed 24로 각각 두 번 완전히 생성하고 vertex·face·origin 배열 SHA-256을
비교했다. 네 환경 모두 두 실행의 hash가 일치했다.

| 환경 | 실제 surface z 범위 | mesh 크기 | 최소 +x 여유 | 재현성 |
| --- | ---: | ---: | ---: | --- |
| seen control | -1.000~0.205 m | 360×200 m | 304 m | 통과 |
| unseen easy | -1.000~0.040 m | 360×200 m | 304 m | 통과 |
| unseen medium | -1.000~0.000 m | 360×200 m | 304 m | 통과 |
| unseen hard | 약 -0.050~0.050 m | 360×200 m | 304 m | 통과 |

Medium의 gap 설정 범위는 5~12 cm이며 고정 difficulty 0.5에서 실제 폭은 약 8.5 cm다.

## 100환경 런타임 검사

정책을 로드하지 않고 각 task를 100환경으로 생성하여 reset과 zero-action 1 step을 수행했다.

- 실제 terrain level: 0~4만 사용
- origin x: -156~-124 m
- origin y: -76~76 m
- +x 외곽: +180 m
- 최소 전진 여유: 304 m
- 각 terrain type column: 5환경씩, 총 100환경
- robot 초기 위치: 모든 환경에서 origin + (0, 0, 0.5 m)
- 관측·보상 NaN/Inf: 없음
- 첫 step terminated/truncated: 0/100
- episode: 16초, 960 policy steps
- 활성 reward: 원본 7개
- 활성 termination: `time_out`, `torso_height`
- 활성 evaluation curriculum term: 0개
- 환경 내부 collision shape 간 material 최대 편차: 0
- 모든 sample에서 dynamic friction ≤ static friction

실측 마찰 범위도 설정 범위 안이었다.

| 환경 | 실측 static | 실측 dynamic |
| --- | ---: | ---: |
| seen control | 0.532–1.462 | 0.412–1.185 |
| unseen easy | 1.000 | 1.000 |
| unseen medium | 0.674–1.301 | 0.559–1.120 |
| unseen hard | 0.390–1.552 | 0.265–1.281 |

## 통과 가능성 smoke test

환경이 물리적으로 실행 불가능하지 않은지 확인하기 위해 한 고정 checkpoint로만 평가했다.
이 값은 조원 간 최종 순위가 아니다.

| 환경 | reward mean ± std | survival | steps mean | +x mean |
| --- | ---: | ---: | ---: | ---: |
| seen control | 39.129 ± 9.075 | 95% | 915.25 | 27.63 m |
| unseen easy | 38.148 ± 11.271 | 92% | 888.99 | 27.19 m |
| unseen medium | 36.074 ± 13.202 | 85% | 848.19 | 26.33 m |
| unseen hard | 22.904 ± 12.152 | 51% | 706.35 | 14.66 m |

설계 난이도와 이 smoke policy의 성능이 Easy > Medium > Hard 순서로 일치했다. 다른 정책에서
세부 순서가 달라질 수 있으며 이는 모델별 일반화 특성으로 분석한다.

## 배포 판정

v2.1은 학습 지형 중복 검사, geometry 재현성, 크기·경계, 스폰, 마찰, 원본 평가 계약,
NaN·즉시 종료, 통과 가능성 검사를 통과했다. 조원 checkpoint hash를 먼저 동결한 뒤 배포한다.
