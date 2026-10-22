# 실패 경로 검증과 후속 학습 진행

2026-10-22. 성능이 낮은 기존 Direct 모델의 추가 epoch 대신, 동일 weight에서
matching posterior와 feature 전달을 분리하는 GPU 진단을 연결했다.
[구현과 가설](2026-10-22-PBR-matching-delivery-diagnosis-ko.md),
[기존 PBR32 결과](2026-10-22-PBR-Direct1905-stage32-analysis-ko.md)를 함께 읽는다.

## 실제로 완료된 검증

CPU 검사8개를 통과했다. 이어 GPU2의 normal4 진단에서 기존 Direct1905의
첫4개 자산과9개 입력 identity,11개 metric, 실제 GT/Pred/Diff pixel,
native pooled aggregation이 **모두 정확히 같았다**. 실제 GPU 실행으로 확인했고,
결과는 file4의 `delivery-intervention-diagnosis-v1/normal4-native-gate-CPU-v2/`
receipt에 보관했다. 검사기 이름의 CPU는 생성된 결과를 대조하는 단계이며,
이미지 생성 자체는 GPU에서 수행했다.

처음 시도는 명시적32개 panel 파일이 `--panel-size 4`보다 우선하여 count guard에서
중단됐다. 실패 로그와 출력 폴더를 보존했다. 동일32개 panel의 첫4개 ID를 담은
별도 파일로 v2 successor를 추가했고 원래 evaluator·worker·checkpoint를 바꾸지 않았다.
실패한 시도를 완료로 표시하거나 기존 queue state를 수정하지 않았다.

이 gate 뒤에 다음 세32개 진단 및 paired CI/W&B 요약이 연결돼 있다.

- Matching prior 복원: appearance와 RAW2048이 같은 probability를 사용한다.
- Direct 전달 제거: 정상 head/DQ 입력과 실제 R 출력을 유지한다.
- 두 개입의 결합: 같은 NN/source1905에서 함께 비교한다.

이는 저장 weight의 개입 결과다. NoScore=Top-1 feature, NoCorr=native spatial,
no-DQ=물리 channel 제거라는 기존 정의를 대체하지 않는다. 이 문서를 기록할 때
세32개 결과는 아직 나오지 않았다.

## SS 후보는 실제 학습 중

Bounded source-contrast 후보는 source-camera ray member query와 source-relative R,
matching prior를 보존하는 attention을 사용한다. 초기 native32 출력 보존과
micro96 backward 검증을 통과했다. TRAIN1536 teacher pilot,7에폭/112step,
GPU0 한 장에서 global96/micro96/accumulation1로 진행한다.

| Step | Native32 voxel IoU |
|---:|---:|
| 16 | 0.4276772 |
| 32 | 0.4286033 |
| 48 | 0.4289900 |

최대 allocated31.65GiB/reserved35.95GiB이고50step까지 OOM이 없었다.
최근16step 평균 계산은 약22초다. 이전 plain source-contrast48step은0.4289440,
최종112step은0.4305255였다. 현재 수치로 새 후보가 더 낫다고 판단하지 않는다.
112step 이후 matching GT mass·feature fit·native quality를 함께 요약하도록 연결했다.
이 pilot은 frozen SS에서 source module을 학습하는 단계이며 전체 SS NN joint 학습,
독립 확인 또는 PBR 학습을 완료한 결과가 아니다.

[현재 SS W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-source-teacher-bounded-contrast-TRAIN1536-7ep-v1)와
[PBR 실패 경로 실측 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr1905-matching-delivery-forensics32-v1)에 기록한다.

## 평가 범위와 운영 복구

Test508 Omega/DINO 생성은 상위 출력 폴더가 없어 native builder 실행 전에 실패했다.
네 실패 로그를 보존하고 stage 부모 폴더만 생성했다. 동일한 SHA-bound worker,
closed plan, cache 목적지와4개 shard를 유지한13개 successor를 추가했다.
복구 후 Omega/DINO의 첫3개 shard381개가 완료됐으며 마지막 shard가 진행 중이다.
DINO/NAF, frozen matches,508개 loader QA는 이어지는 작업이며 완료로 간주하지 않는다.
Test 모델의 품질 평가는 아직 수행하지 않았다.

전체 cascade 생성·render metric·source preservation·source-disjoint 확인과
최종 test508 범위를 유지한다. GPU는0~3만 사용한다. 기존2022개 queue descriptor 중
실패/HELD 기록은 남겼으며 건강한 학습·dispatcher를 재시작하지 않았다.
이 단계에서 full 모델을 채택하거나 목표 성능 개선을 달성했다고 선언하지 않는다.
