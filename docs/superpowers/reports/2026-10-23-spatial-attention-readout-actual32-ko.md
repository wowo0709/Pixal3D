# Spatial480 attention의 실제 조건부 기여

학습된 feature-spatial SS480 하나에서 원래 readout과 네 inference 제어를 실제
완료했다. 저장된 모델 안에서 학습형 재가중을 matching prior로 바꾸면 평균 IoU가
하락했다. 이는 attention 전달이 전혀 사용되지 않는다는 설명에 반하는 근거다.
Semantic Top-K의 추가 효과 또는 attention을 포함한 재학습 arm의 우월성을
입증한 결과는 아니다.

## 실제 입력·모델·재현 범위

DEV32, 한 학습 seed와 한 생성 noise, 동일 SS480 NN/source weight,
edited target RGB8+source RGB9/10/11, native spatial3 후보,
12-step/CFG3/batch8/deterministic sampling이다. 이 spatial arm은 semantic64 후보를
사용하지 않으며 deformation-query channel도 없다.

Normal32의 predicted latent와 voxel은 기존 archive를 정확히 재현했다. 다섯 GPU
실행 모두 정상 종료와 자식 session 정리, 실제 file4 SHA256·크기를 검증했다.
모델 파라미터·native projection/global·RGB/camera/noise·후보·matching prior·
Top-1 carrier·source-use·gate·R target context를 고정했다. R의 입력 feature가
달라지면 R 출력은 달라질 수 있으므로 R 출력까지 고정한 비교라고 부르지 않는다.
추가 optimizer update는 없다.

## 고정32 결과

주 지표는 자산별 voxel IoU 평균이다. 아래32개 결과를 기존128×3 결과의
47.6667%와 직접 섞지 않는다. Normal의 TP/FP/FN 합산 IoU는45.2936%이며
자산별 평균48.8980%와 집계 정의가 다르다.

| Readout | Mean-asset IoU (%) | Normal 대비 (%p) | 네 비교 family95% CI (%p) |
|---|---:|---:|---|
| Normal | **48.8980** | 기준 | — |
| Matching-prior-only | 47.9674 | −0.9307 | **[−1.9074, −0.0029]** |
| Top1-feature-only | 48.1508 | −0.7472 | [−2.0757, +0.3841] |
| Zero-transformed-value | 47.7605 | −1.1376 | [−2.8746, +0.5223] |
| Wrong-projected-V | 46.3685 | −2.5296 | **[−4.3507, −0.9136]** |

Source cluster 단위 paired bootstrap4000회, seed20261023이다. 자산별 차이를
cluster에 합산하고 재표집된 자산 수로 나눠 원래 mean-asset estimand를 유지한다.
표의 CI는 이 네 IoU 비교에만 family 보정하며, 과거 전체 탐색·모델 선택이나
독립 TRAIN seed의 변동을 보정하지 않는다. 부지표의 CI는 탐색적이다.

Matching-prior-only는 원래 비균등 matching prior와 transformed V를 유지하면서
추가 학습형 QK/utility 재가중을 제거한다. Uniform score가 아니며 사용자 정의의
Top-1 No-score 재학습 arm도 아니다. Normal이 이 제어를 평균0.9307%p 넘고,
이번 family CI는0을 간신히 제외한다. 현재 spatial 모델이 학습형 재가중을 유용하게
소비한다는 조건부 근거로 해석한다.

Top1-feature-only는 실제 correspondence 하나의 transformed V를 사용하는
inference 제어다. 평균은 낮지만 CI가0을 포함하므로 Top-K 조합의 필수성은
이 결과로 입증되지 않았다. Zero-transformed-value도 원래 Top-1 carrier와 R을
남기며 R-off가 아니다. 그 IoU 차이의 CI 역시0을 포함한다.

Wrong-projected-V는 spatial3 안에서 projected V의 대응만 순환한다. 후보·prior·
Top-1 carrier는 유지한다. 큰 하락은 현재 readout에서 correspondence와 feature
값의 결합이 중요하다는 근거다. Semantic matching을 원래 spatial matching으로
대체하는 No-corr ablation은 아니다.

## 기존 결론의 갱신과 다음 판단

현재 확인된 것은 세 가지다. Retargeting은 기존 SS 두 학습 seed에서 추가 효과가
재현됐다. Member posterior는 후보 recall을 높였다. 이번 spatial3 진단은 학습된
재가중과 올바른 V 대응의 조건부 기여를 확인했다.

따라서 matching-aware readout에 긍정 근거가 전혀 없다고 표현하지 않는다.
동시에 spatial3 결과를 semantic64의 효과로 확대하거나 ray-wise QKV를 필수
모듈로 확정하지 않는다. 저장된 attention 모델을 다른 readout으로 바꾸면 학습 때의
분포도 달라지므로, 이 의존성이 동일 조건의 재학습 대조군 대비 이득을 뜻하지는 않는다.
Top-1을 넘는 생성 품질, semantic 후보 개선의 downstream 전달, 독립 학습 확인과
공통 PBR cascade의 source/target 지표를 함께 충족해야 Full을 채택할 수 있다.

GPU0·1의 historical retargeting과 GPU2·3의 direct-NoDQ 재현이 실제 시작됐다.
두 arm은 공통96 checkpoint에서480으로 이어가며, 현재 source-camera-ray/native
계열과 다른 target-patch/nearest 경로다. 세 번째 DQ 재현, 나머지 native PBR
feature/Top-K 학습, DEV cascade·독립 seed·선택 조합 및 최종508개10 metric은
계속 진행한다. 이 진단으로 전체 실험을 완료 처리하지 않는다.

Canonical 결과는 `spatial-feature-readout-diagnostics-v1/`의 `actual-v1`,
`summary-v1`, `independent-review-v1`이다. 저장된 다섯 evaluation에서 모델·GT·
실제 입력/consumer 불변식과 원래32개 재현을 재검사하고 모든 집계·CI를 독립
재계산했다. 해당 검토에는 neural forward/backward나 새 학습이 없다.
