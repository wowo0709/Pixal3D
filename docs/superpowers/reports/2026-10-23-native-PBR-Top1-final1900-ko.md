# Native PBR Top-1+R 최종1900 결과

원본 multi-view PBR16850에서 시작한 V1과 Top-1+R이 모두 fresh optimizer로
1900 update를 완료했다. 실제 PBR TRAIN은9139개이며 seed101,
DDP4/global24/micro6/accumulation1이다. 이 Top-1 arm에는 추가 QKV feature head가 없다.

네 rank 모두 frozen base·matcher·anchor 불변, 최종 DDP parameter 일치,
1900 optimizer update를 확인했다. 모든1900 update의 asset·noise·timestep·global
trace가 V1과 같다. 최초 네 예측도 V1과 정확히 같았다. 여섯 validation 기록과
매 epoch NN/source/Adam checkpoint의 실제 file4 SHA256·크기 검증을 완료했다.

## 같은 최종 step의 네 샘플 비교

GT target Shape를 고정하고 target camera8에서 렌더한 개발 진단이다.
Source preservation, 예측 geometry 또는 full-cascade 평가는 아니다.

| Metric | V1 | Top-1+R | Top-1+R − V1 |
|---|---:|---:|---:|
| Pooled-MSE PSNR ↑ | 26.876521 | 26.860112 | −0.016409 dB |
| Mean-asset PSNR ↑ | 30.073555 | 29.986476 | −0.087079 dB |
| Target LPIPS ↓ | 0.00870725 | 0.00853240 | −0.00017485 |
| Foreground pooled PSNR ↑ | 15.765653 | 15.749246 | −0.016407 dB |
| Crop LPIPS ↓ | 0.08645439 | 0.08518629 | −0.00126810 |
| SSIM ↑ | 0.99264704 | 0.99264087 | −0.00000617 |

평균 foreground 비율은7.743%다. 검은 배경이 많은 전체 이미지 PSNR만으로
물체 복원 품질을 판단하지 않는다. GT Shape 조건의 alpha IoU1도 생성 geometry
개선의 증거가 아니다.

| 자산 | PSNR 차이 (dB) | LPIPS 차이 |
|---|---:|---:|
| Axe | +0.225039 | −0.00155799 |
| Bed | −0.597185 | +0.00083404 |
| Bookcase | +0.038620 | +0.00001307 |
| Bottle | −0.014788 | +0.00001148 |

Axe/Bed의 두 모델 최종 GT/Pred/Diff 이미지를 직접 확인했다. 둘 다 GT보다 밝은
경향이 남고 최종 모델 간 차이는 작다. 고정 GT Shape의 시각화로 geometry 차이를
주장하지 않는다. Bed의 하락은 epoch4보다 줄었지만 아직 V1을 넘지 못했다.

## Epoch별 등락과 TRAIN 목적함수

| Step | Pooled PSNR 차이 (dB) | Target LPIPS 차이 |
|---|---:|---:|
| 0 | 0 | 0 |
| 380 | −0.265673 | +0.00022767 |
| 760 | +0.023048 | −0.00056268 |
| 1140 | +0.433802 | −0.00125552 |
| 1520 | −0.280134 | +0.00054984 |
| 1900 | −0.016409 | −0.00017485 |

초기 삽입으로 예측이 달라지는 문제는 없었지만, 학습 이후 개선이 단조롭지 않다.
네 샘플에서 가장 좋았던1140을 사후 선택해 최종 효과로 제시하지 않는다.
고정1900 결과는 PSNR과 LPIPS의 방향이 섞여 있어 일관된 PBR 개선으로 해석하지 않는다.

마지막380 update의 네 rank 평균 TRAIN MSE는 V1 0.2599574203,
Top-1+R 0.2598259192로 차이는−0.0001315010이다. 같은 update의 차이 중앙값은
+0.0000003967이고 Top-1이 낮은 update는49.21%다. 작은 평균 loss 차이는 렌더
품질 향상이나 모집단 overfitting의 증거가 아니다. 해당 epoch의 batch 로그 평균
R residual RMS는0.0208841, fusion gate는0.0264482다. 이는 고정 Bed의 보정량이나
인과적 원인 분석이 아니다.

## 저장된 모델의 전달 경로 제어

추가 학습 없이 실행한 PBR normal 두 GPU 반복은 네 예측·metric·GT/Pred 픽셀이
정확히 같고 원래1900 예측도 모두 정확히 재현했다. SS spatial480 normal32도
기존 archive의 predicted latent·voxel을 정확히 재현했다.

PBR의 세 전달 제어도 실제 완료했고, 다섯 개의 독립 GPU 실행은 모두 자식 session
종료와 file4 저장을 확인했다. 같은 NN/source weight, 후보·matching score·Top-1 carrier,
계산된 R·source-use·fusion gate·RGB·GT·noise·camera를 유지하고 전달 항만 끈다.
두 전달을 모두 끈 조건은 실제 consumer correction RMS가 네 자산 모두0이었다.

| 제어 − Normal | Pooled PSNR 차이 (dB) | Target LPIPS 차이 | Crop LPIPS 차이 |
|---|---:|---:|---:|
| R 전달 off | −0.300740 | +0.00054480 | +0.00348281 |
| Direct Top-1 shift 전달 off | −0.014535 | +0.00001714 | +0.00018126 |
| 두 전달 모두 off | −0.293023 | +0.00053626 | +0.00338017 |

이 고정 모델과 네 샘플에서는 R 전달을 제거하면 평균 PSNR·LPIPS가 나빠진다.
Direct Top-1 전달의 평균 반응은 더 작다. 다만 direct 전달 off에서도 R은 원래
Top-1 evidence를 입력으로 받으므로 semantic matching 자체를 제거한 비교가 아니다.
두 전달 off도 같은 학습된 LoRA를 사용하므로 V1 checkpoint를 뜻하지 않는다.

R-off의 Bed는 PSNR−0.237755dB인 반면 LPIPS−0.00070023으로 지표 방향이
서로 다르다. 평균 R-off 악화를 모든 자산·지표의 개선으로 확대하지 않는다.
이 결과는 현재 모델 내부에서 R 전달의 조건부 기여를 보여주며, 재학습한 R-vs-no-R
효과나 V1 대비 일관된 최종 개선의 증거가 아니다. 저장된 다섯 record로 실제 원본
요약을 독립 재계산했고 모든 조건·집계가 일치했다. SS prior-only/Top-1/value 제어는
후속 실제 결과를 따로 확인한다.

## 후속 비교와 완료 범위

PBR 최종 저장과 중간 로컬 cache 정리 후 historical SS 실행 조건도 실제 통과했다.
공통 R96 checkpoint에서 출발해480까지 이어가는 retargeting 및 direct-NoDQ는
각각 GPU0·1, GPU2·3의 DDP2/global96/micro48/accumulation1을 사용한다.
Retargeting arm은 두 rank에서96 이후 첫 update(step97)를 실제 기록했다. Direct-NoDQ의 학습
진행은 이후 실제 runtime으로 별도 확인한다. Queued 상태를 학습 완료로 세지 않는다.

Retargeting의 기존 SS 개선 근거와 ray-wise QKV의 필수성은 구분한다.
Member matching의 후보 recall 개선도 최종 생성 품질 개선을 대신하지 않는다.
두 native PBR feature/Top-K 학습, historical 세 SS 재현·최종128×3,
native8와 legacy6의 공통 PBR DEV cascade, 독립 TRAIN seed202,
선택 조합 확인 및 최종508개10 metric은 계속 남아 있다. Full 모델은 아직 채택하지 않았다.

Canonical 결과는 `PBR-native-science-runtime-v1/` 아래의
`top1-only-stage1900-completed-evidence-v1`,
`completed1900-2models-fixed4-comparison-v1`,
`top1-final1900-actual-review-v1` 및
`historical-SS-early-boundary-revision-v1/actual-admission-v1`에 있다.
전달 제어는 `Top1-native1900-source-transfer-controls-fixed4-v1/`의
`normal-parity-v1`, `summary-v1`, `independent-review-v1`에서 확인한다.
각 실제 file4 closure와 원래 실패 이력을 유지한다.

[W&B 최종 두 모델 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/native-pbr-stage1900-quick4-2models-v1).
[W&B 고정 모델 전달 제어](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-top1-1900-transfer-controls4-v1).
