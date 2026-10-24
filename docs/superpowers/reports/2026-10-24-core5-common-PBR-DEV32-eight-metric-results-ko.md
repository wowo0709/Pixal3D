# 공통 PBR을 사용한 SS 5개 모델의 DEV32 생성 결과

2026-10-24 10:33 UTC. V1, C_DECODER, C_TOP1_V, C_FIXED_TOPK, FULL 각각의 실제 SS prediction으로 생성한 160개 case와 480개 view의 평가가 완료됐다. 같은 32개 asset·32개 source cluster, training seed101, 생성 noise 1회다. 원본 Shape5124210·Shape102416850과 같은 학습된 vanilla V1 PBR1900을 사용했다. GT geometry는 생성에 사용하지 않았고, 생성 후 원본 material을 편집된 M1 geometry에 렌더링한 동일한 reference와 비교했다.

이 결과는 SS 모델 변경이 downstream 생성에 미치는 효과다. 새 E_FIXED_RAW/E_LEARNED_RAW와 학습된 Raw Top-K PBR을 결합한 결과가 아니다. 서로 다른 SS support에서는 sparse texture noise 자체가 bitwise 동일하다고 주장하지 않는다. 원본 native cascade의 RNG protocol은 같다.

## 실제 평균

Source preservation은 source camera9/10의 평균이다. Full-frame LPIPS·DreamSim·독립 OpenAI CLIP을 사용한다. DreamSim과 CLIP은 원본 official weights, GPU FP32·TF32 off·AMP off로 계산했다.

| SS 모델 | Source LPIPS ↓ | Source DreamSim ↓ | Source CLIP ↑ |
|---|---:|---:|---:|
| V1 | 0.053644 | 0.155405 | 0.940637 |
| C_DECODER | **0.053502** | **0.149162** | 0.942606 |
| C_TOP1_V | 0.054325 | 0.157732 | 0.943192 |
| C_FIXED_TOPK | 0.053833 | 0.151686 | 0.939910 |
| FULL | 0.054224 | 0.154300 | **0.946073** |

Target alignment의 이미지 지표는 edited target camera8이다. PSNR·SSIM은 full-frame이며 foreground PSNR은 별도의 보조 지표다. CD는 source-defined 정규화 좌표에서 양쪽 surface 20k samples의 symmetric L1 Chamfer, F-Score threshold는0.02이며 ICP나 결과별 정합을 수행하지 않았다. Surface F-Score와 voxel F1은 서로 다른 지표다.

| SS 모델 | Target PSNR ↑ | Target SSIM ↑ | Target mask IoU ↑ | Surface CD ↓ | Surface F-Score ↑ |
|---|---:|---:|---:|---:|---:|
| V1 | 24.26623 | 0.950695 | 0.880957 | 0.01245167 | 0.849149 |
| C_DECODER | 24.36422 | 0.950655 | 0.885747 | 0.01241337 | 0.851420 |
| C_TOP1_V | 24.41518 | 0.950808 | 0.886273 | 0.01233150 | 0.856605 |
| C_FIXED_TOPK | **24.41942** | 0.949844 | **0.889989** | 0.01235644 | 0.857143 |
| FULL | 24.31990 | **0.951063** | 0.887708 | **0.01233115** | **0.857392** |

## 모듈별 해석과 남은 불확실성

같은 source cluster를 paired bootstrap 4,000회 resample한 아래95% CI는 탐색용 DEV32의 보정 전 구간이다. 여러 지표를 확인한 뒤 선택한 결과이며, 독립 training seed나 최종 TEST 확인을 대신하지 않는다. 원래 SS128 모듈 contrast family도 별도로 유지한다.

| 비교 | 관측 변화 | 95% CI | 해석 |
|---|---|---|---|
| FULL − V1 | Source CLIP +0.005436 | [+0.001250, +0.009938] | 의미적 유사도 개선 신호 |
| FULL − V1 | Target mask IoU +0.006751 | [+0.001728, +0.012331] | 생성 silhouette 개선 신호 |
| C_TOP1_V − C_DECODER | Surface F-Score +0.005184 | [+0.001497, +0.009446] | Top-1에 V/output 변환을 더했을 때 surface 개선 신호 |
| C_FIXED_TOPK − C_TOP1_V | Target mask IoU +0.003717 | [+0.000258, +0.007653] | Top-K 결합의 silhouette 개선 신호 |
| FULL − C_FIXED_TOPK | Source CLIP +0.006163 | [+0.001412, +0.011157] | Learned Q/K·utility가 고정 결합보다 의미적 유사도를 높인 신호 |
| FULL − C_FIXED_TOPK | Source DreamSim +0.002614 | [−0.004358, +0.009711] | 낮을수록 좋은 지표의 평균은 악화; 구간은0 포함 |
| FULL − C_FIXED_TOPK | Surface F-Score +0.000249 | [−0.003053, +0.003549] | 고정 Top-K 대비 surface의 추가 개선은 미확인 |

V/output 변환, Top-K, learned attention을 추가할 때 모든 지표가 단조롭게 상승하지 않는다. C_DECODER의 source LPIPS·DreamSim 평균이 가장 좋고, FULL의 CLIP 평균이 가장 좋다. 따라서 learned attention이 전혀 효과가 없다고 단정할 수도 없지만 source detail 보존이나 geometry까지 전반적으로 개선했다고 할 수도 없다. FULL과 V1의 PSNR·SSIM·CD·surface F-Score 차이도 위 bootstrap CI가0을 포함한다. CLIP 상승만을 근거로 FULL을 최종 채택하지 않는다.

현재 완료된 SS128×3에서는 같은 raw posterior를 carrier/use/retargeting에 일관되게 사용하는 E_FIXED_RAW가 가장 높은 IoU를 보였다. 이 모델은 learned Q/K·V/O가 없다. E_LEARNED_RAW는 같은 전달 구조와 identity V를 유지한 채 상대 Q/K·utility를 추가하여 학습 중이다. 이 matched 비교와 기존 E_FIXED_RAW의 실제 joint cascade 평가가 attention 및 전달 경로의 설계 판단에 필요하다. 새 주변 arm은 추가하지 않았다.

## 결과의 정합성과 저장

160개 생성 case의 성공한 실제 child terminal, 자기 SS prediction, 원본 Shape checkpoint, 공통 PBR checkpoint, camera/GT와 native guide 계약을 확인했다. 480개 official score는 실제 shared NPZ SHA와 GT float RGB/alpha 및 공통 reference crop에 결합돼 있다. 이전 두 모델의6개 지표와 foreground PSNR을 다시 대조한 최대 평균 차이는2.22e−16으로 반올림 범위 내다. CPU에서 neural metric forward나 학습을 실행하지 않았다.

Canonical file4 경로:

`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/PBR-native-science-runtime-v1/unified-module-pipeline-v1/common-trained-V1-PBR-cascade32-v167/`

`coverage160-v1/`, `official480-v1/native/evaluation.json`, `eight-metric-summary-v1/statistics.json`와 `audit.json`이 실제 결과다. 각 phase의 file4 receipt가 존재한다. [W&B의 모델별 표·5개 실제 GT/Pred 패널·paired delta와 CI](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-core5-common-V1-PBR-cascade32-v167)에 기록됐다.

FID/KID, 다른 training seed, 완료되지 않은 원래 SS 대조군, 공통 생성 geometry에서의 PBR 비교, candidate joint cascade, VAL36 재현 확인, 최종 TEST508과 전체10개 metric은 여전히 남아 있다. 이번 DEV32 평가를 전체 실험 완료나 최종 구조 확정으로 취급하지 않는다.
