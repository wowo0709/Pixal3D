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

M-IoU는 최종 렌더 alpha>0.5의2D mask를 비교한다. Native PBR의6번째 attribute는 opacity이므로 이 지표에는 geometry와 재질 투명도가 함께 반영된다. Geometry만의 개선을 판단할 때는 raw mesh의 CD·surface F-Score를 별도로 확인한다.

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

## 공통 geometry PBR 비교에서 발견한 렌더 교란 요인

후속 PBR4 평가의 첫 공통 case를 실제 배열로 대조했다. 네 모델의 raw vertices/faces와 native guide·초기 texture noise는 같았지만, 원본 renderer가 각 case에서 native mesh simplification을 별도로 실행해 최종 렌더 triangles가 달라졌다. 같은2,966,356-face raw mesh에서 renderer face 수가998,920/998,470/999,948/998,620으로 달랐다. 원본 surface 지표는 유지되지만 PBR만의 렌더 효과를 분리할 때에는 같은 face budget만으로 충분하지 않다.

이를 해결하는 `fixed-render-geometry-PBR4-native32-v203` successor는 asset별 vanilla의 실제 단순화 mesh를 한 번 고정한다. 각 모델의 원래 PBR attributes·sparse support를 그대로 유지한 render-only 입력을 만들고, 기존 renderer를 수정 없이 호출한다. 입력이 이미 budget 안에 있으므로 추가 simplification을 건너뛴다. 실제 렌더 vertices/faces가 고정 mesh와 정확히 같은지 검사하며, 원래 sampling 결과·raw mesh·surface metric은 보존한다. 원래 공통-geometry PBR4 렌더 결과는 같은 budget의 탐색 결과로 보존하고, PBR 선정에는 이 수정 평가를 요구한다. 새 학습 arm이나 SS/Shape/PBR generation forward는 추가하지 않는다.

11:03 UTC에 실제 GPU pilot3개가 모두 성공했다. Baseline을 포함한4개 렌더의 실제 vertices/faces가 모두 동일한998,920-face mesh였고,12개 view의 GT float RGB/alpha buffer도 정확히 같았다. 첫 case에서 기존 모델별2–12 pixel의 binary-alpha 차이는 수정 후0이었다. 이것은 렌더 조건이 통제됐다는 증거이며 모델 성능 순위의 증거는 아니다. Opacity 자체는 각 모델의 원래 값을 유지하므로 다른 asset에서는 재질에 따른 alpha 차이가 가능하다.

관련 CPU50개 검사도 통과했다. Pilot 성공 후 남은93개 GPU render-only case, 전체128개 case의 exact-geometry CPU gate, coverage, official384-view metric 평가와 최종 집계를 실제 큐에 연결했다. Source snapshot과 plan·queue receipt는 file4에 보관했다. 원래 학습·sampling 작업과 source pin은 유지했다.

## 고정 mesh에서의 PBR 3개 모델 DEV32 중간 결과

11:41 UTC에 vanilla, Top-1 retargeting, Top-1+feature PBR 각각32개의 수정 평가가 완료됐다. 같은 V1 SS·생성 Shape와 asset별 고정 렌더 triangles, native guide, 초기 texture noise·sampler를 사용한다.32개 asset/source cluster, training seed101, 생성 noise1회다. 원본 native sampling 성공 기록과 수정 render/shared-score의 실제 성공 terminal을 각각 확인했다. CPU에서96개 실제 렌더 mesh와288개 view의 GT float RGB/alpha를 다시 읽어 일치함을 확인했다. Canonical file4 receipt에 결합된737개 입력을 확인했고, 앞서 측정한 vanilla/Top-1 두 모델의 평균도1e−12 이내로 재현했다.

| Metric | Vanilla | Top-1 R | Top-1 R+feature | Feature − Top-1 R |
|---|---:|---:|---:|---:|
| Source9/10 full-frame LPIPS ↓ | 0.05309933 | 0.05298409 | 0.05301938 | +0.00003530 |
| Target8 full-frame PSNR ↑ | 24.47068 | 24.54852 | 24.51148 | −0.03704 |
| Target8 full-frame SSIM ↑ | 0.9504881 | 0.9505851 | 0.9505358 | −0.0000493 |
| Target8 alpha-mask IoU ↑ | 0.8809446 | 0.8809821 | 0.8810029 | +0.0000208 |
| Target8 foreground PSNR（보조）↑ | 12.36167 | 12.44070 | 12.40322 | −0.03748 |

Top-1 R은 vanilla 대비 source LPIPS −0.00011524, target PSNR +0.07784 dB이며 LPIPS·PSNR·SSIM 각각에서 개선된 asset은17/32개다. 평균 변화가 작아 retargeting의 강한 PBR 성능 향상을 입증한 결과로 취급하지 않는다. Feature head를 더하면 Top-1 R 대비 source LPIPS 개선15/32개, target PSNR·SSIM 개선13/32개이며, 세 평균은 소폭 악화했다. 현재 이 전달 방식에서 learned feature correction의 추가 이득은 확인되지 않았다. Raw mode는 아직 완료되지 않았다.

세 모델의 mask 차이는1개 asset에서 발생했으며, geometry와 raw surface CD/F-Score는 동일하므로 opacity에 따른 차이로 해석한다. DreamSim·CLIP과 원래4개 모델 전체 paired CI는 아직 이 중간 읽기에 포함하지 않았다. 추가 통계 검정이나 모델 선정도 수행하지 않았다.

이 PBR 비교는 원본 cascade에서의 noise를 그대로 재사용하지 않고, 같은 generated geometry에서 공통 seed로 texture noise를 새로 만들어 네 PBR 모델에 동일하게 전달한다. 따라서 위 vanilla 수치를 앞부분 SS5개 모델 평가의 vanilla 수치와 섞어 비교하지 않는다. FID/KID와 독립 training seed, 최종 TEST508도 여전히 필요하다.

## PBR arm 이름과 실제 attention 전달 경로

현재 실행 소스 `pixal3d/modules/r_mediated_pbr_source.py`와 `r_mediated_source_contrast.py`의 `_input_chunk`를 대조했다. PBR query/key는1024-D이고 DINO+NAF value는2048-D다. R 입력은 appearance2048 + target-context1024 =3072채널이며 deformation-query placeholder는 없다.

| PBR mode | R에 전달되는 appearance 입력 | 학습 source tensor 수 |
|---|---|---:|
| `top1-only` | 선택된 단일 semantic feature `v1` | 10 |
| `top1-plus-feature` | `v1 + Δv_attention` | 19 |
| `raw-topk-plus-feature` | `v_raw_weighted + Δv_attention` | 19 |

두 feature mode 모두 같은64후보 기반의4-head learned feature readout을 실행한다. Raw mode의 `v_raw_weighted`는 그 readout의 같은 posterior를 head 평균해 원본 value를 가중 합한 것이며, feature magnitude를 그대로 유지한다. 두 mode 모두 consumer 쪽 carrier와 독립 source-use 판단은 Top-1을 유지하고, feature correction은 R을 통해 전달한다. R 밖으로 직접 더하는 attention correction 경로는 없다.

따라서 `top1-plus-feature`라는 이름은 전체 경로가 후보1개만 사용한다는 뜻이 아니다. 이 mode와 Raw mode의 비교는 **동일한 attention head에서 R에 원본 가중 feature를 전달하는 방식의 효과**다. Attention의 물리적 유무는 `top1-only`와 feature mode를 비교하며, 이 비교에는 learned feature transform도 함께 추가된다. SS의 `C_TOP1_V`와 동일한 arm이라고 해석하지 않는다. 이 구분은 새 실험을 추가하지 않고, 현재 PBR 비교의 인과적 설명 범위를 명확히 하기 위한 것이다.
