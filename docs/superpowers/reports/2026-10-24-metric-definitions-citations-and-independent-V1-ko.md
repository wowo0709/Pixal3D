# 실제 metric 정의·인용 대조와 독립 V1 학습 완료

2026-10-24. 기존 metric numerical code를 바꾸지 않고, 실제 구현과 논문의 metric 정의를 대조했다. 원논문7개를 추가해 현재 초안은 총24개 참고문헌을 인용한다. 최종 checkpoint/model/protocol freeze나 TEST 측정을 완료했다는 뜻은 아니다.

## 독립 V1 학습의 실제 종료

SS seed202 V1의 science480은 실제 NN child PID1539966·birth1792850563.21이 returncode0으로 끝났다. Wrapper의 `PASS_ACTUAL_NATIVE_TASK_PHASE`와 native `PASS_ACTUAL_COLD_NATIVE_TASK_SS480_SCIENCE`를 확인했다.480 optimizer updates·5epochs, 초기 native32 identity, epoch validation32가 기록돼 있다. Wrapper/native/checkpoint의 file4 봉인 후, `seal480`과 cache 정리 job이 완료됐고 GPU1의 final128 noise0 평가 wrapper PID1737214가 실제로 시작됐다.

학습 종료 quick 값 `voxel_iou=0.4347738955`는 epoch용 panel/집계다. 이를 논문의 동일 DEV128×3 per-asset macro IoU와 직접 비교하지 않는다. 후속 관측에서 final128 noise0/1/2의 실제 evaluation 파일3개가 모두 생성됐음을 확인했다. 원래 seed101과의 동일 macro 집계 readout은 남아 있다. R seed202·voxel 대조·DEV128 cascade는 별도 진행 상태이며, V1 완료만으로 독립 확인 전체가 끝난 것은 아니다. Fixed/Learned의 실제 기전 관측은 이후 완료돼 [핵심 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)에 반영했다.

## Metric별 실제 측정 정의

| 목적 | 지표 | 실제 설정/해석 |
|---|---|---|
| Source preservation | LPIPS↓ | Frozen VGG LPIPS, 저장 float RGB를 unit range에서[-1,1]로 변환, source camera9/10의 평균 |
| Source preservation | DreamSim↓ | 공식 trained ensemble, image embedding cosine distance; source camera9/10 |
| Source preservation | CLIP↑ | 별도 **original OpenAI ViT-B/16** image encoder의 image–image cosine; caption/text score가 아님 |
| Global naturalness | FID↓ | Official Inception-v3-compat2048, 전체508×3=1,524 image/pool의 distribution 비교 |
| Global naturalness | KID↓ | 같은 전체 pool, cubic polynomial kernel; subset100회·크기1,000·seed42017, signed estimate 유지 |
| Target alignment | PSNR↑ | Camera8 full-frame float RGB, nominal maximum1, dB |
| Target alignment | SSIM↑ | Camera8, Gaussian11×11·sigma1.5·zero padding5·unit-range constants, RGB/pixel 평균 |
| Target alignment | M-IoU↑ | Camera8 prediction/GT alpha>.5인2D mask의 intersection/union |
| Target alignment | CD↓ | 양방향 평균 최근접 **비제곱 Euclidean 거리**를 더한 뒤2로 나눔 |
| Target alignment | F-Score↑ | Surface point20k씩, precision/recall의 harmonic mean, threshold0.02, fraction[0,1] |

코드 이름 `chamfer_L1`은 좌표별 Manhattan distance를 뜻하지 않는다. `cKDTree.query`의 Euclidean distance를 제곱하지 않고 평균하는 구현이다. 논문에 이를 식으로 명시했다. 원논문의 다른 sampling/threshold나 CD convention을 그대로 사용했다고 쓰지 않는다. Source-defined frame을 유지하고 output별 ICP·centering·scale fitting을 하지 않는다.

Source reference는 **M1 geometry에 original material을 적용한 같은-view render**다. 변형 전 M0 이미지를 pixel GT로 쓰지 않는다. 실제 reference RGB/alpha는 모델 간 동일 bytes를 요구한다. 이 source 지표는 생성 geometry의 오차 영향도 포함하므로, 재질 성능만의 causal effect로 해석하지 않는다. PBR 단독 효과는 이미 수행한 동일 SS/native guide/render triangles의 짝비교를 별도로 사용한다.

## Preprocessing과 통계

- Pixel/LPIPS는 저장 float RGB를 사용한다. DreamSim/CLIP/Inception은 공통으로 clamp[0,1]·round uint8를 적용한 뒤 각 공식 preprocessing을 따른다. Primary는 full frame이며 GT alpha만으로 만든 crop은 보조다.
- 공식 DreamSim·CLIP·Inception forward는 FP32·TF32off·noAMP의 GPU 실행이다. Inception feature의 FID/KID reduction은 `torch-fidelity0.3.0`의 float64 통계다. CPU에서 이 feature 통계를 계산하는 것과 CPU NN forward를 구분한다.
- Main scalar mean은 deformation example에 동일 가중치를 주고 source camera9/10은 asset 안에서 먼저 평균한다. Source-cluster bootstrap과 equal-source mean도 기록한다.
- 현재 final contract는 selected model 대 V1/R의 self-comparison을 제거한 C개 contrast, **8개 scalar endpoint×C**의 primary family를 정의한다. Bonferroni-adjusted cluster interval과 별도의 descriptive contrast를 사용한다. Geometry의 중심 지표 CD/F-Score, appearance의 중심 지표 LPIPS/DreamSim이라는 과학적 우선순위와 통계 보정 family를 구분한다.
- KID subset 표준편차는 source-cluster CI가 아니다. Global FID/KID의 per-asset significance나 인간 선호를 주장하지 않는다. CI가0을 포함하는 것을 equivalence/non-inferiority의 증거로 쓰지 않는다.
- 현재 completed DEV의 training seed101와 독립 seed202 확인을 구별한다. Generation noise3회를 training seed3개로 해석하지 않는다. Final TEST 모델/metric/통계 설정은 실제 선택 이후 freeze한다.

## 추가한 원논문과 범위

| 지표 | 확인한 primary source | 사용하는 근거 |
|---|---|---|
| LPIPS | [Zhang et al.,2018](https://arxiv.org/abs/1801.03924) | Learned perceptual metric의 출처 |
| DreamSim | [Fu et al.,2023](https://arxiv.org/abs/2306.09344) | Human-judgment-tuned visual similarity metric |
| CLIP | [Radford et al.,2021](https://arxiv.org/abs/2103.00020) | Original image representation; 현재 image cosine 사용은 구현으로 명시 |
| FID | [Heusel et al.,2017](https://arxiv.org/abs/1706.08500) | Inception distribution distance의 출처 |
| KID | [Bińkowski et al.,2018](https://arxiv.org/abs/1801.01401) | Kernel Inception Distance의 출처 |
| SSIM | [Wang et al.,2004, 저자 페이지](https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html) | Structural similarity의 원논문과 서지 |
| Surface F-Score | [Tatarchenko et al.,2019](https://arxiv.org/abs/1905.03678) | Surface precision/recall·harmonic mean 정의; 우리 threshold/sampling은 별도 명시 |

Surface 원문의 appendix에서 precision/recall과 threshold 기반 F-score 정의를 확인했다. 문헌상의 평가 protocol 전체나 제안 threshold를 그대로 재현했다고 주장하지 않는다. PSNR과2D mask IoU는 실제 수식/집계로 명시하고 불필요한 최초 출처 주장을 추가하지 않았다.

## 구현 근거

- [Surface sampling/metric](../../../experiments/deformnext_retargeting/surface_metrics.py)
- [Pixel/VGG metric](../../../pixal3d/utils/loss_utils.py)
- [고정 reference scoring](../../../experiments/deformnext_retargeting/score_cascade_fixed_reference.py)
- [DreamSim/CLIP evaluator](../../../experiments/deformnext_retargeting/evaluate_official_perceptual.py), [RGB/crop conversion](../../../experiments/deformnext_retargeting/perceptual_image_inputs.py)
- [Inception extraction](../../../experiments/deformnext_retargeting/extract_official_inception.py), [FID/KID reduction](../../../experiments/deformnext_retargeting/inception_distribution_metrics.py)
- [Final10개 metric/통계 contract](../../../experiments/deformnext_retargeting/selected_final_metric_summary.py)

Metric registry의 실제 CLIP file은 `weights/clip/ViT-B-16.pt`이며 SHA256은 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`다. DreamSim 내부의 adapted CLIP을 별도 CLIP 지표로 잘못 사용하는 것이 아님을 source/registry와 기존 official 결과에서 확인했다. Registry·source identity는 canonical audit에 보존한다.
