# 동일 geometry PBR의 foreground 민감도와 metric 정의

기존 완료된 Fixed SS geometry의 DEV32/32 source PBR4 결과에서 이미 계산된
foreground PSNR·GT-crop DreamSim/CLIP을 내보냈다. 새로운 NN forward, rendering,
학습, endpoint, bootstrap 또는 모델 재선정은 없다. TEST 결과를 사용하지 않았다.

## 원본과 대조한 기존 지표

Source는 camera9/10의 example별 평균이다. Source/target reference는 편집된 M1
geometry의 original material이다. 동일 generated geometry·guide·noise·triangles에서
PBR만 바꾼 비교이며, 모든 행의 mesh CD/F-Score는 같다.

| PBR arm | Source DreamSim full↓ | Source DreamSim crop↓ | Source CLIP full↑ | Source CLIP crop↑ | Source foreground PSNR↑ | Target foreground PSNR↑ |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.158346 | 0.194194 | 0.942179 | 0.935360 | 11.62652 | 12.34999 |
| Actual Top-1 + R | 0.157828 | 0.193302 | 0.941692 | 0.935090 | 11.61447 | 12.41556 |
| Top-1 + learned feature + R | 0.157985 | 0.193285 | 0.941759 | 0.935348 | 11.59947 | 12.38134 |
| Raw Top-K + learned feature + R | 0.158261 | 0.193599 | 0.941559 | 0.935625 | 11.60856 | 12.39391 |

Top-1 R−vanilla의 source DreamSim 변화는 full −0.000517370, crop −0.000892512다.
CLIP 변화는 각각 −0.000486628/−0.000269616이다. Foreground PSNR은 source에서
−0.012047753 dB, target에서 +0.065568164 dB다. 새 추론 구간을 계산하지 않은 기존
descriptive endpoint다. Source primary LPIPS/DreamSim의 기존 CI는 모두0을 포함한다.

따라서 DreamSim의 작은 점추정 개선은 GT crop에서도 관측되지만 semantic cosine과
pixel alignment까지 일관된 appearance 우위를 보이지 않는다. Crop DreamSim은 feature,
crop CLIP은 raw가 가장 좋은 점추정이다. 이 사후 sensitivity 순위를 primary 선정
기준으로 바꾸지 않으며 기존 DEV-only Top-1 R PBR freeze를 유지한다.

고정 geometry는 **PBR arm 사이**의 교란을 통제한다. 그 generated geometry가 GT와
같다는 뜻은 아니므로, 절대 rendering 오차에는 남아 있는 shape 정합 오차도 포함된다.
Foreground PSNR의 낮은 절대값을 PBR material 오류만으로 해석하지 않는다. SS geometry
개선은 CD/F-Score로, PBR appearance 개선은 같은 geometry의 arm 간 차이로 판단한다.

## 구현에서 확인한 metric 정의

- Foreground PSNR: `prediction_alpha > .5`와 `GT_alpha > .5`의 **합집합**에서
  RGB MSE를 구하고, 최소1e-12를 적용한 뒤 −10log10을 계산한다. GT-only common mask가
  아니며 output별 mask가 달라질 수 있다.
- GT crop: GT alpha>.5의 bounding box에 `ceil(.1*max(width,height))`의 padding을
  더하고 image boundary에서 자른다. 모든 모델과 prediction/reference에 같은 crop을
  적용한다. 사각형 내부의 배경은 남을 수 있으며 output별 recentering은 없다.
- Full-frame PSNR/VGG LPIPS/SSIM: 기존 float RGB를 사용한다. PSNR max_val1,
  LPIPS의 [-1,1] affine normalization, SSIM window11·sigma1.5·native zero padding을
  유지한다. DreamSim·독립 OpenAI CLIP·Inception은 원래의 clamp/round uint8 변환을
  적용한다. Source camera11은 conditioning 입력이며 score pool에 더하지 않는다.
- CD: 공통 normalized frame의20k area-weighted surface samples에 symmetric unsquared
  Euclidean distance의 두 방향 평균을 사용한다. Surface F-Score는 거리≤0.02의
  precision/recall harmonic mean이며 두 값이0이면0이다. Per-output ICP는 없다.

정의는 동결된 실제 source의 byte를 확인해 읽었다. 수치/metric code를 바꾸지 않았다.
전체48 DEV family·TEST8C=16 family, primary10 endpoint와 source-cluster protocol은
그대로다. Foreground/crop은 Supplement의 기술적 민감도이며 새로운 primary family가 아니다.

## 실제 원고와 보존 근거

`actual-Fixed-PBR4-existing-foreground-sensitivity-v580`의 table/CSV/정의·수치 JSON은
actual file4 SHA256/size 검증을 마쳤다. 총32개 numeric cell을 closed statistics와
대조했고, 두 role×네 model×아홉 metric의72개 평균을 원본 per-asset row에서 정확히
재현했다. 원본 dtype/asset 순서와 예제32개·source32개를 유지했다.

`tables/dev-selected-pbr-foreground32.tex`와 Supplement에 반영했다. 새 실제 PDF
`stage-separated-main-and-supplement-manuscript-preview-v581`은14쪽·24인용,
compiler0이다. 바뀐9–14쪽을 실제로 시각 검수했다. 기존 검수 v578의1–8쪽과 새
1–8쪽의 rendered PNG byte가 정확히 같다. Overfull/미해결 인용/missing glyph는 없고
underfull33개는 log에 남겼다. 기존14개 numeric table source/caption은 유지하고
실제 sensitivity 표 하나만 더했다. 이는 최종 TEST 논문 완료가 아니다.

실제 TEST508 cascade와 ten-metric/Main/qualitative/W&B chain은 계속 진행한다.
이 결과로 final Main·Supplement·Abstract/Results/Discussion/Conclusion을 완성하고
실제 최종 PDF를 검수·보존해야 한다. 새로운 shared Fixed PBR의 효과나 모든 모듈의
필수성을 기존 feature/raw 실험으로 주장하지 않는다.
