# 완료된 V1/R/Direct의 target/source 역할별 중간 결과

2026-10-21 11:07 UTC. 기존에 생성한 3모델 × 32개 = 96개 cascade와
288개 view를 재사용해 분석했다. GPU 추론이나 학습을 추가하지 않았다.
새 source-ray/bounded attention trial의 결과와 구분한다.

## 동일 조건에서 확인한 수치

Shape512/Shape1024와 V1 PBR step 762를 고정했다. 영상 GT는 원본 source
material을 변형된 M1 형상에 적용하고 동일 카메라·렌더러·조명으로 만든
결과다. Source 평가는 변형 전 M0 영상과의 픽셀 동일성을 뜻하지 않는다.

| 지표와 역할 | V1 | R | Direct | R − V1의 95% 구간 |
|---|---:|---:|---:|---:|
| Target8 PSNR ↑ | 24.41285 | 24.37841 | 24.45846 | [−0.39058, +0.27085] |
| Target8 SSIM ↑ | 0.950730 | 0.950968 | 0.951632 | [−0.000681, +0.001140] |
| Target8 M-IoU ↑ | 0.880467 | 0.885003 | 0.888282 | [+0.000304, +0.008992] |
| 전체 surface CD ↓ | 0.012458 | 0.012115 | 0.012264 | [−0.000739, +0.000033] |
| 전체 surface F-Score ↑ | 0.849527 | 0.861474 | 0.856702 | [+0.001558, +0.024322] |
| Source9/10 LPIPS ↓ | 0.053565 | 0.054451 | 0.053554 | [−0.000748, +0.002716] |
| Source9/10 DreamSim ↓ | 0.152908 | 0.156243 | 0.149921 | [−0.005947, +0.014210] |
| Source9/10 CLIP ↑ | 0.941296 | 0.940763 | 0.943565 | [−0.006135, +0.005242] |

View를 deformation 내부에서 먼저 평균하고 32개 source cluster를 4,000회
paired bootstrap했다. 모두 development 결과이며 training seed는 1개다.
여러 비교에 대한 보정은 하지 않았다. Surface는 source 기준 정규화를
적용하고 20k point, F-Score threshold 0.02를 사용하며 output별 ICP는 없다.

## 해석과 설계에 주는 근거

R은 V1 대비 target M-IoU +0.004536, surface F-Score +0.011947이었다.
이 두 지표의 탐색적 구간은 0보다 높다. Target RGB의 PSNR·SSIM이나
source의 LPIPS·DreamSim·CLIP 개선은 확인되지 않았다. 따라서 현재
근거를 source preservation 전반의 성능 향상으로 확대할 수 없다.

Direct는 R 대비 target M-IoU +0.003279,
구간 [+0.000264, +0.006640]이었다. 그러나 surface F-Score 차이는
−0.004772, 구간 [−0.013073, +0.002316]이며 source perceptual 지표도
모두 차이의 구간이 0을 포함했다. 하나의 IoU 상승으로 full을 정하면
안 되는 이유다. Target 효과와 source 효과의 paired interaction도
PSNR·LPIPS·DreamSim에서 구간이 0을 포함했다.

이 평가는 PBR을 고정한 **SS 변화의 downstream 영향**을 본 것이다.
SS conditioner의 feature가 학습된 PBR conditioner로 이어지는 실험이
아니다. SS의 occupancy 변화가 렌더링에 미치는 효과를 볼 수 있지만,
PBR 모듈 자체의 source appearance 적응 효과는 별도 학습·평가가 필요하다.
현재 계획된 PBR continuation 및 최종 stage 조합 평가를 유지한다.

새 bounded 모델은 기존 member-feature NoDQ와 같은 native480/global96
조건에서 학습 중이다. Learned content가 matching posterior를 과도하게
좁힌다는 가설을 확인하기 위해, 초기 진단 32개에 포함된 source UID를
모두 제외한 72개·61개 source cluster의 prior-only 평가를 추가로 고정했다.
동일한 최종 가중치와 3개 sampling noise를 사용하며 정상 결과는 기존
native128 평가에서 재사용한다. 새 학습은 없다. 이는 validation 내
pilot-source 제외 진단이며 untouched test나 독립 training seed가 아니다.
이 평가의 실제 GPU 결과는 아직 나오지 않았다.

## 실제 산출물과 남은 범위

역할 분리 집계는 기존 all-view 통계와 차이가 정확히 0임을 확인했다.
8개 지표 × 3개 비교의 CSV, PNG/PDF 및 W&B artifact를 생성하고,
파일서버4의 SHA256/size readback을 완료했다.

- W&B: https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-three-references-cascade32-roles-s101-v1
- Canonical 결과: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/cascade-view-roles-v1/three-reference-early-v1`
- 전체 8개 지표 그림: `eight-primary-metrics-v1/eight-primary-metric-effects.png`, 동일 이름의 `.pdf`.
- 원시 수치·구간: `eight-primary-metrics-v1/eight-primary-metric-effects.csv`.
- Target/source 및 interaction 전체 근거: `analysis-v1/audit.json`.

이 결과는 진행 중인 source-ray ablation, 새 bounded trial, 독립 seed
확인, PBR 학습 결과나 최종 test508 평가를 대체하지 않는다. 최종
FID/KID를 포함한 10개 metric table 및 full 모델 선정은 아직 완료되지 않았다.
