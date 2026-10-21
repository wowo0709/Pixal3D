# Matching prior를 보존하는 attention: 실GPU 검증과 본 학습 연결

기준 시각: 2026-10-21 10:54 UTC. 브랜치: `wowo0709/feature-retargeting`.
이는 진행 기록이며 최종 모델 선정이나 성능 향상 결론이 아니다.

## 실패 원인 가설과 한 가지 설계 변경

기존 member 기반 source-ray 모델의 최종 SS128 × 3-noise IoU는
0.438810이었다. V1 대비 차이는 +0.006160, source-cluster bootstrap
95% 구간은 [0.002125, 0.010456]이다. 그러나 member Top-1, 가중합,
scalar attention 및 mean-query spatial 비교군에 대한 차이의 구간은
모두 0을 포함한다. 따라서 semantic matching이나 attention의 독립적인
효과를 입증했다고 해석할 수 없다.

동일한 최종 가중치로 실시한 32개 frozen-weight 진단에서는 learned
attention의 effective K가 약 5.00, matching posterior만 사용했을 때는
약 10.83이었다. posterior-only의 IoU 변화는 +0.002739였지만 구간
[-0.000433, 0.006413]은 0을 포함했다. 이는 learned content가 Top-K
후보를 지나치게 좁힐 수 있다는 가설이며 확정된 실패 원인은 아니다.
target alpha 내부만 utility query에 쓰는 진단의 변화도 +0.000386으로
구간이 0을 포함했으므로 이번 trial에는 visibility 조건을 추가하지 않았다.

새 모델은 matching posterior 위에서 학습하되, 초기 Q/K·utility 함수와
현재 함수의 차이를 중심화한 뒤 `tanh`로 제한한다. 고정 bound는 1이며
후보 간 학습 log-odds 보정 범위는 최대 2이다. 초기 함수는 동결 buffer로
checkpoint에 저장한다. 초기 attention 확률은 matching posterior와 정확히
같고, 기존 utility 가중치와 학습 gradient는 유지한다. 추가 trainable
parameter, 별도 GT 학습, 새 scalar gate, foreground write mask는 없다.

source-ray 단위, Top-64 후보, native multi-view carrier, Top-1 carrier,
retargeter의 target context 및 source-use 경로는 기존 NoDQ와 같다.
변경 요인은 attention의 parameterization 하나다. 이 비교만으로 matching
decoder 자체의 효과를 주장하지 않는다.

## 실제 통과한 검사와 현재 학습

- native32: 기존 V1과 초기 예측 voxel count, GT 및 noise identity 일치.
- SS flow에서 19개 source tensor의 gradient 활성화 확인.
- 실제 global/micro batch 96, accumulation 1로 Adam 3회 업데이트 통과.
- capacity peak allocated 90.23 GiB, reserved 93.57 GiB; 379 optimizer
  states가 모두 step 3. 동결 parameter와 buffer 일치.
- capacity optimizer나 임시 가중치를 재사용하지 않고 seed 101의 새
  480-step 본 학습을 GPU1에서 시작. step 0 validation IoU 0.426715는
  기존 비교군과 같다. 시각 기준 step 23/480; 아직 첫 epoch 결과는 없다.
- GPU0은 physical Top-1, GPU2는 spatial correspondence,
  GPU3은 deformation-query 모델 학습을 계속한다. 네 학습 모두 global
  batch 96, micro 96, accumulation 1이다. 실제 사용 GPU는 0~3뿐이다.
- archive wrapper 및 기존 archive/metric-role 관련 CPU 검증 25개 통과.
  새 archive와 cascade의 실제 GPU 실행은 최종 가중치가 나온 뒤 이루어진다.

본 학습 W&B:
https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-bounded-member-prior-source-s101-v1

## 추가로 연결한 평가

본 학습·native128 × 3·wrong-projected-V32 후속 작업에 더해 독립된 평가
branch 40개를 append했다. 기존 queue descriptor와 실행 중인 neural
process는 변경하지 않았다. 40개는 SS32 archive 1개, cascade case 32개,
공식 DreamSim/CLIP 1개, 6/8-metric 집계 2개, target/source 역할 분석과
W&B 2개, SS128 × 3 비교와 W&B 2개다.

SS의 주 비교는 새 bounded 모델 대 기존 unbounded NoDQ다. 기존
Top-1/가중합/scalar/spatial 결과 및 V1/R/Direct도 함께 제시한다.
후자의 계보 차이를 분리해 해석한다. 3개 sampling noise는 독립된
training seed 3개를 의미하지 않는다.

32개 cascade에서는 기존 V1, 기존 NoDQ, 새 bounded 모델을 같은
Shape512/Shape1024 및 V1 PBR step 762로 생성한다. V1와 NoDQ의
이미 완료된 generation은 원본 그대로 재사용한다. 새 SS 예측을 기존
NoDQ 결과로 대체하거나 relabel하지 않는다. 새 SS32 producer는 새
모델의 native128 첫 32개와 voxel count·GT·noise를 대조한다.

렌더러, 카메라, 조명, 원본 source material을 M1 형상에 올린 GT 및
prediction face budget 1M을 동일하게 적용한다. source 기준 정규화와
raw surface metric을 유지하며 output별 정합/ICP는 사용하지 않는다.

모든 view의 기존 집계를 보존한 뒤, 별도 역할 분석에서 다음을 비교한다.

| 목적 | 카메라 또는 형상 | 지표 |
|---|---|---|
| target preservation/alignment | edited target camera 8 | PSNR, SSIM, M-IoU |
| source identity/material 보존과 edit 적응 | source camera 9, 10 | LPIPS, DreamSim, CLIP |
| target geometry alignment | 전체 생성 surface | CD, F-Score |

각 역할에는 다른 영상 지표도 보조적으로 기록하고, target 효과와 source
효과의 차이도 paired interval로 계산한다. 32개 development panel의
FID/KID로 최종 naturalness 결론을 내리지 않는다. FID/KID를 포함한
최종 10개 지표는 전체 최종 평가 protocol의 남은 작업이다.

## 저장, 실행 순서와 남은 판정

학습 및 평가 데이터는 기존 DeformNeXt split/cache를 사용한다. clean
80k의 재학습이나 데이터 재구축은 필요하지 않다. canonical 결과는
`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/bounded-prior-member-source-ray-v1`
아래에 둔다. 본 학습 checkpoint 공간과 SS 선행 작업을 admission
policy에 반영했다. 새 cascade는 여유 GPU에서 진행하고, 이미 계획된
PBR 4-GPU 구간은 선행 SS 작업 완료 후 기존 drain 규칙을 따른다.

SS validation 상승만으로 full을 선정하지 않는다. 기존 NoDQ 대비 paired
차이, projected-V 진단, target/source 역할별 렌더링 결과, 독립 training
seed 재현을 확인해야 한다. 새 parameterization이 좋아지더라도 semantic
matching의 독립 효과를 주장하려면 같은 parameterization을 사용하는
spatial 비교가 추가로 필요하다. PBR continuation, 기존 전체 ablation과
zero-shot 비교, 최종 test508 cache/평가 및 최종 보고서의 범위를 유지한다.
