# Source-ray control의 같은 epoch 중간 비교

2026-10-21 11:17 UTC snapshot. 최종 비교가 아닌 고정 validation32·고정
noise·seed101의 중간 관측이다. 기존 unbounded member-feature NoDQ의
같은 step을 기준으로 비교했다. 실행 중인 학습이나 config는 변경하지 않았다.

| 모델 | Step | IoU | 같은 step NoDQ 대비 차이 | Paired 95% 구간 |
|---|---:|---:|---:|---:|
| Physical Top-1 | 96 | 0.435399 | −0.001011 | [−0.003888, +0.001382] |
| Physical Top-1 | 192 | 0.434022 | −0.000195 | [−0.003943, +0.003025] |
| Physical Top-1 | 288 | 0.432298 | −0.002331 | [−0.007706, +0.002838] |
| Native spatial | 96 | 0.433640 | −0.002769 | [−0.008358, +0.003208] |
| Deformation query | 96 | 0.436803 | +0.000394 | [−0.000021, +0.000792] |

각 비교는 32개 source cluster를 4,000회 bootstrap했고, paired GT voxel
hash와 noise seed가 32개 모두 일치했다. 여러 비교에 대한 보정은 하지
않았다. 모든 구간이 0을 포함하므로 중간 관측으로 모듈 효과나 full을
확정하지 않는다. 새 bounded-prior 모델은 당시 step 0 검증만 완료했으며,
초기 IoU 0.426715와 per-asset count는 다른 모델과 정확히 같았다.

Physical Top-1은 Top-K에서 하나의 correspondence feature만 사용하고
Q/K·utility-score module을 제거한 arm이다. Native spatial은 원래 카메라
projection에 따른 correspondence를 사용한다. 이 둘을 uniform-score
control이나 correspondence 제거로 해석하지 않는다.

## 집계 방식 확인

Epoch quick IoU와 최종 SS paired comparison의 주 IoU는 모두
`sum(TP) / sum(TP+FP+FN)`이다. Epoch quick IoU도 per-asset IoU의 평균이
아니다. `quick_metrics._ss_metrics`가 32개의 occupancy grid를 한 번에
`binary_metrics`에 넣는 코드와, 저장된 per-asset count 합계를 대조해
확인했다. 이전 내부 인계 메모의 per-asset 평균이라는 설명은 잘못됐고,
수치 자체의 변경은 없다.

예를 들어 NoDQ step288의 저장 quick IoU와 count로 재구성한 값은 둘 다
0.4346284277이며 per-asset 평균은 0.4724400247이다. Physical Top-1은
각각 0.4322975955와 0.4698072724다. 두 집계를 혼용하지 않고 snapshot
CSV 및 W&B에 별도 이름으로 기록했다.

## 실제 산출물과 다음 판정

진행 중인 validation 파일의 읽기 시점 prefix를 별도로 복사해 원본 경로,
SHA256 및 실제 neural PID/birth와 함께 보관했다. 원본 파일은 그대로
계속 append된다. 같은-epoch 곡선 PNG/PDF, count 집계 CSV 및 paired
bootstrap 결과를 파일서버4에 저장하고 SHA256/size readback을 완료했다.

- W&B: https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-member-native-controls-epoch288-s101-v1
- Canonical snapshot: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/member-feature-controls-v1/paired-epoch288-observation-v1`

최종 480-step 가중치의 native128 × 3-noise, projected-V 진단 및 실제
cascade32의 target/source 지표까지 확인한다. 이번 partial 결과로
학습을 중단하거나 모델을 선정하지 않는다. 예정된 PBR, 독립 training
seed 및 최종 test508/10-metric 평가의 범위를 유지한다.
