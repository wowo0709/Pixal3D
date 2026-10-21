# Bounded attention 첫 epoch와 native source control의 중간 검증

2026-10-21 11:55 UTC. 다음 validation 완료를 추적해 bounded96,
spatial192, deformation-query192 결과를 확인했다. 실행 중인 학습을
중단하거나 새 training arm을 추가하지 않았다.

## 같은 step의 결과

모두 같은 validation32와 noise를 사용한 pooled voxel IoU다. 각 arm은
동일한 step의 기존 unbounded member-feature NoDQ와 비교한다.

| Arm | Step | IoU | 같은 step NoDQ | 차이 | Paired 95% 구간 |
|---|---:|---:|---:|---:|---:|
| Bounded prior | 96 | 0.435350 | 0.436409 | −0.001060 | [−0.002877, +0.000674] |
| Native spatial | 192 | 0.440074 | 0.434217 | +0.005857 | [−0.006402, +0.019541] |
| Deformation query | 192 | 0.434763 | 0.434217 | +0.000546 | [−0.000171, +0.001419] |
| Physical Top-1 | 288 | 0.432298 | 0.434628 | −0.002331 | [−0.007706, +0.002838] |

각 비교에서 32개의 GT voxel hash, source UID, noise seed가 일치했다.
Source-cluster bootstrap은 4,000회이며 여러 비교에 대한 보정은 하지
않았다. 모든 구간이 0을 포함한다. Training seed는 1개이고, 이는
480-step의 native128 × 3-noise 평가를 대체하는 결과가 아니다.

Spatial의 192-step point estimate는 높지만 15개에서 개선, 17개에서
하락했다. Per-asset IoU의 평균 차이는 +0.006518로 pooled 값과 구분한다.
하나씩 제외한 pooled 차이는 +0.001467~+0.009221이었다. 특정 asset을
제외하지 않고 전체 32개와 그 불확실성을 그대로 보고한다.

## 실제 학습이 동작하는지 확인

Bounded96의 저장된 conditioner를 CPU에서 strict load해 실GPU capacity
검사에서 기록한 초기 동결 상태와 대조했다.

- Frozen matching parameter 24개와 초기 기준 buffer 8개는 비트 단위로
  일치했고, conditioner 전체의 frozen SHA256도 일치했다.
- Head의 parameter tensor 9개가 모두 초기값에서 갱신됐다. Q/K/V의
  상대 L2 변화는 각각 약 3.03%, 2.39%, 4.17%다. 0으로 초기화한
  output weight도 0이 아니며, 학습 가능한 source tensor는 모두 finite다.
- 초기 Q/K/V 비교에는 고정 seed로 만든 head와 동일한 7-epoch utility를
  사용했다. 부모 retargeter의 임의 초기값을 갱신 판정에 사용하지 않았다.
- 96-step optimizer checkpoint의 state는 379개이며 모두 step96이다.
  LoRA 360개의 LR은 `1e-5`, conditioner 19개의 LR은 `1e-4`였다.
- 검사는 CPU에서 저장 파일을 읽기만 했다. GPU 추론, optimizer update,
  실행 중인 trainer의 변경은 없었다.

Head buffer만의 hash와 동결된 matcher까지 포함한 conditioner 전체의
hash는 서로 다른 범위다. 같은 범위로 대조해 전체 상태의 일치를 확인했다.
Head만 계산한 부분 hash의 차이는 학습 중 상태 변화를 뜻하지 않는다.

이 결과는 가중치가 갱신되지 않거나, 기준 함수가 변경되거나, optimizer와
LR이 누락됐다는 설명을 뒷받침하지 않는다. 다만 candidate가 정확한지,
attention이 유효하게 쓰이는지, 일반화 성능이 개선됐는지를 증명하는
검사는 아니다. 실제 attention entropy와 prior에 대한 KL도 여기서는
측정하지 않았다. 초기 출력 일치와 학습 후 성능 개선을 구분한다.

## 계속 확인할 항목

현재 결과로 full을 선정하지 않고 계획한 480-step까지 유지한다. 최종
native128 × 3, projected-V 진단, cascade32의 target/source별 8개 지표,
pilot source UID를 제외한 72개의 prior-only 진단을 연결해 두었다.
독립 training seed, PBR stage, 최종 test508과 FID/KID를 포함한 10개
지표의 범위도 유지한다. 학습을 연장하면 개선될 것이라고 가정하지 않는다.

좌우 패널에서 같은 모델에 같은 색을 사용한 중간 곡선, CSV, 원본
validation prefix, head 갱신 표와 대조 결과를 W&B 및 파일서버4에 저장했다.

- W&B: https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-member-native-controls-first-bounded96-s101-v1
- 중간 곡선·paired 결과: `member-feature-controls-v1/paired-first-bounded96-observation-v1`
- 실제 checkpoint의 head 검사: `bounded-prior-member-source-ray-v1/checkpoint96-head-audit-v1`

위 상대 경로의 canonical 부모는
`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`이다.
이번 중간 관측을 최종 모델 채택이나 전체 실험의 완료로 취급하지 않는다.
