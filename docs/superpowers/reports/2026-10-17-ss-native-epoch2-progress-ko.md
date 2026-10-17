# SS P1/P2: 2에폭 중간 결과

2026-10-17 23:44 KST / 14:44 UTC 기준이다. 두 run 모두 학습 중이며 5에폭 최종 결과가 아니다.

## 동일 에폭 비교

고정32개 asset·32 source cluster, sample별 같은 noise, sampler12/CFG3, 생성 batch8을 사용했다. P1은 기존 R9 matcher와 native bilinear/support-normalized projection 및 C1 adapter, P2는 같은 consumer에 calibration으로 선택한 H3/Top256/T0.1을 적용했다. 두 run 모두 rejection은 OFF다.

| 시점 | P1 voxel IoU | P2 voxel IoU | P2−P1, %p | paired cluster 95% CI, %p |
| --- | ---: | ---: | ---: | --- |
| 학습 전 | 0.426715 | 0.426715 | 0 | 동일 초기 출력 |
| Epoch1 / step96 | 0.434282 | 0.437257 | +0.297 | [−0.077,+0.715] |
| Epoch2 / step192 | 0.438175 | 0.441612 | +0.344 | [+0.036,+0.645] |

2에폭에서는 precision도 P1 0.557612 → P2 0.561878, recall은0.671667 →0.673543, F1은0.609348 →0.612664로 함께 높았다. P2의 중간 우세가 단순 precision/recall 한쪽의 상승만으로 생긴 것은 아니다.

P1 자체의 epoch1→2 gain은 +0.389%p였고 paired CI는[−0.424,+1.215]%p였다. 따라서 모든 에폭 증가가 통계적으로 확정됐다고 표현하지 않는다. P2의 epoch1→2 point gain은+0.435%p다.

## 해석 범위

개선된 matcher bundle을 같은 native consumer에 넣었을 때 SS 생성에서도 도움이 될 가능성이 보인다. 그러나 이번 비교는32개·한 sampling noise·한 training seed의 중간 결과이며 여러 에폭을 관찰한 exploratory CI다. Proposal 수, ranking 학습, feature loss와 temperature를 함께 바꾼 bundle 비교이므로 H3의 특정 loss 하나의 효과로 분해하지 않는다.

P1/P2는 각각 step103/56에서 사용자 중단 후 optimizer/scheduler/sampler를 복원했다. Native checkpoint에는 rank별 RNG가 없어 두 run이 같은 stochastic trajectory를 따랐다고 주장할 수 없다. Full 채택은 예정한 final128×3-noise 비교 및 필요한 training-seed 확인 이후 결정한다. 기존128개 aggregate를 이32개 중간 수치와 직접 차감하지 않는다.

## 실행·정합 확인

- 두 에폭 결과는 동일 panel 순서, per-asset seed와 GT foreground 수를 유지했다. Epoch2의32개 GT occupancy 전체 hash도 P1/P2 사이에 모두 일치했다. GT는 정합된 M1 SS latent의 decoder 복원 occupancy다.
- GPU2·3의 P1, GPU0·1의 P2가 rank당 micro48/accumulation1/global96으로 진행 중이다. 최근 compute는 약17.2초/update, peak allocated는 rank당47.75GiB다. 당시 실패 작업/OOM은 없었다.
- P1/P2 모두 step192 checkpoint 세 파일을 생성했다. P1의 optimizer step192와 파일서버 저장은 확인했으며, P2의 원격 보존 확인도 별도 provenance receipt에 남긴다.
- Epoch 비교는 CPU reporter가 양쪽의 동일 step 결과가 생기면 자동 산출한다. W&B에 epoch2 paired CI까지 기록된 것을 확인했다. 최종 모듈 행렬의 다섯 학습은 큐에서 대기 중이며 이 중간 결과로 취소하거나 대체하지 않는다.

## 남은 실험

[전체 모듈 비교 복원 계획](2026-10-17-ss-full-module-restoration-ko.md)에 따라 P1/P2의480 updates를 마치고, 같은 H3에서 Top1 → confidence/rejection → attention → retargeter → physical RAW2048 query의 다섯 arm을 독립 학습한다. Zero-shot 두 조건과 기존 V1/B1/B2/B3/C1을 포함한14개 구성을 동일128×3으로 최종 집계한다.

Authoritative data: file4의 `rgb_matching_v2/retarget-native-v3/runs_ss64/*/validation.jsonl`, `epoch-comparisons/epoch-0{1,2}-p2-vs-p1.json`. [P1 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-v3-p1), [P2 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-v3-p2).
