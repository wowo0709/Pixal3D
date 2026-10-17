# SS P1/P2: 에폭별 중간 결과

2026-10-18 00:41 KST / 2026-10-17 15:41 UTC 기준이다. 두 run 모두 학습 중이며 5에폭 최종 결과가 아니다.

## 동일 에폭 비교

고정32개 asset·32 source cluster, sample별 같은 noise, sampler12/CFG3, 생성 batch8을 사용했다. P1은 기존 R9 matcher와 native bilinear/support-normalized projection 및 C1 adapter, P2는 같은 consumer에 calibration으로 선택한 H3/Top256/T0.1을 적용했다. 두 run 모두 rejection은 OFF다.

| 시점 | P1 voxel IoU | P2 voxel IoU | P2−P1, %p | paired cluster 95% CI, %p |
| --- | ---: | ---: | ---: | --- |
| 학습 전 | 0.426715 | 0.426715 | 0 | 동일 초기 출력 |
| Epoch1 / step96 | 0.434282 | 0.437257 | +0.297 | [−0.077,+0.715] |
| Epoch2 / step192 | 0.438175 | 0.441612 | +0.344 | [+0.036,+0.645] |
| Epoch3 / step288 | 0.442308 | 0.441529 | −0.078 | [−0.668,+0.567] |
| Epoch4 / step384 | 0.448729 | 0.445345 | −0.338 | [−1.320,+0.681] |

2에폭에서는 precision도 P1 0.557612 → P2 0.561878, recall은0.671667 →0.673543, F1은0.609348 →0.612664로 함께 높았다. P2의 중간 우세가 단순 precision/recall 한쪽의 상승만으로 생긴 것은 아니다.

3에폭에서는 P2가2에폭 대비−0.008%p로 거의 같은 수준에 머물렀고 P1은+0.413%p 상승했다. P2−P1은−0.078%p이며 CI가0을 포함한다. 따라서2에폭의 양의 CI가 이후 에폭에도 유지됐다는 주장은 성립하지 않는다. P2의 초반 개선이 더 빠른 학습인지 final quality 차이인지는 아직 분리되지 않았으며, final128×3 비교 전에는 H3 bundle의 SS 생성 이득을 확정하지 않는다.

P1의4에폭은 precision0.569479, recall0.679104, F1 0.619479다. 3에폭 대비 IoU point gain은+0.642%p이며 같은32개 source cluster의 paired CI는[−0.334,+1.696]%p다. P2의4에폭은 precision0.565638, recall0.676803, F1 0.616247이며3에폭 대비IoU+0.382%p, paired CI[−0.240,+1.031]%p다. 두 run 모두 point estimate는 상승했으나 각 증가량의 CI는0을 포함한다.

같은4에폭의 P2−P1은−0.338%p, CI[−1.320,+0.681]%p다. P2의2에폭 우세가3·4에폭에도 유지되지는 않았다. H3 bundle의 SS 추가 이득은 현재까지 확인되지 않았으나, 반대로 P2가 확정적으로 열세라고 결론 내릴 근거도 부족하다. 4에폭의 두 run 비교와 각 run의3→4에폭 비교 모두32개 GT occupancy 전체 hash가 일치했다.

P1 자체의 epoch1→2 gain은 +0.389%p였고 paired CI는[−0.424,+1.215]%p였다. 따라서 모든 에폭 증가가 통계적으로 확정됐다고 표현하지 않는다. P2의 epoch1→2 point gain은+0.435%p다.

## 해석 범위

개선된 matcher bundle의 초반 이득은 관측됐지만3·4에폭에서 우세는 확인되지 않았다. 이번 비교는32개·한 sampling noise·한 training seed의 중간 결과이며 여러 에폭을 관찰한 exploratory CI다. Proposal 수, ranking 학습, feature loss와 temperature를 함께 바꾼 bundle 비교이므로 H3의 특정 loss 하나의 효과로 분해하지 않는다.

P1/P2는 각각 step103/56에서 사용자 중단 후 optimizer/scheduler/sampler를 복원했다. Native checkpoint에는 rank별 RNG가 없어 두 run이 같은 stochastic trajectory를 따랐다고 주장할 수 없다. Full 채택은 예정한 final128×3-noise 비교 및 필요한 training-seed 확인 이후 결정한다. 기존128개 aggregate를 이32개 중간 수치와 직접 차감하지 않는다.

## 실행·정합 확인

- 각 에폭 결과는 동일 panel 순서, per-asset seed와 GT foreground 수를 유지했다. Epoch2와3의32개 GT occupancy 전체 hash도 P1/P2 사이에 모두 일치했다. GT는 정합된 M1 SS latent의 decoder 복원 occupancy다.
- GPU2·3의 P1, GPU0·1의 P2가 rank당 micro48/accumulation1/global96으로 진행 중이다. 최근 compute는 약17.2초/update, peak allocated는 rank당47.75GiB다. 당시 실패 작업/OOM은 없었다.
- P1/P2 모두 step192/288/384 checkpoint 세 파일을 생성했다. 파일서버 보존과 optimizer step 확인은 `provenance/*-file4-checkpoint-receipt.json`에 기록한다.
- Epoch 비교는 CPU reporter가 양쪽의 동일 step 결과가 생기면 자동 산출한다. W&B에 epoch4 paired CI까지 기록된 것을 확인했다. 시각화 검토 기록이 통계 JSON 폴더에 들어가 reporter가 종료된 건은 해당 기록을 `visual-reviews/`로 옮기고 CPU reporter만 재개해 복구했다. 학습 두 개는 중단하지 않았다. 최종 모듈 행렬의 다섯 학습은 큐에서 대기 중이며 이 중간 결과로 취소하거나 대체하지 않는다.

## 남은 실험

[전체 모듈 비교 복원 계획](2026-10-17-ss-full-module-restoration-ko.md)에 따라 P1/P2의480 updates를 마치고, 같은 H3에서 Top1 → confidence/rejection → attention → retargeter → physical RAW2048 query의 다섯 arm을 독립 학습한다. Zero-shot 두 조건과 기존 V1/B1/B2/B3/C1을 포함한14개 구성을 동일128×3으로 최종 집계한다.

Authoritative data: file4의 `rgb_matching_v2/retarget-native-v3/runs_ss64/*/validation.jsonl`, `epoch-comparisons/epoch-0{1,2,3,4}-p2-vs-p1.json`, `training-trends/p{1,2}-epoch4-vs3.json`. [P1 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-v3-p1), [P2 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-v3-p2).
