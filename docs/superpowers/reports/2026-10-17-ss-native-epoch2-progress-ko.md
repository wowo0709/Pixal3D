# SS P1/P2: 에폭별 경향과 최종128×3 결과

2026-10-18 01:20 KST / 2026-10-17 16:20 UTC 기준이다. P1/P2 모두480 updates·5에폭 검증·최종128×3-noise 평가를 완료했다. 전체 다섯 모듈 ablation은 진행 중이다.

## 동일 에폭 비교

고정32개 asset·32 source cluster, sample별 같은 noise, sampler12/CFG3, 생성 batch8을 사용했다. P1은 기존 R9 matcher와 native bilinear/support-normalized projection 및 C1 adapter, P2는 같은 consumer에 calibration으로 선택한 H3/Top256/T0.1을 적용했다. 두 run 모두 rejection은 OFF다.

| 시점 | P1 voxel IoU | P2 voxel IoU | P2−P1, %p | paired cluster 95% CI, %p |
| --- | ---: | ---: | ---: | --- |
| 학습 전 | 0.426715 | 0.426715 | 0 | 동일 초기 출력 |
| Epoch1 / step96 | 0.434282 | 0.437257 | +0.297 | [−0.077,+0.715] |
| Epoch2 / step192 | 0.438175 | 0.441612 | +0.344 | [+0.036,+0.645] |
| Epoch3 / step288 | 0.442308 | 0.441529 | −0.078 | [−0.668,+0.567] |
| Epoch4 / step384 | 0.448729 | 0.445345 | −0.338 | [−1.320,+0.681] |
| Epoch5 / step480 | 0.451872 | 0.453171 | +0.130 | [−0.609,+0.825] |

2에폭에서는 precision도 P1 0.557612 → P2 0.561878, recall은0.671667 →0.673543, F1은0.609348 →0.612664로 함께 높았다. P2의 중간 우세가 단순 precision/recall 한쪽의 상승만으로 생긴 것은 아니다.

3에폭에서는 P2가2에폭 대비−0.008%p로 거의 같은 수준에 머물렀고 P1은+0.413%p 상승했다. P2−P1은−0.078%p이며 CI가0을 포함한다. 따라서2에폭의 양의 CI가 이후 에폭에도 유지됐다는 주장은 성립하지 않는다. P2의 초반 개선이 더 빠른 학습인지 final quality 차이인지는 아직 분리되지 않았으며, final128×3 비교 전에는 H3 bundle의 SS 생성 이득을 확정하지 않는다.

P1의4에폭은 precision0.569479, recall0.679104, F1 0.619479다. 3에폭 대비 IoU point gain은+0.642%p이며 같은32개 source cluster의 paired CI는[−0.334,+1.696]%p다. P2의4에폭은 precision0.565638, recall0.676803, F1 0.616247이며3에폭 대비IoU+0.382%p, paired CI[−0.240,+1.031]%p다. 두 run 모두 point estimate는 상승했으나 각 증가량의 CI는0을 포함한다.

같은4에폭의 P2−P1은−0.338%p, CI[−1.320,+0.681]%p다. P2의2에폭 우세가3·4에폭에도 유지되지는 않았다. H3 bundle의 SS 추가 이득은 현재까지 확인되지 않았으나, 반대로 P2가 확정적으로 열세라고 결론 내릴 근거도 부족하다. 4에폭의 두 run 비교와 각 run의3→4에폭 비교 모두32개 GT occupancy 전체 hash가 일치했다.

P1의5에폭은 precision0.572939, recall0.681370, F1 0.622468이다. 4에폭 대비IoU+0.314%p, paired CI[−0.209,+0.945]%p로 point estimate는 계속 상승했으나 증가량의 CI는0을 포함한다. P2의5에폭은 precision0.574432, recall0.682212, F1 0.623700이다. 4에폭 대비IoU+0.783%p, paired CI[+0.180,+1.481]%p로 마지막 에폭에서 개선됐다. 각 비교의32개·seed·GT occupancy hash를 확인했다.

같은5에폭의 P2−P1은+0.130%p, CI[−0.609,+0.825]%p다. P2가2에폭에서 앞서고3·4에폭에서는 뒤처진 뒤5에폭에 다시 작은 양의 차이를 보였다. 한 에폭만 골라 H3 bundle의 SS 우세나 열세를 주장하지 않는다. 마지막 epoch의 P2 개선도5에폭에서 수렴했다는 근거는 아니며, 학습 길이를 바꾸는 후속은 공통 budget의 주 비교와 구분해야 한다. 기존 V1/C1의128개 집계와 이32개 수치를 직접 차감하지 않는다.

P1 자체의 epoch1→2 gain은 +0.389%p였고 paired CI는[−0.424,+1.215]%p였다. 따라서 모든 에폭 증가가 통계적으로 확정됐다고 표현하지 않는다. P2의 epoch1→2 point gain은+0.435%p다.

## 해석 범위

개선된 matcher bundle의 초반 이득은 관측됐지만 최종5에폭의32개 비교와 아래128×3 비교 모두 우세는 확인되지 않았다. Epoch 표는32개·한 sampling noise·한 training seed이며 여러 에폭을 관찰한 exploratory CI다. Proposal 수, ranking 학습, feature loss와 temperature를 함께 바꾼 bundle 비교이므로 H3의 특정 loss 하나의 효과로 분해하지 않는다.

P1/P2는 각각 step103/56에서 사용자 중단 후 optimizer/scheduler/sampler를 복원했다. Native checkpoint에는 rank별 RNG가 없어 두 run이 같은 stochastic trajectory를 따랐다고 주장할 수 없다. Full 채택은 후속 다섯 모듈 비교와 필요한 training-seed 확인 이후 결정한다. 기존128개 aggregate를 이32개 수치와 직접 차감하지 않는다.

## 실행·정합 확인

- 각 에폭 결과는 동일 panel 순서, per-asset seed와 GT foreground 수를 유지했다. Epoch2와3의32개 GT occupancy 전체 hash도 P1/P2 사이에 모두 일치했다. GT는 정합된 M1 SS latent의 decoder 복원 occupancy다.
- P1/P2의 두 rank 모두1~480 update가 누락·중복 없이 남았고, 여섯 검증(step0/96/192/288/384/480), finite loss, 초기 정합·gradient와 final checkpoint가 완료 검사를 통과했다. `provenance/p{1,2}-final-training-audit.json`에 전체 근거를 보존했다. 실제 micro48/rank, accumulation1, world2, global96이다. Median compute는 P1 rank0/1 17.21985/17.22072초, P2 rank0/1 17.25140/17.25370초/update이며 peak allocated는 모두47.74934GiB다.
- Zero-shot 두 조건의 추가 noise 네 개와 P1/P2 final 평가가 완료됐다. GPU3 backfill과 독립 B2의 평가/학습 병행 조건은 `provenance/p2-noise2-backfill.json`, `p2-next-arm-overlap.json`에 남겼다. 현재 GPU0·1의 b2-top1과 GPU2·3의 b2-top1-r가 시작됐다. 두 arm 모두 초기 global/proj 오차0, gate gradient, 두 rank의2-update micro48/accumulation1 memory probe를 통과했으며 peak47.02867GiB다. 실제 시작 전32개 생성 IoU도 모두0.42671546671444294로 같았다. 당시 실패 작업/OOM은 없었다.
- P1/P2 모두 final480의 denoiser·conditioner·optimizer/misc 세 파일이 저장됐고, CPU에서 optimizer480과 moment 상태를 확인했다. 두 모델 모두 로컬과 file4의 파일 전체 SHA256이 일치했다. 근거는 `provenance/p{1,2}-final-checkpoint-checksum.json`이다. 중간 checkpoint의 크기·step 확인 receipt와 최종 전체 hash 대조를 구분한다.
- Epoch 비교는 CPU reporter가 양쪽의 동일 step 결과가 생기면 자동 산출한다. W&B에 epoch5 paired CI까지 기록된 것을 확인했다. 시각화 검토 기록이 통계 JSON 폴더에 들어가 reporter가 종료된 건은 해당 기록을 `visual-reviews/`로 옮기고 CPU reporter만 재개해 복구했다. 학습 두 개는 중단하지 않았다. 모듈 행렬의 첫 두 arm은 실행 중이며 나머지 세 arm도 그대로 큐에 유지한다.

## 동일128×3-noise 최종 interface 비교

동일128개·93 source cluster·noise 세 개, sampler12/CFG3, generation batch8에서 TP/FP/FN을 합산했다. 동일 source의 variant와 모든 sampling noise를 함께 묶은4,000회 paired bootstrap이다.

| 모델 | Voxel IoU | F1 |
| --- | ---: | ---: |
| 기존 V1 | 0.431926 | 0.603280 |
| 기존 C1 | 0.442214 | 0.613243 |
| P1: R9 + native bilinear/support + adapter | 0.441459 | 0.612517 |
| P2: H3 + 같은 consumer | 0.440920 | 0.611997 |

| 대비 | IoU 차이, %p | Paired95% CI, %p |
| --- | ---: | --- |
| P1−V1 | +0.953 | [+0.484,+1.422] |
| P2−V1 | +0.899 | [+0.390,+1.444] |
| P1−기존 C1 | −0.075 | [−0.346,+0.170] |
| P2−기존 C1 | −0.129 | [−0.432,+0.165] |
| P2−P1 | −0.054 | [−0.317,+0.209] |

두 모델 모두 V1보다 높은 결과를 유지했으나, native projection 보정 또는 새 matcher bundle이 기존 C1에 추가 이득을 준다는 근거는 확인되지 않았다. P2−P1도 CI가0을 포함한다. 우세를 확인하지 못한 것과 formal equivalence를 입증한 것은 구분한다. Head Top1 개선을 SS 생성 이득으로 대신 설명하지 않는다.

P1/P2 사이128개×3-noise의384개 paired GT hash가 모두 일치했다. 기존 V1/C1은 과거 전체 hash가 없어 동일 foreground count·cluster·seed 검사 범위를 유지한다. 모든 대비는 exploratory이며 한 training seed의 checkpoint를 sampling noise만 반복한 것이다. 이 두 interface run으로 confidence/rejection·attention·adapter·query의 개별 효과를 판정하지 않는다.

P1의noise0 IoU는0.440296이다. P1/P2 모두 동일 checkpoint의32개 epoch 검증과128개 final 평가의 공통32개 TP/FP/FN·seed·source cluster·GT voxel hash가 정확히 재현됐다(`provenance/p{1,2}-final128-epoch32-replay.json`). P1의32개 IoU0.451872와의 차이는 평가 population 차이이며 같은32개의 생성 결과가 바뀐 것이 아니다. 전체 근거는 `retarget-native-v3/interim-interface-pair-report.json`에 보존한다. 해당 보고서는 두 interface run의 완료이며14개 구성과 전체 goal의 완료가 아니다.

## 남은 실험

[전체 모듈 비교 복원 계획](2026-10-17-ss-full-module-restoration-ko.md)에 따라 같은 H3에서 Top1 → confidence/rejection → attention → retargeter → physical RAW2048 query의 다섯 arm을 독립 학습한다. 첫 두 arm이 시작됐고 나머지 세 arm은 예약돼 있다. Zero-shot 두 조건과 기존 V1/B1/B2/B3/C1을 포함한14개 구성을 동일128×3으로 최종 집계한다.

Authoritative data: file4의 `rgb_matching_v2/retarget-native-v3/runs_ss64/*/validation.jsonl`, `epoch-comparisons/epoch-0{1,2,3,4,5}-p2-vs-p1.json`, `training-trends/p{1,2}-epoch4-vs3.json`, `training-trends/p{1,2}-epoch5-vs4.json`, `evaluations/`, `noise-confirmation/`. [P1 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-v3-p1), [P2 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-v3-p2).
