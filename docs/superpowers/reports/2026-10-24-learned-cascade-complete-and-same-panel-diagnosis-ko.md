# Learned cascade 완료와 동일 panel 진단

2026-10-24 16:50 UTC 이후 실제 결과다. E_LEARNED_RAW SS480과 공통 Top-1 R PBR1900을 연결한 DEV32 생성·render·공식 DreamSim/CLIP·8개 metric·W&B 집계를 완료했다. 새 training arm과 TEST forward는0이다.

## 실제 파이프라인 비교

V1/R의 모든 per-asset metric 값이 D1-cos·Fixed·Learned 세 study에서 정확히 같음을 검증한 뒤 합쳤다. 아래는 training seed101·generation noise1회의 DEV32다.

| 실제 SS/PBR 조합 | Source LPIPS↓ | DreamSim↓ | CLIP↑ | Target PSNR↑ | SSIM↑ | M-IoU↑ | CD↓ | F-Score↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1/V1 | 0.053099 | 0.155692 | 0.942434 | 24.47068 | 0.950488 | 0.880945 | 0.012452 | 0.849149 |
| Top-1 R/Top-1 R | 0.052104 | 0.152435 | 0.940636 | 24.57466 | 0.951284 | 0.885678 | 0.012413 | 0.851420 |
| D1-cos/Top-1 R | 0.053362 | 0.155373 | 0.942157 | 24.57729 | 0.951170 | 0.886600 | 0.012504 | 0.851308 |
| E_FIXED_RAW/Top-1 R | 0.053290 | 0.157828 | 0.941692 | 24.54433 | 0.951563 | 0.886574 | 0.012328 | 0.856566 |
| E_LEARNED_RAW/Top-1 R | 0.052894 | 0.156364 | 0.943132 | 24.62475 | 0.951161 | 0.885769 | 0.012498 | 0.854233 |

Learned−R의 탐색적 source-cluster CI95:

- Source LPIPS +0.000790 [−0.000014,+0.001637].
- DreamSim +0.003929 [−0.000371,+0.008541].
- CD +0.00008462 [−0.00014377,+0.00030834].
- F-Score +0.002813 [−0.004362,+0.010418].

모두0을 포함한다. Learned−Fixed는 source LPIPS −0.000396, DreamSim −0.001464로 점추정이 개선되지만 CD +0.00017041, F-Score −0.002333으로 악화하며 이 추가 paired 비교의 CI도0을 포함한다. 이 비교는 원래 core12/SS의 사전 contrast가 아닌 추가 탐색적 pipeline 비교다. 따라서 높은 SS128 IoU나 CLIP 하나로 Learned를 최종 모델로 채택하지 않는다.

## 중요한 실패 원인 구분: 같은 panel에서 순위가 같은가?

기존 SS128×3 평균과 cascade32/noise1 결과를 직접 대조하면, 서로 다른 population/noise의 순위를 비교하게 된다. 실제 cascade에 사용한 **동일32 asset·generation seed·GT**만 뽑아 SS를 다시 읽었다. 추가 NN은 실행하지 않았다.

| SS 모델 | 같은32/noise의 macro IoU |
|---|---:|
| V1 | 0.473279 |
| Top-1 R | 0.476967 |
| Fixed | 0.478940 |
| Learned | 0.477442 |

이 panel에서는 Learned가 Fixed보다−0.001498로 낮다. Full128×3에서는+0.001323이므로 순위가 반대다. Learned가 cascade에서 Fixed보다 좋지 않은 현상을 전부 attention 구현이나 SS loss의 문제로 단정할 수 없다. **Panel/noise 차이와 실제 단계 간 metric 차이를 먼저 분리해야 한다.** R 대비 Learned의 같은32 SS 증가도+0.000475로 작다.

## 다음 핵심 평가 범위

현재32개의 CI로 승자를 고르거나 TEST를 이용해 구조를 선택하지 않는다. 새로운 module 조합·training arm을 늘리는 대신, 이미 학습된 **R·Fixed·Learned의 같은3개 파이프라인**을 기존 DEV128의 동일 noise로 확장하는 것이 다음 핵심 측정이다. 첫32개는 그대로 재사용하고 나머지96개만 추가한다. 이는 새로운 조합 탐색이 아니라 같은 panel에서의 비교를 넓히는 작업이다.

새 평가를 연결하기 전에 각 SS의 실제128 prediction archive·동일한 noise/GT·원래 frozen Shape512/1024·같은 PBR1900·reference와 sampler의 정합 및 필요한 runtime을 확인한다. 아직 새128-case 실행을 등록하거나 완료했다고 주장하지 않는다. Source-cluster 단위 분석에서128 asset/93 source와 generation noise1회를 명시하고 기존32개 screening 결과와 구분한다. D1-cos와 모든 PBR 조합까지 확장하지 않는다.

최종 모델 결정과 실제 모듈별 matched ablation은 이 핵심 비교, 진행 중인 voxel 대조, 필요한 독립 확인을 함께 보고 진행한다. 후보 seed202 config는 준비돼 있지만 현재 이 결과만으로 새로운 E science 학습을 추가하지 않는다.

## 실제 근거

결과 root `unified-module-pipeline-v1` 아래:

- `learned-DEV32-metrics-proxy-recovery-v234/summary-v1/audit.json`: 실제8개 metric,288개 official view, source/target별 paired 통계와 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/consistent-learned-Top1-joint-native32-proxy-recovery-v234).
- `actual-five-pipeline-DEV32-comparison-v235/audit.json`: baseline per-asset exact 일치와 추가 Learned−Fixed 통계.
- `actual-same-panel-SS-cascade-diagnostic-v236/audit.json`: 같은32개의 SS seed·GT·metric 일치.

모두 실제 file4 SHA256/size closure가 있다. 원래 v216 coverage 실패는 보존했고, 수정한 완료 기록 검증을 거쳐 성공한 v234를 최종 결과로 사용한다. 모든 생성 결과는 재사용했으며 실패를 성공으로 덮어쓰지 않았다.
