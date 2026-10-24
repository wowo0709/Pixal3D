# SS/PBR 4에폭 중간 결과

2026-10-24 07:13 UTC의 실제 로그 기준. 진행 중인 세 학습의 기존 PID와 시작 시간을 확인했다. GPU0–3은 PBR DDP4, GPU6은 E_FIXED_RAW SS, GPU7은 voxel-query 비교 SS가 사용한다. GPU4/5는 사용하지 않는다.

## SS: 동일 epoch·평가 표본 비교

DEV32, 생성 noise 1개, training seed101, optimizer384 update의 case-macro IoU다. 최종 DEV128×3 결과와 직접 섞지 않는다.

| Arm | Case-macro IoU ↑ | Global-micro IoU ↑ |
|---|---:|---:|
| Vanilla V1 | 0.470704 | 0.432527 |
| Spatial correspondence | 0.469731 | 0.429782 |
| Raw semantic correspondence | 0.471939 | 0.436794 |
| Decoder cosine correspondence | 0.471786 | 0.436916 |
| Decoder correspondence + retargeting | 0.472579 | 0.436983 |
| Decoder correspondence, no retargeting | 0.470945 | 0.432334 |
| Fixed Top-K | 0.474012 | 0.440188 |
| Full | 0.474813 | 0.440635 |
| Full Q/K only | **0.477176** | **0.443992** |
| E_FIXED_RAW | 0.476221 | 0.443039 |

E_FIXED_RAW는 carrier/use/retargeting에 같은 raw Top-K posterior를 사용하고 V는 identity이며 learned Q/K·V/O가 없다. V1 대비 +0.005517이지만 Q/K-only보다 낮다. 학습 가능한 상대 Q/K·utility를 추가하는 E_LEARNED_RAW가 다음 matched 비교다. E_FIXED_RAW와 기존 Full은 여러 요소가 달라 순수 attention 효과로 해석하지 않는다.

원래 완료된 480-update DEV128×3 결과에서는 Decoder cosine이 0.473008로 높았다. 위 384-update DEV32 순위는 중간 결과이며 최종 순위로 대체하지 않는다. V1의 0.43대 micro와 0.47대 macro는 집계 차이이므로 모듈 향상으로 보고하지 않는다.

384개 update의 실제 training input trace와 평가 RGB·noise·GT 대응을 확인했다. 모든 실제 metric은 보존했다. 첫 CPU readout의 두 rank-label 필드에 이전 epoch 이름이 남은 것을 게시 전에 발견하여, 원본 artifact를 보존하고 `actual-v2`에서 실제 argmax로 바로잡았다. 두 집계 모두 Q/K-only가 최고다. 수정된 결과로 W&B 28개 scalar와 표·curve·artifact의 원격 완료를 확인했다.

## PBR: 3에폭 개선이 4에폭에서 유지되지 않음

GT Shape 조건, target camera8, 고정 4개 asset, seed101, optimizer1,520 update다. 생성 geometry·source preservation·전체 cascade 결과는 아니다.

| Arm | Foreground PSNR ↑ | Crop LPIPS ↓ | Crop SSIM ↑ |
|---|---:|---:|---:|
| Vanilla V1 | **15.63350** | **0.087684** | **0.952987** |
| Top-1 + retargeting | 15.35337 | 0.091200 | 0.952008 |
| Top-1 + feature | 15.43449 | 0.090897 | 0.952390 |
| Raw Top-K + feature | 15.47607 | 0.090345 | 0.952665 |

Raw Top-K는 Top-1 retargeting 대비 LPIPS가 0.000856 낮고 PSNR이 0.12270 높다. 그러나 V1보다 LPIPS가 0.002661 높고 PSNR이 0.15743 낮다. V1 대비 LPIPS 차이는 Bed 한 사례의 악화가 크다. 표본4개로 과적합이나 실패 원인을 확정하지 않는다. 초기 prediction latent와 평가 입력 대응, 네 rank의 실제 첫1,520개 training trace가 동일함을 확인했다. 이는 단순 입력 불일치 설명을 줄이지만 설계의 최종 효과를 입증하지 않는다.

## 실행 상태와 다음 판정

관측 시점 PBR은1,540/1,900, E_FIXED_RAW는395/480, voxel-query SS는121/480이다. 현재 세 프로세스는 살아 있고 update가 증가한다. GPU6/7 사용 메모리는 약94.0/92.9 GiB로 여유가 작다. Voxel-query는 FFN chunk를 줄인 재실행이며 이전147-update OOM 지점을 아직 통과하지 않았다. 장기 OOM 해결을 선언하지 않는다.

먼저 현재 5에폭을 완료하고 동일 DEV128×3에서 SS를 비교한다. PBR은 동일 생성 geometry의 DEV32와 SS→Shape→PBR 연결 평가를 수행하여 target alignment와 source preservation을 함께 확인한다. 필수 PBR V1/Top-1 seed202 학습 및 실제 두-seed 결과 집계는 큐에 연결했지만 아직 pending이다. 최종 후보를 고정한 뒤 독립 TEST508 및 10개 metric을 계산한다. 현재 어떤 arm도 최종 full로 채택하지 않았다.

실제 file4 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `consistent-fixed-epoch384-matched-v198/`, `consistent-fixed-epoch384-all-available-controls-v198/actual-v2/`, `PBR-epoch1520-four-arm-fixed4-matched-v198/`에 있다. 완료된 seed101 V1/Top-1의 frozen tensor700개가 원본 PBR16850 initializer와 정확히 같은 것도 별도로 확인했다. 수치 source나 진행 중인 NN은 이번 readout에서 변경하지 않았다.
