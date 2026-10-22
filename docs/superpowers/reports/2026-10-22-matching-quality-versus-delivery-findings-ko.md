# 매칭 정확도와 feature 전달을 분리한 실측 결론

## PBR: 저장 weight에서 direct 전달 제거가 품질을 회복시킨다

동일 Direct1905 NN/source,32개 개발 자산/source cluster, 동일 RGB·noise·GT Shape·
target08에서 비교했다. 초기 normal4 GPU 검사에서9개 입력 identity,11개 metric,
GT/Pred/Diff pixel이 기존 결과와 정확히 같았다. 개입은 optimizer update 없이 수행했다.

| 조건 | Foreground PSNR↑ | Target crop LPIPS↓ | Target crop SSIM↑ |
|---|---:|---:|---:|
| V1 별도 학습 baseline | 15.605368 | 0.1204092 | 0.9208989 |
| R-only 별도 학습 | 15.891211 | 0.1164771 | 0.9227996 |
| Direct1905 정상 | 14.624105 | 0.1261741 | 0.9100598 |
| 같은 Direct, matching prior 복원 | 14.607129 | 0.1268505 | 0.9065113 |
| 같은 Direct, direct 전달만 제거 | 15.742467 | 0.1186069 | 0.9219504 |
| 같은 Direct, 두 개입 결합 | 15.742945 | 0.1184150 | 0.9220249 |

Direct 전달 제거−정상의 PSNR 차이는+1.118363dB, paired source-cluster
CI95[+0.272607,+1.914741]다. SSIM 차이도+0.011891,
CI[+0.001832,+0.024153]다. LPIPS 차이는−0.007567이나 CI는0을 포함한다.
Matching prior 복원만으로는 회복되지 않았다. Direct를 제거한 뒤 prior 복원을
추가한 PSNR 차이는+0.000477dB, CI[−0.006662,+0.008014]로 작고 불확실하다.
Bootstrap4000회/seed20261022의 보정되지 않은 개발 interval이다.

따라서 **이 저장 모델에서는 R을 우회하는 feature correction 경로가 PSNR/SSIM을
해친다는 근거**가 있다. 매칭 posterior의 GT mass 저하가 실제로 관측됐더라도,
그것만 복원하는 것이 이 모델의 품질 회복을 설명하지 못한다. 이것은 matching이
불필요하다는 결론도 아니다. 기존 outside-proposal value 진단에서는 올바른
correspondence value의 perceptual 이점이 있었다.

전달 제거 조건도 정상 head와 RAW DQ aggregation, R 입력/출력,5120/2048 physical
channel을 유지한다. 이를 새로 학습한 R-only, NoScore, NoCorr 또는 no-DQ라고
부르면 안 된다. V1 대비 평균이 높다는 것만으로 개선이 입증된 것도 아니다.
Target crop LPIPS는 이 단계의 target 진단이며 논문의 source preservation이 아니다.

## SS: 매칭을 개선해도 현재 전달 구조에서는 native 이득이 작다

Bounded source-contrast19는 TRAIN1536/7에폭112회, global96/micro96/acc1을
완료했다. Frozen SS4210, 기존 plain contrast와 같은112개 batch 순서와 teacher
supervision을 사용했다. 실제 native32 초기 출력 보존·micro96 backward와 학습 후
matching/feature/native 관측을 모두 완료했다.

| 동일32개 개발 자산 | Plain contrast | Bounded contrast |
|---|---:|---:|
| Native IoU | 0.4305255 | 0.4305918 |
| Exact ray-set Top-1 hit | 18.5650% | 22.8752% |
| Exact ray-set GT mass | 15.8846% | 17.7564% |
| Source foreground feature MSE / 초기 MSE | 0.9924867 | 0.9920840 |
| Source foreground correction cosine | 0.08935 | 0.09092 |

매칭은 source-camera ray의 set-valued GT를 사용한 CPU cached-feature 관측이다.
Top-1은+4.3102pp, GT mass는+1.8718pp이며 각 CI는 양수다. 그러나 IoU 차이는
**+0.0000663, CI[−0.0004594,+0.0006357]**로 개선이 확인되지 않는다. 올바른
candidate에 확률을 더 주는 것과, source feature를 필요한 방향으로 바꾸는 것,
최종 3D 결과를 개선하는 것은 별개다. 단순 correction 배율 증가는 이전 진단에서
feature MSE를 악화시켰으므로 해결책으로 채택하지 않는다.

Plain contrast의 source-disjoint72/61clusters×3noise 확인도 완료됐다.
Original19 대비 IoU는0.4125542→0.4145314, 차이+0.0019772이나
CI[−0.0005664,+0.0043781]가0을 포함한다. 사전 조건을 충족하지 않아 warm
joint480 admission은 `HOLD_NO_REPLICATED_SOURCE_INPUT_GAIN`이다. 실행을
완료한 것으로 바꾸거나 gate를 우회하지 않는다. 새 bounded 후보를 이72개에서
확인한 결과는 아니다. 독립 학습 seed 확인도 남아 있다.

## 다음 구조에서 확인할 두 변경 — 아직 구현·학습 전

Source-camera ray query, target stream 보존, own-source contrast, candidate64와
matching decoder/score를 유지하면서 다음 두 구조를 순서대로 비교한다.

1. R-mediated feature: head가 만든 변화를 R 입력으로 전달하고, R을 우회하는
   추가 head correction 전달을 제거한다. 기존 SourceContrast19와 비교해 전달
   경로의 효과를 SS에서 직접 확인한다.
2. R-mediated raw Top-K: 같은 구조에서 실제 Top-K weighted appearance도 R 입력에
   제공한다. Learned value transform만으로 필요한 source displacement를 표현하는
   병목인지 확인한다. 직접 bypass를 다시 추가하거나 correction 크기만 키우지 않는다.

두 후보는 같은 frozen SS/teacher pilot의 source module 예산과 batch 순서로
비교한다. R의 cold zero output으로 native 초기 함수를 보존하고, 새 구조의 실제
micro96 backward·초기 native32 identity·유효 gradient 경로를 검증한 뒤 학습한다.
기존 학습 source weight를 의미가 다른 구조에 강제로 로드하지 않는다. 새 schema로
구분한다. Matching/depth/latent cache는 재사용한다. 이 목록은 다음 구현 범위이며
이미 실행된 결과가 아니다. 품질 향상이 확인되면 user 정의의 Top-1 NoScore와
native-spatial NoCorr 및 필요한 물리 channel ablation으로 효과를 확인한다.

## 전체 평가의 진행

Generated PBR160개에 대한 native LOD 재사용·shared M1 material GT·480-view
official DreamSim/CLIP·target/source8metric 비교를 연결했다. 처음 reference
범위 오류는 [정정 기록](2026-10-22-trained-PBR-render-reference-correction-ko.md)에
남겼다. 올바른 same-replay native GPU proof는3camera의 모든 array/metric이
정확히 같아 통과했고, 실제 case 렌더와 shared scoring이 진행 중이다.
물리GPU3의 이 새 후속 작업들은 기존 학습과 cache 작업을 재시작하지 않는다.
실제 queue manifest와 receipt로 생성·렌더·metric의 완료 범위를 각각 추적한다.
FID/KID와 최종 test508, joint/독립 seed/전체 cascade의 최종 판단은 남아 있다.

[PBR 개입 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr1905-delivery-intervention32-v1),
[SS matching/native join W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-bounded-source-contrast112-matching-native-join32-v1),
[Source-disjoint72 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-teacher-source-disjoint72x3-s101-v1)에
수치·CI·관측 범위를 기록했다. Full 모델은 아직 채택하지 않았다.
