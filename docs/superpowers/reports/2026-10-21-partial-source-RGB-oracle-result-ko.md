# Partial M1 source RGB oracle 결과와 다음 원인 분리

2026-10-21 14:10 UTC. GPU0에서 진단을 실행해 정상 종료와 실제 file4 검증을
확인했다. 기존 학습이나 실행 중이던 cascade case를 중단하지 않았다.

## 실제 입력을 바꾸면 개선 여력이 있다

고정32개, native original multi-view SS64 step4210, sampling12/CFG3,
generation batch8, 같은 GT/noise를 사용했다. Camera8 M1 target과 camera11
M0 source를 유지하고 camera9·10만 실제 M1 RGB로 교체했다.

| 조건 | Pooled voxel IoU | 원래 입력 대비 차이 |
|---|---:|---:|
| M1 target8 + M0 source9/10/11 | 0.426715 | — |
| M1 target8 + M1 source9/10 + M0 source11 | 0.458495 | +0.031780 |

Paired source-cluster bootstrap95% 구간은 **[+0.010986, +0.051550]**다.
32개 source cluster, sampling noise1개, bootstrap4,000회이며 탐색적 비교다.
Scoring GT hash와 seed32쌍이 일치했다. Deformation 학습/optimizer update는
0회이고 matching/Omega/conditioner를 사용하지 않았다. Native 실행은
약90.67초, 최대 allocated5.53GiB였다.

모델에 실제 들어간 RGB tensor가 CPU preflight와32개 모두 정확히 같았다.
Target8과 source11이 유지되고9·10만 teacher RGB인 것을 hash로 확인했다.
Checkpoint, 카메라, mesh scale와 GT SS latent는 그대로다. Actual input contact
sheet와 SS GT/prediction panel을 보존했다.

추가 정답 RGB를 쓰는 진단이며 competitive 모델의 성능 개선은 아니다.
Camera11은 여전히 M0이므로 full four-M1 oracle이나 엄밀한 상한도 아니다.
이 결과는 source 입력 불일치를 줄이는 방향에 개선 여력이 있다는 근거다.
입력만으로 그런 feature를 예측할 수 있다는 증명은 아니다.

## Global/projection 경로를 먼저 나눠야 한다

Native DINO의 RGB 교체는 두 경로를 함께 바꾼다.

- CLS/register 기반 multi-view global conditioning.
- View projection으로 전달하는 grid feature.

현재 source-ray 모듈은 native global을 그대로 두고 source projected feature만
보정한다. 따라서 이번 RGB oracle의 이득 전체를 현재 retargeter가 달성할 수
있다고 해석하면 안 된다. Teacher feature loss를 즉시 추가하기 전에 같은
두 native encoding에서 global/proj를 따로 선택하는 진단이 필요하다.

Projected-only는 원래 global과 negative를 정확히 유지하고 두 M1 source의
projection만 사용해야 한다. Global-only는 원래 projection과 negative를
정확히 유지하고 global만 교체해야 한다. 같은 RGB/카메라, checkpoint,
GT/noise와 sampler를 사용하고 feature path의 tensor hash/동일성을 확인한다.
Cache precision 차이로 인한 추정 교체 대신 실제 native encoder 출력을 사용한다.
아직 이 두 경로 진단은 구현·실행하지 않았다.

Projected-only의 개선이 확인되면 source feature 보정의 방향·크기와 train-only
teacher supervision을 검토할 근거가 된다. Global-only가 주로 개선된다면
현 projection 전용 모듈로는 주요 정보를 바꾸지 못한다는 대안 설명을 우선한다.
어느 경우에도 matching prior의 온도/분산을 다시 넓게 탐색하지 않는다.

## Physical Top-1 대조도 최종 집계했다

같은480 update/seed101, micro96/accum1의128개×3-noise 비교다.

| 조건 | 최종 pooled IoU |
|---|---:|
| Physical Top-1 feature + V/output + R | 0.437880 |
| Top-K Q/K/V/output NoDQ + R | 0.438810 |

Top-K−Top-1 차이는 **+0.000929**, 구간**[−0.001578,+0.003828]**다.
전체384 GT hash가 일치했다. 독립 학습 seed는1개다. 현재 Top-K 추가 이득을
확정할 수 없으며 spatial/DQ/bounded 및 full-cascade 대조는 아직 필요하다.

Top-1의 별도32개 value sensitivity에서는 원래 Top-1 carrier와 source-use를
유지하면서 V branch에 넣는 선택 feature만 중간 matching rank로 교체했다.
실제 코드는 V projection 직전 feature를 바꾸며 head의 hard selection 표시는
유지한다. 이때 IoU는0.437548에서0.445501로 오히려 높아졌다.
Normal−perturbed 차이는−0.007952, 구간[−0.015176,−0.000993]다.
이는 탐색적32개/1-noise saved-weight 진단이며 재학습 NoCorr나 decoder의
독립 효과가 아니다. 현 V/output feature branch가 정확한 correspondence를
유용한 생성 정보로 전달하는지 더 검토해야 한다는 신호로 기록한다.

- [부분 RGB oracle 실제 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-partial-M1-source-rgb-oracle32-v1)
- [Physical Top-1 대 Top-K 최종128×3](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-physical-top1-vs-topk-final128x3-s101-v1)

파일서버4의 result root 아래
`failure-driven-diagnostics-v1/partial-M1-source-RGB-oracle32-v1/actual-v1` 및
`member-feature-controls-v1/physical-Top1-final128x3-v1`에 실제 근거를 저장했다.
최종 모델 채택과 최종508개 평가 완료를 뜻하지 않는다.
