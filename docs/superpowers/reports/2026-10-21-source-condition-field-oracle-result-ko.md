# Source projection의 개선 여력과 global 경로 분리 결과

2026-10-21 14:37 UTC. 두 native SS32 진단과 CPU 집계가 모두 정상 종료했고
실제 file4 SHA/크기 검증과 W&B 업로드를 통과했다. 추가 학습은 없었다.

## Source projection만 바꿔도 개선된다

같은 original multi-view SS4210 checkpoint와 고정32개/noise0,
sampling12/CFG3/generation batch8을 사용했다. Camera9·10의 실제 M1 RGB를
추가 정답 입력으로 사용하고 target8과 M0 source11은 그대로 유지했다.

| 조건 | Pooled voxel IoU | 원래 입력 대비 차이 | Paired95% 구간 |
|---|---:|---:|---:|
| 기존 native 입력 | 0.426715 | — | — |
| Global+projection RGB oracle | 0.458495 | +0.031780 | [+0.010986,+0.051550] |
| Projected-only oracle | 0.461177 | +0.034461 | [+0.013365,+0.054736] |
| Global-only oracle | 0.426766 | +0.000050 | [−0.001140,+0.001526] |

32개 모두 서로 다른 source cluster이며 각 비교의32쌍 GT hash/noise seed가
일치했다. Source-cluster bootstrap4,000회이며, 여러 탐색적 비교에 대한 보정은
하지 않았다. Sampling noise는1개이고 독립 학습이나 untouched test가 아니다.

Projection이 oracle일 때 global도 교체한 차이, RGB-both−projected-only는
−0.002682, 구간[−0.005184,−0.000414]였다. 이 특정 개발 진단에서는 global
교체가 추가 이득을 주지 않았다. Global adaptation이 언제나 불필요하다는
일반화나 기존 matching/retargeter가 이미 개선됐다는 주장으로 확장하지 않는다.

모든 oracle은 추가 GT RGB를 사용하는 진단이며 competitive 모델, strict 상한,
최종 모델 선정 후보가 아니다. Camera11의 M1 RGB는 사용하지 않았다.

## 실제 native tensor로 검증한 범위

Cache feature를 근사 대입하지 않고 두 입력을 실제 native encoder로 인코딩했다.
Projected-only는 원래 global tensor와 oracle projection을, global-only는 원래
projection tensor와 oracle global을 선택했다. 원래 source9·10 M0 RGB는 같은
32개 row와 CPU preflight의 tensor SHA로 재확인했다.

- 조건마다 batch8×4회의 actual conditioning을 확인했다.
- 실제 global[5,1024], projection[4096,1024] 출력 크기와 FP32 dtype을 확인했다.
- 교체하지 않는 condition field는 원래 출력과 hash/byte가 정확히 같았다.
- Negative conditioning, target8/source11 RGB와 CPU/CUDA RNG도 동일했다.
- 원래 sampler, weights, 카메라, scale, scoring GT/noise와 optimizer update0을
  유지했다. Process-local encoder wrapper는 종료 시 복원했다.
- Native 실행 시간은 projected-only97.27초, global-only97.23초였으며 최대
  allocated는 각각5.53GiB였다. 기존 학습과 cascade case를 중단하지 않았다.

CPU6개 테스트가 이 입력/경로 adapter의 불변 조건과 예외 복구를 확인했고,
위32개 실제 GPU 기록이 native 경로 실행을 별도로 검증했다. SS panel은
voxel projection 시각화이며 PBR rendering metric으로 취급하지 않는다.

## 원인 분석과 다음 설계의 초점

현재32개에서는 global을 유지해도 원하는 source projection으로 바꾸면
native prior가 개선된 geometry를 생성했다. 따라서 기존 projection 전용
interface 자체에 개선 가능성이 없거나 global 미변경이 주된 장애라는
설명은 지지되지 않는다. 올바른 source feature 적응을 학습하는 쪽에
개선 여력이 있다는 근거다.

다만 이것은 입력만으로 그런 feature를 예측할 수 있다는 증명이 아니다.
다음에는 실제 NoDQ480 conditioner가 만든 source map 보정
`delta_pred = source_numerator - semantic_weight * original_source_values`를
M1 source9·10의 native DINO feature 차이와 대조할 필요가 있다. 같은 native
projection으로 보낸 delta도 비교해 방향, 크기와 query/write coverage를 분리한다.
이 측정은 아직 실행하지 않았다.

Feature MSE가 높다는 것만으로 구현 오류라고 판단하지 않는다. Task-specific
conditioning이 literal M1 DINO와 다를 수 있기 때문이다. Teacher feature loss를
검토한다면 정규화/계수는 TRAIN으로만 정하고, R-only/실제 Top-1/semantic Top-K
대조에 같은 추가 목적과 학습 예산을 적용해야 한다. Inference에는 원래
M1 target8+M0 source9/10/11만 사용하고 GT source feature를 넣지 않아야 한다.
Teacher loss/cache나 새 학습 arm은 아직 추가하지 않았다.

현재 attention 분포를 넓히는 것만으로는 별도72×3에서 이득이 확인되지 않았고,
physical Top-1 대비 Top-K의128×3 차이도 CI가0을 포함했다. Temperature/bound
sweep이나 epoch 연장을 늘리기보다 실제 source feature retargeting의 목표와
전달 경로를 위 근거에 맞춰 확인한다. 과거 target-patch query/R96 재현과 현재
source-camera ray 이식의 차이도 계속 분리한다.

[실제 field 비교와 SS panel W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-global-proj-oracle32-v1)

Canonical result root 아래
`failure-driven-diagnostics-v1/source-condition-field-oracle32-v1/{proj-only-v1,global-only-v1,comparison-Wandb-v1}`에
실제 근거와 PNG/PDF를 저장했다. 최종 full 모델 채택과508개/10개 metric 평가는
아직 완료되지 않았다.
