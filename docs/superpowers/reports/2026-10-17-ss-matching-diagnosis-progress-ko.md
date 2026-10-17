# SS matching 원인 진단과 후속 학습 진행

2026-10-17 KST 실행 중간 결과다. 새 P1/P2의 5-epoch final 성능은 아직 없다. 물리 GPU 0~3만 사용하며 기존 SS C2(step90), PBR B3(step979)는 중단 상태로 file4에 보존했다.

## 1. 기존 모델이 전부 악화됐다는 결론은 수정해야 한다

같은 128개 asset, sampler12/CFG3, sample별 동일 noise에서 global voxel-count IoU를 다시 계산했다. 아래 CI는 93개 source cluster를 재표집한 exploratory paired bootstrap 4,000회다. 여러 contrast에 대한 family-wise 보정이나 추가 training seed 결과는 아니다.

| 모델 | IoU | V1 대비 %p | 95% CI, %p |
| --- | ---: | ---: | --- |
| Source-only zero-shot | 0.379171 | −4.893 | 입력 효과 진단 |
| Edited-RGB zero-shot | 0.416098 | −1.200 | 입력/학습 기준 |
| V1 | 0.428099 | 기준 | 기준 |
| B1 | 0.430146 | +0.205 | [+0.007,+0.411] |
| B2 | 0.430537 | +0.244 | [+0.048,+0.439] |
| B3 | 0.430925 | +0.283 | [+0.087,+0.474] |
| C1 | 0.440281 | +1.218 | [+0.460,+2.007] |

B1~B3도 이 single-noise panel에서 작은 양의 차이가 있다. 그러나 B2−B1은 **+0.039%p, CI[−0.215,+0.289]**, B3−B2는 **+0.039%p, CI[−0.015,+0.086]**로 개별 matching/attention 추가의 증분은 아직 확인되지 않는다. C1−B3는 **+0.936%p, CI[+0.156,+1.735]**다. C1과 B 계열의 기존 microbatch/accumulation 차이에 따른 stochastic training nuisance도 있으므로 architecture만의 정확한 인과 효과로 확대하지 않는다.

기존 C1과 V1을 추가 noise 두 개로 반복했다. 세 noise의 gain은 각각 **+1.218/+1.013/+0.851%p**였다. 동일 asset의 반복을 source cluster 안에 묶은 합산 IoU는 V1 **0.431926**, C1 **0.442214**, 차이 **+1.029%p, CI[+0.586,+1.498]**다. 따라서 4개 quick sample의 하락을 전체 C1의 실패로 일반화할 수 없다. 이는 generation-noise 안정성 확인이며 training-seed 재현성 검증은 아니다.

## 2. 확인된 병목과 아직 분리되지 않은 원인

### Matching과 value의 소비는 다른 문제다

Top64 후보 recall은 validation 63.86%였고 Top256에서는 93.16%다. Train-internal calibration은 57.18% → 88.39%로, 고정한 상한 K256을 사용했다. Calibration 목표 90%를 달성했다고 표현하지 않는다.

새 고정-final 7-epoch Top-1은 H1 **21.79%**, H2 **22.78%**, H3 **23.71%**다. 기존 matcher의 이력 값은 21.19%다. 기존 matcher와 새 head는 train/calibration 및 budget이 달라 단일 변수 비교가 아니므로, 같은 새 protocol의 Top64 H0 replay도 완료했다. H0 replay Top1은 **20.71%**, near1은 **49.26%**다. H*는 validation으로 고르지 않고 calibration의 transported feature error로 **H3 / T0.1**을 선택했다.

같은 508개/132 source cluster의 paired bootstrap에서 H1−H0 Top1은 **+1.083%p, CI[+0.108,+2.066]**, H2−H1은 **+0.996%p, CI[+0.481,+1.529]**, H3−H2는 **+0.926%p, CI[+0.359,+1.479]**였다. H3−H2 near1은 **+0.450%p, CI[−0.370,+1.255]**로 불확실하다. H0/H1의 feature error 평균은 각각 자신의 proposal-hit query를 분모로 써 모집단이 다르므로 직접 차감하지 않는다.

H2 confidence loss의 ranking 입력은 detach되어 있다. 실제 calibration descriptor의 CPU autograd 검사에서도 ranking parameter의 direct confidence gradient는 0이었다. Confidence 추가로 전체 gradient clipping의 효과는 달라질 수 있으므로 H2의 Top1 상승을 inference rejection의 인과 효과로 해석하지 않는다. 생성 성능에서의 효과는 아래 별도 SS R arm으로 확인한다.

H1/H2 최종 감사에서 근접 1-patch(Chebyshev, 같은 source view)의 정확도는 51.64%/53.50%다. H2의 weighted DINO feature cosine error는 **0.01670**, 같은 후보의 uniform 진단은 **0.04348**이었다. Source foreground mean을 제거한 feature error도 **0.20456 대 0.68056**으로, 단순 공통 feature 평균만으로 설명되지는 않는다. 여기서 uniform은 feature utility 진단일 뿐 no-score arm의 정의가 아니다. No-score arm은 Top-1을 유지한다.

새 temperature에서 실효 후보 수는 H2 약 **8.47**, 평균 최대 weight는 **27.58%**다. 기존 T0.005의 1.26개/92.71%보다 여러 후보를 실제로 사용한다. 이것은 SS 생성 개선의 증거가 아니며 P2에서 확인한다.

H2의 calibration-fixed rejection은 validation에서 exact-match precision **32.96%**, coverage **11.26%**, ECE15 **0.01224**였다. 확률 보정이 좋아도 정확한 주소 자체의 불확실성은 크고, 이 정책은 대부분의 foreground query를 거부한다. P1→P2 첫 비교에서는 rejection을 꺼서 matcher와 gate 정책을 섞지 않는다. 근접 patch의 유용성과 exact Top-1 correctness를 같은 개념으로 취급하지 않는다.

### Projection operator 차이가 있었다

Native Pixal3D는 bilinear ProjGrid를 쓰지만 추가 semantic 경로는 nearest patch gather를 썼다. Cached target self-map의 anchor 대비 상대 RMSE는 약 **6.9% 대 0.6%**였다. 같은 canonical camera를 쓴다는 것만으로 연산 동등성을 주장할 수 없다. 남은 작은 오차에는 cached feature 정밀도와 기존 arithmetic 차이도 포함되며 bit-exact cache라고 표현하지 않는다.

Calibration32, 고정 scale0.25, V1 frozen GT-address 결과:

| 조건 | IoU |
| --- | ---: |
| Native | 0.426813 |
| GT nearest | 0.421874 |
| GT bilinear | 0.424428 |
| GT support-normalized bilinear | 0.425286 |
| Bilinear wrong-object source value | 0.402297 |

Projection을 맞추면 nearest보다 좋아지지만 native를 넘지는 못했다. Correct source value는 wrong-object보다 유용하나, raw 주소 교정만으로 강한 native 4-view consumer를 개선한다는 근거는 없다. 기존 raw-only matching-v2 gate는 계속 HOLD다.

또한 GT 주소 조건은 teacher-visible query로 지원 영역도 좁힌다. 이를 분리한 C1 calibration32의 동일-support nearest 비교는 learned **0.444285**, GT **0.441959**였다. 정확한 point address가 학습된 region/value 소비보다 항상 유리하지 않았다. 이 결과는 teacher mask를 사용하는 진단이며 deployable inference 결과가 아니다.

### 단순 target 가중치 확대만으로 해결되지 않았다

네 RGB와 native global을 유지하고 `native_proj + w*(anchor_proj-native_proj)`를 비교했다. Calibration에서 고정 선택한 w0.25는 validation128 IoU **0.431926**, V1 대비 **+0.383%p, CI[−0.495,+1.348]**였다. CI가 0을 포함하므로 채택하지 않는다. w0.5는 calibration에서도 악화됐다. Source 세 장을 없애는 target-only arm은 실행하지 않았다.

## 3. 실제로 진행하는 성능 개선 학습

[후속 계획](2026-10-17-ss-native-retargeting-followup-ko.md)에 따라 **P1은 기존 R9 attention + native/support-normalized projection + C1 adapter**, **P2는 동일 consumer에 calibration-selected H3 matcher를 적용**한다. 두 모델 모두 원본 mixed385 base+LoRA에서 독립적으로 시작하며, 기존 C1 weight를 continuation하지 않는다. Dataset/latent/Omega는 재구축하지 않았다.

기존 R9 weighted evidence를 SS batch에서 K64로 펼치지 않도록 사전 집계했다. 9,657개를 처리했고 8개 audit에서 원래 online soft evidence와 **최대 오차 0**을 확인했다. FP32 weighted feature를 유지하고 frozen scorer 제거에 따른 retargeter 초기 RNG 소비도 보존했다. Teacher 파일을 이 inference cache에 넣지 않았다. 새 H3 cache도 별도 namespace에 9,657개 구축을 완료했고 별도 8개 online audit의 최대 오차 0을 확인했다.

실제 GPU에서 native-bilinear B2/B3/C1/C2 경로의 초기 global/proj 최대 오차 **0**, gate 및 adapter gradient를 확인했다. P1 실제 R9-reference cache에서도 같은 검사가 통과했다. P1 시작 전 생성 IoU는 **0.42671546671444294**로 기존 edited-RGB zero-shot 32개 값과 정확히 같았다. 초기 성능 하락이 없는 것을 생성 지표로도 확인했다.

| 실행 | Micro / world / accumulation | 평균 compute/update | Peak allocated |
| --- | --- | ---: | ---: |
| Single GPU, 최적화 후 | 96 / 1 / 1 | 37.49초 | 91.57 GiB |
| Single GPU, 최적화 후 | 48 / 1 / 2 | 34.63초 | 47.71 GiB |
| **P1 DDP 검사** | **48 / 2 / 1, global96** | **약17.5초** | **각 rank47.75 GiB** |

큰 microbatch가 항상 빠르지는 않았다. 두 GPU를 나눠 accumulation 없이 처리하는 방식으로 전체 latency를 줄인다. P1은 **GPU2·3**, P2는 **GPU0·1**에서 진행 중이며 steady-state 약17.2초/update, 각 rank peak47.75GiB다. 두 모델의 시작 전 공통32 IoU는 모두0.426715로 일치했다. P1의 첫 epoch(step96)는0.434282였다. 사용자 중단 이후 step103/56의 optimizer/scheduler/sampler까지 복원했으며 rank별 RNG가 없어 bitwise uninterrupted replay는 아니다. 5 epochs/480 updates, 매 epoch32개와 final128개, W&B loss/metric/GT-pred-diff logging을 유지한다.

P1/P2와 별도로 원래 계획의 Top1 → rejection → attention → adapter → physical query **다섯 SS arm을 native projection에서 실행하도록 복원**했다. Retarget-only query-study의 미실행 두 run은 이 다섯 arm 중 C1/C2로 대체해 중복을 없앴다. [전체 범위 복원 및 실행표](2026-10-17-ss-full-module-restoration-ko.md)를 기준으로 진행하며, P1/P2만으로 원래 모듈 비교를 완료했다고 판정하지 않는다.

Pipeline에서 완료 의존성 대기가 빠진 문제로 H3/H0 최종 감사가 일찍 시도됐지만, 감사의 7-epoch guard가 이를 거부했다. 의존성 완료 대기를 수정하고 해당 감사만 재예약했다. 불완전한 head 결과를 최종 수치로 사용하지 않았고, 진행 중 SS/head process는 보존했다.

## 4. 확인해야 할 남은 결과

- P1/P2의 5-epoch final 생성 지표와 기존 V1/C1 대비 paired 차이. 현재 새 full 성능 향상을 주장하지 않는다.
- 복원한 다섯 SS arm의 final 결과와 Top1/rejection/attention/adapter/query 인접 대비. H0/H3 head 최종 감사는 완료했다.
- 실제 학습 속도/메모리의 안정성, epoch별 생성 경향과 loss의 관계.
- 같은 H3/rejection/native projection에서 물리적인 deformation-query 채널을 추가하는 C2 비교. C1에는 해당 채널이 없으며 retargeter/query 2×2는 구성하지 않는다.

SS quick voxel 지표의 GT는 정합된 M1 SS latent를 decoder로 복원한 occupancy다. Raw mesh 표면 또는 렌더링 지표와 혼동하지 않는다. 후속 결과에는 이 occupancy의 해시와 shape도 저장하고, 비교 시 동일 GT foreground 수·hash(양쪽에 존재할 때)·cluster·sampling 설정과 noise 반복 중복을 검사한다. 기존 V1/C1 공통128×3 noise의 GT foreground 수는 모두 일치했고, 보강한 comparator에서도 기존 gain/CI가 정확히 보존됐다. Legacy 결과에는 당시 저장한 전체 GT voxel hash가 없어 count 검사 범위를 넘어선 hash 검증을 주장하지 않는다.

Authoritative outputs: file4의 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/`. 통계 근거는 `diagnostics/*common128-bootstrap.json`, `noise-confirmation/c1-v1-three-noise.json`, `head-audit/`, `projection-audit/`, `role-balance-probe/`에 있다. 새 학습은 `retarget-native-v3/`, 실행 큐는 `runtime/queue-state.json`이다. [W&B P1](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-v3-p1), [공통 평가](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-matching-v2-evaluations).
