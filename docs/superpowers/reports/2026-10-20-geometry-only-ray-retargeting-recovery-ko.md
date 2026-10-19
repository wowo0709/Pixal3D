# Appearance 경로 실패를 분리한 ray-wise geometry retargeting 재학습

이 문서는 SS 32개 개발 패널에서 확인한 실패 원인과 그에 따라 추가한 학습 후보를 기록한다. 새 후보의 성능 향상은 아직 입증되지 않았다. 기존 SS/PBR 실험을 취소하거나 최종 full model을 선택한 문서가 아니다.

## 관측된 실패와 해석 범위

동일한 FullDirect+DQ step288 체크포인트를 사용하고, NN/R/LoRA·GT·noise·source·12 sampling steps·generation batch8을 고정하여 네 경로 조건을 비교했다. Normal의 32개 TP/FP/FN·metric은 실제 학습 validation과 정확히 일치한다.

| 저장된 가중치에서의 경로 조건 | pooled voxel IoU |
| --- | ---: |
| Normal: appearance 보정과 DQ 사용 | 0.43913141 |
| appearance와 DQ 보정 모두 끔 | 0.44760567 |
| DQ projection만 끔 | 0.43601077 |
| appearance 보정만 끄고 학습된 DQ 유지 | 0.44905614 |

Appearance 보정만 끈 경우 Normal보다 **+0.9925 percentage point**, source-cluster paired bootstrap 95% CI **[+0.4403,+1.6057]pp**다. 현 체크포인트와 패널에서 추가 appearance 경로가 해롭다는 근거다. 반면 appearance를 끈 조건과 matched-R288의 차이는 +0.3136pp, CI [−0.1360,+0.7247]pp로 R보다 우수하다고 결론 낼 수 없다. DQ 자체의 추가 효과도 아직 불확실하다. 이 검사는 저장 가중치의 민감도 검사이며, 경로를 제거하여 다시 학습한 ablation과 구분한다.

임의의 learned V/O residual이 native DINO feature 공간을 흐트러뜨리거나, appearance와 RAW geometry에 서로 다른 대응을 전달할 수 있다는 설명은 **가설**이다. 경로를 끄면 개선된다는 결과만으로 내부 원인까지 확정하지 않는다. step384에서도 Direct IoU 0.44038553은 matched-R 0.44748878보다 낮다. 이를 단순 초기 성능 하락만으로 설명할 근거는 없다.

Matching도 독립적인 병목이다. 실제 native VAL32에서 추가 Q/K를 사용한 Direct288의 near1 관계별 probability mass는 pretrained utility보다 −0.7296pp, CI [−1.4043,−0.0151]pp다. 추가 Q projection만 끄면 +0.6858pp 회복된다. Ray-union 분모로 바로잡아도 utility mass 0.46609545와 Top64 coverage 0.88661833 사이의 차이는 남는다. Coverage는 후보에 정답이 포함되는 비율, mass는 정답에 부여한 확률이며 동일한 통계나 SS IoU의 상한이 아니다.

추가 Q/K-only 7epoch warmup은 정해진 마지막 epoch까지 진행한다. 별도 SS-TRAIN calibration의 near1 mass는 epoch0/1/2/3/4/5에서 0.45333/0.45566/0.43559/0.43287/0.43412/0.43601이다. Top1 coverage와 mass가 같은 방향으로 움직이지 않는다. Training loss 감소를 Top-K 확률 조합 개선으로 해석하지 않는다. Epoch1을 사후 선택하거나 검증되지 않은 epoch7을 현재 SS 학습에 덮어쓰지 않는다. 실제 SS-VAL32/native256 분할의 fixed7 평가가 별도로 연결되어 있다.

## 추가한 구조

`RayGeometryRetargetingConditioner`는 기존 target-patch ray grouping, nearest projection, Top1 appearance carrier, pretrained D1 candidate/score bank와 native retargeting adapter R을 보존한다. 같은 target patch의 voxel들에 전달되는 기존 ray 단위를 사용하며, 개별 voxel feature를 새 query로 넣지 않는다.

1. Matching decoder가 contextual Omega query를 만들고, pretrained utility score와 추가 4-head Q/K가 Top64 확률 alpha를 계산한다.
2. 같은 alpha로 실제 source RAW Omega2048을 조합한다. DQ는 `RAW_target − Σ alpha_i RAW_source_i`이다.
3. Native Top1 appearance와 target DINO query는 기존 R에 전달한다. DQ2048에는 별도의 zero-initialized projection을 추가한다. 실제 RGB/Omega 입력은 0으로 만들지 않는다.
4. 추가 appearance value/output 행렬은 모듈과 state_dict에서 **물리적으로 제거**한다. R에 arbitrary transformed appearance residual을 더하지 않는다.
5. 추가 Q의 초기 가중치는 0이므로 pretrained utility prior에서 시작한다. K와 5개 utility tensor를 포함한 matching7은 이후 SS loss로 학습한다. 진행 중인 Q/K auxiliary warmup의 가중치는 사용하지 않는다.

추가 trainable tensor는 matching7+DQ projection1이다. 공통 NN LoRA360+R/fusion6과 합쳐 **374 optimizer states**다. 기존 Direct376 증명을 재사용하여 374 상태를 검증했다고 보고하지 않는다. DQ를 제거하면 이 새 구조의 추가 geometry 경로가 사라지므로, 향후 동일한 R-only 대조군과 중복되는 arm을 주의한다.

## 실제로 통과한 검사

- CPU 3 checks, 18.60초: SS1024/PBR2048 readout 확률이 기존 prior-anchored native 수식과 정확히 일치하며 VO 행렬이 없다. 실제 RAW 정보의 가중 조합과 matching7 gradient를 확인했다. PBR 전체 학습 통합을 검증한 결과는 아니다.
- Native SS TRAIN micro2: engineering R480 기준 초기 loss **0.10466600209474564**와 conditioning evidence가 정확히 일치한다. 실제 비영(非零) Omega 입력으로 8개 tensor 모두 update3까지 gradient를 받는다. NN/R은 고정했으며 임시 update3은 폐기했다. Peak5.59644GiB, 측정 구간55.64초, native/wrapper exit0, 실제 file4 closure 완료.
- Native joint continuation micro2: **R96의 공통 Adam366을 복원**했다. NN360/R6은96→99, matching7/DQ1은0→3으로 진행했고 374개 모두 변경되었다. Frozen NN base와 D1은 정확히 유지됐다. Peak5.84160GiB, 측정 구간40.66초, native/wrapper exit0, file4 closure 완료. 전용 DDP2/micro48 capacity나 학습 성능 향상을 뜻하지 않는다.
- 추가 CPU 2 checks, 2.98초: Q/K와 utility 출력이 비영 값으로 학습된 상태에서도 기존 native readout과 matching probability 및 7개 matching gradient가 정확히 같다. VO 제거가 이후 matching 수식을 바꾸지 않음을 확인했다.
- 실제 R96 checkpoint·GT·noise·generation batch8로 초기 SS32 검사를 완료했다. **IoU 0.43718718977498344**, 32개 모두 TP/FP/FN·4개 metric·GT·noise·source가 원래 R96 validation과 정확히 같다. Peak5.83506GiB,236.25초, native/wrapper exit0 및 file4 closure를 확인했다. R480 engineering 초기화 파일을 과학 실험의 R96 상태로 착각하지 않도록, 공통 R96의 30개 state tensor를 모두 명시적으로 복사한 독립 initial-state를 사용했다.

## 학습·평가·자원 연결

원본 DeformNeXt split과 이미 구축된 RGB/Omega/latent/Top64 cache를 사용한다. Raw download, 렌더 재구축, Omega 재추론은 추가하지 않았다. Canonical 결과는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/geometry-only-retargeting-v1`에 저장한다.

새 arm은 현재 Direct SS가 끝난 다음 GPU **2,3**에서 실행한다. 초기32 native parity와 자체 joint profile, 자체 DDP2/micro48 capacity를 모두 통과해야 본 학습을 시작한다. Global96, accumulation1, BF16, NN LR1e-5/R LR1e-4, 새 matching/DQ LR1e-4와 5-update warmup, seed101을 사용한다. R96에서 공통 optimizer/sampler를 이어받아384회 추가하여 총480 update로 맞춘다. 기존 NoDQ0/1, scalar composer2/3, PBR capacity/train/eval descriptor는 보존했다. 새 geometry 실험이 scalar보다 먼저 2/3을 사용하도록 순서를 추가했다.

본 학습·capacity·final closure·128×3 sampling noises·replay32·summary의 8개 job을 연결했다. 기존469 descriptor는 그대로 두고 queue는477개가 되었다. 새 결과는 V1, historical strong-R, matched-R, FullDirectDQ, inference-only appearance suppression, physical Direct-NoDQ와 비교한다. Source-only undeformed4 RGB와 edited-target 포함4 RGB의 두 zero-shot 결과도 함께 보고한다.

최종 checkpoint는480으로 고정한다. 128개/93 source clusters/3 generation noises에서 TP/FP/FN을 모아 IoU를 계산하고 source cluster 단위 paired bootstrap을 사용한다. Generation noise3회는 training seed3회가 아니다. Strong-R·matched-R·V1 대비 CI 하한과 각 noise의 개선을 확인한 뒤 독립 training seed 후속 실험을 판단한다. Matching decoder/Top-K/score의 독립 효과는 새 모델 자체의 적절한 학습 대조군으로 후속 검증해야 한다. No-score=Top1 correspondence feature, No-corr=native spatial correspondence, No-DQ=채널의 물리적 제거라는 사용자 정의를 유지한다.

진행 중인 FullDirect480에는 appearance-only suppression을 fixed480/128×3으로 평가하는 별도 5개 job도 연결되어 있다. 기존 PBR의 native parity/warmup 검사는 반복하지 않고, 준비된 actual capacity/train/eval 계획을 유지한다. 이 문서 작성 시점에는 성능 향상 달성, full model 채택 또는 전체 실험 완료를 선언하지 않는다.

## 본 학습 시작 확인 — 2026-10-19 23:56 UTC

자체 DDP2 capacity가 두 rank 모두 통과했다. Micro48, accumulation1에서 peak50.02418/50.01889GiB,374 optimizer states, 공통366 Adam96→99와 추가8 state0→3, 모든 trainable tensor 변경 및 DDP 일치를 확인했다. 첫 회 초기화가 지난 capacity update98/99는 약17.6초였다.

**GPU2,3 본 학습도 시작하여 step99까지 진행했다.** 별도 evaluator의 초기 일치뿐 아니라, 실제 과학 실험 프로세스의 native 초기32 결과와 두 rank의 첫97 update에 기록된 모든 loss 항목이 closed matched-R 기록과 정확히 같다. Sampler(epoch1/idx48), 공통 Adam366, stage RNG101/102와 공통 optimizer group 설정도 일치한다. Source와 초기화 때문에 시작하자마자 성능이 하락한 상태는 배제했다. 이 확인은 이후 학습의 성능 향상을 보장하지 않는다. W&B: https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-ray-geometry-only-prior-anchored-s101-joint480-v1

기존 input-DQ와 FullDirect는480을 완료했다. 같은32 개발 패널에서 input-DQ0.44830316, FullDirect0.44678401, matched-R0.44655414, historical strong-R0.45402236, V10.43483711이다. FullDirect의 중간 하락 일부가 회복됐지만 matched-R보다 나아졌다는 증거는 없다(+0.0230pp, CI [−0.8433,+1.0055]pp). Strong-R를 넘지도 못했다. 따라서 새 geometry 구조를 이미 성공 모델로 선언하지 않고 고정480/128×3 평가까지 진행한다.

Actual scientific startup 감사 1개 CPU job을 추가하여 queue478개로 연결했다. 추가 전477개 descriptor는 그대로다. 이 감사도 실제 기록을 비교하여 terminal0/실제 file4 closure까지 완료했다.
