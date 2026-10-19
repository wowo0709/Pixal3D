# Semantic matching의 효과와 Top-K feature readout 수정

현재 증거는 **semantic matching 전체가 무효라는 결론을 지지하지 않는다.**
저장된 PBR checkpoint를 동일한 32개 source cluster에서 다시 평가하면, 학습된
decoder/score를 함께 사용하는 Top-1은 vanilla보다 소폭 좋아진다. 그러나 기존
Top-K 가중합은 이 Top-1보다 추가 이득이 없다. SS에서 확인했던 retargeting의
이득과 이번 PBR의 신호를 보존하면서, 후보 feature를 조합하는 부분을 수정한다.

## PBR의 동일 조건 비교

네 학습 모델은 epoch2/step762 raw checkpoint다. 두 zero-shot은 원래 mixed385
multi-view base+LoRA PBR step16850을 사용한다. 모든 평가에서 target GT Shape
latent를 고정하고 target anchor08만 렌더링했다. 전체 cascade나 novel view 결과가
아니다. 32개 asset은 서로 다른 source cluster이고 generation noise는 한 종류다.

| 모델 | Foreground PSNR ↑ | Target crop LPIPS ↓ | Target crop SSIM ↑ |
|---|---:|---:|---:|
| Zero-shot, M0 RGB 4장 | 15.06174 | 0.128776 | 0.914208 |
| Zero-shot, M1 RGB + M0 RGB 3장 | 15.29189 | 0.122529 | 0.921306 |
| V1 vanilla | 15.49508 | 0.121535 | 0.920779 |
| B1 raw Top-1 | 15.49518 | 0.121453 | 0.920714 |
| B2 learned decoder/score Top-1 | 15.56750 | 0.120846 | 0.921666 |
| B3 learned score Top-K 가중합 | 15.56732 | 0.120803 | 0.921568 |

PSNR은 asset별 PSNR의 단순 평균이 아니라 foreground 면적을 반영한 MSE를 먼저
합쳐 계산했다. 같은 source cluster를 함께 뽑는 4,000회 bootstrap에서:

- B2−V1 PSNR: **+0.07242 dB**, 95% CI **[+0.02027, +0.12348]**.
- B3−B2 PSNR: **−0.00018 dB**, 95% CI **[−0.00362, +0.00219]**.
- B3−B2 crop LPIPS: −0.000043, CI [−0.000120, +0.000029].
- B3−B2 crop SSIM: −0.000098, CI [−0.000202, −0.000016].

B2는 decoder와 score를 함께 바꾸므로 각각의 독립적 효과를 분리한 결과는 아니다.
독립 training seed 반복도 아니다. LPIPS는 여기서 **target crop 진단값**이며,
논문에서 정의한 source preservation LPIPS와 혼용하면 안 된다. V1−edited zero의
PSNR/LPIPS 점 추정은 좋아졌지만 이 32개 source bootstrap CI는 0을 포함한다.

각 asset의 실제 입력 RGB, noise, GT PBR/Shape latent SHA를 기록해 비교했다.
Source-only의 RGB가 달라지는 것은 의도한 입력 대조군이다. 여섯 모델의 GT 렌더
PNG는 전부 같았다. V1의 네 asset을 GPU2에서 GPU3로 옮겨 다시 평가했을 때
숫자와 PNG도 모두 같았다. 이 replay는 네 case에 한정한다.

## 실패 원인에 대한 판단

확인된 구현 제약은 기존 attention이 학습한 scalar weight로 **원래 value feature를
평균**한다는 점이다. 후보별 정보를 head마다 변환해 조합하는 value/output projection이
없다. Top-K 안에 유용한 후보가 있어도 단일 평균으로 희석될 수 있다. 새로 도입한
scalar-gated residual에서는 gate가 0일 때 내부 attention으로 SS gradient가 전달되지
않는다. 두 현상은 개선 가설의 근거이며, 실패의 유일한 인과 원인으로 확정하지 않는다.

이전 positive retargeting과 후속 ray arm 사이에는 입력 전달 방식도 달랐다. Positive
R은 edited-target patch를 같은 camera ray상의 voxel들로 nearest 전달한다. 후속
source-camera ray pooling은 그 계산과 동일하지 않다. 새 실험은 positive R의
**target patch/ray + nearest 전달 + 기존 Top-1 carrier**를 유지한다.

## 새 구현과 실제 검증

새 `RayFeatureAttentionConditioner`는 frozen Omega epoch7 및 D1 context, native
RGB DINO query/value, semantic Top-64, matching score prior를 사용한다. 네 head가
각각 query/key/value를 변환하고 output projection을 거쳐 feature residual을 만든다.
원래 Top-1 feature에 이 residual을 더한 뒤 기존 retargeter를 통과한다.

Output projection만 0으로 초기화한다. 따라서 초기 출력은 기존 Top-1+R과 같고,
output projection은 첫 SS backward부터 학습된다. Scalar zero gate는 사용하지
않는다. Matching/GT depth/GT correspondence를 모델 입력에 추가하지 않았다.
Deformation query 추가 channel도 없다. 실제 PBR backend의 새 모듈 검증은 아직
하지 않았고, 2048차원 DINO+NAF 입력의 CPU 계약만 확인했다.

실제로 완료한 검사:

- 관련 CPU 검사 **13개 통과**.
- 두 TRAIN asset의 실제 SS loss와 conditioning evidence가 초기 reference와 정확히
  일치. 임시 세 update에서 새 9개 tensor 모두 SS gradient를 받음.
- 별도 32개 평가에서 IoU **0.454022361676035**, per-asset TP/FP/FN·GT hash·seed·
  source cluster 및 GT/Pred PNG가 기존 positive R과 전부 일치. 실제 RGB input panel은
  저장한 네 example에서 동일. Noise tensor SHA는 별도로 수집하지 않았으며 pinned
  sampler/GT shape·dtype/seed를 동일하게 사용했다.
- 실제 native joint update에서 기존 LoRA 360개, R/fusion 6개, 새 feature tensor 9개가
  모두 바뀜. 기존 366개 Adam state를 step96에서 보존했고, 세 임시 update 후 공통
  state는 step99, 새 state는 step3. Frozen NN base와 matcher는 hash가 동일.
- 위 joint correctness 검사는 single GPU/micro2, peak **5.8003 GiB**다. 전용 DDP2/
  micro48 메모리 검사나 성능 향상 증거로 해석하지 않는다. 임시 모델은 폐기했다.

## 연결한 학습과 판정 기준

기존 NoDecoder/NoPrior 학습은 유지한다. 이후 common epoch1/raw96부터 R-only
control과 새 feature attention을 비교한다. 총 raw480이므로 추가 학습은 384 update다.
두 모델의 common Adam state, sampler, 학습률, data split과 stage RNG reset을 맞춘다.
기존 checkpoint에 RNG가 없어 old training의 bit-exact resume라고 주장하지 않는다.

Global96, DDP2, GPU당 micro48, accumulation1을 유지한다. 먼저 전용 GPU0/1에서
actual DDP capacity를 검사하고 두 rank의 weight 동등성 및 file4 저장을 확인한다.
이후 R-only는 GPU0/1, feature attention은 GPU2/3에서 실행하도록 우선순위를 연결했다.
기존 scalar composer arm도 제거하지 않고 대기열에 보존했다. 새 모델은 아직 본 학습
전이며, 초기 identity·learnability 검사와 실행 연결이 완료된 상태다.

매 epoch raw validation32와 시각화/W&B를 유지한다. 판정은 최종480 checkpoint의
128개 asset·93개 source cluster × generation noise3 비교 및 32개 replay로 한다.
Generation noise 반복을 독립 training seed라고 부르지 않는다. Matching decoder,
score 및 value correspondence를 따로 검증할 추가 control과 PBR 새 구조 학습은
후속으로 남아 있다. 현재 성능 향상을 달성했거나 full model을 확정한 상태가 아니다.
세 matched arm 각각의 최종 native checkpoint를 actual file4에 저장한 뒤 평가하는
15개 closure/evaluation job도 연결했다. Source/recipe와 final weight의 SHA를 검사하고
기존 학습 job은 그대로 둔다. 현재 전체 대기열에는 과거 completed/retired 항목을
포함해 384개 descriptor가 있으며, 이를 384개 새 실험이라는 뜻으로 세면 안 된다.

## 저장된 근거

공통 root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1`이다.

- `pbr-matched32-v1/six-model-summary-v2/summary.json` 및 `file4-closure.json`:
  230개 artifact, 7,255,027 bytes, actual file4 SHA/size 확인.
- `pbr-matched32-v1/cross-GPU-replay-v1/audit.json` 및 receipt:
  네 case cross-GPU 정확 재현, 11개 artifact.
- `ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/implementation-v1/`:
  새 source 및 실제 gradient/joint 검증, 16개 artifact, 13,788,252 bytes.
- 같은 `feature-readout-v1/initial-parity-v1/`:
  새 feature attention의 생성 초기화32 완전 일치, 5개 artifact.
- 같은 `feature-readout-v1/prelaunch-v1/`와 `queue-install-v1/`:
  source/recipe 및 DDP capacity→학습 실행 연결. Actual file4 SHA/size 확인.
- 같은 `feature-readout-v1/evaluation-plan-v1/`:
  세 arm final128×3/replay32 평가 연결 및 runtime/weight guard, 19개 artifact.
