# Source teacher 대조의 실제 시작과 판단 기준

## 완료한 검증

기존 source-ray NoDQ19와 source-reference NoDQ20을 같은 cold 함수로 초기화했다.
Source-reference는 R에 해당 source ray 자신의 native feature를 추가로 참조한다.
Semantic Top64, matching decoder/score, four-head attention의 query와 candidate,
source-use, native projection은 변경하지 않았다. 따라서 이 실험은 새 attention-query
설계나 semantic matching 자체의 효과를 증명하는 실험이 아니다.

두 모델의 common19 state SHA는 모두
`23ed975eebaf88e9a156ccd0557423cb317ae0f8e31c793ffaf32add9ad5c1cc`였다.
실제 GPU0에서 global96/micro96/accumulation1 forward/backward를 모두 통과했다.
초기 source delta는 정확히0이고, 모든 source parameter에 finite gradient가 연결됐다.
추가 분기의 첫 gradient가0인 것은 zero-output R의 cold 초기화에 따른 정상 동작이다.

| 대조 | 실제 backward compute | 최대 할당 VRAM | 초기 TRAIN96 feature MSE |
|---|---:|---:|---:|
| NoDQ19 | 10.77초 | 26.04GiB | 0.00626161834225 |
| source-reference NoDQ20 | 11.28초 | 26.44GiB | 0.00626161834225 |

별도의 실제 RGB→SS native32 검증에서 두 모델 모두 원본 SS4210의
global/projection/negative conditioning과 RNG를 보존했다. 원본과32개 각각의
TP/FP/FN·GT hash·noise seed도 일치했다. 초기 IoU는 동일한0.426715466714였다.
이 검증은 optimizer update0이며, 두 결과를 실제 file4 SHA receipt로 닫은 뒤 학습을 시작했다.
CPU 검증은 본 실험16개와 별도 validation replay 검증2개가 통과했다.

## 데이터와 학습

원본 DeformNeXt TRAIN9149 중 source별 하나를 고른1536 TRAIN asset만 쓰는
조건부 source-module pilot이다. 원래 전체9149 joint SS/PBR 실험을 대체하지 않는다.
M1 source9/10 RGB3072장,331,307,950byte는 원래 file4에 있던 데이터다.
로컬 작업 캐시 복사 후 모두 실제 file4 SHA와 일치함을 확인했다.

원래 SS DINO를 FP32로 실행하고 FP16으로 저장한 teacher1536개는
GPU0에서101.88초, 최대1.95GiB에 인코딩했다. 파일4의 실제 SHA 검증까지 완료한
artifact는1544개,6,452,594,084byte다. 원래 source cache의 사전 지정3개에 대한
재인코딩 상대 RMS 오차는1.44–2.16×10⁻⁵로 기준을 통과했다. Bit-exact라고 표현하지 않는다.

두 arm 모두7 epoch·112 update, 같은1536개·같은 순서·같은 teacher 목적,
source LR1e-4·AdamW·adaptive clipping·micro96/accumulation1이다.
원래 multiview SS4210 denoiser는 업데이트하지 않는다. Cached FP16 target DINO를
native grid에 투영하여 source module을 학습하고, 매 epoch 별도의 실제 RGB/BF16
native SS 생성으로 IoU와 voxel panel을 평가한다. W&B에도 loss·VRAM·IoU·panel을 기록한다.

Teacher MSE는 TRAIN의 M1 source9/10 추가 supervision이다. 추론 입력은 계속
M1 target8 + M0 source9/10/11이다. Validation/test RGB를 학습 label로 쓰지 않는다.
Source11에 존재하지 않는 teacher나 identity label을 만들지 않는다.
이는 feature 목적과 SS flow 목적의 동일한 full-joint ablation이 아니다.
Full joint에서 teacher loss의 기여를 주장하려면 같은 구조·LoRA·budget에서 별도로 대조해야 한다.

## 첫 중간 결과

기존 NoDQ19 teacher pilot의 동일 validation32·noise 결과다. 아직 source-reference20은
학습 대기이므로 두 구조의 효과를 비교할 수 없다.

| module epoch | native IoU | frozen SS4210 대비 delta | paired CI95 |
|---|---:|---:|---|
| 0 | 0.4267155 | 0 | 초기 예측 일치 |
| 1 /16 update | 0.4283577 | +0.0016422 | [−0.0002317,+0.0038099] |
| 2 /32 update | 0.4270653 | +0.0003498 | [−0.0007578,+0.0016502] |

두 CI는 모두0을 포함한다. 초반 성능 개선이 입증됐다고 말하지 않는다.
실제 optimizer 학습은 약7–8초/update, 최대 할당 약31.34GiB를 관측했다.
서로 다른 minibatch의 loss를 앞뒤로 비교하여 감소 추세라고 판단하지 않는다.

## 종료 후 무엇을 구분할 것인가

기존32개 validation-only teacher replay는 `not_for_training=True`로 유지한다.
두 최종112 checkpoint에 대해 별도의 읽기 전용 GPU feature 진단을 연결했다.
Optimizer를 만들지 않고, 전체 state가 전후 동일한지 확인한다. 이는 cached target
projection을 사용하는 feature 진단이며 실제 RGB native 생성 평가와 구분한다.

- TRAIN fitting 실패: gradient·학습량·보정 bound·표현 용량을 확인한다.
- TRAIN fitting만 개선: validation feature 방향과 오차로 일반화 실패를 확인한다.
- Validation feature만 개선: 실제 SS 생성으로 전달되지 않는 이유를 조사한다.
- Native 생성까지 개선:128×3·독립 training seed·전체9149 joint 실험으로 확인하고,
  같은 loss의 physical Top1·spatial correspondence·decoder 대조로 matching/attention 기여를 구분한다.

Source-reference 이득만으로 decoder·matching score·correspondence·Top-K attention의
효과를 주장하지 않는다. NoScore는 실제 Top1 feature, NoCorr는 spatial correspondence라는
기존 정의를 유지한다. 원래 target-patch query/R96 재현도 이 source-camera ray 실험과 구분한다.

기존 spatial/DQ/bounded SS 학습,128×3 final 평가와 checkpoint 보존,
4-GPU PBR Direct, generated cascade와 독립 seed, heldout508 및10 metric 요구는 유지한다.
기존 NN·dispatcher·admission keeper는 중단하거나 재시작하지 않았다.

원본 결과는 `feature-readout-v1/source-feature-teacher-pilot-v1/` 아래에 있다.
Source 방향 진단 시각화는 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-source-feature-direction-CPU32-v1),
현재19-tensor pilot은 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-source-teacher-original-TRAIN1536-7ep-v1)에서 확인할 수 있다.
