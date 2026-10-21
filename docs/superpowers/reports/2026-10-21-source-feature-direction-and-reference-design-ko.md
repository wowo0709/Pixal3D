# Source feature 보정 방향 진단과 native-source 참조 대조

## 판단의 근거

현재 모듈에 epoch만 추가하기 전에, 실제 M1 source feature로 이동하는지 확인했다.
기존 NoDQ step480을 고정하고 동일한 validation32에서 M1 camera9/10 RGB를
원래 frozen SS DINO로 인코딩했다. 이 RGB는 진단용 추가 GT이고 실제 추론에는 쓰지 않는다.
CPU FP32 재생 후 teacher를 기존 cache와 같은 FP16으로 양자화했다.
FRAME, input-alpha support, world-v4 patch map, native source-camera ray를 사용했다.
학습·optimizer·CUDA native denoiser 실행은 없었다.

| 범위 | 보정 전 MSE | 보정 후 MSE | 후/전 | 예측 변화와 M1 변화 cosine |
|---|---:|---:|---:|---:|
| source9/10 foreground union | 0.0281613 | 0.0535396 | 1.9012 | 0.02191 |
| 그중 쓰기 가능한 ray | 0.0267518 | 0.0547210 | 2.0455 | 0.02360 |
| source9/10 변화만 native projection | 0.00149659 | 0.00459553 | 3.0707 | 0.01205 |

Foreground에서 예측 보정 norm/M1 변화 norm은 0.9715다. 따라서 보정이 단순히
작은 것이 아니라, literal M1 DINO 변화 방향과 잘 맞지 않는다. Foreground patch의
90.74%, foreground teacher 변화 에너지의 86.20%는 현재 쓰기 가능한 ray에 있다.
32개 모두 foreground MSE가 커졌다. 전체 pixel 변화 에너지 coverage 55.64%를
물체 foreground coverage로 해석하면 안 된다. 배경 descriptor 변화도 포함되기 때문이다.

이는 구현 버그나 생성 성능 부재의 증명이 아니다. NoDQ는 native SS final128×3에서
V1보다 IoU +0.00616을 보였고, joint LoRA가 literal M1과 다른 conditioning code를
활용했을 수 있다. CPU 재생은 실제 CUDA BF16 forward와도 구분한다.
Source11 M1 teacher는 없으므로 source11까지 포함한 부분 oracle 비교를 주지표로 쓰지 않는다.

반면 앞선 실제 frozen SS4210 oracle32는 projection field에만 GT9/10을 넣어도
IoU 0.4267155→0.4611766, +0.0344611, paired CI95 [0.0133646,0.0547356]이었다.
Global field만 바꾼 효과는 +0.0000505, CI가 0을 포함했다. 이 결과는 native projection
경로의 여지를 보여주지만, 추가 GT 진단이므로 경쟁 모델의 성능이나 strict upper bound는 아니다.

## 분리해서 확인할 두 가설

1. 생성 loss만으로 source 보정 방향을 학습하기 어렵다.
2. 기존 R은 `(matched appearance + feature delta, target context)`를 받지만,
   보정 대상 source pixel 자신의 native feature S를 직접 받지 않는다. S는 바깥의
   `g*(E-S)`에만 등장한다. 명시적 source 참조가 같은 teacher 목적에서 도움이 될 수 있다.

둘 다 아직 가설이다. 현재 결과만으로 matching 자체가 불가능하다거나 source 참조가
반드시 성능을 올린다고 결론 내리지 않는다.

## 추가 구조와 검증

`SourceReferencedMemberFeatureConditioner`는 기존 source-camera ray query,
semantic member Top64, matching score, four-head QKV, Top1 carrier, source-use,
native source projection을 그대로 둔다. R 첫 hidden layer에 해당 source ray의
실제 native source feature를 선형 투영하여 더한다. 새 projection weight만 0으로
초기화한다. 기존 input matmul과 weights를 유지하여 warm parent의 함수도 보존한다.
새 trainable tensor는 1개, 총20개다. 기존19-tensor 구조가 실제 NoSource 대조다.
이는 deformation-query 채널을 0으로 대체하는 arm이 아니다.

Source background ray도 실제 native descriptor를 참조한다. 새 visibility write gate는
없다. GT teacher는 추론 입력이나 매칭에 들어가지 않는다. Target/global/negative native
carrier는 기존 경로를 사용한다. CPU 테스트는 common state/RNG, cold zero correction,
warm parent 출력과 선택의 정확 일치, own-ray source provenance, gradient, empty-support,
exception 이후 transient source context 복구를 확인했다. 실제 CUDA/native SS identity와
full-batch capacity는 별도 검증이 필요하다.

## TRAIN-only teacher pilot

전체 TRAIN9149 중 원래 M1 RGB9/10 참조가 있는9143에서 source별 하나를 seed101 hash로
고르고,50 category round-robin으로1536개를 고정했다. 결과나 오차에 기반한 선택은 없다.
선택된1536 source cluster와 validation132 cluster의 교집합은0이다. 실제 file4에서
teacher3072 PNG의 SHA/512²와 raw/export RGB4608쌍 일치를 확인했다. 기존 원본과
cache는 변경하지 않는다. 이 작은 source-module pilot이 전체9149 SS/PBR 실험을 대체하지 않는다.

첫 대조는 기존 NoDQ와 source-reference NoDQ에 같은 teacher 목적을 적용하는 것이다.
계획은7 epoch, global96/micro96/accumulation1,16 update/epoch, 총112 update다.
같은 source LR1e-4, AdamW, native adaptive clip을 사용하며 denoiser는 original
multiview SS4210 상태를 고정한다. 기존 joint480의 임의 conditioning code를 teacher
목적으로 이어 학습하지 않고, 두 arm을 같은 cold native 함수에서 시작한다.

Teacher 목적은 `MSE(S + emitted_delta, M1_DINO)`를 source9/10 전체 patch에서
동일 가중으로 계산한다. Foreground/가시성/매칭 성공에 따른 새 loss mask는 없다.
Source11에는 존재하지 않는 identity teacher를 만들지 않는다. Validation replay는
TRAIN label로 재사용하지 않는다. 코드가 TRAIN split, camera9/10, detached teacher,
source 중복·heldout 중복을 거부하도록 검증한다.

학습 목적 변경의 효과와 source 입력 추가 효과를 먼저 구분하고, 실제 native SS 생성
평가에서 이득이 확인될 때 matching score/Top1/spatial/decoder 대조를 같은 조건으로 확장한다.
Feature MSE만 줄어든 결과는 채택 근거가 아니다. 독립 training seed, full-cascade,
정해진10 metric과 heldout508 최종 평가 요구는 유지한다.

## 기존 학습 중간 관측

동일 validation32·noise에서 spatial step384 IoU는0.449176,
NoDQ step384는0.434842로 delta+0.014334, paired CI95 [0.000160,0.032881]이다.
DQ step384 delta+0.000804, bounded step288 delta−0.000304는 CI가0을 포함한다.
Spatial은 탐색적 중간 관측의 CI가 처음으로0을 약간 벗어났지만, 여러 arm/epoch 비교에
대한 보정 전이며 training seed1/noise1이다. Final128×3와 독립 seed를 확인하기 전에
semantic matching의 열세나 spatial의 최종 우위를 확정하지 않는다.

원본 진단·32 replay·260 input SHA readback·actual file4 closure는
`feature-readout-v1/failure-driven-diagnostics-v1/source-projection-alignment-CPU32-v1/`,
TRAIN catalog는 `feature-readout-v1/source-feature-teacher-pilot-v1/TRAIN1536-readiness-v2/`,
중간 비교는 `feature-readout-v1/member-feature-controls-v1/paired-spatial-DQ384-bounded288-observation-v1/`에 있다.
새 teacher encoding과 pilot training의 실제 시작/완료는 이 설계 기록과 별도 receipt로 판단한다.
