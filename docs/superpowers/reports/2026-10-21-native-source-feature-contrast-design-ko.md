# Native-source feature contrast를 직접 retargeter에 전달하는 대조

## 관측과 한계

기존19-tensor source module의 TRAIN1536 teacher 학습7 epoch는 완료됐다.
실제 native SS32 최종 IoU는0.4282802이고, frozen SS4210은0.4267155다.
초반부터 소폭 변화만 보여 아직 성능 개선이 입증됐다고 판단하지 않는다.

추가 source projection을0으로 초기화한20-tensor 모듈도 학습 중이다.
Checkpoint16/32의 projection Frobenius norm은 같은 R의 matched-input block 대비
각각1.252%,2.502%였다. 이 값은 weight norm이고 실제 기능적 영향의 증명은 아니다.
분기가 학습되지 않는다는 뜻도 아니다. 다만 R output도0에서 시작하므로 새 source 정보가
첫 output gradient에 직접 반영되지 않는 초기 경로를 별도로 점검할 근거가 있다.
진행 중인 두 대조는 중단하지 않는다.

TRAIN1536 원본 source와 teacher 전체를 NN 없이 계산한 zero-correction MSE는
0.00703520825다. 이 통계는 validation/test를 사용하지 않고 학습 설정을 바꾸지 않는다.

## 하나의 직접적인 표현 변경

기존 R 입력은 `[E + d, T]`다. 여기서 E는 semantic Top1 carrier, d는 실제
Top64 matching-aware QKV feature change, T는 target native context,
S는 보정 대상 source ray 자신의 원래 DINO feature다.

새 후보는 R 입력을 `[E + d - S, T]`로 바꾼다. 외부 source delta는 계속
`u * (g*(E-S) + R + bounded(d))`이며 native4-view carrier·target·global·negative를 유지한다.
R에 source feature의 차이를 전달하는 것이며, 외부 carrier에서 source를 두 번 빼는 것이 아니다.

Source 정보가 기존에도 외부 `-u*g*S`에 나타난다는 점을 구분해야 한다.
새 후보의 가설은 nonlinear R에 local feature contrast를 직접 주는 것이 도움이 된다는 것이다.
기존 모델에 정보가 전혀 없거나 문제를 풀 수 없다는 수학적 증명이 아니다.

새 후보는 기존19개 trainable tensor의 값·shape·개수를 그대로 유지한다.
추가 source projection은 만들지 않는다. R output의 기존 zero initialization으로
cold-start native 함수를 보존하고, 첫 output-weight gradient부터 실제 S를 참조한다.
Warm parent의 동일 함수를 보존하는20-tensor 후보와 목적을 구분한다.
Warm parent migration은 허용하지 않고, 별도 checkpoint marker를 저장하여 기존19-tensor
클래스로 잘못 strict-loading하는 것을 막는다.

Source-camera ray query, member posterior, Top64 support, matching decoder/score,
four-head attention와 source-use는 유지한다. RAW2048 deformation-query 채널을
추가하는 실험도 아니다. NoDQ는 계속 물리적으로 그 채널과 projection이 없다.

## 검증과 학습 조건

CPU5개 검증은 common19 weights 동일, cold zero correction, 실제 own-source 차이와
target context 유지, 첫 output gradient의 source 의존성, checkpoint marker,
warm migration 거부, empty-support zero-gradient 연결을 확인했다.
실제 GPU96 backward와 RGB→SS32 native 초기 tensor/count/GT-noise 일치는 별도 확인해야 한다.

GPU 검증을 통과하면 앞선 두 teacher 대조와 같은 TRAIN1536·순서·seed101·7 epoch·112 update,
global96/micro96/accumulation1·source LR1e-4·AdamW·adaptive clip으로 비교한다.
동일한 frozen SS4210과 TRAIN source9/10 teacher 목적을 사용하고 매 epoch 실제 native SS32
IoU와 voxel panel을 측정한다. 새 데이터, 새 teacher, 추가 GT 추론 입력은 필요하지 않다.

이득이 확인되더라도 이것만으로 semantic matching, matching score/decoder,
Top-K attention 기여를 주장할 수 없다. 같은 loss의 physical Top1·spatial correspondence와
fixed-use value 대조,128×3·독립 seed·full9149 joint/PBR/cascade·최종508 평가가 필요하다.
현재 이 문서는 설계 및 CPU 검증 기록이며 GPU 실행·학습·성능 성공 보고가 아니다.
