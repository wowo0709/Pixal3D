# SS deformation-query 후속 비교 실행 계획

P1/P2의 5-epoch 학습과 동일 128개 최종 평가가 끝난 다음 단계다. 현재 parent가 결정됐거나 C2 학습이 시작됐다는 뜻이 아니다. 준비 중 SS 학습은 계속 진행한다.

## Parent 선택을 결과 전에 고정

P1/P2 각각 기존 C1과 동일한 sampling noise 세 개로 final128 평가를 수행한다. Model checkpoint는 모두 480 updates/5 epochs다. 원래 C1의 세 noise 결과는 재사용한다. 같은 asset의 noise 반복은 source cluster에 묶어 TP/FP/FN을 합산하고 paired bootstrap한다. 이것은 development model selection이며 미사용 test 또는 training-seed 재현성으로 표현하지 않는다.

1. P2−기존 C1과 P2−P1의 CI lower가 모두 양수면 P2를 parent로 고른다.
2. 그렇지 않고 P1−기존 C1의 CI lower가 양수면 단순한 P1을 고른다.
3. 그렇지 않으면 기존 C1 구조를 유지한다. 새 matcher를 head 정확도나 작은 mean gain만으로 강제 채택하지 않는다.

이 선택은 다음 query 실험의 parent를 고르는 실행 규칙이다. 최종 full 모델 승격을 뜻하지 않는다.

## 두 개의 matched arm

| Arm | 구조 | 초기값과 학습 |
| --- | --- | --- |
| C1-control | 선택한 parent의 matcher/projection/retargeter, deformation 입력 채널 없음 | 원본 mixed385 base+LoRA, 5 epochs |
| C2-query | C1과 동일 + 물리적인 RAW2048 deformation context 채널 | 같은 원본 초기값, 5 epochs |

두 arm을 GPU 두 장씩 병렬로 실행한다. Global96, GPU당 micro48, accumulation1을 기본으로 같은 2-update memory 검사를 반복한다. 학습한 parent의 weight를 이어받지 않는다. V1을 재학습하지 않는다. C1 재실행은 두 query arm의 DDP 실행 방식과 최신 코드 경로를 맞추기 위한 통제이며, 별도 training seed의 재현성 검증이라고 부르지 않는다. Retargeter/query 2×2는 만들지 않는다.

C1 adapter 첫 linear는 2048 입력, C2는 4096 입력이다. C1에서 2048 deformation 채널을 0으로 채우는 구현을 쓰지 않는다. C2 context는 Omega epoch7 RAW FRAME+GLOBAL의 target−weighted source feature이며 GT mesh deformation vector가 아니다.

- H3 parent는 이미 구축한 H3 inference cache의 RAW2048 context를 재사용한다.
- 기존 R9 parent는 원래 immutable R9 match cache의 context sidecar를 별도 경로로 읽는다. Head SHA와 RGB fingerprint/attention contract가 맞아야 하며, 새 H3 parent에 R9 context를 섞으면 실패한다. 큰 evidence cache를 다시 복제하지 않는다.
- 실제 validation sample을 CPU loader로 읽어 양쪽 context의 shape/finite/nonzero와 fencing을 확인했다. 이 확인을 GPU 또는 생성 성능 검증으로 표현하지 않는다.
- 실제 GPU의 초기 native condition/gradient 검사를 다시 수행한다. Zero-output adapter에서 처음에는 첫 layer gradient가 0인 것이 정상이다. Output layer를 잠시 perturb한 뒤 추가 query weight gradient를 확인하고 완전히 복원하는 진단으로 채널 연결도 확인한다. 이 진단에는 optimizer update가 없다.

## 평가와 완료 조건

각 epoch 같은 32개, final 같은 128개와 세 sampling noise를 사용한다. C2−matched C1, 각 arm−기존 C1, 각 arm−V1을 paired 비교한다. Loss, gate/feature RMS, voxel IoU/F1/precision/recall과 GT-pred-diff 이미지를 W&B에 기록한다. 작은 차이와 0을 포함한 CI를 성능 향상으로 승격하지 않으며 필요하면 추가 training seed를 실행한다. Epoch 추세가 더 긴 학습을 지지하는지도 최종 checkpoint를 확인한 뒤 판단한다.

실행 위치: file4의 `rgb_matching_v2/retarget-native-v3/query-study/`. `decision.json`이 없으면 parent는 미정이다. 실제 `completed.json`과 final evaluation을 확인하기 전에는 실험 완료를 주장하지 않는다. 기존에 중단한 SS C2(step90)/PBR B3(step979)는 재개하지 않는다.
