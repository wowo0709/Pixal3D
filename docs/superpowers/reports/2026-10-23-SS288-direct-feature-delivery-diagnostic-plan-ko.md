# NoDQ288 direct feature 전달 강도 진단 계획

현재 두 SS 학습을 유지하고 NoDQ288 저장 checkpoint의 작은 제어 평가를 준비한다.
실제 결과나 모델 개선이 확인된 계획은 아니다. 같은288에서 retargeting 대비
평균 IoU−0.5584%p와 FP 증가를 관측해, delivery 경로를 분리하는 것이 목적이다.

## 가설과 고정 범위

현재 attention feature F는 R 입력과 별도 bounded correction C(F)에 사용된다.
후자는 semantic fusion gate 밖에서 전달된다. Matching의 상대적 확률이 있어도
직접 feature 전달량을 원래 semantic gate가 줄여주지는 않는다.
이 전달이 pretrained prior를 지나치게 바꾸는지가 검증할 가설이다.

같은 NoDQ288 NN/source weight, 데이터·RGB·camera·GT·sampling seed,
Top64 후보·matching logits·head 확률·Top1 carrier·F·R 입력/출력·query validity·
fusion gate g를 고정한다. Deformation-query channel은 계속 물리적으로 없다.
추가 학습이나 cache 재구축은 하지 않는다.

| 조건 | 별도 C(F) 전달 | F를 사용하는 R 입력 |
|---|---|---|
| Normal / independent repeat | C(F) | 원래 Top1+F 유지 |
| Direct-correction-zero | 0 | 동일 |
| Direct-correction-fusion-gated | g*C(F) | 동일 |

둘 모두 matching-aware attention 전체나 retargeter를 끄는 조건이 아니다.
사용자가 정의한 Top1 No-score 또는 native spatial No-corr arm으로 부르지 않는다.
Gated 조건은 같은 bounded C(F)의 전달량만 바꾼다. 기존 unbounded RayFeature
구조의 bit-exact 재현이라고 주장하지 않는다.

## 실행과 검증

GPU0·1은 현재 retargeting science가 실제 완료·file4 검증된 뒤 사용한다.
실행 중인 학습을 중단하거나 cohost하지 않는다. SS 최종128×3 평가를 우선하고
비는 슬롯에서 진단한다. GPU2·3의 NoDQ 학습과 예정된 third-DQ는 유지한다.
CPU metadata와 통계는92/93 두 core·nice10, neural inference는 할당 GPU만 쓴다.

네 GPU 실행(Normal 두 독립 load와 두 제어), normal parity CPU gate,
최종 CPU 요약으로6개 job을 구성한다. 두 normal load의 실제 input/noise/GT/
prediction latent·voxel·metric이 같아야 제어를 시작한다. Original quick288과의
count/metric 일치는 별도로 기록하며, 과거에 기록하지 않은 hash를 소급해 만들지 않는다.
독립적인 진단 수치 계약과 원래 quick validation을 혼용하지 않는다.

SS64·고정 DEV32·12step/CFG3·하나의 생성 seed map을 사용한다.
실제 batch·AMP/TF32/deterministic 설정과 NN/source tensor hash를 기록한다.
Normal은 기존 함수를 그대로 실행해 동일성을 먼저 검증한다. 제어는 원래
consumer의 semantic carrier·global/native projection·R 계산을 유지하고
residual의 C(F) 항만 바꾼다. Mask·dtype·반올림으로 다른 입력이 바뀌면 실제
불변식 검사에서 실패시키고 실패를 보존한다.

주 지표는 mean-asset voxel IoU다. 두 제어−Normal의 source-cluster paired
bootstrap4000회/seed20261023, 개별95% CI와 두 비교 family 보정 CI를 기록한다.
Precision/recall/F1과 개별 asset 변화는 탐색적 보조다. GT/Pred/Diff projection과
W&B·actual file4 closure를 남긴다. GPU 작업은 소유한 child session을 종료·정리한
뒤 예약을 반환한다. 개선을 통과 조건으로 두지 않는다.

## 결과가 결정할 다음 행동

C(F)를 줄이거나 제거했을 때 일관되게 좋아지면 ungated direct delivery를
우선 수정 대상으로 삼아, 공통 R96의 matched 재학습을 검토한다. 같은 weight의
진단 결과를 재학습 효과로 대신하지 않는다. 두 제어가 좋아지지 않으면 이 경로를
주원인으로 단정하지 않고 candidate/value와 최종480 결과를 함께 분석한다.
현재의480 예산과 기존 최종128×3·third-DQ·native PBR·DEV/독립 seed·최종508개
10 metric을 축소하거나 이 진단으로 대체하지 않는다.
