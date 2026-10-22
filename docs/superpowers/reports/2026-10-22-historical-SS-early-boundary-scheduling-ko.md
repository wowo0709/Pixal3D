# 현재 PBR 완료 뒤 기존 query 재현 SS를 먼저 비교

현재 Top-1+R PBR1900 학습은 끝까지 유지한다. 그 완료 및 canonical checkpoint
정리 이후, 이미 준비된 historical target-patch/R96 재현 세 SS arm을 먼저
실행한다. 이어서 예정된 feature+R와 raw Top-K+feature+R PBR1900을 진행한다.
새 arm을 추가하거나 기존 학습을 중단하지 않는다.

이 순서는 이전 양성 R 경로의 query·전달 위치를 보존한 비교를 더 일찍 얻기
위한 것이다. 현재 source-camera ray 모델과 기존 target-patch query는 같은
구조가 아니다. Native480에서 Top-K 경로의 소비는 확인했지만 추가 IoU 이득은
불확실하므로, 과거 query 설계의 실제 재현 결과를 이후 설계 판단에 활용한다.
이전 QKV의 독립적 성공이나 재현 세 arm의 미래 성능을 미리 주장하지 않는다.

| 순서 | 실행 | GPU |
|---|---|---|
| 현재 | Top-1+R PBR, 고정1900 update | 0–3 |
| 다음 | Historical retargeting, direct-noDQ 동시 학습 | 각각0–1, 2–3 |
| 이어서 | Historical direct-DQ, 완료된 arm의128×3 평가 | DQ는0–1, 평가는 가용 GPU |
| 이후 | 재현 세 arm 비교·중간 checkpoint 정리 완료 후 나머지 PBR | 0–3 |

세 SS의 원래 R96 NN/R/Adam, 추가384 update→480, global96/DDP2/micro48,
accumulation1, data·optimizer·config·command·output·평가 예산을 그대로 유지한다.
두 rank의 첫97 loss 및 Adam 복원 재현 capacity 검사는 세 arm 모두 이미
통과했다. 재현을 위해 현재 single-GPU SS와 다른 원래 DDP2 실행을 유지한다.

새 CPU admission은 현재 Top-1 PBR1900의 실제 종료0·완료 snapshot·checkpoint
정리 이후에만 실행된다. 세 SS의 warm checkpoint·capacity·미실행 상태와 실제
남은 디스크를 확인한다. 두 arm의 동시 저장을 보수적으로 모델당 checkpoint5개,
warm NN/R/Adam 크기 및 모델당1GiB 여유로 계산하고 workspace8GiB를 더했다.
필요 여유는 약36.604GiB다. 준비 시점의 공간은 실제 미래 admission으로 취급하지
않으며, 실행 시 공간이 부족하면 학습을 시작하지 않고 해당 증거를 남긴다.

Direct-DQ가 먼저 두 arm의 실제 완료·중간 checkpoint 정리를 기다리는 기존
조건도 유지한다. 다음 feature+R PBR에는 재현 세 arm의 실제128×3 요약과 세
checkpoint 정리를 추가 prerequisite로 연결한다. Pending job의 순서·dependency만
수정하며, queue-state·실행 중 descriptor·이미 SHA로 고정한 numerical plan과
source는 변경하지 않는다. 전체 dependency DAG 및 기존 command/config 불변을
canonical 변경 기록에서 검증한다.

Native 독립 seed202, PBR isolation 및 DEV full cascade, DEV 모델 결정과 최종508
전체10 metric 범위는 그대로 남는다. Historical 모델이 더 좋더라도 native
모델의 독립 seed 결과로 이를 대신하지 않는다. 필요한 해당 모델의 확인과
선택된 SS/PBR 조합의 DEV 검증을 거친 뒤 test를 진행해야 한다.
