# 완료된 PBR에서 직접 전달항과 R 보정항을 분리하는 진단

Top-1+R PBR 첫 epoch의 고정4 평가에서는 Axe가 개선되고 Bed가 악화됐다.
Vanilla와의 비교만으로는 직접 Top-1 전달과 R의 기여를 분리할 수 없다.
현재 Top-1 PBR의 예정된1900 학습을 완료한 뒤, 같은 가중치에서 다음 진단을
실행한다. 현재 학습을 중단하거나 새 학습 arm을 추가하지 않는다.

Source camera마다 실제 보정은 `u*g*(Top1−S) + u*R`을 native projection한 값이다.
`g`는 signed fusion gate이며 matching confidence 자체가 아니다.

| 진단 | 유지하는 실제 전달항 | 목적 |
|---|---|---|
| Normal | 직접 Top-1 전달 + R | 새 진단 프로토콜의 기준 |
| 독립 Normal 반복 | 위와 동일, 독립 model load | 생성·render·feature 경로의 정확한 재현 |
| R-off | `u*g*(Top1−S)` | 저장된 모델이 R 보정에 의존하는가 |
| Direct Top-1 shift off | `u*R` | 직접 feature 교체가 도움이 되는가 |
| Both off | Native conditioning | 같은 학습 NN에서 source 보정 전체를 제거한 결과 |

직접 전달항을 끌 때 numerator뿐 아니라 source를 빼는 scalar mass도0으로
만든다. R-off에서는 이 mass를 유지한다. 원래 module은 모든 경우 그대로
forward하고, 후보·score·Top-1 carrier·계산된 R·use·gate를 고정한 뒤 전달항만
바꾼다. R이나 channel을 제거한 재학습 ablation으로 해석하지 않는다.

## 실행·재현 조건

실제 Top-1 PBR1900 종료0·네 rank의 budget/frozen/DDP 증명·canonical 완료
snapshot 이후에만 GPU0–3에서 실행한다. 두 Normal을 먼저 실행하고 나머지
세 개입을 연결한다. 모델당 고정4개·GT Shape 조건·target camera8·native12 step,
CFG3, render256을 사용한다. 현재 학습 및 기존 재현 SS/PBR의 budget은 유지한다.

이 진단은 deterministic algorithms를 명시한 별도 프로토콜이다. 독립 Normal
두 실행의 실제 prediction latent·metric·GT/Pred pixels·consumer가 정확히 같아야
한다. 원래 epoch validation과의 prediction/GT pixel 일치 여부는 따로 기록한다.
차이가 있으면 원래 epoch의 exact replay라고 이름 붙이지 않는다. 개입 차이는
오직 이 새 프로토콜의 Normal을 기준으로 계산한다.

각 개입에서 NN/source/고정 encoder의 hash, RGB·noise·GT PBR·GT Shape·camera,
candidate·matching score·use·기존 R 출력·global/negative conditioning을 대조한다.
Both off는 실제 projected correction이0이고 native conditioning과 값이 정확히
같아야 한다. Fresh eval optimizer는 사용하지 않으며 실제 trained step1900과
진단 optimizer update0을 구분한다. Source weight나 데이터는 바꾸지 않는다.

Owned GPU child와 session을 실제 종료·정리한 뒤 reservation을 놓는다. 모델
hash나 정상 반복 검사가 실패하면 이력을 보존하며, 통과로 바꾸지 않는다.

## 해석 제한과 준비 검증

이 검사는 저장된 한 모델의 조건부 의존성이다. R 없이 처음부터 학습한 모델,
LoRA만 따로 학습한 모델, spatial NoCorr 또는 사용자 정의 NoScore=Top-1의
추가 training arm이 아니다. Frozen same-model counterfactual의 분포 변화도
포함하므로, 개입으로 좋아졌다고 바로 새로운 최종 모델로 채택하지 않는다.

n=4의 기술적 경향으로 기록하고 CI나 전체 데이터 개선을 주장하지 않는다.
Source preservation·predicted-geometry isolation·full cascade·최종508/10 metric과
남은 PBR feature/Top-K 비교는 계속 필요하다.

Signed gate·source subtraction·R 분리·native 복원, 입력/weight/negative/GT 변경,
독립 Normal 불일치 및 잘못된 step/update를 거부하는 CPU 검사24개가 통과했다.
새 GPU entry와 wrapper는 구문 검증했다. 실제 trained1900 GPU 실행·Normal 반복
및 최종 진단 결과는 아직 완료되지 않았으며 canonical 준비/queue와 구분한다.
