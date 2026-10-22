# R-rawTopK112의 실제 native SS 기여 진단

## 확인할 문제

동일 TRAIN1536/112-step feature-teacher 학습에서 R-rawTopK의 native IoU는 .431005,
bounded control은 .430592였다. 차이 CI가 0을 포함한다. 매칭/feature 지표만으로
큰 joint 학습이나 추가 에폭을 정당화하지 않고, 완성된 112-step 가중치가 어떤 정보를
실제로 사용하는지 먼저 분리한다. 이 평가는 frozen SS4210의 **추론 개입 진단**이다.
재학습 ablation이나 최종 모델 선택을 대신하지 않는다.

## 같은 가중치에서 비교할 네 개입

| 개입 | 고정하는 항목 | 바꾸는 항목 | 해석 범위 |
|---|---|---|---|
| wrong-retrieved-values-fixed-use | candidate, key/semantic score, 네 head 확률, Top1 carrier, source-use | retrieved raw V와 projected V의 donor를 동일하게 순열 변경 | 올바른 correspondence value 연결이 실제 출력에 필요한가 |
| semantic-prior-only | candidate, V, semantic prior, Top1 carrier, source-use | 학습 후 bounded attention 확률을 원래 semantic prior 확률로 교체 | 학습된 attention logit 보정이 원래 matching score보다 유용한가 |
| top1-feature-only | candidate, V, semantic Top1, Top1 carrier, source-use | 네 head 모두 semantic Top1 하나만 선택 | soft Top-K 조합이 Top1 feature만 쓰는 것보다 유용한가 |
| no-transformed-value | candidate, 확률, raw Top-K appearance, carrier, source-use | learned value/output feature change만 0으로 제거 | raw mixture 외에 learned value 변환이 R에서 기여하는가 |

모든 개입에서 raw appearance와 transformed value는 동일한 실제 확률을 사용한다.
Top1은 uniform이 아니다. Prior-only도 uniform이 아니며 semantic matching score를 유지한다.
Wrong-V는 spatial matching으로 바꾸는 NoCorr arm이 아니다. Value 변환 제거 역시 전체 matching
decoder를 제거하는 모델이 아니다. 이 결과로 decoder architecture 자체의 독립 효과를 주장하지 않는다.
사용자가 정의한 NoScore(Top1 feature), NoCorr(native spatial), 물리적 NoDQ의 재학습 비교는 별도 범위다.

## 증명 후 실행

새 코드가 기본 동작을 바꾸지 않는지 normal-shadow native32를 먼저 실행한다. 각 실제 module
forward에서 같은 입력의 normal shadow와 actual 출력을 비교하고, normal에서는 source numerator,
R appearance/출력, attention probability가 정확히 같아야 한다. 네 개입에서도 candidate/semantic prior,
Top1 carrier, source-use, semantic weight, source row, CPU/CUDA RNG 및 parameter version은 동일해야 한다.

Normal native32의 모든 기록된 sample/cluster/noise/GT hash/TP/FP/FN/metric과 sampler 설정이
기존 완료된 step112 결과와 정확히 같아야 한다. Raw voxel mask는 이 평가에서 보관하지 않았으므로
mask bit identity까지 검증했다고 쓰지 않는다. 이 증명이 PASS인 경우에만 네 추론 개입을 실행한다.
증명 실패 시 후속 개입을 실행하지 않고 실패 이력을 보존한다.

개입은 scoped hook으로 설치하고 종료 시 해제한다. 실제 checkpoint loader/기존 evaluator,
native trainer registry, input RGB, camera, native target/global/negative 경로는 유지한다.
기존 평가 파일이나 SHA-bound 코드를 수정하지 않고 별도 entry에서 process-local constructor만
잠시 바인딩한다. Eval/no_grad 밖에서는 즉시 실패하며 optimizer update는 없다.

동일 32개 source cluster, 학습 seed101, generation noise 1회, generation batch8,
12-step/guidance3 조건을 사용한다. Native IoU의 intervention−normal 차이와 source-cluster CI를
보고한다. 반복 사용한 개발 패널이며 다중 비교 보정 전이므로 탐색적 결과다.

## 실행과 이후 판단

GPU는 지정된 0~3만 사용하고 각 진단은 한 GPU를 독점한다. 진행 중인 zero-shot 평가를
중단하지 않으며 정상 종료 후 비는 GPU에서 시작한다. Normal proof 뒤 네 개입을 가능한 만큼
동시에 실행하고, CPU 두 core에서 paired 통계와 W&B 표/CI를 집계한다. 새 neural training arm은 없다.
Zero-shot 두 조건과 기존 cascade/8-metric 평가도 그대로 완료한다.

올바른 V 또는 soft Top-K가 native 품질을 개선하지 못하면 단순한 matching 지표 상승을
성공으로 취급하지 않는다. 원래 score, learned attention, raw mixture, value transform 중
소비 단계에서 유효한 부분을 이 진단으로 좁힌 뒤 새로운 학습 설계를 결정한다.
72개 source-disjoint/3-noise 검증, native SS/PBR joint objective 학습, 독립 학습 seed,
full cascade와 final508/10-metric 평가는 여전히 필요하다. 기존 source72 HOLD는 변경하지 않는다.

CPU 회귀 검증 8개는 normal 동등성, 네 개입의 변경 범위, source-use/score/RNG/weight 고정,
Top1 비균등 선택 및 invalid/빈 support 처리, 학습 중 실행 거부와 hook 해제를 확인한다.
실제 GPU/native 동등성은 별도 선행 증명에서 확인해야 한다.
