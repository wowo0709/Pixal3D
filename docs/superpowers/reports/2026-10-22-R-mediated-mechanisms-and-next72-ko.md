# R-mediated 추론 진단 결과와 고정 72개 검증

## 실제 완료된 진단

R-rawTopK112의 normal-shadow native32는 실제 GPU에서 기존 결과와 같았다.
기록된 전체 sample/GT/noise/occupancy count/metric, 동일 입력에서의 module 출력 및 RNG 검증을
통과했고 다섯 native 실행과 통계 집계는 모두 terminal 0 및 file4 SHA closure를 남겼다.

| 동일 가중치 개입 | native IoU | normal 대비 차이 | source-cluster 95% CI |
|---|---:|---:|---|
| Normal | .431005 | — | — |
| Top-K 내부 retrieved V 순열 | .431019 | +.000014 | [−.001353, +.001303] |
| Learned residual 없이 semantic prior 확률만 | .430853 | −.000153 | [−.000785, +.000673] |
| Semantic Top1 feature 하나만 | .430724 | −.000281 | [−.001665, +.001136] |
| Learned V/output feature change 제거 | .429968 | −.001037 | [−.002761, +.000613] |

네 개입 모두 candidate/key/semantic prior/Top1 carrier/source-use/semantic weight/RNG/parameter version을
고정했다. Value-only 두 개입은 네 head 확률도 동일했다. Top1은 uniform이 아니라 한 correspondence만 사용했다.
이 개발 32개에서 학습된 attention 보정·soft mixture·value 변환의 native 품질 기여는 아직 입증되지 않았다.

Wrong-V는 **같은 semantic Top-K 안에서** 값 순서를 바꿨다. 후보들이 서로 비슷할 수 있으므로
이 결과를 semantic matching 전체가 불필요하다는 증거로 확대할 수 없다.
Prior-only도 semantic score를 유지한다. NoCorr의 spatial correspondence 재학습이나 decoder 구조 자체의
독립 효과를 검사한 결과가 아니다. 재학습 ablation과 same-weight 개입을 혼동하지 않는다.

## zero-shot SS와의 구분

기존 완료된 두 원본 SS4210 archive를 실제 canonical receipt로 확인하고 동일 GT/noise로 비교했다.
입력은 모두 RGB 4장이며 source4에는 deform 전 camera8을 포함한다.

| 모델/입력 | native32 IoU |
|---|---:|
| 원본 MV zero-shot, M0 source4 | .394997 |
| 원본 MV zero-shot, M1 target8 + M0 source9/10/11 | .426715 |
| R-rawTopK112, 같은 edited-target 입력 | .431005 |

Edited-target zero-shot − source4는 +.031719, CI [+.011109, +.055051].
R-rawTopK112 − edited-target zero-shot은 +.004290, CI [+.000966, +.007743].
후자는 frozen SS 위 source module 전체 학습 효과에 관한 개발 관측이며,
matching-aware attention의 독립 효과나 full NN LoRA 학습 결과가 아니다.
Bounded control 대비 R-rawTopK 차이 +.000413의 CI가 0을 포함한다는 결과도 유지한다.

## 다음 단일 후보 검증

여러 teacher epoch나 추가 후보를 탐색하지 않고, **완료된 R-rawTopK의 고정 step112 하나**를
기존 source-disjoint72/61 cluster 패널, 기존 세 generation noise로 평가한다.
32개 pilot과 source cluster가 겹치지 않는 이미 구축된 split/seed/GT를 그대로 사용한다.
이 72개도 앞선 모델 설계 과정에서 결과를 본 개발 패널이다. 새로운 untouched holdout이나
confirmatory test로 부르지 않으며, 최종 test508은 계속 모델 선택 이후까지 유지한다.

이미 완료된 Original19 teacher112와 Contrast19 teacher112의 같은72/세 noise 결과를 재사용한다.
동일 base SS4210, TRAIN1536, source 학습 112 updates/7에폭 예산을 비교하고,
각 noise마다 sample/cluster/GT hash/seed가 같은지 검사한다.
세 source checkpoint는 구조가 여러 부분에서 다르므로 단일 matching decoder/score의 인과 효과라고 하지 않는다.
새 neural training은 없고 세 단일 GPU 평가만 추가한다. 정상 실행 중인 zero-shot 작업은 유지한다.

기존 plain-contrast72 admission의 HOLD를 수정하지 않는다. 새 후보 결과는 별도 lineage로 기록한다.
이 결과와 위 소비 단계 진단을 토대로 full NN joint objective에서 검증할 구조를 결정해야 하며,
proxy feature fit만으로 효과적인 full 모델이 완성됐다고 처리하지 않는다.
Native capacity/SS-PBR joint/독립 seed/full cascade/final508/10-metric 범위는 남아 있다.
