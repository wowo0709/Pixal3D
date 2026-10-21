# 새 조합의 matching 개선과 native 생성 품질을 함께 판정하기

새 bounded-source-contrast teacher112의 native32 IoU만으로 matching-aware attention의
효과를 판단하지 않는다. 같은 final112 weight에 대해 CPU 관측을 연결한다. 추가 GPU
학습이나 생성 없이 원래 검증32의 cached feature로 normal forward32회를 실행한다.
기존 Plain Contrast112의 같은 관측과 비교한다. 새 모델의 학습·관측 결과는 아직 없다.

## 학습 후 관측

Frozen matcher의 state hash, 각 자산의64개 ray 후보·prior hash, GT topology label hash,
현재 RGB·geometry·camera fingerprint가 기존 Plain Contrast의 관측과 같아야 한다.
Query 단위는 동일 ray의 distinct target-patch member에 속한 visible correspondence의
집합이며, unique surface point GT가 아니다. GT label과 검증용 M1 source feature는
모델의 normal forward가 끝난 뒤 읽으며 입력이나 loss에 사용하지 않는다.

각 모델의 matching prior, 실제 네 head의 평균 확률을 대상으로 exact/Near1 GT-set
Top-1 hit와 GT 확률 질량을 기록한다. All-source 및 own-source-camera 범위를 따로
측정한다. 후보 coverage가 바뀌지 않았는지도 검사한다. Source-use, 유효 K, source
delta 구성, source feature teacher와의 방향·MSE도 함께 기록한다. 모든 state tensor는
관측 전후 같아야 하고 encoder·denoiser·optimizer·CUDA는 실행하지 않는다.

새 후보−Plain Contrast와 새 후보−자신의 matching prior를 자산/source cluster32개
단위 paired bootstrap10,000회로 비교한다. Seed20261022를 고정한다. 수만 개 ray를
독립 데이터로 resample하지 않는다. Native32의 GT/noise를 맞춘 같은 final112 비교와
함께 표·CSV·PNG/PDF·W&B에 기록한다. 기존 source teacher 세 모델을 다시 학습하지 않는다.

## 결과가 가르는 원인

| 실제 관측 | 다음 해석과 조치 |
|---|---|
| GT 질량과 native IoU 모두 개선 | 이 조합의 작은 개발 패널 근거. Source-disjoint 확인 후 joint 학습으로 검증 |
| GT 질량 개선, native 개선 불확실 | Correspondence 순위만의 문제가 아닐 수 있다. Value 전달·R·teacher objective와 native task의 차이를 조사 |
| GT 질량 악화, native IoU 개선 | GT-set 확률이 task optimum과 같다는 가정을 재검토. GT 질량만으로 head를 제거하지 않음 |
| 둘 다 개선되지 않음 | 현재 교정 가설을 지지하지 않음. 반복 epoch 연장 대신 source adaptation 방향과 표현 경로를 재검토 |

위 표는 관측을 해석하는 기준이며 CI가0을 포함하는 경우를 개선 또는 동등성의 증거로
간주하지 않는다. Repeated development32의 보정되지 않은 interval이며 모델 채택
기준이 아니다. 개별 decoder·score·NoScore Top-1 feature·NoCorr native spatial의
재학습 ablation이나 full9149 joint·cascade·test508을 대신하지 않는다.

통계 연결의 사전 검증은 이미 완료된 실제32개 결과로8개 contrast의 값과 bootstrap
interval을 기존 publication과 수치적으로 정확히 재현하는 방식이다. 이때 기존
analytic `prior_only`는 `log π`를 softmax로 다시 정규화했으므로, 저장된 해당 control의
통계를 사용한다. `exp(log π)`를 그대로 집계하면 float32 재정규화에 따른 작은 차이가
있다. 새 비교의 prior는 원래 저장된 matching logit의 exp로 정의하고 이를 명시한다.
첫 사전 검사에서 이 차이를 잡아낸 실패 log를 보존했으며, 같은 정의로 맞춘 두 번째
검사가 통과했다. 이 검사는 새 후보의 성능 결과가 아니다.
