# Semantic matching + Top-K + matching-aware attention 모델 탐색

목표는 exact patch Top1 정확도를 높이는 것이 아니라, 여러 semantic 후보에서 생성에 유용한 feature를 전달해 SS 성능을 개선하는 것이다. 이전 C1의 개선은 기준선으로 보존하며, Top-K 자체의 추가 효과와 동일시하지 않는다. 새 namespace `rgb_matching_v2/semantic-topk-search-v1`에서 실행하고 과거317개 queue와 저장본을 변경하지 않는다.

## 고정 조건과 첫 비교

DeformNeXt world-v4 train9149/validation508, 실제 사용자 SS4210 base+LoRA, frozen Omega epoch7을 유지한다. 새 run은 seed101, global96, DDP2/micro48/accumulation1,5epochs/480updates다. GPU0/1과2/3을 사용한다. 매 epoch 같은32개 quick voxel metric/GT-Pred를 W&B에 남긴다. Final은 raw480, 기존 fixed128/93source clusters/서로 다른3noise,12steps/CFG3/generation8로 비교한다. Development 비교이고 미사용 test라고 표현하지 않는다.

첫 wave는 같은 R9 head와 같은64개 후보에서 `top1-reference`와 `soft-topk`를 독립 학습한다. Rejection과 retargeter는 둘 다 없다. Temperature는 head-train 내부 calibration의 고정 첫64 asset에서 alpha-valid query만 사용하고, 고정 grid의 평균 entropy 실효 후보 수가4에 가장 가까운 값으로 선택한다. GT·exact-match label·validation 성능은 선택에 사용하지 않는다. 이 기준은 collapse를 피하기 위한 실행 설정이지 성능 개선 증거가 아니다.

## 세 구성요소의 실제 사용과 효과를 구분

| 요소 | 사용 검증 | 성능 검증 |
| --- | --- | --- |
| Matching decoder | 실제 후보 context가 query/score를 바꾸는지, 학습 가능한 경우 gradient와 weight 변화 검사 | 동일 후보·같은 budget에서 contextual decoder와 pairwise/no-context 비교. Top1 대 soft를 decoder ON/OFF로 부르지 않음 |
| Matching score | score를 바꾸면 candidate weight와 전달 feature가 바뀌는지, entropy·실효K·gradient 검사 | Top1 대 softTopK, 그리고 학습 attention의 score-prior ON/OFF. 사용자 No-score는 Top1을 뜻하므로 prior-OFF를 같은 이름으로 부르지 않음 |
| Correspondence | 실제 semantic candidate 주소와 source value의 연결 검사 | Native spatial 기준 및 같은 후보/support를 보존하는 주소 개입. Wrong-address만으로 semantic 후보 검색 전체의 효과를 판정하지 않음 |

현재 cache의 `spatial_indices`는 GT target depth를 unproject해 만들어졌다. 이를 inference용 No-corr에 넣지 않는다. 원래 Pixal3D의 native camera projection을 spatial 기준으로 사용한다. Same-support/address 개입과 native spatial 비교는 서로 다른 질문이며 함께 보고한다.

## 후속 모델 탐색

Frozen score의 soft weighting 이후에는 생성 loss를 통해 후보의 유용성을 학습하는 attention을 비교한다. 초기 native 출력을 정확히 보존하고, frozen candidate/value 계약과 CFG·BF16·training/inference parity를 먼저 검사한다. 성공한 C1 retargeter와 결합한 모델은 같은 static-attention C1과 비교해 adapter와 attention의 효과를 분리한다. Decoder context와 matching-score prior를 각각 제거해 실제 기여를 확인하며, no-query channel을 물리적으로 유지한다. 세부 새 구조·arm은 구현 전 계약과 검사 결과에 추가 기록한다.

고정 결과에서 vanilla V1 및 matched Top1 대비 source-cluster CI와 noise별 방향을 확인하고, 개선 후보는 추가 training seed로 재검증한다. 후보를 선택한 development panel의 효과와 독립 seed의 재현을 구분한다. 작은 quick32 개선이나 attention이 넓어졌다는 사실만으로 모델을 채택하지 않는다. 음의 결과도 보존하고 효과를 강제로 양수로 만들지 않는다. SS에서 근거를 얻기 전에는 PBR 전체 ladder로 확대하지 않는다.

이 문서의 작성·config 생성은 학습 완료를 뜻하지 않는다. 첫 두 run만 완료해도 decoder/score/correspondence 전체 검증과 최종 모델 탐색이 끝난 것은 아니다.
