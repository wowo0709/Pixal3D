# Native transport projection 진단

기존 matching-v2 GT transport gate는 HOLD다. 이를 우회해 본 학습을 시작하지 않고, 실패 원인 중 하나일 수 있는 projection operator 차이를 먼저 분리한다. 기존 run과 cache는 변경하지 않는다.

코드 확인 결과 native Pixal3D는 32×32 DINO map을 `ProjGrid`의 bilinear/border/align_corners=False로 투영한다. 추가 semantic 경로는 canonical camera의 최근접 patch ID로 값을 gather한다. 좌표계가 같아도 이 두 연산이 같다는 뜻은 아니다. 이전 reports_final2의 GT source-address 양의 결과는 실제 native ProjGrid를 통과했으므로 이 차이를 확인할 필요가 있다.

동일 V1 final checkpoint, 네 RGB 입력, sampler12/CFG3, 고정 calibration32/sample별 seed를 사용한다. Gate scale은 이전 calibration sweep에서 이미 사용한 0.25로 고정한다. 이 진단에서 weight를 업데이트하지 않는다.

| 조건 | 변경 |
| --- | --- |
| native | 추가 경로 OFF; cached target self-map의 native anchor projection 오차도 측정 |
| nearest | 기존 GT-address 추가 경로 그대로 |
| bilinear | GT-address evidence를 native ProjGrid로 투영; 기존 nearest validity 유지 |
| normalized | bilinear feature를 valid support로 정규화; coverage로 전체 update 조절 |
| wrong-object | bilinear와 같되 source value만 같은 batch의 다른 asset 값으로 교체 |

Self-map 검사는 feature/projection 동등성 진단이며 target-only 생성 arm이 아니다. 모든 생성 조건은 native 4-view/global condition을 유지한다. Calibration에서 native 대비 이득과 wrong-object 대비 주소/물체 정보의 효용이 확인된 연산만 고정해 validation32에서 재확인한다. 개선이 없으면 source value/interface의 다른 원인을 계속 조사한다. GT 진단의 성공을 production matcher 또는 fine-tuning 성공으로 표현하지 않는다.

실행은 명시적으로 할당된 GPU 0~3 내 빈 장치에서 수행한다. 결과는 file4의 `rgb_matching_v2/projection-audit/`에 별도 보존하고 W&B 공통 평가 run에 기록한다.

## 독립적인 source/target balance 진단

C1 wrong-object 진단에서 효과가 상당 부분 남았으므로 adapter가 정확한 source address 외의 보정을 수행했을 가능성도 조사한다. `native_proj + w*(anchor_proj-native_proj)`, 고정 w=0/0.25/0.5를 calibration32에서 비교한다. Native global condition은 바꾸지 않고 모든 경우 source 세 장의 projection 가중치도 양수다. w=0.5에서도 target/source별 가중치는 0.625/0.125/0.125/0.125로, target-only arm이 아니다. GT geometry/address를 이 진단의 추론 입력으로 사용하지 않는다. 기존 V1 checkpoint를 고정한 추론 개입이며 새 독립 학습의 효과와 구분한다. Calibration에서 선택된 nonzero w가 native를 개선하면 고정한 뒤 validation128에서 V1과 비교한다. 결과는 `role-balance-probe/`에 보존한다.
