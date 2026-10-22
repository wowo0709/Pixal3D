# 과거 양성 R 경로와 현재 native SS의 동일 지표 비교

과거 target-patch Top-1+R도 현재 모델과 같은 지표로 비교하면 강한 기준선이다.
현재 source-camera-ray 모델이 과거 기준보다 개선됐다는 근거는 아직 없다.
Ray 단위 feature 공유와 학습형 QKV attention의 필요성을 같은 주장으로 취급하지 않는다.

실제 저장된8모델×128개×3생성 noise의24개 evaluation에서 sample ID·source cluster·
생성 seed·64³ GT voxel hash/shape·GT positive count가 모두 같음을 확인했다.
모두 raw480,12-step/CFG3, generation batch8, edited-target RGB+source RGB3장이다.
새 inference나 optimizer update는 수행하지 않았다.

| 모델 | 자산별 평균 IoU (%) | 전체 TP/FP/FN 합산 IoU (%) |
|---|---:|---:|
| 과거 V1 | 46.7750 | 43.2650 |
| 과거 Top-1 reference | 46.9369 | 43.4529 |
| 과거 C1 | 47.4841 | 44.2193 |
| 과거 target-patch Top-1+R | **47.5245** | **44.2757** |
| 현재 native V1 | 46.7355 | 43.1957 |
| 현재 source-ray Top-1+R | 47.2600 | 43.7038 |
| 현재 feature+R | 47.0473 | 43.5735 |
| 현재 raw Top-K+feature+R | 47.2078 | 43.7031 |

자산별 평균은 각 자산의 세 noise IoU를 평균하고128개를 평균한다. 합산 IoU는
128개·세 noise의 TP/FP/FN을 모두 합한 뒤 비율을 구한다. 세 noise의 pooled IoU를
단순 평균하는 방식과도 다르다. 과거 .44대와 현재 .47대를 직접 비교하면
집계 방식의 차이를 성능 향상으로 잘못 읽을 수 있다.

93개 source cluster의 paired bootstrap4000회/seed20261023으로 계산한 탐색적
개별95% CI는 다음과 같다. 차이 단위는 percentage point다.

| 비교: 자산별 평균 IoU | 차이 | 95% CI |
|---|---:|---|
| 과거 C1 − 현재 Top-1+R | +.2241 | [−.2451, +.6777] |
| 과거 target-patch Top-1+R − 현재 Top-1+R | +.2645 | [−.2083, +.7181] |
| 과거 target-patch Top-1+R − 과거 Top-1 reference | +.5876 | [+.2201, +.9483] |
| 현재 Top-1+R − 현재 V1 | +.5245 | [+.1605, +.9077] |
| 현재 raw Top-K+feature+R − 현재 Top-1+R | −.0522 | [−.3014, +.2004] |

과거 두 기준선은 평균에서 현재 Top-1+R보다 높지만 차이의 CI는0을 포함한다.
이 비교에서 어느 쪽의 우월성이나 동등성이 입증된 것은 아니다. 현재 원래 native
최종128×3 보고서의 family 보정 CI를 이 탐색적 CI로 대체하지 않는다.

## 서로 다른 학습·수치 조건은 그대로 남는다

GT·기록된 sampling seed가 같다는 사실이 실제 초기 noise tensor hash나 학습 RNG·
optimizer trajectory의 동일성까지 증명하지는 않는다. 과거 target-patch/nearest 전달과
현재 source-camera-ray/bilinear 전달은 query·source 입력·보정 위치 및 실행 방식이
다르다. 현재 native 평가는 deterministic algorithms를 사용했으며 과거 record는
그 값을 명시하지 않는다. 따라서 두 계열 사이의 차이를 ray grouping, member matching,
R 또는 QKV 하나의 순수 효과로 해석하지 않는다. 세 생성 noise를 독립 학습 seed로
세지도 않는다.

이 결과는 예정된 faithful target-patch/R96 재현 세 arm을 유지할 근거다. 현재
Top-1 PBR 학습 종료 이후 이 세 SS를 먼저 비교하는 순서도 유지한다. 기존 양성
경로를 기준에서 제외하거나, QKV가 포함된 조합의 V1 대비 이득만으로 attention을
필수로 채택하지 않는다. 최종 SS/PBR 조합은 DEV cascade와 독립 학습 근거로 검증한다.

Canonical 비교는 `native-flow-cold3-v1/`
`legacy-and-native-SS128-common-metric-audit-v1/`이다. 실제 사용한24개 evaluation,
2,575,260bytes의 SHA256·크기를 SSH로 file4에서 읽어 대조했다. 원래 파일을 서버로
다시 쓰거나 수정하지 않았다. 새 audit·worker 두 파일307,869bytes도 실제 file4 저장을
확인했다.

## 이전 결과의 보존·완료 범위 재확인

별도 `completed-legacy-SS-evidence-scope-audit-v1/`에서 과거 SS 캠페인의20항목·
58개 metadata 증거를 원래 digest와 대조하고, primary14개 구성의42개 평가와
semantic seven-arm 보고서에 포함된11개 구성의33개 평가를 다시 계산했다.
두 보고서의 TP/FP/FN 기반 최종 IoU/F1은 정확히 재현됐다. Eleven-model 쪽은
128개·93cluster·세 noise의 GT/hash/seed 및 source-only M0 RGB4장,
edited M1 target+M0 source3장의 입력 계약도 확인했다.

관련127개 파일13,841,166bytes를 실제 file4에서 읽어 SHA256/크기를 대조했다.
이 검사는 bounded metadata와 저장된 count의 검증이다. 새 GPU 재현이나 모든 대형
가중치·원본 데이터의 재검증이 아니다. 과거 캠페인의 완료를 현재 목표의 완료로
취급하지 않는다. 기존14모델 DEV448 cascade·zero64·PBR3의 coverage audit,
진행 중인 새 PBR/SS 재현·독립 seed·선택된 조합·최종508개10 metric은 별도로 남는다.
