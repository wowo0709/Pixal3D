# SS 고정 checkpoint의 sampling 민감도 진단

[최초 실험 계획](2026-10-17-ss-matching-v2-experiment-plan-ko.md)의 8.1은 flow loss와 12-step 생성 경향이 엇갈리면 대표 checkpoint에 같은 더 긴 sampling을 적용하도록 정했다. 이 조건을 실제 본 비교에서 확인해 후속 진단을 추가한다. 기존 학습과 fixed-final128×3의 primary protocol은 유지한다.

## 실제 실행 근거

완료된 새 B3-attention-R의 고정32 validation에서 평균 K4 flow loss는 step96 **0.1203734 → step480 0.1177244**로 줄었지만, 12-step voxel IoU는 **0.4356038 → 0.4336378**이었다. 이는 loss 개선을 생성 개선으로 해석할 수 없는 관측이다. 작은 패널의 변동인지 sampling 민감도인지 아직 구분하지 못했다. 이 관측만으로 optimizer나 모듈의 오류를 확정하지 않는다.

## 고정할 비교

대표 checkpoint 네 개에 각각 **12-step/50-step**을 적용한다. Vanilla V1과 기존 R9 C1은 이전 양의 경로를, 새 B3-attention-R과 C2-query-R은 새 matching/Full 후보의 경향을 대표한다. 모두 해당 run의 **step480 raw weight와 동일 conditioner**를 사용한다. 새로운 학습이나 best-epoch 선택은 없다.

고정32 source cluster·한 sampling noise·CFG3·generation batch8·M1 camera8 RGB와 M0 camera9/10/11 RGB를 공유한다. 각 checkpoint의 50−12 변화와 각 sampler에서 V1 대비 차이를 계산한다. 같은 model의 12/50 비교에서 바뀌는 것은 sampling 횟수다. 서로 다른 checkpoint의 학습 seed/실행 조건 차이가 제거됐다는 주장은 하지 않는다. C2와 기존 C1의 구조 차이가 많으므로 이 네 개만으로 query의 독립 효과도 판단하지 않는다.

실제 비교 전 source recipe·checkpoint·conditioner·panel·seed map의 SHA256을 기록한다. 새 12-step 평가가 기존 primary noise0의 같은32 TP/FP/FN·seed·source cluster를 재현해야 한다. 여덟 평가 모두 실제 GT occupancy hash, raw step480, zero optimizer update, 네 카메라 RGB·CFG·generation batch를 검사한다. 동일 weight의 12/50 비교는 step 차이만 의도적으로 허용하며 GT/noise/source identity 검사는 그대로 유지한다. CPU 검증 12개로 이 계약과 recipe의 제한된 변경을 확인했다. GPU 평가 전의 설정·검증 통과를 성능 완료로 표시하지 않는다.

## 실행과 해석 범위

현재 C1/V1 및 후속 B3 학습은 유지한다. 원본 constant-gate 두 arm·3개 seed의 집계가 끝난 뒤 비는 **GPU2/3**에서 네 평가씩 순차 실행한다. GPU0/1의 B3와 병행할 수 있으며 허용 장치는 계속 0~3뿐이다. 진단 여덟 개, strict CPU 집계, W&B publication을 큐에 추가한다. Source는 `sampler_sensitivity.py`, canonical 결과는 file4의 `rgb_matching_v2/sampler-sensitivity-v1/`이다.

결과의 source-cluster bootstrap CI는 한 sampling noise·저장된 특정 checkpoint에 조건부인 탐색적 결과다. 반복 비교에 대한 보정이나 training-seed 모집단 CI는 아니다. `selection_use=false`, `production_sampler_changed=false`를 유지하며 primary final128×3를 덮어쓰거나 각 arm에 유리한 sampler를 골라 본 표를 만들지 않는다. W&B에는 수치·GT/Pred/Diff 여덟 이미지·paired 표·source fingerprint와 summary artifact를 기록한다.
