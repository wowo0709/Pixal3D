# 원본 constant-gate network를 보존한 geometry 학습 재현

현재 `runs-v2`의 geometry 두 arm·3개 seed 학습은 완료됐지만, gate의 학습 parameterization은 원본과 달랐다. 이 차이를 수정한 별도 검증을 진행한다. 기존 결과만으로 원본 모듈이 현재 데이터에서 효과가 없다고 확정하지 않는다.

## 확인한 차이와 수정

원본 R4-content는 geometry-independent gate도 `LayerNorm9 → Linear9→32 → SiLU → Linear32→1`의 371개 parameter로 학습했다. 이전 pilot은 이를 학습 가능한 scalar logit 1개로 단순화했다. 두 표현 모두 상수 gate를 학습하지만, AdamW·weight decay·gradient clipping이 parameter별로 작동하므로 학습 과정은 같지 않다. 추론 함수가 같다는 검증만으로 학습 재현을 주장할 수 없었다.

`GeometryContentRetargetAdapter`에 명시적 `historical-network` 모드를 추가했다. 원본 content/gate/residual parameter를 보존하고, 원본 reference initializer를 통해 공통 parameter를 맞춘다. No-query는 content encoder와 residual의 query 입력 열을 물리적으로 제거한다. 두 arm에 공통인 constant-gate network는 실제 geometry를 입력받지 않는다. Query tensor를 0으로 대체하는 ablation이나 geometry-gate 2×2를 추가하지 않는다.

CPU에서는 이식한 R4와 원본 authority에 동일 초기 state·입력·loss·AdamW·clip1을 주고 **6회 연속 update의 매 step에서 모든 parameter가 bit-exact**임을 검증했다. 별도 검증에서는 scalar 변환이 최초 추론 함수와 일치해도 첫 optimizer update부터 달라짐을 확인했다. No-query의 3075 channel, query arm의 3107 channel, 공통 weight prefix·gate state·initializer RNG 보존도 검증했다. 이 PASS는 GPU 학습 완료나 성능 향상을 뜻하지 않는다.

## 기존 scalar pilot의 3개 seed 결과

32개 calibration source·3개 training seed·3개 sampling noise의 고정 결과다. GT는 실제 SS encoder 입력 support이며 metric은 macro 집계다.

| Method | Macro IoU | Source recall32 | Stale leak32 ↓ |
| --- | ---: | ---: | ---: |
| Frozen native | 0.445530 | 0.825740 | 0.601524 |
| Scalar-gate adapter | 0.445856 | 0.819366 | 0.573571 |
| Scalar-gate adapter+query | 0.444496 | 0.820379 | 0.573087 |

Query−adapter의 IoU는 −0.136pp, 95% source-cluster CI [−0.501, +0.213]pp였다. Source recall32는 +0.101pp [−0.256, +0.440]pp, stale leak32는 −0.048pp [−0.885, +0.769]pp로 추가 효과를 확인하지 못했다. Seed별 IoU 차이는 +0.400/+0.102/−0.910pp로 방향도 일치하지 않았다.

Intact−joint-shuffle은 IoU +0.006pp [−0.037, +0.046]pp였다. 첫 seed의 작은 양의 차이가 3개 seed 집계에서 유지되지 않았다. 결과는 scalar parameterization의 pilot으로 보존한다. 원본 Q2C 24개 학습의 수치 재현, 기존 G2 144개 endpoint 재현, 본 14개 비교와 진행 중인 SS matched-seed 학습은 이 차이에 영향받지 않는다.

## 새 검증의 고정 조건과 자원

기존 native condition 64개, predicted O/D9, physical GT 64개를 재사용한다. Backbone은 기존 mixed385 SS4210 frozen base+LoRA이며, train32/cal32, 128 updates, AdamW 1e−4/WD 0.01/clip1, 원본 regional SS64/32 BCE와 correction penalty 0.001, 12 steps/CFG3/rescale1을 유지한다. 데이터 도메인·consumer와 No-query의 물리적 제거가 과거 CAGE 실험과 다르다는 한계도 유지한다.

CPU 검증 뒤 실제 GPU에서 native initial loss/condition 일치, frozen backbone/decoder의 gradient 부재, query 및 constant-gate의 gradient 연결을 검사한다. Gate 모드가 다른 proof를 학습에 사용할 수 없다. 그 뒤 adapter-only와 adapter-query를 seed101/202/303에서 각각 128 update 학습하고 cal32×3 noise, source projection OFF·joint-query shuffle로 평가한다. 새 checkpoint/summary는 gate mode를 명시하고 strict collector도 실제 weight key를 검사한다. Scalar 결과와 새 network 결과를 한 집계로 합치지 않는다.

GPU 0~3만 사용한다. 진행 중인 SS C1/V1 학습은 유지하고, V1의 final 평가가 끝난 GPU 2/3에서 새 검증을 실행한다. GPU 0/1의 후속 B3 학습과 병행한다. 이전 scalar pilot 실측은 No-query 약 30~33분/run, query 약 33~36분/run이었다. 새 6개 학습은 2개 GPU에서 3차례 병렬 실행하므로 약 2시간 규모이며, native GPU proof와 실제 runtime으로 추정을 갱신한다. 신규 원본/80k 다운로드나 전체 데이터 재전처리는 필요 없다.

새 namespace는 `current-geometry-regions-v1/runs-network-gate-v3`, GPU proof는 `regional-gradient-proof-network-gate-v3`, 최종 집계는 `all-six-network-gate-summary-v3.json/.md`다. 기존 `runs-v2`와 원래 checkpoint를 수정하거나 resume하지 않는다. Full/PBR 채택은 원본 gate 학습 결과와 SS matched 비교를 함께 검토한 뒤 결정한다.

## 실제 GPU 검증과 seed101 시작

V1의 final128×3가 끝난 뒤 GPU2에서 `historical-network` proof가 정상 완료됐다. 두 arm 모두 초기 projection과 regional loss가 native와 exact였다. No-query는 raw query0·residual3075 channel, query arm은 raw query9·residual3107 channel이었다. 두 번째 update의 constant-gate MLP gradient L1은 각각 **0.0160192 / 0.0081770**, query encoder의 gradient L1은 **0.00240139**로 연결돼 있었다. Frozen SS/decoder에는 parameter gradient가 없었다. 이 PASS는 원본371-parameter gate의 실제 GPU 경로 검증이며 성능 평가가 아니다.

GPU2/3에서 adapter-only와 adapter-query의 seed101 본 학습이 시작됐다. 두 실제 train log의 첫8 update가 연속·finite이고 sample ID·noise seed·epoch 순서가 정확히 같음을 확인했다. 이후 update10까지 실제 진행했으며 train peak allocated는 약6.730/6.736GiB였다. 같은 sample/noise에서 두 loss가 다른 것은 모듈의 학습 상태 차이이며, calibration 품질의 우열로 해석하지 않는다.

실제 GPU proof와 immutable first8 main-update prefix를 `provenance/current-regional-original-network-GPU-start-v3.json`에 보존했다. `current-regional-original-network-GPU-start-v3-file4.json`은 두 closed artifact의 local/actual-file4 SHA256·크기 검증을 기록한다. 128 updates, checkpoint freeze 후 cal32×3noise와 source-OFF/joint-query-shuffle 평가, seed202/303 및 strict six-run 집계는 계속 남아 있다. Main SS B3는 GPU0/1에서 global96/DDP2/micro48/accumulation1 학습을 병행한다.

## Seed101 두 checkpoint의 실제128-update 고정

두 arm의 학습128 updates가 완료돼 calibration GT를 연 뒤 평가하기 전에 각각 checkpoint를 고정했다. Train loop 시간은 No-query **1,147.87초**, query **1,145.68초**였다. 이 약19분은 후속 cal32×3noise·control 평가를 포함한 run 전체 시간이 아니다.

실제 checkpoint의 optimizer 상태는 No-query12개/query18개가 모두 step128이고 모든 weight·moment tensor가 finite였다. Scalar `gate_logit`은 없고 두 weight 모두 원본 constant-gate MLP의371개 parameter를 포함한다. No-query는 content encoder 자체가 없고 residual LayerNorm 폭3075, query arm은 content encoder와 폭3107을 가졌다. 고정32 train population, 조건/actual-SS-GT-v4 hash도 대조했다. 두 actual train log의128개 전체 sample ID·epoch·noise seed 순서가 정확히 같았다.

고정된 두 checkpoint·optimizer, freeze marker, 닫힌128-update log와 검사 기록 등7개 산출물 **12,998,909 bytes**의 actual-file4 SHA256·크기를 확인했다. `provenance/current-regional-original-network-seed101-train-freeze-v3.json`은 이 학습 고정 검사이며 `calibration_quality_complete=false`, `all_six_runs_complete=false`다. 각 freeze marker의 checkpoint hash도 실제 파일과 일치한다. 저장 검증은 같은 이름의 `*-file4.json` receipt에 있다.

현재 두 process는 fixed checkpoint의 cal32×3noise와 source-OFF/joint-query-shuffle 평가를 진행한다. 성능 차이는 해당 평가가 끝나야 계산하며 seed202/303까지 완료한 뒤 원본 network-gate의 전체 추가 효과를 판정한다. 같은32/3noise라도 primary14의 decoded-GT micro metric과 이 regional actual-SS-input GT/macro metric은 합쳐 집계하지 않는다.
