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

## Seed101 평가 완료: cleanup과 source 보존의 tradeoff

첫 seed의 두 run이 학습·고정 checkpoint·평가를 모두 마쳐 총6개 중2개가 완료됐다. 전체 시간은 adapter-only **1,786.88초(29분47초)**, query **1,996.44초(33분16초)**였다. GPU2/3에서 각seed202가 이어서 시작했다. 아래는 한 training seed의 cal32·3 sampling noise 결과이며 아직 전체 재현 판정이 아니다.

| Method | Macro IoU | Source recall32 ↑ | Stale leak32 ↓ | Regional BCE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Frozen native | 0.445530 | 0.825740 | 0.601524 | 95.616471 |
| Original-network adapter-only | 0.446752 | 0.823651 | 0.584793 | 94.315253 |
| Original-network adapter+query | 0.446010 | 0.789958 | 0.469601 | 88.252432 |

Query−adapter의 macro IoU 차이는 **−0.074pp,95% CI[−1.123,+1.041]pp**로 추가 개선을 확인하지 못했다. 반면 stale leak는 **−11.519pp [−15.051,−8.376]pp**로 줄었고 source recall도 **−3.369pp [−4.632,−2.276]pp** 낮아졌다. Regional BCE는−6.063 [−9.922,−2.584]로 개선됐지만, 이를 source 보존을 포함한 생성 품질 개선으로 읽을 수 없다. 첫 seed에서는 잘못 남은 geometry 제거와 유지해야 할 source 영역의 손실이 함께 나타났다.

Query intact−joint-shuffle의 IoU는 **+0.0453pp [+0.0096,+0.0823]pp**였다. 같은 input 정보를 유지한 공간 정렬 control에서 작은 신호가 있으나, 이것이 query−No-query의 전체 개선을 뜻하지 않는다. Projection-source OFF 대비 ON의 IoU 차이는 두 arm 모두 약+8.5pp였다. 이 OFF는 native source projection도 제거하므로 learned adapter만의 source-use 효과로 해석하지 않는다. Query branch의 원래 camera08 추가 정보와 macro/GT 정의의 한계도 유지한다.

7개 case마다32 source cluster·3noise의96개 기록, 원본128-update pair/noise schedule, 실제 SS encoder-input GT voxel hash, 두 frozen-native prediction hash 일치를 검사했다. GT hash는 actual SS-input label authority에서 독립 재구성해 대조했다. W&B의 두 run 설정·7개 metric set과7개 PNG도 서버에서 내려받아 대조했다. 이 CI는 한 model seed에 조건부이며 반복 비교 보정이나 model-seed 모집단 추정이 아니다. `selection_use=false`다.

근거는 `seed101-network-gate-interim-v3.json`, `wandb-seed101-readback-v3/audit.json`, 상위 `provenance/regional-network-seed101-{complete,remote}-file4.json`이다. Closed 결과/이미지/one-seed 검사와 대기 스케줄 변경18개 산출물765,154bytes, 원격 readback8개 산출물의 actual-file4 SHA256·크기 검증을 완료했다. Seed202/303와 strict six-run 집계 뒤 재현성과 최종 채택을 판단한다. 이 첫 결과로 학습 budget이나 loss를 변경하지 않는다.

## Seed202의128-update 학습 고정 검사

두 번째 seed도 두 arm의128-update 학습을 완료했다. Adapter-only의 train loop는 **1,163.27초**, query는 **1,148.23초**였고 peak allocated는 각각6.730/6.736GiB였다. 전체 calibration·control 평가 시간은 포함하지 않는다. GPU0/1의 B3와 병행하는 GPU2/3의 실제 실행이며, 새 OOM이나 학습 재시작은 없었다.

독립 CPU 검사에서 두 실제 checkpoint의 constant-gate371 parameter, scalar logit 부재, 물리적인 residual 입력 폭3075/3107과 content encoder의 유무를 확인했다. Optimizer12/18개 상태는 모두 step128이었고 weight·moment tensor가 모두 finite였다. 저장된 consumer·condition·actual-SS-input region authority·TRAIN32와128개 actual pair/epoch/noise 순서도 일치했다. 두 arm의 최초 native regional loss도 exact였다. Freeze marker의 calibration GT 미공개 상태와 실제 checkpoint SHA를 대조했다.

근거는 `provenance/network-v3-training-checkpoints-independent-seed202.json`이다. 두 checkpoint·freeze marker·closed training log·검사 JSON 등7개 산출물 **12,999,497 bytes**의 actual-file4 SHA256·크기를 `network-v3-training-checkpoints-independent-seed202-file4.json`에서 확인했다. 검사 source는 `provenance/source-components/regional-checkpoint-audit-v3.py`에 보존했다. 새 검사를 seed101에도 적용해 기존 검사에서 기록한 모든 arm별 field가 동일함을 확인했고, source·검사·Git branch 증거까지9개 파일은 `network-v3-independent-seed101-and-branch-file4.json`에서 별도로 저장 검증했다.

이는 **학습 고정 검사**이며 seed202의 calibration 품질 완료가 아니다. 두 process는 fixed-final cal32×3noise, projected-source-OFF와 query joint-shuffle 평가를 계속한다. Seed303·strict six-run 집계 및 최종 채택 판단은 그대로 남는다.

## 고정 O/D query 입력의 별도 진단

첫 seed에서 source 보존과 stale cleanup의 tradeoff가 관측돼, 실제 controller에 주는 O/D 예측 자체도 확인했다. 이미 고정한 calibration32의 RGB-only query cache를 그대로 읽었고 cache·학습 loss·epoch·threshold를 변경하지 않았다. O는 M0 camera8/9/10/11 네 RGB, D는 M1 camera8 한 RGB로 같은 SS4210 consumer와 같은 pair별 초기 noise에서 생성한 값이다. GT는 여기서만 실제 SS encoder-input의 regional mask를 사용했다.

| 고정 query prior | Source recall32 ↑ | Stale leak32 ↓ |
| --- | ---: | ---: |
| O, M0 RGB4장 | 0.802326 | 0.832130 |
| D, M1 RGB1장 | 0.718446 | 0.405455 |

32 source cluster의 한 fixed query-noise pair에 대한 macro 값이다. D의 stale 점유율은 낮지만 보존할 source-positive 영역의 recall도 O보다8.388pp 낮다. 전체 source-positive 영역 중 O에는 있으나 D에는 없는 cell 비율은 평균19.91%였다. 이는 추가9채널의 입력 예측에도 source 보존 오류가 있음을 보여준다. **Continuous logit/probability/SDF를 쓰는 controller가 binary D support를 그대로 따른다는 증거는 아니며**, 이 진단으로 query 모듈의 성능 하락 원인을 확정하지 않는다. O/D의 view 수가 다르고 query-cache noise와 controller calibration noise도 다르므로 공정한 model-arm 비교로 읽지 않는다.

Torch maxpool과 별개로 NumPy boolean coarsening·coordinate mask로32개 source/stale count 및192개 metric을 재계산했다. Count는 exact이고 최대 metric 차이2.947e−8은 float32 mean의 반올림 범위였다. 원본 query/GT NPZ는 모든 행에서 SHA를 대조했다. W&B의 독립 diagnostic run `ss-current-g2-query-prior-audit-v1`에서24개 macro 수치, 다운로드한32-row table의 ID·192개 값, artifact3개 파일의 bytes를 확인했다. 9개 closed 결과·source·readback 파일181,458bytes의 actual-file4 SHA256·크기도 검증했다.

근거는 `predicted-prior-visibility-audit-v1.json`, `wandb-prior-visibility-readback-v1/{audit,table-readback}.json`, 상위 `provenance/predicted-prior-visibility-{independent-validation-v1,audit-v1-file4}.json`이다. 이는 GT를 모델 입력에 추가하는 작업이 아니며 남은 seed303나 기존 SS 학습의 조건을 바꾸지 않는다. Net 생성 이득과 최종 query 채택은 원래 six-run 결과로 판단한다.

## No-query seed202 평가 완료와 seed303 시작

Adapter-only의 두 번째 seed가128-update 학습과 native/normal/projected-source-OFF의 cal32×3noise를 완료했다. 전체 시간은 **1,802.19초(30분2초)**였다. Normal의 macro IoU는0.446520, source recall32는0.811916, stale leak32는0.539452였다. Frozen-native 세 수치는각0.445530/0.825740/0.601524로 seed101과 동일했다. 이 한 run의 수치로 query 추가 효과나 전체3seed의 개선을 판단하지 않는다.

실제 완료된 job command의 W&B ID를 읽어 remote의 gate mode·arm·seed·128 updates·frozen consumer를 확인했다. 세case의18개 metric과 다운로드한3개 GT/Pred/Diff PNG가 로컬과 일치했다. GPU2에서는 seed303의 새로운 실제 process가 시작됐고, 첫8개 update의 finite loss/nonzero gradient·원본 pair/epoch/noise schedule을 확인했다. Process의 CUDA_VISIBLE_DEVICES도 실제GPU2로 제한돼 있었다. Mutable log를 저장 증거로 삼지 않고 닫힌 first8 prefix를 별도로 고정했다.

No-query seed202의 완료 결과·readback, 검사 source와 seed303 시작 증거 등14개 산출물 **306,538bytes**의 actual-file4 SHA256·크기도 검증했다. 근거는 `wandb-network-readback-v3/adapter-only/seed202/audit.json`, 상위 `provenance/regional-network-NoQuery-seed303-{start,first8-prefix}-v1.json`, `provenance/regional-network-NoQuery-seed202-eval-and-seed303-start-file4.json`이다. Query seed202의 남은 평가, 두 arm의 seed303, strict six-run 집계는 계속 필요하다.

## Seed202 두 arm 평가 완료: query의 seed 간 불안정성

Query seed202도 전체128-update 학습·네calibration case를 완료했다. 전체 시간은 **1,998.26초(33분18초)**였다. 이제6개 중4개 run이 완료됐으며, GPU2/3에서 seed303의 두 실제 process가 모두 학습 중이다.

| Seed202, cal32×3noise | Macro IoU | Source recall32 ↑ | Stale leak32 ↓ | Regional BCE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Frozen native | 0.445530 | 0.825740 | 0.601524 | 95.616471 |
| Adapter-only | 0.446520 | 0.811916 | 0.539452 | 91.989813 |
| Adapter+query | 0.432891 | 0.836455 | 0.652164 | 100.633080 |

Query−adapter의 IoU 점추정치는 **−1.363pp**였다. 첫 seed에서는 query가 stale을 줄이면서 source recall도 낮췄지만, seed202에서는 source recall과 stale leak가 함께 높아졌다. 따라서 첫 seed의 cleanup/보존 tradeoff나 고정 D prior의 source 누락만으로 query 실패를 설명할 수 없다. 이 표의 차이는 단일 model seed의 점추정치이며,3seed 전체의 paired CI와 최종 결론을 대신하지 않는다.

같은32 TRAIN pair와 같은 noise를 사용한 각 pass의 평균 training loss는 아래와 같았다. Epoch 간 noise는 바뀌지만 같은 pass의 두 arm은 정확히 짝지어진다.

| TRAIN pass | Adapter-only | Adapter+query |
| --- | ---: | ---: |
| 1 | 80.443 | 82.436 |
| 2 | 81.637 | 83.669 |
| 3 | 80.167 | 89.861 |
| 4 | 82.464 | 93.673 |

Seed202 query는 calibration뿐 아니라 matched training loss도 불리했다. 이 경우를 validation 과적합만으로 설명할 수 없다. 원본 LR/objective가 현재 consumer에서도 안정적으로 작동한다는 보장은 없지만, 이 관측 하나로 최적화 문제나 구현 오류를 확정하지 않는다. Seed303의 recipe·budget은 변경하지 않는다.

두 arm의7개case·672개 GT/source 기록을 실제 SS-input label에서 독립 재구성한32개 occupancy hash와 대조했다. 두 frozen-native96개 예측도 seed101과 exact였고, 저장된 macro metric을 실제96개 row에서 재계산했다. Query202의24개 W&B metric과4개 다운로드 PNG도 로컬과 일치했다. 두303 process의 실제 GPU2/3 제한·first8 sample/epoch/noise 순서·finite loss/gradient와 최초 native loss의 arm 간 exact를 확인했다.

추가 정밀도 검사에서는 **실제 query202 trained weight와 고정된 실제 calibration 입력1개**를 원본 R4-CONTENT에 매핑했다. CPU BF16 autocast에서 active projection·correction·고정 upstream vector-Jacobian loss와18개 parameter gradient가 모두 bit-exact였다. 이는 앞선6-update CPU optimizer 검사에 추가한 local-module 증거이며, CUDA 전체 rollout/128-update trajectory 동등성을 입증하는 검사는 아니다. 검사 때문에 학습 source나 weight를 바꾸지 않았다.

Query202 완료 결과·remote readback,672-row 검사·training 통계, 두303 시작 prefix, active BF16 검사와 source 등19개 파일 **414,950bytes**의 actual-file4 SHA256·크기를 검증했다. 근거는 `wandb-network-readback-v3/adapter-query/seed202/audit.json`, 상위 `provenance/regional-network-seed202-complete-data-audit-v1.json`, `regional-network-both-seed303-{start,first8-prefix}-v1.json`, `current-regional-trained-BF16-reference-parity-v1.json`, `regional-network-seed202-complete-and-both303-start-file4.json`이다. Seed303·strict six-run과 Full/PBR 판단은 계속 남아 있다.
