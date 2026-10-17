# SS 전체 모듈 비교 복원과 실행 범위

원래 [matching v2 계획](2026-10-17-ss-matching-v2-experiment-plan-ko.md)의 핵심은 Top-1, rejection, matching-aware attention, retargeting adapter, deformation query를 분리하는 것이다. Frozen V1에서 GT feature 주입도 개선되지 않아 필수 B 세 arm이 HOLD됐고, 실행이 P1/P2와 retarget-only query 비교로 이동했다. P1/P2는 유효한 interface 진단이지만 이 다섯 비교를 대체하지 못한다. 사용자가 지적한 범위 축소를 다음과 같이 복원한다.

## 전체 실험과 상태

| 구분 | 실험 | 상태/역할 |
| --- | --- | --- |
| 공통 기준 | Source-only RGB 4장 / edited-RGB 1+3장 zero-shot, V1, 기존 B1/B2/B3/C1 | 공통128 noise0 완료; zero-shot/V1/C1 세 noise 완료; B1/B2/B3 추가 noise 예약; V1 재학습 없음 |
| 원인 진단 | Native/current/GT 주소, semantic 경로 OFF/wrong correspondence, adapter OFF, projection 보간/support, source/target 비중 | 완료 자료 보존; GT 지원 영역과 주소의 confound를 구분 |
| Matching head | 같은 protocol의 Top64 H0 replay, Top256 H1, confidence H2, feature-task H3 | 7 epochs 완료; train-internal calibration으로 H3/T0.1 고정 |
| Interface 진단 | P1: R9 + native bilinear + adapter / P2: H3 + 같은 consumer | 5 epochs·final128×3 완료; P2−P1의 추가 SS 이득 미확인 |
| SS 본 비교 | 아래의 다섯 arm | Top1/Top1-R 480 updates·final128×3 완료; attention/adapter도480 updates 완료·final128×3 평가 중; query 예약 |
| 후속 | 작은 gain의 training-seed 재현, 필요한 동일 sampler 진단, SS 채택 후 PBR 확대 판단 | 최종 생성 결과에 따라 결정 |

P1/P2를 포함하면 새 SS run은 총 일곱 개다. P1/P2는 사용자 중단 당시 같은 run의 저장 상태부터 재개해 완료했고, 나머지 다섯 개는 독립 원본 initialization이다. 이전 SS C2와 PBR B3는 재개하지 않는다. Retarget-only query-study의 미실행 두 학습은 아래 C1/C2로 대체하고 기존 예약·provenance를 보존한다. 별도의 두 query run을 중복 실행하지 않는다.

## 복원한 다섯 arm

| 새 run ID | 원래 계획 대응 | Correspondence value | Confidence/rejection | Adapter | RAW query |
| --- | --- | --- | --- | --- | --- |
| b2-top1 | B2-v2 | Learned Top-1 하나 | OFF | 없음 | 채널 없음 |
| b2-top1-r | B2-v2-R | 같은 Top-1 | 고정 H3 p_use | 없음 | 채널 없음 |
| b3-attention-r | B3-v2 | 같은 후보/score의 weighted attention | 같은 p_use | 없음 | 채널 없음 |
| c1-retarget-r | C1-v2 | 동일 attention | 동일 | 추가 | 채널 없음 |
| c2-query-r | C2-v2 | 동일 attention | 동일 | 동일 | 물리적 RAW2048 추가 |

V1은 native spatial correspondence 기준이다. No-score 비교는 b2-top1-r 대 b3-attention-r이며 uniform 평균을 사용하지 않는다. C1에는 deformation channel 자체가 없고 C2에만 추가한다. Retargeter/query 2×2가 아니라 두 개의 추가 arm이다.

모든 arm은 같은 frozen H3 checkpoint/hash, candidate set, FP32 cached Top1/soft evidence, temperature0.1, native bilinear/support-normalized projection을 쓴다. R arm의 threshold0.276316212는 기존 train calibration에서 잠근 값이다. Validation 결과로 다시 조절하지 않는다. Teacher address/visibility/GT deformation은 inference 입력에 넣지 않는다.

과거 nearest raw-transport gate의 HOLD와 수치를 그대로 남긴다. 이 연구의 config-ready PASS는 실행 준비를 뜻하며 성능 PASS가 아니다. Negative frozen GT probe는 주입 문제의 증거이지만, LoRA를 함께 학습하는 모듈별 비교를 생략하는 근거로 사용하지 않는다. 기존 C1의 공통128 세-noise 이득도 있으므로 adapter가 raw 경로의 약점을 회복하는지 다섯 arm에서 직접 확인한다. 주입 구조가 실패한다는 가설도 최종 결과로 명시할 수 있어야 한다.

## 학습·자원

- DeformNeXt RGB/world-v4 train9149/val508, frozen Omega epoch7, 기존 정합 latent와 cache를 재사용한다. Clean80k를 추가하지 않는다.
- 원본 mixed385 complete base+LoRA checkpoint에서 각 arm을 독립적으로 시작한다. 구조의 순서는 weight continuation을 뜻하지 않으므로 고정한 arm은 병렬 학습할 수 있다.
- 480 updates/약5 epochs, global96, 2-GPU DDP, rank당 micro48, accumulation1. 원래 optimizer/LoRA/conditioner LR/warmup/CFG dropout을 유지한다.
- GPU0·1에는 P2 → b2-top1 → b3-attention-r → c2-query-r, GPU2·3에는 P1 → b2-top1-r → c1-retarget-r를 배치한다. 각 run의 final 세-noise 평가를 마치면 같은 pair의 다음 학습을 실행하는 것이 기본이다. 이21개 final 평가 job은 기본적으로 해당 학습 pair에 배치하고 noise0은 validation rank0의 물리 GPU에 고정한다. 실제 빈 GPU의 backfill은 별도 receipt에 남긴다. P1 noise2가 GPU2, P2 noise0/1이 GPU0/1에서 실행되고 GPU3이 빈 시점에는 P2 noise2 하나를 GPU3으로 옮겨 세 학습 wave가 남은 GPU0/1의 전환을 앞당긴다. 독립 초기화인 b2-top1은 GPU0/1의 두 평가가 끝나면 시작할 수 있고 GPU3의 P2 noise2는 최종 집계의 필수 의존성으로 유지한다. GPU2/3의 다음 학습은 이 backfill의 잔여 시간도 기다릴 수 있음을 명시한다. 생성 평가는 학습 GPU와 겹치지 않는다. 마지막 학습 중 다른 pair는 평가와 진단에 사용한다.
- 실제 native-start/gradient 증명과 2-update per-rank memory probe를 통과한 run만 본 학습으로 진행한다. 메모리·속도·loss·gate/update RMS·metric·이미지를 W&B와 file4에 남긴다.

P1/P2는 사용자 중단 당시 step103/56 checkpoint에서 optimizer/scheduler/sampler까지 복원했다. Native checkpoint에는 rank별 RNG가 없어 bitwise uninterrupted replay라고 주장하지 않는다. 현재 설정의 steady-state는 약17.2초/update, 각 rank peak47.75GiB였다.

### 1~5에폭의 Top1 / confidence 비교

2026-10-18 03:46 KST에 두 arm의480-update 학습과5에폭 검증이 완료됐다. 같은32개·32 source cluster·한 sampling noise이며, 각 에폭의32개 GT voxel hash가 모두 일치했다. 두 arm의 초기 IoU는 모두0.42671546671444294였다. 아래 수치는32개 패널의 경향이며, 완료된final128×3 결과는별도표에기록한다.

| Epoch / step | Arm | Voxel IoU | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 / 96 | b2-top1 | 0.435432 | 0.552400 | 0.672817 | 0.606691 |
| 1 / 96 | b2-top1-r | 0.435453 | 0.552483 | 0.672745 | 0.606712 |
| 2 / 192 | b2-top1 | 0.434025 | 0.553075 | 0.668475 | 0.605324 |
| 2 / 192 | b2-top1-r | 0.432750 | 0.551961 | 0.667076 | 0.604083 |
| 3 / 288 | b2-top1 | 0.435287 | 0.553983 | 0.670139 | 0.606550 |
| 3 / 288 | b2-top1-r | 0.434038 | 0.552076 | 0.669971 | 0.605337 |
| 4 / 384 | b2-top1 | 0.435127 | 0.552991 | 0.671218 | 0.606395 |
| 4 / 384 | b2-top1-r | 0.434820 | 0.552641 | 0.671001 | 0.606096 |
| 5 / 480 | b2-top1 | 0.434014 | 0.551235 | 0.671157 | 0.605314 |
| 5 / 480 | b2-top1-r | 0.433655 | 0.553065 | 0.667613 | 0.604964 |

Confidence/rejection의 차이는1에폭+0.002%p, source-cluster paired95% CI[−0.108,+0.098]%p였다. 2에폭은−0.127%p, CI[−0.293,+0.021]%p로 point estimate는 낮지만0을 포함한다. 현재 추가 이득은 구분되지 않으며 formal equivalence나 최종 무효를 입증한 것은 아니다.

3에폭의 R−Top1은−0.125%p, CI[−0.332,+0.070]%p로2에폭과 비슷한 음의 point estimate이며0을 포함한다. Top1의2→3에폭 변화는+0.126%p, CI[−0.282,+0.533]%p이고 Top1-R은+0.129%p, CI[−0.237,+0.517]%p다. 두 run 모두2에폭보다 소폭 회복했으나 이 변화도 확정되지 않았다. 아직 confidence/rejection의 추가 이득은 확인되지 않았으며, 이 중간 결과로 나머지 arm을 취소하거나 설정을 바꾸지 않는다.

4에폭의 R−Top1은−0.031%p, CI[−0.257,+0.186]%p다. 3에폭보다 점추정치의 차이가 줄었지만 추가 이득은 여전히 확인되지 않는다. Top1의3→4에폭 변화는−0.016%p, CI[−0.423,+0.376]%p이고 Top1-R은+0.078%p, CI[−0.337,+0.456]%p다. 두 변화 모두0을 포함하며 큰 방향 변화의 근거가 없다. 예정한5에폭 final checkpoint와128×3 비교를 유지한다.

5에폭의 R−Top1은−0.036%p, CI[−0.276,+0.192]%p로 이32개 패널에서도 confidence/rejection의 추가 이득은 확인되지 않는다. 4→5에폭 변화는Top1 −0.111%p, CI[−0.459,+0.182]%p, Top1-R −0.116%p, CI[−0.466,+0.165]%p다. 두 run 모두 소폭 하락한 점추정치이나 변화의 CI는0을 포함한다. Final128×3 결과가 나오기 전에 최종 모듈 효과나 학습 연장을 결정하지 않는다.

1→2에폭의 Top1 변화는−0.141%p, CI[−1.232,+0.967]%p이고 Top1-R은−0.270%p, CI[−1.327,+0.819]%p다. 두 point estimate가 소폭 내려갔으나 각 변화의 CI도0을 포함한다. 두 arm 모두 여전히 초기 IoU보다 높지만, 그 증가에는 공통 LoRA 학습 효과도 포함되므로 semantic matching만의 효과로 해석하지 않는다. 중간 하락만으로 구현 오류나 과적합을 확정하지 않고 예정한480 updates와final128×3을 유지한다.

각 스냅샷에서 기록된 모든 update를 검사해 두 rank의 연속 step·finite loss·micro48·accumulation1·global96을 확인했다. 4에폭 스냅샷의 warmup25 이후 median compute는각 rank에서약17.21~17.23초/update이며 peak allocated는각47.02867GiB다. Step96/192/288/384 optimizer의361개 moment state가각각96/192/288/384이었고, 양쪽의denoiser/conditioner/misc 파일 크기도file4와일치했다. Step384의optimizer moment 값도모두finite였다. 중간 checkpoint는크기·CPU optimizer 검증 범위이며전체 SHA256 검증으로표현하지 않는다.

최종 audit는 두 rank의480회 연속 update, step0/96/192/288/384/480 검증, native 초기 출력·gradient proof, optimizer361개 상태의step480·finite moments를 확인했다. 양쪽 모두최종 denoiser/conditioner/misc 세 파일의로컬/file4 SHA256·크기가일치했다. 큐job 기준 preflight·validation·저장을포함한시간은Top1 약2시간29분, Top1-R 약2시간30분이다. 각 rank의median compute는약17.21~17.24초/update, peak allocated는47.02867GiB였다. 이 완료는 두 학습에 대한 것이며, 세-noise 평가와 나머지 세 arm은 아직 남아 있다.

학습된 tanh gate는Top1의96→192→288→384→480에서0.008080→0.017181→0.026108→0.035086→0.043826, Top1-R은0.007325→0.015744→0.024041→0.032035→0.039978로 바뀌었다. 추가 경로의 parameter가0에 고정된 상태는 아니다. R은 별도로 query를 mask/weight하므로 gate 크기만으로 전체 source 기여나 유용성을 비교하지 않는다.

근거는 `module-study-native-v4/interim-{first,second,third,fourth,fifth}-epoch-top1-rejection.json`, `epoch-comparisons/epoch-0{1,2,3,4,5}-b2-top1-r-vs-b2-top1.json`, `provenance/native-ladder-epoch{1,2,3,4}-checkpoints.json`, `provenance/b2-top1{,-r}-final-training-audit.json`, `provenance/b2-top1{,-r}-final-checkpoint-checksum.json`에있다. 다섯 에폭의 paired CI는W&B에도기록됐다. 한 training seed·한 sampling noise·반복 관찰의 exploratory 결과이며14개 구성의최종 표나채택 판단을대체하지 않는다.

### 완료된 여덟 구성의 final128×3

2026-10-18 03:57 KST에Top1 두 arm의세-noise 평가가완료됐다. 아래는같은128개·93 source cluster·샘플별서로다른noise3개, sampler12/CFG3의TP/FP/FN 합산IoU다. Source-only도동일한M1 desired-state GT와비교한다. 한training seed의개발패널결과이며14개구성의전체표는아직미완료다.

| 구성 | Voxel IoU |
| --- | ---: |
| Source-only zero-shot, M0 RGB 4장 | 0.380711 |
| Edited-RGB zero-shot, M1 1장 + M0 3장 | 0.419275 |
| V1 | 0.431926 |
| 기존 C1 | 0.442214 |
| P1 native-grid + adapter | 0.441459 |
| P2 H3 native-grid + adapter, No-R | 0.440920 |
| 새 b2-top1 | 0.433644 |
| 새 b2-top1-r | 0.432806 |

Top1−V1은+0.172%p, source-cluster paired95% CI[+0.005,+0.371]%p다. 이seed에서는작은양의차이지만경계에가까운exploratory 결과이고training-seed 재현은없다. Top1-R−V1은+0.088%p, CI[−0.039,+0.228]%p이며R−Top1은−0.084%p, CI[−0.190,+0.011]%p다. 현재confidence/rejection의추가이득은확인되지않는다.

실제V1 runtime은world1·micro96·accumulation1이고새arm은world2·rank당micro48·accumulation1이다. Global96은같지만rank별RNG·학습trajectory까지같은대조는아니다. 위CI는source cluster의불확실성이며training-seed 변동이나이실행차이를포함하지않는다. 작은Top1 gain을matching만의확정적인인과효과로해석하지않으며필요한matched seed 확인은최종채택단계에남긴다. 주모듈다섯arm끼리는같은2-GPU/micro48 조건이다. 근거는 `provenance/native-top1-v1-execution-nuisance.json`이다.

기존C1은Top1보다+0.857%p, CI[+0.362,+1.331]%p이고P2는+0.728%p, CI[+0.183,+1.268]%p다. 이차이에는attention·adapter 등여러변경과training trajectory 차이가포함되므로adapter 단독효과로해석하지않는다. 새B3/C1의동일조건비교를마쳐분리한다. Head 개선이나Top1의작은gain만으로full을확정하지않는다.

128 패널의공통32개는같은checkpoint의epoch 검증TP/FP/FN·seed·cluster·GT voxel hash를양쪽모두정확히재현했다. 새두arm끼리는384개paired record의GT voxel hash가모두일치했다. 기존V1/C1에는전체voxel hash가없으므로count/identity 검사로구분했다. 기존V1/C1 noise1의sampler metadata 두필드누락은저장된recipe/command/source 근거와별도로명시했고, strict primary 표에는아래의추가재평가를요구한다. 역사파일에필드를채워넣지않았다.

근거는 `module-study-native-v4/interim-top1-final128-three-noise-report.json`, `provenance/native-top1-eight-model-panel-contract.json`, `provenance/b2-top1{,-r}-final128-epoch32-replay.json`이다. 기존B1/B2/B3의추가noise와새attention/adapter/query 결과는진행·예약상태이며이여덟구성표를전체완료로표현하지않는다.

### Attention / adapter 학습 전환

2026-10-18 04:01 KST에GPU0·1의`b3-attention-r`와GPU2·3의`c1-retarget-r`에서본학습update가확인됐다. 두arm 모두native 조건차이0·유효gradient·2-rank2-update memory probe를통과했다. 초기32개IoU는둘다0.42671546671444294이며sample별TP/FP/FN·seed·cluster·GT voxel hash도앞선native 초기값과일치했다. Native 초기성능하락은관찰되지않았다.

시작스냅샷은각12/4 updates였고rank별micro48·accumulation1·global96·finite loss가확인됐다. 관측peak allocated는attention46.84117GiB, adapter47.74934GiB다. C1의adapter gradient가연결돼있으며RAW deformation query는C2에만추가한다. 이는시작검증결과이며학습완료나성능향상을뜻하지않는다. 근거는 `provenance/native-ladder-second-pair-startup.json`이다.

### Attention / adapter 첫 epoch

2026-10-18 04:30 KST에 두 arm의 step96 검증이 완료됐다. 아래는 같은32개·32 source cluster·한 sampling noise의 같은 epoch 비교다. 각 대비의32개 GT voxel hash도 모두 일치했다.

| Arm | Voxel IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| b2-top1 | 0.435432 | 0.552400 | 0.672817 | 0.606691 |
| b2-top1-r | 0.435453 | 0.552483 | 0.672745 | 0.606712 |
| b3-attention-r | 0.435604 | 0.552723 | 0.672749 | 0.606858 |
| c1-retarget-r | 0.431068 | 0.544840 | 0.673663 | 0.602442 |

Attention−Top1-R은 +0.015%p, source-cluster paired95% CI[−0.068,+0.105]%p로 첫 epoch의 추가 이득은 구분되지 않는다. Adapter−attention은 −0.454%p, CI[−0.814,−0.150]%p다. 이 패널·seed에서는 adapter 추가 후 첫 epoch 성능이 낮다. Precision은 낮아지고 recall은 소폭 높아졌으며, 이를 초기 native 성능 붕괴와 구분한다. 두 arm의 학습 전 출력은 정확히 같았고 adapter도 초기 IoU보다 높다. 이 결과로 학습을 중단하거나 query arm을 취소하지 않고 480 updates와 final128×3을 유지한다.

P2-NoR의 같은 epoch IoU는0.437257이다. C1-R−P2-NoR의 보조 비교는 −0.619%p, CI[−1.362,+0.174]%p이며0을 포함한다. P2의 중단/재개 RNG 차이가 있어 rejection만의 인과 효과로 확정하지 않는다.

CPU checkpoint 검사에서 B3의 optimizer361개, C1의366개 상태가 모두step96이고 moment 값도finite였다. C1의 실제 `retargeter.net.0.weight`는256×2048이며 RAW query 추가 채널이 없다. Zero-init·gradient·rejection 시 추가 경로 전체 차단은 검토한 구현에서 계획과 일치했다. 각 denoiser/conditioner/misc 파일 크기도file4와 일치했다. 중간 checkpoint의 전체SHA256 검증이나 source feature의 유용성을 입증한 것은 아니다.

별도CPU constructor probe에서는 adapter 추가가 난수 소비를 바꾸는 것도 확인했다. `sample_t`는CPU 난수 생성기를 사용하고 validation은RNG를 저장·복원하므로, 같은seed·world·microbatch만으로 서로 다른 구조의 학습 난수 경로까지 일치한다고 볼 수 없다. 이는 격리된 초기화 검사이며 실제 rank별 timestep을 캡처한 결과는 아니다. 한seed의 source-cluster CI가 training stochastic variation을 포함하지 않는다는 해석 한계로 기록하고, 기존R9-compatible 초기화와 실행 설정은 유지한다.

기록된 모든 update에서 micro48·accumulation1·global96·finite loss를 확인했다. Warmup25 이후 median compute는attention 약17.26초/update, adapter 약17.22초/update였고 rank별peak는46.84117/47.74934GiB였다. GT/Pred/Diff 투영 이미지도 확인했으며 두 모델 모두 큰 형상은 유지되고 경계·가는 구조 등에 차이가 남았다. 한 방향 투영으로3D 정확도를 대체하지 않는다.

근거는 `module-study-native-v4/interim-first-epoch-attention-adapter.json`, `epoch-comparisons/epoch-01-{b3-attention-r-vs-b2-top1-r,c1-retarget-r-vs-b3-attention-r}.json`, `provenance/native-second-pair-epoch1-checkpoints.json`, `provenance/native-second-pair-constructor-rng-probe.json`이다. 두 주 대비의CI는W&B에도 기록됐다. 이 첫 epoch 결과는 최종 모듈 채택 판단을 대체하지 않는다.

### Attention / adapter 두 번째 epoch

2026-10-18 04:59 KST에 두 arm의 step192 검증이 완료됐다. 같은32개·한 sampling noise에서 얻은 결과이며, 대비별32개 GT voxel hash도 일치했다.

| Arm | Voxel IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| b2-top1 | 0.434025 | 0.553075 | 0.668475 | 0.605324 |
| b2-top1-r | 0.432750 | 0.551961 | 0.667076 | 0.604083 |
| b3-attention-r | 0.432622 | 0.552025 | 0.666679 | 0.603959 |
| c1-retarget-r | 0.435896 | 0.554282 | 0.671145 | 0.607142 |

Attention−Top1-R은 −0.013%p, paired95% CI[−0.060,+0.021]%p로 추가 이득이 구분되지 않는다. Adapter−attention은 +0.327%p, CI[−0.006,+0.787]%p다. 첫 epoch의 −0.454%p에서 양의 점추정치로 바뀌었으나 CI는 아직0을 포함한다. 초기 하락이 계속된다고 볼 수 없으며, 첫 epoch만으로 adapter 실패를 결정하지 않은 이유를 보여준다. 최종 이득의 확정이나 채택은 아니다.

같은 arm의1→2에폭 변화는attention −0.298%p, CI[−1.357,+0.813]%p, adapter +0.483%p, CI[−0.624,+1.592]%p다. 두 변화 모두0을 포함한다. Adapter는precision이0.544840→0.554282로 높아졌고recall은0.673663→0.671145로 소폭 낮아졌다. 같은 epoch의attention보다precision·recall 점추정치가 모두 높지만, 단일seed와작은패널의 불확실성을 유지해 해석한다.

P2-NoR의 같은 epoch IoU는0.441612다. C1-R−P2-NoR의 보조 비교는 −0.572%p, CI[−1.469,+0.403]%p이며0을 포함한다. P2의 중단/재개 및 학습 난수 경로 한계는 그대로 적용한다.

B3/C1 optimizer의361/366개 상태가 모두step192이고 moment가finite임을 확인했다. Gate tanh는step96→192에서B3 0.007442→0.015897, C1 0.007071→0.015212로 변했다. C1의입력weight는여전히256×2048이다. 각checkpoint 세 파일의file4 크기가일치했고, 기록된양rank의연속update·micro48·accumulation1·global96·finite loss도확인했다. 이검사는실행정상성을뒷받침하며모듈효과나source 활용을입증하지않는다.

근거는 `module-study-native-v4/interim-second-epoch-attention-adapter.json`, `epoch-comparisons/epoch-02-{b3-attention-r-vs-b2-top1-r,c1-retarget-r-vs-b3-attention-r}.json`, `provenance/native-second-pair-epoch2-checkpoints.json`이다. 두주대비의CI는W&B에기록됐다. 두run의480 updates·final128×3, C2 및나머지baseline 평가를계속유지한다.

### Attention / adapter 세 번째 epoch

2026-10-18 05:28 KST에두run의step288 검증이완료됐다. 같은32개·동일seed·cluster·GT voxel geometry를유지한결과다.

| Arm | IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| b2-top1 | 0.435287 | 0.553983 | 0.670139 | 0.606550 |
| b2-top1-r | 0.434038 | 0.552076 | 0.669971 | 0.605337 |
| b3-attention-r | 0.434564 | 0.552481 | 0.670628 | 0.605849 |
| c1-retarget-r | 0.433627 | 0.553038 | 0.667585 | 0.604937 |

Attention−Top1-R은+0.053%p, source-cluster paired95% CI[−0.001,+0.113]%p다. Adapter−attention은−0.094%p, CI[−0.335,+0.159]%p다. 두효과모두0을포함한다. Adapter의epoch별부호가음/양/음으로바뀌므로초기하락의지속이나확정된추가이득으로해석하지않는다. 2→3에폭변화는B3 +0.194%p, CI[−0.173,+0.592]%p, C1 −0.227%p, CI[−0.719,+0.193]%p다. 같은시점P2-NoR IoU는0.441529이며C1-R−P2-NoR는−0.790%p, CI[−2.303,+0.608]%p다. 기존중단/재개·구조별난수경로한계가적용된다.

Optimizer의361/366개상태가모두288이고moment 값은finite였다. C1 adapter 입력은256×2048로RAW query 채널이없다. Tanh fusion gate는B3 0.024315, C1 0.022862다. 세종류중간checkpoint 크기는file4와일치했다. 전체SHA나source 유용성증명은아니다. 실제epoch3 GT/Pred/Diff 이미지에서도전체silhouette는유지되지만경계·얇은부분오차가있고, 이투영만으로3D 정합전체를증명하지않는다. 두주대비CI는W&B logging receipt에서도확인했다.

별도CPU 조사에서고정threshold0.276316212를변경하지않고validation508 전체의실제cache와edited-RGB alpha를검사했다. 유효foreground query38,366개중4,348개(11.33%)가rejection을통과했고, `p_use * accepted`의평균은유효foreground 기준0.04277이었다. Common32/128의통과율은각10.59%/9.44%다. 이는추가경로가제한될수있다는정량근거이며, confidence의정확도·SS 성능하락의인과효과·projected voxel 기여를증명하지않는다. 현재R arm의threshold를사후조절하지않고P2-NoR 보조대조와고정checkpoint source/adapter 진단으로확인한다.

근거는`module-study-native-v4/interim-third-epoch-attention-adapter.json`, `epoch-comparisons/epoch-03-{b3-attention-r-vs-b2-top1-r,c1-retarget-r-vs-b3-attention-r}.json`, `provenance/native-second-pair-epoch3-checkpoints.json`, `provenance/rejection-validation-coverage.json`이다. 두run은480 updates와final128×3을계속진행하며C2와나머지평가·후속검증도유지한다.

3에폭의 steps193~288, 양 rank의 192개 pack을 비교해 실제 추가 경로의 크기도 집계했다. 모든 대응 rank/step에서 native condition RMS와 유효 query 비율이 정확히 같았다. 아래 값은 로그에 기록된 strided RMS 비율의 pack 중앙값이다.

| Arm | 추가 update RMS / native RMS |
| --- | ---: |
| Top1, No-R | 0.735% |
| Top1-R | 0.090% |
| Attention-R | 0.083% |
| Adapter-R | 0.928% |
| P2 adapter, No-R | 9.875% |

Gate의 중앙값은 대략0.019~0.022로 비슷하지만 R arm의 유효 query당 평균 usage weight 중앙값은0.04722이고 No-R은1이다. R을 사용한 추가 경로가 작다는 실제 로그 근거다. P2/새 C1의 학습 이력과 학습된 adapter도 다르므로 위 크기 차이를 rejection만의 인과 효과나 생성 품질 개선량으로 해석하지 않는다. Confidence는 선택한 Top1의 정확성을 감독받지만 attention consumer는 soft evidence를 사용하므로, 정확한 patch 판정과 유용한 feature 전달 사이의 차이도 후속 진단에서 확인한다. Threshold를 비교 도중 바꾸지는 않는다. 근거는`provenance/native-epoch3-path-strength.json`과`module-study-native-v4/figures/interim-epoch3-transport-strength.{png,svg}`다. Boxplot은 pack 분포이며 신뢰구간이 아니다.

### Attention / adapter 네 번째 epoch

2026-10-18 05:57 KST에 step384 검증이 모두 완료됐다. 같은32개·동일 순서·seed·cluster·GT voxel geometry, sampler12/CFG3/batch8을 유지했다.

| Arm | IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| b2-top1 | 0.435127 | 0.552991 | 0.671218 | 0.606395 |
| b2-top1-r | 0.434820 | 0.552641 | 0.671001 | 0.606096 |
| b3-attention-r | 0.434544 | 0.552388 | 0.670716 | 0.605828 |
| c1-retarget-r | 0.433987 | 0.551289 | 0.671013 | 0.605287 |

Attention−Top1-R은−0.028%p, source-cluster paired95% CI[−0.127,+0.054]%p다. Adapter−attention은−0.056%p, CI[−0.474,+0.386]%p다. 두 모듈 모두 추가 이득이 확인되지 않았다. 3→4에폭 변화도 B3 −0.002%p, CI[−0.423,+0.390]%p, C1 +0.036%p, CI[−0.218,+0.313]%p로0을 포함한다. 초기 성능 붕괴나 최종 모듈 무효의 확정으로 해석하지 않고, 예정한480 updates와final128×3을 유지한다.

같은 epoch의 P2-NoR IoU는0.445345이며 C1-R−P2-NoR은−1.136%p, CI[−2.331,+0.072]%p다. 앞서 관찰한 작은 R usage와 함께 확인할 보조 비교지만, P2 중단/재개 RNG와 구조별 초기화 난수 경로 한계가 있어 rejection만의 인과 효과로 확정하지 않는다. Threshold·LR·학습 budget을 중간에 변경하지 않았다.

CPU 검사에서 optimizer361/366개 상태가 전부384이고 moment 값도finite였다. C1 adapter 입력은256×2048이며 RAW query 채널이 없다. Tanh fusion gate는B3 0.032525, C1 0.030677이다. 모든 중간 checkpoint 파일 크기는file4와 일치했다. 두 rank의 모든 연속 update에 global96·micro48·accumulation1·finite loss가 유지됐다. Steady median은B3 약17.24초, C1 약17.22초/update이며 peak allocated는46.841/47.749GiB다. 중간 PT 전체SHA 검증으로 표현하지 않는다.

W&B는 client receipt뿐 아니라 원격API에서도 검증했다. 두 training run의4에폭 IoU와 paired run의 gain/CI가 실제 기록과 일치했고, 서버의 GT/Pred/Diff 두 이미지를 다운로드해 로컬 이미지와SHA256이 같은지 확인했다. 두 run의 서버 상태는running이었다. 근거는`module-study-native-v4/interim-fourth-epoch-attention-adapter.json`, `epoch-comparisons/epoch-04-{b3-attention-r-vs-b2-top1-r,c1-retarget-r-vs-b3-attention-r}.json`, `provenance/native-second-pair-epoch4-checkpoints.json`, `provenance/native-second-pair-epoch4-wandb-remote.json`이다. C2·나머지baseline 평가·경로 진단·필요한 seed/채택 후속은 아직 남아 있다.

### Attention / adapter 다섯 번째 epoch와 학습 완료

2026-10-18 06:26 KST에 두 run의480 updates·5에폭 검증이 모두 완료됐다. 아래는 같은32개·32 source cluster·한 sampling noise의 결과이며, final128×3 표와 구분한다. 대비별32개 GT voxel hash도 일치했다.

| Arm | IoU | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| b2-top1 | 0.434014 | 0.551235 | 0.671157 | 0.605314 |
| b2-top1-r | 0.433655 | 0.553065 | 0.667613 | 0.604964 |
| b3-attention-r | 0.433638 | 0.553084 | 0.667545 | 0.604948 |
| c1-retarget-r | 0.436316 | 0.553788 | 0.672870 | 0.607549 |

Attention−Top1-R은−0.002%p, source-cluster paired95% CI[−0.058,+0.062]%p다. Adapter−attention은+0.268%p, CI[−0.090,+0.680]%p다. Adapter의점추정치는높지만추가이득은아직확정되지않았다. 4→5에폭변화도B3 −0.091%p, CI[−0.436,+0.183]%p, C1 +0.233%p, CI[−0.153,+0.603]%p로0을포함한다. 이32개중간패널로고정한실험을연장하거나취소하지않는다.

같은32개에서P2-NoR의IoU는0.453171이다. C1-R−P2-NoR은−1.686%p, CI[−3.208,−0.383]%p다. 이seed·패널에서는P2가높지만, P2의중단/재개RNG·구조초기화의난수소비차이가있으므로rejection만의인과효과로확정하지않는다. CI는training-seed변동을포함하지않는다. Confidence통과율·경로RMS와함께최종평가·고정checkpoint진단·필요한matched follow-up으로검토한다.

두rank의1~480연속update·global96·micro48·accumulation1·finite loss, step0/96/192/288/384/480검증, native 초기출력·gradient proof가모두PASS다. Optimizer361/366개상태는전부step480이고moment도finite다. C1첫weight는256×2048로RAW query채널이없다. 최종tanh fusion gate는B3 0.040660, C1 0.037955다. 각run의denoiser/conditioner/misc 세파일은로컬과file4에서전체SHA256·크기가일치했다.

큐wrapper기준preflight·validation·저장을포함한시간은각약2시간29분27초다. Rank별median compute는B3 약17.24초, C1 약17.22초/update이며peak allocated는각46.841/47.749GiB다. W&B원격API의최종IoU·paired gain/CI가로컬과일치하고두training run상태도finished다. 서버에서실제다운로드한최종GT/Pred/Diff 이미지두개도로컬PNG와SHA256이일치했다.

B3의final128 noise0은같은checkpoint의epoch32개를seed·cluster·TP/FP/FN·GT voxel hash까지정확히재현했다. 이검증은noise0의재현성범위이며세-noise평가나C2완료를뜻하지않는다. C2·나머지baseline·actual path controls·필요한seed/최종채택후속은계속진행한다.

근거는`module-study-native-v4/interim-fifth-epoch-attention-adapter.json`, `epoch-comparisons/epoch-05-{b3-attention-r-vs-b2-top1-r,c1-retarget-r-vs-b3-attention-r}.json`, `provenance/{b3-attention-r,c1-retarget-r}-final-training-audit.json`, `provenance/{b3-attention-r,c1-retarget-r}-final-checkpoint-checksum.json`, `provenance/native-second-pair-epoch5-wandb-remote.json`, `provenance/b3-attention-r-final128-epoch32-replay.json`이다. 전체14개구성의final128×3와모듈채택은아직미완료다.

### 저장된 checkpoint의 추가 경로 진단 준비

기존 진단의 일부는 legacy candidate cache를 전제로 하므로 새 matching-v2 cache에 그대로 적용하지 않는다. `evaluate_native_checkpoint --transport-control`에 기존 inference control 경로를 연결했다. 기본값은`normal`이고 정상 평가의 tensor 흐름은 유지한다. 모델 forward·학습 설정·진행 중 프로세스는 변경하지 않았다.

`source-off`는 추가 transport/adapter 경로만 끄고 native multi-view와M1 RGB1장+M0 RGB3장을 유지한다. 별도로 학습한V1이나target-only arm과 같지 않다. `wrong-corr`는 같은 object 안에서 correspondence value를한query slot 이동하는 약한 국소 민감도 진단이다. 변화가 없더라도source/주소를 사용하지 않는다는 증거로 단정하지 않는다. `shuffled-deform`은C2의물리적RAW context를batch 내asset 사이에서순환시키며batch당최소2개를요구한다. C1의채널제거 ablation을0벡터로대체하는기능이아니다.

`adapter-off`는동일checkpoint의retargeter 출력만scoped hook으로제거한다. Semantic evidence·학습된fusion gate·native RGB조건·입력채널은유지하며, 예외발생시에도hook을제거한다. 원래D-C의adapter 단독진단에해당하고학습arm의채널제거비교를대체하지않는다. 모델forward와weight는변경하지않는다.

비정상control 결과에는`diagnostic_only`와control 정의를명시하며, primary collector는이를정상14개구성의평가로받지않는다. 같은voxel count로위장된diagnostic도거부하는경우를포함해관련CPU30개테스트가통과했다. Wrapper에서실제`get_inference_cond`·transport 추출·conditioner 호출까지control이전달되고sampler에는남지않는dispatch도검증했다. 실제C1/C2 validation cache에서각2개batch를읽어RGB4장·camera·GT·evidence tensor가정확히유지되고C2 RAW context가2×1024×2048임도확인했다. CPU 렌더러 import에는프로젝트테스트의초기화방식만사용했고GPU 연산은하지않았다.

현재는실행경로와입력전달검증완료이며counterfactual 생성결과나source 유용성입증은아직없다. 후속진단은동일checkpoint·panel·seed의별도output으로허용된GPU0~3의빈시간에진행한다. 정상학습arm 수나필수14개구성은늘리거나줄이지않는다. 근거는 `provenance/transport-control-cpu-smoke.json`이다.

2026-10-18 05:25 KST에`queue_transport_diagnostics`로11개common32/noise0 생성과8개CPU paired 비교를실제큐에추가했다. 기존142개job의내용·의존성은그대로이며총161개다. P2-NoR와새C1의normal/source-off/adapter-off/wrong-corr, 새B3의normal/source-off/wrong-corr를각각동일물리GPU에서비교한다. 모든생성은C1 학습종료를기다린뒤비는GPU2·3에서실행하고, GPU0·1의C2 학습경로를유지한다. 기존baseline·final 평가는더높은우선순위를유지한다. 정상14개구성·새SS학습7개는변경하지않았다. Hook의실제residual 제거·weight/semantic gate 보존·예외후복원과queue의동일device/checkpoint/panel/seed 조건을포함한관련CPU32개테스트가통과했다. 생성결과는아직없으며예약완료와실행완료를구분한다. 근거는`provenance/transport-control-queue.json`이다.

2026-10-18 05:32 KST에는 C2 학습 종료 뒤의 normal/shuffled-RAW-context 생성 두 개와 paired 비교 한 개도 예약했다. 기존 161개 작업은 유지되며 총 164개다. 두 생성은 비는 GPU2에서 동일 checkpoint·common32·batch8·noise로 실행한다. 실제 panel의 32개 source cluster가 서로 다르고, 각 batch8 안에서도 전부 다른 source임을 확인했다. RGB 네 장·카메라·GT·correspondence evidence·학습 weight는 유지하고 RAW context만 다른 asset과 순환시킨다. CPU 테스트에서는 C2의 물리적 RAW 입력에 연결된 출력이 context를 따라 바뀌고 semantic evidence·usage는 보존됨을 확인했다. 관련 테스트 34개가 통과했으며 GPU 생성 결과는 아직 없다. 정상 구성 14개와 새 SS 학습 7개는 그대로다. 근거는 `provenance/transport-query-control-queue.json`, `provenance/transport-query-control-panel-contract.json`이다.

2026-10-18 05:39 KST에는 P2/B3/C1의`wrong-object` 생성 세 개와 paired 비교 세 개를 추가했다. 기존164개 작업은 유지되며 총170개다. 이미 예약한 동일 device의 normal32를 재사용하고 각 모델의 correspondence evidence만 다른 source cluster의 asset과 순환시킨다. 기존 legacy 진단에서 사용한`source_values`는 새 cache의 실제 전달값이 아니므로, matching-v2가 소비하는`evidence_top1/evidence_soft`에 scoped pre-hook을 적용한다. Recipient target query·confidence·validity·RGB·camera·weight는 유지한다. CPU 테스트는 target-query-only residual이 유지되는 경우에도 source evidence만 바뀌고, 실패 후 hook과 caller tensor가 원래대로 남는지 확인했다. 관련 테스트36개가 통과했다. Donor feature의 유효 영역/zero support도 달라질 수 있어 결과만으로 주소 정확성의 효과를 고립시켰다고 주장하지 않는다. 약한 within-object 이동과 source-off·adapter-off를 함께 해석한다. 새 학습은 없으며 GPU2·3의 C1 학습 뒤 빈 시간에 실행한다. 근거는`provenance/transport-wrong-object-control-queue.json`이다.

05:40 KST의 실제 C1 cache/step288 conditioner 검사에서는 위 단순 roll이 유효 query150개 중136개, R 통과 query25개 중22개에 donor background zero를 전달했다. GPU 진단6개가 전부 pending임을 확인하고, 실행 전에 donor foreground의 token 순서를 recipient의 유효 query 개수에 맞춰 결정적으로 resample하는 방식으로 수정했다. Recipient support·confidence·native target query·RGB/camera는 유지하며 donor의 실제 feature만 선택한다. Teacher·새 RNG·feature 합성은 사용하지 않고 빈 donor foreground는 오류로 중단한다. 원래 roll 구현과 CPU 관찰은`campaign-source-5b1a96f.tar.gz`, `transport-wrong-object-real-cache-smoke.json`에 보존했다.

수정 후 같은 실제 두 asset/학습된 conditioner에서 zero donor feature는0/150이고, 정상 출력 복원·parameter 보존·recipient usage 일치를 확인했다. 실제 common32의 batch8 네 개도 전부 검사해2,511개의 유효 query와266개의 R 통과 query를 유지하고 donor zero 주입이0개임을 확인했다. 관련 CPU37개 테스트가 통과했다. 작업 수170개·명령·checkpoint·noise·정상 학습은 그대로이며 GPU 생성 결과는 아직 없다. Cross-object appearance도 바뀌므로 주소 정확성만의 단독 인과 비교는 아니다. 근거는`provenance/transport-wrong-object-support-preserved-smoke.json`, `provenance/transport-wrong-object-common32-support-audit.json`이다.

## 비교·완료 조건

매 epoch 공통32의 voxel IoU/F1/precision/recall과 GT/Pred/Diff를 확인한다. 최종은 같은128개, sampling12/CFG3, sample별 고정 noise 세 개다. 다섯 인접 대비와 각 arm의 V1/기존 C1 대비 차이를 source cluster bootstrap으로 계산한다. 같은 source의 variant/noise를 독립 표본으로 세지 않는다.

Zero-shot 두 조건과 기존 B1/B2/B3도 추가 noise 두 개를 생성해 최종 세-noise 표에 함께 넣는다. V1/기존 C1은 이미 세 noise가 있다. 이 보충은 저장된 checkpoint의 평가 열 개이며 재학습이 아니다. P1이 끝난 GPU2·3에서 zero-shot 반복 평가를 먼저 실행한다. 기존 B1/B2/B3의 반복 여섯 개는 C1 학습이 끝난 뒤 비는 GPU2·3에 배치해, P1/P2 final 평가와 새 SS 학습의 시작을 지연시키지 않는다. 해당 여섯 job은 `train-native-module-c1-retarget-r`를 기다리며 GPU2/3에만 배정한다. 명령·checkpoint·sample·noise·완료 조건은 바꾸지 않는다. 최종 집계는 baseline 일곱 개와 새 run 일곱 개, 총14개 구성을 같은128×3 조건으로 다룬다. Source-only의 네 M0 RGB도 동일 M1 target과 비교하는 입력 효과 진단임을 명시한다.

최종 집계 사전 점검에서 기존 V1/C1 noise1 두 JSON에 `sampling_steps`와 `guidance`가 없음을 확인했다. 저장된 recipe·command·source의 별도 근거는 유지하되, strict primary collector에는 필드가 명시된 새 평가를 사용한다. 같은 checkpoint·128 panel·noise1 seed map으로 두 평가만 `baseline-noise-confirmation/protocol-replay/noise-1/{v1,c1}`에 추가하며 원래 파일은 보존한다. 새 학습 arm은 늘리지 않는다. 두 평가도 C1 학습 뒤 비는 GPU2·3에서 실행한다. 최종 집계는 새 결과의 sampler12/CFG3·optimizer update0과 기존 결과의 sample별 seed·cluster·TP/FP/FN 일치까지 검사한다. 불일치하면 집계를 통과시키지 않고 원인을 확인한다. 새 평가가 실행되기 전에는 replay 검증 완료로 표시하지 않는다.

핵심 대비는 `b2-top1−V1`, `b2-top1-r−b2-top1`, `b3-attention-r−b2-top1-r`, `c1-retarget-r−b3-attention-r`, `c2-query-r−c1-retarget-r`다. P1/P2와 기존 C1도 최종 비교에 포함한다. Head Top1 향상만으로 SS module 효과를 입증하지 않는다. H2 confidence loss는 ranking branch에 detach되어 있어 H2−H1 순위 향상을 inference rejection의 인과 효과로 설명하지 않으며, 실제 SS R 대조로 판단한다.

추가 설정 대조에서P2와새C1은rejection·threshold를제외한모델·dataset/cache·학습설정이일치했다. P2를adapter가있는No-R 대조로재사용해C1-R과의보조비교도확인한다. 새학습을추가하는뜻은아니다. P2의중단/재개시rank별RNG가복원되지않았으므로엄밀히같은training trajectory라고주장하지않으며, 작은차이로채택을결정하려면seed 재현이필요하다. 현재는설정일치만검증했고C1의성능비교는미완료다. 근거는 `provenance/p2-c1-rejection-config-control.json`이다.

Full은 C2로 미리 확정하지 않는다. 다섯 대비를 모두 보고 best final 구성과 확인되지 않은 효과를 기록한다. 근소한 차이는 training-seed 재현을 추가한다. 개발용128을 선택에도 사용한 결과는 unseen test 성능으로 부르지 않는다. 과거 voxel P1/G2의 cleanup 개선도 source utilization 성공과 구분한다. 기존 보고서는 두 실험 모두 DEV projected-source 효용을 입증하지 못했다고 명시했다.

추가 seed는 fresh run의 JSON에 `training_seed`를 지정한다. 기본0은 기존 `rank` seed 정책과 정확히 같으며, 값101이면 각 rank의 model initialization·diffusion noise/t·CFG RNG가101+rank가 된다. Dataset sampler 순서는 그대로 고정한다. JSON이 CLI보다 우선하므로 저장된 JSON의0을 CLI만으로 덮어썼다고 가정하지 않는다. 같은 run의 resume에서 seed를 변경할 수 없으며, seed 재현 run은 별도 output/optimizer/W&B ID로 시작해야 한다. 이 옵션 준비는 현재 주 비교의 seed나 학습 설정 변경을 뜻하지 않는다.

완료 시 원래 계획 항목별 결과/미확인 가설, 같은-panel final 표, 학습 경향, GT/Pred 이미지, head/cache/checkpoint provenance, 실제 시간·메모리 및 PBR 확대 근거를 한 리포트에 정리한다. 실행 중이거나 config만 준비된 상태를 완료로 표시하지 않는다.

`summarize_ss_module_study`를 마지막 CPU job으로 연결한다. 완료 marker뿐 아니라 두 rank의1~480 연속 update, 여섯 epoch 검증, 초기 generation parity, native/physical-query gradient 증명과 final checkpoint를 확인해야 primary report를 만든다. 이 report의 `primary_matrix_complete`는 최초 비교 행렬의 완료이며 추가 seed/최종 채택/PBR 또는 전체 goal 완료를 뜻하지 않는다. 자동 산출물은 file4와 W&B artifact로 보존하고 최종 채택과 필요한 후속 실험은 실제 결과를 검토해 결정한다.
