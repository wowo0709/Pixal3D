# 기존 SS 향상 경로의 실행 조건을 맞춘 추가 seed 확인

본14개 비교와원본Q2C/G2 재현뒤의확인실험이다. 이전에관측한성능향상이새seed와같은실행조건에서도유지되는지확인한다. 기존모든arm을반복하거나현재regional 두arm·3seed를대체하지않는다.

## 추가 확인이 필요한 근거

기존fixed-final128×3에서C1은V1 대비IoU +1.029pp,source-cluster95% CI[+0.586,+1.498]pp였다. 같은결과로추가계산한기존R9의인접비교는다음과같다.

| 비교 | IoU차,pp [95% CI] | 당시실행 |
| --- | --- | --- |
| B2 learned Top1−B1 raw Top1 | +0.129 [−0.012,+0.271] | 양쪽world1/micro96/accumulation1 |
| B3 score attention−B2 Top1 | +0.015 [−0.013,+0.046] | 양쪽world1/micro96/accumulation1 |
| C1 retargeter−B3 attention | +0.686 [+0.252,+1.130] | C1:world1/micro48/accumulation2,B3:world1/micro96/accumulation1 |

Retargeter 비교가가장큰양의추가차이지만실행조건도달랐다. Global96은같아도난수소비·microbatch별연산·학습trajectory의차이가남는다. Attention/learned ranking의작은점추정치도확정된효과로취급하지않는다. 추가seed에서vanilla 대조군을함께학습해전체C1 이득과B3→C1의module 이득을둘다확인한다.

실제common32의target-alpha/source-valid query2,511개를같은R9 cache로검사했다. Temperature0.005에서Top1 probability 평균0.9336,중앙값0.9992이며80.4%의query가0.9를넘었다. Soft attention과Top1의DINO feature 상대L2 차이는평균1.48%였다. 입력자체의차이가작다는관측은attention 추가효과가작은결과와양립한다. 이를확정적인원인이나성능동등성의증거로취급하지않고재현설정의temperature를그대로유지한다. GT/topology teacher를읽지않은inference-only 분석이다.

근거는`ss-seed-confirmation-v1/historical-cumulative-comparisons-v1.json`의세paired 대비·다섯run480-update execution hash와`provenance/original-R9-attention-versus-Top1-audit.json`이다.

## 고정한 세 run

| Arm | Frozen matching | Adapter | Query | 학습 |
| --- | --- | --- | --- | --- |
| V1 | native spatial | 없음 | 없음 | 원래LoRA만 |
| B3 | 원래R9 D1,Top64,T0.005,No-R | 없음 | 없음 | 원래LoRA+fusion gate |
| C1 | B3와동일 | 원래R9 retargeter | 물리적query channel 없음 | 원래LoRA+fusion gate+adapter |

실제완료된run의saved config를명시적으로읽고단계별source config SHA를보존한다. B3/C1의head/operator/temperature/cache는동일하다. 원래mixed385 SS complete base+LoRA step4210에서각run을fresh로시작한다. 원래DeformNeXt train9149/val508,world-v4 latent/RGB/descriptor/cache를재사용한다. Clean80k 추가학습이나data 재구축은없다.

세run 모두training seed101,world2,rank당micro48,accumulation1,global96,480updates/약5epochs로맞춘다. 원래optimizer·LoRA rank/LR·conditioner LR1e-4·warmup·CFG dropout·sampler를유지한다. Data/model/optimizer/validation 설정은transport module에필요한차이를제외하고구조적으로동일해야한다. 저장된pilot config의96-step 제한은480으로고정하고pilot 자동연장·과거resume 필드를제거한다.

매run의실제native condition 일치·LoRA backward를먼저검사한다. B3/C1은추가gate/adapter gradient도검사한다. 두rank의실제2-update memory probe를통과한후학습한다. Micro48을실행할수없으면다른arm과조건을다르게자동축소하지않고해당확인실험을중단해원인을검토한다. 초기common32 생성도기존native IoU0.4267154667과일치해야한다.

Training seed는JSON에101을기록하고rank별101+rank로설정한다. 실제validation callback이CPU/CUDA/NumPy RNG를저장·복원하는것도확인했다. 같은seed가서로다른architecture의모든난수소비와학습trajectory까지bitwise 같다는주장은하지않는다.

## GPU와 예상 시간

GPU0~3만사용한다. GPU0/1은현재seed202 평가·기존C1 source진단·C2 shuffle 진단이끝난뒤C1을학습하고그final 평가뒤B3를학습한다. GPU2/3은현재regional seed303 두run이끝난뒤V1을학습한다. 원본본비교·현재regional 실행은중단하지않는다. 각run의final noise0/1은해당두GPU에동시에,noise2는첫GPU에이어실행한다.

기존같은SS/DDP48 실행은steady 약17초/update였다. Run당compute 약2시간18분이며setup/validation/final generation이추가된다. 세run은두wave여서학습wave만약5시간규모이고현재남은작업과겹쳐실행한다. 정확한시간·peak는새실제probe/runtime로갱신한다. 조건을바꾸어좋은수치를얻기위한장기epoch 연장은하지않는다.

실제C1 native proof는global/proj 차이0,LoRA gradient L1 248.4244,fusion-gate gradient0.03636,adapter-output gradient L1 10.0385로PASS였다. 두GPU의실제2-update probe도각micro48/global96/accumulation1·finite loss를통과했다. 양rank peak allocated는47.999GiB였고step2 compute는17.83/17.23초였다. Main 초기common32 IoU도0.4267154667과exact였다. 이는실행·초기정합증명이며학습후성능향상은아직판단하지않는다.

## 평가·완료 조건

각epoch common32의voxel IoU/F1/precision/recall·GT/Pred/Diff를W&B에남긴다. Final은같은128개·93source cluster·noise3개·12steps/CFG3·generation8이다. Decoded-target SS GT와micro IoU를사용하므로기존14개표의protocol을따른다. 현재regional pilot의actual-SS-input GT/macro 집계와섞지않는다.

같은 epoch에 도달한 V1/B3/C1 사이의 세 paired 대비도 별도 `epoch-comparisons/`에 기록한다. 새 실행의 seed101/world2/micro48/accumulation1, 같은 32개 target voxel hash·sample noise·source cluster를 검사한다. 이 한 noise의 중간 결과는 `selection_use=false`이며 fixed-final128×3를 대체하지 않는다. 원래 seed0의 epoch 경향과 섞지 않는다.

두rank480개연속finite update·micro48/accumulation1·초기native proof·0/96/192/288/384/480 검증·실제final checkpoint를감사한다. Final128/noise0의common32 subset은동일checkpoint의epoch480 저장metric과일치해야한다. 모든새평가에target voxel hash를요구하며3noise를source cluster 안에두고C1−V1/B3−V1/C1−B3의paired CI를계산한다.

기존seed0과새seed101은실행조건까지동일한두반복이아니므로독립seed 모집단CI로통합하지않는다. 새seed의matched 비교와기존관측의방향을나란히기록한다. 설정준비PASS와학습·성능PASS를구분한다. Full/PBR 채택은새결과·source활용진단·전체module 비교를검토한뒤결정하며C1으로미리확정하지않는다.

`prepare_ss_seed_confirmation --mode prepare/queue/summarize`가설정생성·queue 등록·strict 최종집계를담당한다. Canonical file4 namespace는`rgb_matching_v2/ss-seed-confirmation-v1`이며source만Git에저장한다. W&B는기존`youngwoojeon-unist/pixal3d-deformnext-retargeting`의개별training run과공통evaluation reporter를사용한다.

## 첫 epoch의 실제 matched 비교

2026-10-18 10:46 KST에 V1의 step96 검증도 완료돼 C1과 같은 epoch를 비교했다. 두 모델 모두 native 시작 IoU는 0.42671546671444294였으며, 아래는 같은 32개 source cluster·한 sampling noise의 중간 결과다.

| Arm | Step96 voxel IoU | 학습 전 대비, pp |
| --- | ---: | ---: |
| V1 | 0.43214776 | +0.543 |
| C1 | 0.43633516 | +0.962 |

C1−V1은 +0.419pp, paired source-cluster 95% CI [−0.583, +1.320]pp다. 점추정치는 양수지만 추가 효과를 확정하지 못했다. Sample noise·source cluster·GT voxel hash 32개가 모두 일치했다. B3와 fixed-final128×3가 아직 남아 있으므로 초기 개선만으로 기존 C1의 이득이 새 seed에서도 재현됐다고 결론내리지 않는다.

각 rank의 step1~96이 연속이며 finite loss·global96/micro48/accumulation1임을 확인했다. 실제 step96 checkpoint의 optimizer 상태는 V1 360개, C1 366개가 모두 step96이고 모든 moment tensor가 finite였다. W&B 원격 history에서도 step0/96의 IoU와 paired gain/CI가 로컬 기록과 일치했다. 두 GT/Pred/Diff 이미지도 서버에서 내려받아 로컬과 SHA256 일치를 확인했다. 그림은 SS occupancy projection이며 rendering-level 품질 평가를 뜻하지 않는다.

근거는 `epoch-comparisons/epoch-01-c1-vs-v1.json`, `epoch-snapshots/epoch-01-{v1,c1}-{runtime,validation}.json`, `figures/epoch-01-{v1,c1}-GT-Pred-Diff.png`, 상위 `provenance/matched-seed101-first-epoch-{audit,wandb-remote}.json`이다. 학습은 480 updates까지 계속하며 중간 snapshot을 final 결과로 대체하지 않는다.

두 모델의 step96 checkpoint와 위 snapshot·이미지·source delta 등 17개 산출물 약 5.70GB는 실제 file4와 local의 SHA256·크기가 모두 같았다. `provenance/matched-seed101-first-epoch-file4.json`에 기록했다. 아직 쓰고 있는 runtime/validation 전체 파일을 immutable 파일로 해시한 것은 아니다.

## 두 번째 epoch의 matched 비교

2026-10-18 11:15 KST에 V1의 step192 검증도 완료됐다. 같은 32개 source cluster·동일 noise·같은 decoded-target GT를 유지한다.

| Epoch / step | V1 voxel IoU | C1 voxel IoU | C1−V1, pp [95% CI] |
| --- | ---: | ---: | --- |
| 1 / 96 | 0.43214776 | 0.43633516 | +0.419 [−0.583, +1.320] |
| 2 / 192 | 0.43444352 | 0.44060209 | +0.616 [−0.541, +1.900] |

두 모델 모두 첫 epoch보다 점추정치가 올랐지만, C1의 추가 이득은 여전히 확정되지 않았다. C1 자체의 1→2 epoch 변화도 +0.427pp, paired 95% CI [−1.095, +1.911]pp로 0을 포함한다. 다른 epoch의 V1/C1을 직접 비교하거나 이 경향으로 학습을 연장하지 않는다. 예정한 480 updates와 final128×3를 유지한다.

두 rank의 step1~192가 모두 연속이고 finite loss·global96/micro48/accumulation1이다. Step192의 optimizer 상태는 V1 360개, C1 366개가 전부 step192이고 moment tensor도 모두 finite였다. Peak allocated는 V1 45.620GiB, C1 47.999GiB다. 두 모델의 GT voxel hash 32개와 sample noise가 같고, W&B의 실제 step192 history·paired gain/CI·두 GT/Pred/Diff 이미지도 대조했다. 원격 summary의 부동소수점 직렬화 차이는 절대오차 1e−12 이내로 검사하며, 이미지는 다운로드한 bytes의 SHA256이 일치해야 한다.

근거는 `epoch-comparisons/epoch-02-c1-vs-v1.json`, `epoch-comparisons/trend-epoch02-vs-epoch01-c1.json`, `epoch-snapshots/epoch-02-*`, `figures/epoch-02-*`, 상위 `provenance/matched-seed101-{C1-epoch2,second-epoch}-{audit,wandb-remote}.json`이다. `C1-epoch2` receipt는 C1의 epoch별 변화, `second-epoch` receipt는 같은 epoch의 모델 간 비교이며 합쳐 집계하지 않는다. 두 checkpoint의 실제 file4 저장 검증은 각 `*-file4.json` receipt로 구분한다. B3와 원본 constant-gate 6개 학습은 후속 대기 상태이며 Full/PBR 채택은 아직 결정하지 않았다.

## 후속 epoch의 자동 검증과 C1 step288

남은 C1/V1 epoch3~5, B3 epoch1~5의 검증을 별도 CPU 프로세스로 연결했다. 실제 step192의 두 checkpoint로 native proof·GT/noise/source identity·연속 rank별 update·optimizer 감사를 먼저 실행해 통과했다. W&B history의 특정 step에 연결된 이미지가 기존에 다운로드한 이미지와 같은 것도 확인했다. 이 프로세스는 GPU를 사용하거나 trainer·queue를 제어하지 않는다. Source와 실행 기록은 상위 `provenance/runtime-verifiers/`, `matched-epoch-auditor-start.json`, `matched-epoch-auditor-source-file4.json`에 있다.

C1의 실제 step288이 이 절차의 첫 전체 검증을 통과했다. IoU는 **0.44362372**로 epoch1/2/3의 점추정치가 0.43633516 → 0.44060209 → 0.44362372였다. Epoch2→3은 +0.302pp, paired 95% CI [−0.167, +0.827]pp로 0을 포함한다. V1의 epoch3가 아직 없으므로 다른 epoch의 V1과 비교하지 않고 예정한 480 updates를 유지한다.

두 rank의 step1~288은 모두 연속·finite·global96/micro48/accumulation1이고, optimizer 366개 상태가 모두 step288·finite였다. GT hash·noise·source identity 32개가 native 시작과 일치한다. W&B의 실제 step288 지표·이미지를 내려받아 대조했고, checkpoint 등 7개 산출물 약 2.86GB의 local/actual-file4 SHA256·크기도 일치했다. 근거는 `epoch-audits-v2/c1/epoch03/{audit,file4-closure}.json`, 같은 폴더의 validation/runtime snapshot·이미지, `epoch-comparisons/trend-epoch03-vs-epoch02-c1.json`이다. 이 namespace의 PASS는 해당 epoch의 실행·보관 검증이며 fixed-final128×3 성능 채택을 뜻하지 않는다.

## 세 번째 epoch의 동일 조건 비교

2026-10-18 11:43 KST에 V1의 step288 검증도 완료됐다. 아래는 seed101·고정32 source cluster·한 sampling noise의 같은 epoch 비교다.

| Epoch / step | V1 voxel IoU | C1 voxel IoU | C1−V1, pp [95% CI] |
| --- | ---: | ---: | --- |
| 1 / 96 | 0.43214776 | 0.43633516 | +0.419 [−0.583, +1.320] |
| 2 / 192 | 0.43444352 | 0.44060209 | +0.616 [−0.541, +1.900] |
| 3 / 288 | 0.43187184 | 0.44362372 | +1.175 [−0.051, +2.641] |

C1의 점추정치 격차는 커졌지만 CI는 여전히 0을 포함한다. V1의 2→3 epoch 점추정치는 내려갔으나 이를 구현 오류나 과적합의 확정으로 취급하지 않는다. B3·480 updates·final128×3가 남아 있으며 중간 경향으로 arm이나 학습 길이를 변경하지 않는다.

두 모델 모두 실제 optimizer가 step288이고 native 시작과 GT hash·noise·source cluster가 일치한다. V1도 자동 검증을 통과해 optimizer 360개 상태가 finite이고, W&B 지표/이미지와 checkpoint를 포함한 6개 산출물의 actual-file4 SHA256·크기를 확인했다. C1의 같은 감사는 앞 절의 7개 산출물 검증을 따른다. `epoch-comparisons/epoch-03-c1-vs-v1.json`, `epoch-audits-v2/{v1,c1}/epoch03/`, 상위 `provenance/matched-seed101-third-epoch-audit.json`에 paired 지표와 서버 검증 근거를 기록했다. Full/PBR 채택은 아직 결정하지 않았다.

## C1의 네 번째 epoch 검증

2026-10-18 11:59 KST에 C1 step384의 실행·보관 검증을 마쳤다. 고정32의 IoU는 **0.45024880**, F1은 **0.62092629**다. C1의 epoch1~4 IoU 점추정치는 0.43633516 → 0.44060209 → 0.44362372 → 0.45024880이었다. 같은 32개 GT voxel hash·source cluster·sampling noise를 확인한 epoch3→4의 paired 차이는 **+0.663pp**, 95% source-cluster bootstrap CI **[+0.012, +1.284]pp**였다. 이는 한 noise의 중간 경향이며 반복 관측에 대한 보정이나 training-seed 모집단의 추정이 아니다. `selection_use=false`를 유지하고 fixed-final128×3를 대체하지 않는다. V1의 step384가 아직 없으므로 다른 epoch의 V1과 비교하지 않는다.

두 rank의 step1~384가 연속·finite·global96/micro48/accumulation1임을 확인했다. Optimizer 366개 상태가 모두 step384이고 moment tensor도 모두 finite였다. W&B의 실제 history step384 수치를 검사하고 해당 GT/Pred/Diff 이미지를 내려받아 로컬 bytes와 SHA256을 대조했다. Checkpoint 3개와 검증 snapshot·이미지 등 7개 산출물 **2,857,601,449 bytes**도 actual-file4의 SHA256·크기와 일치했다. 근거는 `epoch-audits-v2/c1/epoch04/{audit,file4-closure}.json`과 `epoch-comparisons/trend-epoch04-vs-epoch03-c1.json`이다. 학습은 480 updates까지 진행 중이며 B3·원본 constant-gate 6개 학습도 후속 범위로 남아 있다.

## 네 번째 epoch의 동일 조건 비교

V1 step384 검증과 원격 저장 검증도 완료됐다. 아래는 같은 training seed101·world2/micro48/accumulation1·고정32 source cluster·한 sampling noise의 비교다.

| Epoch / step | V1 voxel IoU | C1 voxel IoU | C1−V1, pp [95% CI] |
| --- | ---: | ---: | --- |
| 1 / 96 | 0.43214776 | 0.43633516 | +0.419 [−0.583, +1.320] |
| 2 / 192 | 0.43444352 | 0.44060209 | +0.616 [−0.541, +1.900] |
| 3 / 288 | 0.43187184 | 0.44362372 | +1.175 [−0.051, +2.641] |
| 4 / 384 | 0.43429997 | 0.45024880 | +1.595 [+0.153, +3.134] |

Epoch4의 paired CI는 0보다 높지만, 이 표는 반복 관측에 대한 보정이 없는 중간32 평가이며 training-seed 모집단 추정도 아니다. C1의 추가 이득을 최종 확인했다는 결론이나 조기 종료·epoch 연장에 사용하지 않는다. B3와 480-update fixed-final128×3가 남아 있고 Full/PBR 채택도 보류한다.

두 run 모두 실제 optimizer 상태 V1 360개/C1 366개가 전부 step384·finite였다. GT voxel hash·sampling noise·source cluster 32개가 일치하며, W&B의 실제 step384 수치·다운로드한 두 GT/Pred/Diff 이미지와 공통 reporter의 paired 수치·CI를 확인했다. 두 epoch의 checkpoint와 snapshot 등 13개 산출물 **5,701,868,309 bytes**는 각 자동 closure receipt에서 actual-file4 SHA256·크기 검증을 통과했다. `epoch-comparisons/epoch-04-c1-vs-v1.json`, `epoch-audits-v2/{v1,c1}/epoch04/`, 상위 `provenance/matched-seed101-fourth-epoch-{audit,wandb-remote,file4}.json`에 근거를 보존했다.

## C1의 5 epoch 학습 완료와 final 평가 시작

C1이 480 updates를 마쳤고 마지막 고정32 IoU는 **0.45384906**이었다. 같은 C1의 epoch4→5 변화는 +0.360pp, paired source-cluster 95% CI [−0.168, +0.939]pp로 0을 포함한다. 학습 전 0.42671547보다 점추정치는 높지만, 아직 완료되지 않은 V1 epoch5나 fixed-final128×3 결과를 대신하지 않는다. 32개 경향으로 epoch를 연장하지 않고 계획한 5 epoch에서 종료했다.

Strict training audit로 두 rank의 step1~480 연속 update·finite loss·global96/micro48/accumulation1, 여섯 validation, 초기 native 정합/gradient proof와 실제 final checkpoint를 확인했다. 두 rank의 steady compute 중앙값은 17.280/17.278초이며 peak allocated는 각각 47.999GiB였다. Queue의 실제 시작부터 정상 종료까지는 **9,041.93초(2시간 30분 42초)**다. Native/memory proof·초기화·epoch 검증을 포함하며 후속 final128 평가 시간은 포함하지 않는다. Run은 정상 종료했고 같은 process를 재시작하거나 과거 weight에서 resume하지 않았다.

Step480 optimizer 366개 상태도 전부 step480·finite였으며 W&B의 실제 step480 metric과 다운로드한 GT/Pred/Diff 이미지를 검증했다. Final checkpoint·snapshot·이미지 7개 **2,857,836,515 bytes**는 actual-file4 SHA256·크기와 일치했다. 근거는 `epoch-audits-v2/c1/epoch05/`, `epoch-comparisons/trend-epoch05-vs-epoch04-c1.json`, 상위 `provenance/matched-C1-five-epoch-training-{audit,file4}.json`이다.

학습 정상 종료 뒤 GPU0/1에서 final128 noise0/1의 실제 두 process가 시작됐고 각40개 생성까지 확인했다. Noise2와 후속 matched B3는 선행 평가 완료 후 같은 GPU를 사용한다. V1 학습, B3, 원본 constant-gate 두 arm·3개 seed와 [계획된 sampler 진단](2026-10-18-ss-sampler-sensitivity-ko.md)이 남아 있다. Full/PBR 채택이나 전체 실험 완료로 표시하지 않는다.
