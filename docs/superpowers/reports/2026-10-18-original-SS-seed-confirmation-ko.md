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
