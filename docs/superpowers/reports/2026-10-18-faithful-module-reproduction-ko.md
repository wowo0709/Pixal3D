# 이전 양의 모듈의 재현 구현과 실행 계약

사용자의 “이전에 성능 향상을 보였던 모듈들을 재현” 요청에 따른 실행 문서다. [차이 분석](2026-10-18-historical-module-transfer-gap-analysis-ko.md)의 확인된 불일치를 복구하며, 현재14개 구성의 결과를 삭제하거나 이름을 바꿔 성공한 재현으로 취급하지 않는다.

2026-10-18 현재 원본 Q2C 24개 학습의 모든 loss/최종 weight와 기존 G2 checkpoint의 144개 평가 endpoint를 재현했다. [현재 RGB task](2026-10-18-task-rgb-matcher-transfer-ko.md)에서도 Q2C의 RGB error 6.01% 감소를 3개 seed에서 확인했다. SS 생성 효과는 별도이며 [본 14개 결과](2026-10-18-ss-primary14-final-ko.md)는 완료됐다. 현재 9채널 geometry의 기존 scalar-gate 6개 학습도 완료됐지만, query 추가 이득은 확인되지 않았고 원본 optimizer parameterization과의 차이가 남았다. 이를 복구한 [원본 constant-gate network 재실험](2026-10-18-original-network-gate-reproduction-ko.md)은 실제 GPU native/gradient 검증과 seed101 두 arm의128-update·calibration 평가를 마쳤고, seed202가 진행 중이다. 첫 seed에는 stale cleanup 개선과 source recall 하락이 함께 있어 나머지 seed를 확인한다. [새 seed의 V1/B3/C1 matched 실행](2026-10-18-original-SS-seed-confirmation-ko.md)에서는 V1/C1의480-update final128×3가 완료돼 C1−V1 IoU **+0.954pp,95% CI[+0.542,+1.350]pp**를 확인했다. 이는 C1 전체 경로의 추가 이득이며, retargeter 분리를 위한 B3·원본 network-gate6개 전체·최종 conditioning 검사·sampler 진단과 Full/PBR 판단은 남아 있다. 아래 단계별 준비/진행 상태는 당시 이력이며 최신 결론은 각 결과 문서를 따른다.

## 재현 대상

| 대상 | 이전에 유효했던 정확한 비교 | 재현 계약 |
| --- | --- | --- |
| Source spatial routing | 세 source-view map의 GT 주소 대 wrong 주소/동일 projection-source-off | 3×32² write slot, 정확한 read address, view별 value multiset·support 고정, 각 source의 native ProjGrid, global/negative 불변 |
| Q2C | 같은64 candidate·task RGB loss의 D0 pairwise 대 D1 query→candidate context | Top1/soft 양쪽에 Q2C가 들어간 현 대비와 구분; candidate·objective·seed·fixed-final checkpoint를 맞춤 |
| Geometry content | 3D predicted O/D logit·occupancy·SDF9채널을 residual content에 넣은 효과 | 16³ native xyz, target/mismatch/difference, frozen SS full rollout→decoder regional SS64/32 BCE, correction penalty0.001 |

이전 retargeter는 모든 DEV source-use gate를 통과하지 않았다. 따라서 과거의 geometry cleanup 상대 개선과 실제 source-retargeting 성공을 구분한다. RAW Omega2048 feature delta를 이전9채널 occupancy/SDF geometry와 같다고 취급하지 않는다.

## 이식한 reference 구현

`historical_reproduction.py`와 `historical_geometry.py`는 이전 원본에서26개 함수/클래스의 AST를 그대로 옮겼다. `historical_reproduction_authority.json`에 각 원본 파일 SHA256·line·함수 AST hash를 고정했다. 실행 시 과거 worktree나 `runtime_bundle`을 import하지 않는다. 저수준 Q2C/scalar는 기존 `r9_matching.py`에 이미 같은 AST로 이식돼 있다.

Reference G2의 네 mode는 과거 수식의 감사용 API다. 사용자가 요청한 production의 No-query arm을 zero channel로 대체하거나 새로운2×2 학습을 실행하는 뜻이 아니다. Production query ablation은 입력 채널의 물리적 유무를 유지한다.

CPU 검증은 native identity/fallback, soft validity 단일 가중, identity sampling, view별 wrong-address derangement/value multiset, pixel-ray 왕복, geometry 축/채널, zero adapter, frozen rollout의 gradient, regional loss, scoped hook 복원, frame fingerprint를 포함한다. 관련30개 테스트가 통과했다. 이 PASS는 수식·통합 계약의 검증이며 생성 성능 PASS가 아니다.

## 1차 실제 검증: 현재 common32의 per-view oracle

Data는 기존 DeformNeXt validation common32·world-v4를 유지한다.32개 모두 M0 geometry·M1 vertices·RGB가 로컬 file4 mirror에 있었으며 추가 원본 다운로드는 하지 않았다. 과거와 같은 Open3D raycast·barycentric route·pixel7.5+16n·visibility tolerance0.002를 사용해 세 source-camera route를 구축했다. CPU affinity를2 core로 제한했고 실제 계산은약3초였다.32개 모두 각 source-view에서2개 이상 active slot이 있어 wrong-address control이 정의됐다.

Route-v2는 RGB/mesh fingerprint 외에 mesh scale·render-world schema·center/extent도 고정한다. 기존 v1 route는 보존하며 v2를 실행에 쓴다. 모든 GT cache는 `diagnostic_only`이며 inference-safe matcher cache와 namespace가 다르다.

실제 encoder의 native per-view ProjGrid 입출력을 scoped hook으로 캡처한다.4개 view의 native online mean이 원래 joint projection과 bit-exact인지를 검사하고 global/negative는 유지한다. Source-on case는 세 source map을 각각 교정해 각 카메라의 원래 ProjGrid로 투영한다. Anchor-camera map 하나로 합친 뒤 voxel residual을 더하는 이전 구현을 재사용하지 않는다.

Cases는 `native`, `source-off`, `gt-corr`, `wrong-corr`다. 여기서 source-off는 **source projection만 제거하고 네 RGB로 얻은 native global/negative를 유지**하는 과거 reference control이다. 현재 primary study의 추가 모듈 전체 OFF와 정의가 다르므로 별도 reproduction output에 기록한다. Target-only 모델 학습 arm을 추가하지 않는다.

Checkpoint는 원본 multi-view mixed385 complete base+LoRA SS64이며, loader·architecture는 현재 dev-jaehyeok 기반 runtime을 사용한다. 같은32개·같은 sample noise·12 steps·CFG3·generation batch8에서 frozen consumer를 비교한다. 먼저1개 sample의 encoder preflight를 통과해야 전체 생성을 시작한다. GPU0~3만 사용하고, queue가 비는GPU3에 이 작업을 배정한다. 실행 중인 C2와 충돌시키지 않는다.

초기 metric은 global SS IoU/F1과 `route_sampled_hidden_gt_recall64/32`다. 후자는 raycast로 샘플링한 hidden surface cell과 decoded target SS의 교집합이며, **과거의 full source-exclusive SS region authority를 재구축한 metric은 아니다.** 결과 파일에 이 한계를 명시한다. 이전 regional primary의 정확한 재현에는 full region authority를 별도로 구성해야 한다. 생성 검증을 과거 수치의 재현 완료로 확대하지 않는다.

## 후속 연결

1. Actual GPU preflight와 full32 correct/wrong/source-off를 완료해 native per-view 경로의 headroom을 판별한다.
2. 같은 upstream/task loss에서 D0/D1을 직접 비교하는 작은 head 학습·evaluation runner를 연결한다. 기존 H3를 Top1과soft 양쪽에 쓰는 비교로 이 검증을 대체하지 않는다.
3. Predicted SS O/D geometry·full region authority를 준비하고, 옮긴 full-rollout regional objective의 실제 native GPU gradient·zero-regression을 확인한다.
4. 재현 가능한 source-use/geometry gain이 확인된 경로에 adapter와 physical query를 순서대로 연결한다. 기존 데이터의 전체 재전처리나 장기 fine-tuning을 먼저 반복하지 않는다.

현재 성능 gate가 열린 것으로 간주하지 않는다. 준비·수식 검증·GPU preflight·생성 endpoint·모듈 채택은 별도 상태로 기록한다. 기존14개 비교의 완료와 위 faithful reproduction·필요한 후속의 완료도 구분하며, Full/PBR 결정까지 추적한다.

## Q2C의 통제된 task 재현 실행

`historical_q2c_training.py`는 과거 RGB task의10개 함수/자료형과 initializer·sampler·zero-check·optimizer-step5개 method를 AST 그대로 이식한다. 별도 authority에 원본 SHA와 method hash를 기록했다. `reproduce_q2c.py`는 과거 TRAIN source array로 packed data를 다시 대조한 뒤 file4 reproduction namespace로 옮긴다. 기존 worktree를 실행 중 import하지 않는다.

같은63개 component-disjoint4-fold,42017/42018/42019,각1200 update×256query,동일Top64·L12 RAW·TRAIN-only tau0.005를 사용한다. D0/D1은 같은 upstream P0와 같은 scalar-head 초기화·pair/query sampler·원본 RGB carrier L1을 사용하며 correspondence CE를 추가하지 않는다. 각1200개 pair/query hash를 과거 log와 모두 대조하고 loss 차이를 기록한다. Frozen final checkpoint와 SHA를 쓴 다음 held-out evaluator cue를 연다. 총24개 tuple을 빠른 head job으로 등록하며, 현재 SS DDP와 겹치는 GPU는 배정하지 않는다.

새5개 CPU 검증을 추가해 관련16개 검증을 통과했다. AST·초기 score parity·head 초기화·candidate/query 동일성·RGB가 scorer 입력에 들어가지 않는 계약·visibility-matched wrong map·파일 변조 거부를 확인했다. 이 실행은 **과거의 controlled RGB task 재현**이다. 여기서 얻은 양의 결과를 현재 DeformNeXt SS 생성 성능 향상으로 보고하지 않으며, native per-view consumer 연결 뒤 별도 검증한다.

## 실제32개 transport 결과와 G2 consumer 차이

원본 mixed385 SS64 checkpoint, 동일32개·1noise·12steps/CFG3의 생성이 완료됐다.

| Case | Global SS IoU64 |
| --- | ---: |
| Native four RGB | 0.42671547 |
| Source projection off | 0.37160198 |
| GT-corr, 원래 target fallback | 0.32404648 |
| Wrong-corr, 같은 support/value multiset | 0.27693184 |

GT−wrong은 **+4.711pp**, source-cluster bootstrap95% CI[+2.865,+6.675]pp이다. 주소의 유효한 효과가 존재한다. 그러나 GT−native는−10.267pp CI[−13.745,−7.075]pp로, 이 replacement를 그대로 production에 채택할 수 없다. GT cache는 여전히 diagnostic-only다. Global/negative는 모두 보존했고, native replay projection maxabs=0·wrong source value multiset exact·active address changed100%를 actual GPU에서 확인했다. 원래 target fallback은 native source의 일부 지원도 제거하므로, 주소 효과와 native 대비 향상을 구분해야 한다.

G2 archived checkpoint의 runtime을 열어 추가 차이를 확인했다. 당시 consumer는 full fine-tuning SS64 EMA step20000(`c2700988…`)이고, sampler는12steps·CFG7.5·guidance rescale0.7·interval[0.6,1]·rescale_t5였다. 현재 production의 mixed385 base+LoRA step4210·CFG3와 같지 않다. **현재 학습의 checkpoint를 과거20k로 바꾸지 않는다.** 과거 consumer는 별도 namespace에서 원래 양의 결과가 이식 코드로 재현되는지 검증하는 reference로만 사용한다.

`historical_geometry_runtime.py`에 native sampling/scoring·noise·condition10개 함수를 AST 그대로 옮겼다. `reproduce_geometry_content.py`는 실제 predicted O/D9채널·full SS regional masks·원래 frozen flow/decoder·훈련된 adapter를 file4 reproduction namespace로 고정한다. 두 reference arm(feature/content)만 실행한다. 각4-update 실제 gradient smoke와 기존3seed fixed-final checkpoint의 DEV8×3noise endpoint 재생을 수행해 수치 차이를 직접 기록한다. 이미 존재하는128-update checkpoint를 재생하므로 같은 학습을 처음부터 모두 반복하지 않는다.

추가5개를 포함해 관련21개 CPU 검증과 CLI 검증이 통과했다. 이 reference의 historical zero-geometry control은 과거 수치 검증용이다. 향후 production의 No-query는 기존 사용자 정의대로 query 입력·encoder·채널을 물리적으로 제거한다. Reference 결과와 현재 데이터의 module 채택 결정을 혼합하지 않는다.

## Native fallback의 별도 bridge control

원래 reproduction의 `historical-target` 출력을 보존한다. 추가 `native-spatial` control은 각 source camera의 **valid write map에서만** `selected - native`를 투영해 원래 four-view projection에 더한다. GT/wrong의 support·source value multiset은 고정하며, 나머지 native source 정보와 global/negative를 보존한다. source-weight0은 실제 native projection을 그대로 반환한다. 이 control은 과거 성공 수식을 재현했다고 이름을 바꾸는 작업이 아니라, 현재 consumer로 옮길 때의 fallback 문제를 분리하는 새 진단이다. 고정 weight1 및0.25의 common32 비교를 별도 namespace에서 실행한다. GT 경로이므로 production 학습·추론 cache로 승격하지 않는다.

## Physical No-query 구현과 원본 weight 위치 복구

`GeometryContentRetargetAdapter`는 두 additive arm용 구현이다. Query-enabled는 원래 R4-CONTENT의 9채널 encoder·target/mismatch/difference/xyz·zero-initialized residual을 사용한다. 최초 pilot은 geometry와 무관한 원본 gate를 scalar logit으로 단순화했다. 출력과 입력 gradient는 같지만 **optimizer trajectory는 같지 않으므로 학습 재현에는 충분하지 않았다.** 현재 `historical-network` 모드는 원본의 371-parameter constant-gate MLP를 그대로 보존한다. 6회 연속 CPU optimizer update의 parameter가 원본 R4와 매 step bit-exact이며, 실제 GPU 검증과 학습은 별도 namespace로 실행한다. 기존 scalar 결과를 수정하거나 원본 학습의 재현 완료로 해석하지 않는다.

No-query는 content encoder가 아예 없고 residual 입력 폭에서도 geometry embedding 채널을 뺀다. Geometry tensor를 전달하면 zero tensor라도 거부한다. 두 arm은 초기 projection이 native와 exact이며, 출력 layer의 gradient가 연결돼 있다. Reference old-zero arm의 수치를 production physical-No-query arm의 수치로 재사용하지 않는다.

과거 flow의 기록 경로 `/root/data3/pixal3d/ckpts/...`가 비어 있어 최초 reference export는 실패했다. 파일은 `/root/data3/delete-at-261010/pixal3d/ckpts/ss64/20260730T160615Z/ckpts/denoiser_ema0.9999_step0020000.pt`에서 발견했다. 새 실행에는 이 위치를 **명시적** `--flow-checkpoint`로 전달하고 원래 SHA `c2700988…`와 대조한다. 임의 checkpoint fallback이나 현재 production initializer 교체는 하지 않는다. 실패 기록은 보존하고 별도 v2 export/job으로 다시 실행한다.

## Native fallback bridge 실제 결과

동일32개·1noise에서 두 사전 명시 weight의 생성이 완료됐다. 모든32개 native prediction/GT occupancy SHA가 최초 reproduction과 일치하고, correct/wrong은 같은 GT mask·sample noise·global/negative·write support를 쓴다. Wrong의 active address는100% 바뀌며 view별 source value multiset은 exact다.

| Native-spatial bridge | GT IoU64 | Wrong IoU64 | GT−native, pp [95% CI] | GT−wrong, pp [95% CI] |
| --- | ---: | ---: | --- | --- |
| weight1 | 0.44631358 | 0.38078197 | +1.960 [+0.517,+3.457] | +6.553 [+4.934,+8.261] |
| weight0.25 | 0.43296665 | 0.42611825 | +0.625 [+0.205,+1.060] | +0.685 [+0.075,+1.355] |

이는 **현재 mixed385 consumer에서도 correspondence 주소의 headroom이 있고, native source 보존이 중요함**을 보여준다. 과거 target-fallback GT case의0.32405를 native-spatial bridge로0.44631까지 바꾼 원인을 모듈 유무와 분리해서 볼 수 있다. 단, GT 주소를 쓰므로 학습된 matching 모듈이나 unseen-test 향상으로 발표하지 않는다. Final 모델에 GT route cache를 넣지 않는다. Learned correspondence의 per-view 연결과 source-use 검증은 별도 gate로 남는다.

Physical No-query·reference runtime을 포함한 관련44개 CPU 검증이 통과했다. Q2C의 실제 GPU 학습도 시작됐고, 첫 D0 run의 중간 optimizer loss들이 과거 log와 bit-exact였다. 전체1200 update·held-out 결과와 D1까지 완료한 뒤 정확한 재현 범위를 판정한다.

## Q2C 전체 수치 재현 완료와 geometry GPU 검증

Controlled RGB task의 **24개 run(4-fold×3seed×D0/D1)**이 모두 완료됐다. 총28,800개 update의 pair/query sampling hash·loss가 과거 기록과 exact이고, 모든 최종 learned parameter tensor와 held-out RGB component도 원본과 exact다. 집계/불확실성 함수5개 역시 원본 AST를 이식했으며, cue→seed→pair 순서와 pair bootstrap을 그대로 사용했다.

| Historical OOF63 | RGB L1 |
| --- | ---: |
| D0 pairwise | 0.03676911540385993 |
| D1 Q2C | 0.035128986239626396 |

D0−D1 error 감소는 **0.0016401291642335306**, paired95% CI[0.0009215979517582304,0.002400408386399977]로 과거 결과를 완전히 재현했다. 이 결과는 **과거63개 controlled task**에서 matching-aware Q2C의 효과를 재현한 것이다. 현재 DeformNeXt 전체 SS 생성 향상은 별도 검증 대상이다. 결과/tuple별 parameter·component audit는 file4 `faithful-reproduction-v1/q2c-exact-reproduction-summary.json`에 있다.

Geometry reference는 original TRAIN32/DEV8의 full region·predicted O/D9채널·6개 trained checkpoint와 원래5.36GB SS20k weight를 모두 SHA 검증해 export했다. 원본 split manifest에 따로 있던6개 confirm pair는 export/검증에서 제외했다. 총48개 관련 CPU 검증을 통과했다.

Actual GPU4-update smoke는 두 arm 모두 초기 native projection·첫 loss·첫 gradient norm이 exact다. 모든 step에서 adapter gradient가 finite/nonzero이며 frozen SS weight에는 gradient가 없다. 후속 step loss의 원본 대비 최대 절대 차이는 feature0.00187683, content0.00189209였다. **4-step optimizer trajectory가 bit-exact라는 주장은 하지 않는다.** 최초 실행의 seed 순서 차이를 발견해 seed→frozen flow→decoder→frozen DINO→adapter 순서로 복구했다. 실제 DINO 로딩이 CPU RNG를 소비하는 것도 확인했고, 앞으로는 export된 conditioner 초기화 config를 읽어 과거 worktree config에 실행 의존하지 않는다.

기존6개 G2 fixed-final checkpoint의 DEV8×3noise 재생이 **144개 조합 모두 완료**됐다. Normal·projected-source-off·geometry-shuffled의 모든 regional/global metric과 nested BCE 항까지 원본과 exact이며 최대 절대 차이는0이다. `geometry-exact-reproduction-summary.json`은 원본8-pair bootstrap 함수로 model/noise9개를 각 pair 안에서 먼저 평균한다. 기존 trained checkpoint의 endpoint 재현이며,128-update 전체 재학습 trajectory 재현과는 구분한다.

| 원본 DEV8 재현 대비 | Source-exclusive recall32 차이, pp [95% CI] |
| --- | --- |
| Geometry content−feature-only | +4.075 [+1.851,+5.959] |
| Geometry intact−shuffled | +0.953 [+0.296,+1.922] |
| Content source projection ON−OFF | −3.701 [−6.628,−0.131] |

Geometry content의 상대 개선과 query를 섞으면 악화되는 경향은 재현됐다. 동시에 source projection OFF가 더 좋은 한계도 원본 그대로 재현됐다. OFF는 네 RGB의 native global/negative를 유지한다. 따라서 이를 source RGB 전체의 무용성으로 읽거나, 상대 cleanup 개선만으로 useful retargeting 성공/Full 채택을 선언하지 않는다. 이 한계는 현재 dataset 이식에서도 source-use 검증을 별도로 요구하는 이유다.

현재 DeformNeXt에서 원본RGB task objective로 학습한 Q2C의 추가 대조도 완료됐다. 251 train/61 calibration query-eligible asset, D0/D1×3seed에서RGB error가0.09008583→0.08467440으로 감소했다. Paired CI는양수지만 DINO teacher error·exact-patch 정확도의 추가 개선 CI는0을 포함한다. 자세한 protocol/수치는 [현재 dataset task 대조](2026-10-18-task-rgb-matcher-transfer-ko.md)에 기록한다. 이 결과와 과거63-pair OOF 수치를 합쳐 집계하지 않으며, 현재SS 생성 효과는 별도 gate다.

사용자 데이터/결과의 실제 file4 저장도 확인했다. `faithful-inputs-and-bridge-file4.json`은224개 immutable input·bridge output·source 파일(약9.52GB)의 local/actual-file4 SHA256 일치를 기록한다. Q2C/geometry reference와 현재14개 production 비교는 namespace·성능 주장 모두 분리한다. GPU0/1의 C2는 계속 진행하며, reference 작업은GPU2/3을 사용한다.

## 현재 dataset의 실제 regional 학습으로 전환

RGB-only predicted O/D9채널은pilot320+common32 총352개에서완료·재계산exact·actual-file4 SHA까지통과했다. Common32 O의occupied count도기존source-only baseline과모두같다. 현재native condition64와actual SS-input/physical-cell authority64를구축했다. 실제full-rollout regional gradient gate를통과한뒤native adapter /adapter+query 두arm의seed101 학습을GPU2/3에서시작했다. Seed202/303도같은128-update·fixed-final3noise protocol로등록했다. 자세한입력정보량·GTauthority·학습/eval·자원·실패보존범위는 [현재regional transfer실험](2026-10-18-current-geometry-endpoint-pilot-ko.md)에기록한다.

본14개표의C2-RAW2048과새9채널frozen-controller 실험을구분한다. Query/no-query의실제차원3075/3107·initial loss exact·query gradient연결을검증했다. 새regional 실험의향상과source-use는최종calibration 뒤판정하며아직Full/PBR채택완료로표현하지않는다.

## 과거 실패 이력과 현재 미해결 오류의 구분

과거 queue의 실패 8건은 원본 flow/cache 경로, W&B API 호출, GT voxel 재구성, sampler 인자 오류였다. 각 실패 log를 보존하고 실제 완료된 후속 job과 결과 SHA256을 대조해 `provenance/resolved-historical-queue-failures-v1.json`에 기록했다. 원래 `FAILED` 상태를 지우거나 성공으로 바꾸지 않는다. GT 재구성 실패의 후속은 actual SS-input support와 physical-cell 등록을 모두 확인한 V4이며, 이전 불일치 label을 사용하는 뜻이 아니다.

최종 campaign 집계는 archived log·완료된 replacement·immutable PASS evidence가 확인된 과거 실패와 미해결 오류를 구분한다. 새로운 실패는 이 검토로 자동 해소되지 않는다. 해당 검사는 7개 CPU 테스트를 통과했다. 이 기록은 운영 오류의 수정 이력이며, 원본 constant-gate 학습·새 SS seed의 완료나 Full/PBR 채택을 뜻하지 않는다.
