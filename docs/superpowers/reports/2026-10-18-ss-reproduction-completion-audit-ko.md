# SS matching·원본 모듈 재현의 완료 확인표

2026-10-18 04:41 UTC 최초 작성 이후 **진행 중**이다. 후속 완료 상태는 마지막 절을 따른다. 본 문서는 [원래 SS 계획](2026-10-17-ss-matching-v2-experiment-plan-ko.md), [복원한 전체 모듈 비교](2026-10-17-ss-full-module-restoration-ko.md), [원본 재현 계약](2026-10-18-faithful-module-reproduction-ko.md)의 범위를 함께 확인한다. 큐의 성공 개수나 일부 양의 결과만으로 전체 완료를 선언하지 않는다.

아래 runtime 경로의 기준은 실제 file4의 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/`다. 로컬 mirror의 존재와 실제 file4 SHA256·크기 검증은 구분한다. 각 단계의 PASS는 표에 적힌 범위에 한정한다.

| 요구사항 | 현재 증거와 유효 범위 | 완료 상태 |
| --- | --- | --- |
| 초기 모델·데이터·입력 | Dev-jaehyeok 계열 SS4210의 complete base+LoRA continuation, DeformNeXt world-v4 train9149/val508. Clean80k는 초기 모델의 학습 이력이다. Source-only는 M0 RGB4장, edited 입력은 M1 camera08와 M0 camera09/10/11이며 동일 M1 GT로 평가한다. Actual config/native proof와 strict primary·matched 검사가 이 계약을 확인한다. | 본 비교 완료. 진행 중 run도 같은 계약 유지 필요 |
| 전체 SS 모듈 비교 | `module-study-native-v4/final-primary-report.json`의 `primary_matrix_complete`: 14개 구성, final128×3. 새 Top1 → rejection → score-weighted attention → adapter → RAW query의 다섯 arm을 모두 실제480 updates 학습했다. P1/P2만으로 대체하지 않았다. | 완료. 모듈의 양의 효과를 뜻하지 않음 |
| Arm의 정확한 의미 | No-score는 Top1 선택, native V1은 spatial 기준. Query가 없는 adapter는 입력 채널 자체가 없으며 RAW2048 또는 geometry9 query arm과 구분한다. Retargeter/query의 추가 두 arm을 쓰고2×2로 확대하지 않는다. Actual native/gradient proof와 원본 network6개 actual final weight3075/3107·371gate를 확인했다. | 완료 |
| Matcher·cache 계약 | H0 replay/H1/H2/H3의7-epoch head 실험, train-internal calibration, 고정 H3/T0.1/rejection과 실제 head/cache fingerprint를 본 표에 기록했다. 원본 R9/T0.005/head `4f8771f2…`의 matched 비교와 H3 `066738c9…`를 혼합하지 않는다. H2/H3의38044개 trustworthy-query precision/coverage 전체 curve·같은확률 tie 처리·frozen GPU operating point와 W&B/actual-file4 검증도 완료했다. | 완료. C1 직접 cache/AMP/CFG도 아래 별도 범위로 완료 |
| 원본 Q2C 학습 재현 | `faithful-reproduction-v1/q2c-exact-reproduction-summary.json`: 24개 tuple,28,800개 update의 sampling/loss와 final parameter·held-out RGB component가 원본과 exact. | 완료. 현재 SS 생성 이득과 별개 |
| 과거 geometry 결과 재현 | `faithful-reproduction-v1/geometry-exact-reproduction-summary.json`: 기존6개 checkpoint의144개 endpoint에서 metric 최대 차이0. Full128-update GPU trajectory 재현은 아니다. 원본에서도 projected-source OFF가 더 좋은 한계를 보존한다. | Endpoint 재현 완료. 유용한 source retargeting 성공으로 확대하지 않음 |
| 현재 RGB task 이식 | `task-rgb-transfer-v1/matched-six-run-summary.json`: D0/D1×3seed, 원본 task loss의 RGB error 감소를 확인했다. Teacher exact-patch/feature 개선과 SS 생성 개선은 별도다. | 완료 |
| 현재 per-view 주소의 headroom | Native source를 보존한 GT/wrong bridge에서 동일 support·source value multiset과 actual native ProjGrid를 검사했다. GT−native의 양의 차이는 diagnostic-only다. | GT 진단 완료. 학습된 per-view WHERE의 개선 증거로 대체하지 않음 |
| C1 전체 이득과 retargeter 분리 | `ss-seed-confirmation-v1/c1-vs-v1-final128-audit.json`은 동일 seed101/DDP2/micro48/accumulation1의 V1/C1 final128×3를 확인한다. C1−V1 IoU+0.954pp [95% CI+0.542,+1.350]pp. B3는384 update 검증 뒤 마지막 epoch를 진행한다. | **미완료:** B3 학습·final128×3·strict three-arm 집계 |
| 원본371-parameter gate의 현재 학습 | Scalar pilot을 별도 보존하고 원본 constant-gate MLP를 복구했다.6개 실제128-update checkpoint·90optimizer state finite·physical width와 paired schedule,21case/2016개 prediction·strict summary를 확인했다. Query−adapter macro IoU−0.955pp로 현재 추가 효과는 실패했다. | **완료:** 전체 실제 W&B/actual-file4 closure 포함. Module gain/PBR 채택을 뜻하지 않음 |
| 현재 C1의 active-path cache/AMP/CFG | [직접 검증 결과](2026-10-18-ss-conditioning-contract-ko.md)의실제 trained C1 두 사례·BF16·cache/online·CFG p0/p1/mixed 검사가 완료됐다. v1은 assertion 통과 뒤 NaN 통계 표시 문제였고 verifier만 수정한v2는 finite 값·동일mask를 기록했다. CPU12개 회귀 PASS와 W&B22개 지표/4개 artifact 실제 download·file4 receipt가 있다. | **완료:** `conditioning-contract-v2/summary.json`·`independent-audit.json`. 두-case 계약 범위 |
| Sampler 민감도 | [실제8개 진단](2026-10-18-ss-sampler-sensitivity-ko.md)과 strict 집계가 끝났다. 네 fresh12가 primary count를exact 재현했고50−12 IoU는모두−1.1~−1.4pp였다. W&B32개 metric·8개 PNG·4행 table·11개 artifact 실제 download와 actual-file458개 산출물을 검증했다. | 완료. Primary12-step/final128×3 유지 |
| 빠른 평가·자원·로그 | SS는 voxel metric과 GT/Pred/Diff를 사용한다. GPU0~3만 배정했고 새 본 학습은 global96/DDP2/micro48/accumulation1이다. 완료 산출물의 W&B metric/image download와 actual-file4 SHA receipt가 있다. B3 epoch2 이후와 후속 run의 closure는 계속 필요하다. | 완료된 부분만 검증됨 |
| Git·소스 보존 | 실제 local/remote `wowo0709/feature-retargeting`과 기존 remote `feature-retargeting`에verifier 수정9bb2643까지 push했다. Source archive+patch를 실제 임시 checkout에 적용해9bb2643의17개 변경 파일과 byte 일치를 확인했다. | 해당 snapshot 완료. 최종 문서 commit의 push/readback은 계속 필요 |
| 과거 FAILED/HELD 처리 | 원래8개 FAILED log와 완료된 후속을 SHA로 재검토해8 resolved/0 unresolved였다.59개 HELD의 원래 JSON state도 모두 그대로이고,17/14/22/6개 군의 required replacement가 모두 완료됐다. 마지막6개군도 network-gate6개 실제 학습/평가/remote/file4 closure를 확인했다. | 해당 replacement 검토 완료. 전체 campaign 최종 상태에서 재확인 필요 |
| 최종 SS 구성·Full/PBR 판단 | 기존14개, matched V1/B3/C1, 원본 geometry3seed, source/address control, QA 및 sampler 진단을 함께 보고 채택/보류와 남은 한계를 명시해야 한다. | **미완료:** 최종 판정·통합 보고 |

## 최종 판정에서 구분할 내용

Final128×3는 checkpoint480의 고정 평가이고,32개 epoch 관측으로 best epoch를 다시 고르지 않는다. Cluster bootstrap은 동일 source의 deformation와 noise 반복을 함께 두며, 한 training seed의 조건부 CI를 training-seed 모집단 CI로 표현하지 않는다. 기존 seed0은 microbatch/DDP 실행 조건이 달라 새seed101과 하나의 엄밀한 two-seed 추정으로 합치지 않는다.

C1−V1의 전체 경로 개선, B3−V1의 fusion 효과, C1−B3의 retargeter 추가 효과를 각각 보고한다. Same-support wrong-address control의 CI가0을 포함하는 사실과 wrong-object/adapter-OFF 결과를 혼합해 정확한 correspondence 주소의 유용성을 입증했다고 쓰지 않는다. 원본 Q2C RGB task 개선도 이 SS 세 비교의 대체 근거가 아니다.

Geometry controller의 macro IoU/actual SS-input GT, source recall/stale leak와 본 SS 표의 micro IoU/decoded aligned GT를 분리한다. Cleanup이 좋아져도 유지할 source 영역을 잃고 net IoU가 개선되지 않으면 그 tradeoff를 그대로 보고한다. Scalar pilot의 결과를 원본 network-gate6개의 완료나 성능으로 재사용하지 않는다.

Full을 C2로 미리 확정하지 않는다. SS에서 확인된 추가 이득과 재현 범위를 근거로 구조를 선택하고 PBR 확대 또는 보류 이유를 명시한다. 이 SS 실험은 논문의 rendering10개 metric/PBR 평가를 수행했다는 증거가 아니다. Negative 결과도 보존하며, 미완료 검증을 생략하거나 추가 epoch/loss 변경으로 양의 결과를 강제하지 않는다.

## 현재 큐와 남은 실행

실제 큐316개 가운데 completed228/running3/pending18/HELD59/FAILED8이었다. Running은 matched B3와 regional seed202의 두 arm이다. Pending18개에는 B3 final3·three-arm 집계, regional seed303 두 arm·six-run 집계, current C1 QA, sampler8개·집계·publication이 모두 들어 있다. 이 숫자는 당시 snapshot이며 raw59/8개를 미해결 실패 수로 읽지 않는다. 최종 완료에는 큐 밖의 실제 checkpoint/logging/storage 확인과 위의 통합 판단도 필요하다.

04:44 UTC 후속: B3의 epoch2는 실제 optimizer·W&B history/image·actual-file4 checkpoint closure와 두 paired 비교36개 remote numeric field까지 확인했다. Regional seed202도 두 arm의128-update checkpoint·optimizer 전체 finite·실제 channel/gate/schedule 검사와 actual-file4 저장을 통과했다. 평가와 seed303는 계속 남는다. 근거는 `provenance/matched-B3-second-epoch-{audit,paired-wandb-remote,file4}.json`, `provenance/network-v3-training-checkpoints-independent-seed202{,-file4}.json`이다.

05:02 UTC 후속: 원본 network-gate seed202의 두 arm 평가가 모두 끝나6개 중4개가 완료됐다. 실제 SS-input GT672개 기록·macro metric 재계산·두run W&B 지표와 이미지·actual-file4 저장을 확인했다. Seed303 두 process도 실제GPU2/3에서 first8 native/paired schedule/finite 검사를 통과했다. 실제 trained query의 CPU BF16 local-module output·18개 gradient parity도 추가 확인했다. `provenance/regional-network-seed202-complete-and-both303-start-file4.json`을 따른다. Seed303 평가·six-run 집계, B3 fixed-final, current C1 QA, sampler 진단과 최종 구성/PBR 판단은 아직 미완료다.

05:11 UTC 후속: B3 epoch3의 실제 optimizer/GT/noise/두rank 계약과 W&B metric/image·actual-file4 checkpoint closure, 두 paired 비교36개 수치를 확인했다. Step288 IoU는V1/B3/C1 순서로0.431872/0.434638/0.443624다. Pending final noise2는GPU2 진단 종료 뒤GPU2를 사용하도록 자원 배치만 변경했고316개job과 sampler/input 계약을 보존했다. No-query seed303는128-update 고정 marker가 생성돼 평가 중이며 query303는 계속 학습한다. 두303 checkpoint가 모두 고정되면 독립 CPU optimizer/channel/schedule 및 actual-file4 검사를 수행하도록 대기 중이다.

최근 계약의 실제 test 실행 증거도 보완했다. `cpu-contract-validation-3f0f8ec`의7개 test file에서 **39개 PASS, failure/error/skip0**였고 stdout·JUnit·소스 commit·CUDA 비가시·CPU2개 제한을 저장했다. 실행 source는 검증한c165372 이후 문서만 바뀐 상태다. 세 증거 파일8,476bytes의 actual-file4 SHA256·크기는 `provenance/cpu-contract-validation-3f0f8ec-file4.json`에서 확인했다. 이 CPU 검사로 pending 실제 GPU QA·성능 검증이나 전체 완료를 대체하지 않는다.

05:18 UTC 후속: 원본 network-gate6개 run의 실제128-update checkpoint가 모두 고정됐고,90개 optimizer 상태·전체 trained weight finite·물리적 query 폭·원본371 gate·seed별128 paired schedule/native loss와 actual-file4 저장 검사가 완료됐다. `provenance/regional-network-all-six-training-only-closure-v1.json`과 `regional-network-all-six-training-only-closure-file4.json`이 근거다. 마지막 두 calibration 평가와 six-run 통계는 아직 진행 중이며, 이 training-only closure를 module-gain/Full/PBR 또는 전체 goal 완료로 읽지 않는다.

05:40 UTC 후속: 원본 network6개 모두 calibration·strict 집계까지 끝났고,W&B126개 metric/21개 실제 PNG/54행 aggregate table과2개 artifact 파일을 독립 readback했다. Actual-file4101개 산출물41,602,877bytes가 일치했다. Query−adapter macro IoU−0.955pp로 추가 개선은 실패했고,train-loss/gradient/schedule까지 분석했다. 현재 trained C1의 QA-v2도 실제 GPU PASS,CPU12개 회귀 PASS,W&B22개 scalar와4개 artifact 파일·file49개 산출물36,963bytes 검증을 완료했다. v1의 NaN은 기존 assert_close 통과 뒤의 통계 표시 문제였으며 모델·학습 수식은 바꾸지 않았다.

B3는epoch4 검증과 actual361 optimizer/step384·file4 checkpoint closure까지 통과했고 마지막 epoch를 진행한다. 같은 epoch의 V1/B3/C1 quick IoU는0.434300/0.436509/0.450249다. 두 paired 비교는 각각+0.221pp[−0.295,+0.712]pp와+1.374pp[−0.139,+2.963]pp로 탐색적중간값이며 fixed-final을 대신하지 않는다. Sampler 마지막 pending C2 pair는 반환된GPU2로 함께 이동했고 모든317개 required DAG/command/weight/panel을 보존했다. 현재 남은 required 작업은 B3 fixed-final128×3·strict three-arm, sampler의 나머지 평가/집계/remote closure,최종 SS/PBR 판단과source/FAILED·HELD 후속 확인이다. [성공/실패 원인 분석](2026-10-18-module-reproduction-cause-analysis-ko.md)은 확인된 구현/계약 차이와 아직 분리되지 않은 원인을 구분한다.
