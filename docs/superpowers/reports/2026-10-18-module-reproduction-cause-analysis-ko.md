# 모듈 재현: 성공한 경로와 실패한 이식의 원인

2026-10-18의 실제 실행 결과를 기준으로 한다. 이전 양의 결과의 수치 재현, 현재 데이터에서의 task 개선, 현재 SS 생성 개선을 각각 판정한다. 같은 모듈 이름이나 비슷한 layer만으로 재현했다고 판단했던 부분을 수정했다. 최종 matched B3의128×3 평가와 sampler 진단은 아직 진행 중이므로 retargeter 단독 효과와 최종 Full/PBR 판단은 그 뒤 확정한다.

Runtime 기준 경로는 실제 file4 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/`다. 원본 multi-view mixed385 base+LoRA를 이어받고 DeformNeXt world-v4 train9149/val508을 사용한다. Clean80k를 deformation 학습에 추가하지 않는다. 학습은 GPU0~3만 사용한다.

## 무엇이 재현됐고 무엇이 재현되지 않았나

| 대상 | 실제 재현 결과 | 허용되는 결론 |
| --- | --- | --- |
| 원본 Q2C matching-aware context | 24개 학습·28,800 update의 sampling/loss·final parameter·OOF RGB component가 과거와 exact. RGB L1 0.03676912→0.03512899 | 과거 RGB task의 Q2C 추가 이득 재현 완료 |
| 현재 DeformNeXt의 같은 RGB task | D0/D1×3seed에서 L1 0.09008583→0.08467440,6.01% 감소,3seed 모두 개선 | 현재 task에서도 이득. DINO 정확도나 SS 생성 이득의 증명은 아님 |
| 과거 G2 geometry-content | 기존6개 checkpoint의144 endpoint, 모든 metric/nested BCE가 원본과 exact. Recall32+4.075pp 재현 | 과거 checkpoint의 상대 geometry 효과 재현. 전체128-update 재학습 trajectory exact는 주장하지 않음 |
| 현재 원본 R9 C1 전체 경로 | 동일 seed101/DDP2/micro48/accumulation1,final128×3에서 V1 IoU0.432650→C1 0.442193 | 전체 경로+0.954pp,CI[+0.542,+1.350]pp. Retargeter만의 이득은 B3 final 대기 |
| 새 H3/rejection/RAW-query ladder | 본14개 비교의 final128×3 완료. Attention/adapter/RAW query 추가 이득 미확인 | 과거 성공 조건을 그대로 재현한 실험이 아니었음 |
| 현재 predicted O/D9 geometry query | 원본371-gate·regional loss로 두 arm×3seed를128 updates 학습. Query−adapter macro IoU−0.955pp | 원본 parameterization 복구 뒤에도 현재 calibration에서 추가 효과 실패 |

Q2C는 [수치 재현 기록](2026-10-18-faithful-module-reproduction-ko.md), 현재 task는 [61-cluster 결과](2026-10-18-task-rgb-matcher-transfer-ko.md), C1은 [matched SS 기록](2026-10-18-original-SS-seed-confirmation-ko.md), geometry는 [원본 network-gate 결과](2026-10-18-original-network-gate-reproduction-ko.md)를 따른다. 각 표의 GT·metric·source population이 달라 수치를 합쳐 집계하지 않는다.

## 확인된 실패 원인: 재현 조건과 ablation 질문이 달랐다

### Q2C 효과와 Top1 대 soft 효과를 혼동했다

기존 새 ladder의 Top1과 soft-attention은 **같은 contextual H3/Q2C head**를 썼다. Top1에도 Q2C score가 이미 들어가므로 두 arm의 작은 차이는 Q2C ON/OFF 검사가 아니다. 사용자의 No-score 정의인 matched Top1 하나를 유지하면서, Q2C 자체는 원래처럼 동일 후보·동일 task loss의 D0-pairwise/D1-Q2C로 별도 검증하도록 구현했다. 이 직접 대조에서는 과거와 현재 task 모두 개선이 재현됐다.

R9의 원래 T0.005에서는 Top1 weight 평균이0.9336이고64후보 중 winner에 weight가 강하게 몰린다. Top1 대 soft의 작은 입력 차이도 확인했다. 이것은 작은 추가 효과와 양립하는 설명이며 attention 자체의 무효나 수학적 동등성을 뜻하지 않는다.

### 학습 목표와 confidence 사용 방식이 달랐다

과거 성공한 head는 RGB task L1을 최적화했다. 새 H3는 ranking CE·선택 patch의 exact correctness confidence·DINO value loss를 사용한다. 정확한 patch 선택과 consumer에 유용한 soft evidence는 같은 목표가 아니다. 실제 현재 task에서도 Q2C의 RGB 개선은 유의했지만 DINO error/Top1 정확도 추가 개선의 CI는0을 포함했다.

R의 threshold0.276316212는508개 validation의 foreground query 중11.33%만 통과시켰고 foreground 평균 usage weight는0.04277이었다. Fixed128에서는9.44%만 통과했다. 기존 실제 pack 진단에서도 No-R에 비해 추가 projection의 크기가 작아졌다. **이것은 경로를 강하게 억제했다는 직접 관측**이다. H3 objective·후보 수·temperature·rejection이 함께 달라진 비교에서 R 하나만을 최종 SS 하락의 원인으로 확정하지는 않는다. Validation을 보고 threshold를 낮추거나 다시 학습하지 않았다.

### 2D anchor residual이 과거 per-view correspondence 경로를 재현하지 않았다

과거 성공한 GT-CORR은 source camera별 target-state write slot에 각 source DINO 값을 놓고 각 camera의 native ProjGrid를 사용했다. 현재 patch conditioner는 evidence를 target anchor map으로 모아 투영한다. 보간 방식을 같게 만든 것만으로 세 source-camera의 공간적 재배치를 재현한 것이 아니다.

같은 현재 frozen SS4210·32개·noise·CFG3에서 이 차이를 직접 검사했다. 원래 target-fallback GT replacement IoU는0.32404648로 native0.42671547보다 낮았다. 같은 GT routing을 native source 보존 방식으로 연결한 bridge에서는0.44631358이었다. GT−native는+1.960pp,CI[+0.517,+3.457]pp이고 GT−wrong-address는+6.553pp였다. **Native fallback/소비 경로가 결과를 바꿀 수 있음**은 직접 대조로 확인됐다.

이 결과는 GT 주소를 사용한 diagnostic이다. 현재 학습된 matcher의 precise WHERE 이득이나 deployable per-view bridge가 완성됐다고 쓰지 않는다. 원본 C1의 동일-support wrong-address control도 normal 이득의 CI가0을 포함했다. Wrong-object 또는 adapter-OFF의 하락만으로 정밀한 correspondence 주소의 효과를 증명할 수 없다.

### RAW2048 query와 원본 geometry9 query가 달랐다

기존 C2는 Omega FRAME+GLOBAL delta2048을 patch MLP에 추가했다. 과거 G2는 predicted original/desired SS logit·occupancy·SDF와 차이,총9채널을16³ voxel content generator에 넣고 frozen full rollout/decoder의 regional BCE로 학습했다. RAW feature2048을 단일-timestep flow MSE로 학습한 결과는 이 G2의 재현이 아니었다.

이 차이를 수정해 predicted O/D9·원본 regional loss·constant-gate MLP를 구현하고 actual GPU native/gradient proof를 거쳤다. No-query는 encoder와32개 query embedding 열 자체를 제거해3075채널이고, query는3107채널이다. 0을 의미 있는 geometry 관측으로 넣는 ablation을 사용하지 않는다. Scalar1 gate로 원본371-parameter MLP를 대체했던 차이도 수정했다. 원본과6회 CPU AdamW update의 모든 parameter exact와 실제 trained BF16 local-module output/18개 gradient exact를 확인했다.

## 수정 후에도 query가 실패한 이유는 어디까지 확인됐나

원본 parameterization으로 실행한6개 controller는 실제 train32/cal32·3model seed·3sampling noise·128 updates의 고정 비교다. 모든 checkpoint·90개 optimizer 상태가 finite/step128이고 paired asset/noise schedule과 초기 native loss가 동일했다. Full backward 연결도 실제 GPU로 확인했다.

| 현재 original-network 방법 | Macro IoU | Source recall32 ↑ | Stale leak32 ↓ | Regional BCE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Frozen native | 0.445530 | 0.825740 | 0.601524 | 95.616471 |
| Adapter-only | 0.447883 | 0.816588 | 0.549600 | 91.920679 |
| Adapter+query | 0.438332 | 0.814673 | 0.554092 | 93.545370 |

Query−adapter IoU는 seed101/202/303에서−0.074/−1.363/−1.428pp였다. 세 seed 평균−0.955pp,source-cluster CI[−1.400,−0.528]pp다. Intact−joint-query-shuffle은−0.027pp,CI[−0.082,+0.020]pp로 공간 query 사용의 추가 이득도 확인되지 않았다. 이 CI는 이3개 학습 seed에 조건부인32 source-cluster 추정이며 training-seed 모집단 CI나 unseen-test 결과가 아니다.

실제 같은 asset/noise로 학습한 마지막32 updates의 query−adapter 평균 train loss는 seed별−3.708/+11.209/+3.536이었다. Seed202는 query loss가 pass1의82.436에서 pass4의93.673으로 올라갔다. Seed303도 마지막 pass에서 No-query보다 나빠졌다. 따라서 **validation에서만 나쁜 단순 과적합이나 에폭 부족으로 설명되지 않는다.** Seed101은 regional BCE와 stale 제거는 좋아졌지만 source recall도3.369pp 낮아지는 tradeoff가 있었다. 기존 source-prior 누락률만으로 모든 seed의 실패를 설명하는 가설도 seed202의 반대 방향 source recall 때문에 충분하지 않다.

전체768 update가 clip norm1에 걸렸다. Query의 마지막 pass gradient 중앙값은588/741/880으로 No-query380/427/599보다 컸지만, 양쪽 모두 clipping돼 이것만으로 query 구현 오류나 clipping이 단독 원인이라고 판단하지 않는다. Gate도 약0.46~0.50으로 유지됐고 gradient가 끊기거나 모두0이 된 상태가 아니다. 근거는 `provenance/regional-network-paired-training-diagnosis-v1.json`이다.

현재 확인할 수 있는 판정은 **원본 구조를 이식한 이 local controller가 현재 consumer·데이터·loss 조합에서 안정적인 net gain을 만들지 못했다**는 것이다. 과거의 consumer는 full-FT SS20k/CFG7.5/rescale_t5,현재는 LoRA SS4210/CFG3/rescale_t1이다. 데이터와 physical No-query 정의도 다르다. 각각을 고정한 교차 실험을 하지 않았으므로 consumer·dataset·학습 목표 중 하나를 단독 원인으로 단정하지 않는다. 현재 initializer를 과거20k로 바꿔 양의 수치를 만드는 방식도 사용하지 않는다.

## 성공 원인과 지금 적용한 재현 조건

아래는 같은128개·93 source cluster·3sampling noise·decoded aligned GT/micro 집계의 기준점이다. 첫 두 행은 같은 multi-view 모델의 입력만 바꾼 zero-shot이며, 뒤 두 행은 같은 seed101/480-update 학습이다.

| 같은 SS 평가 기준 | Voxel IoU |
| --- | ---: |
| Zero-shot source-only: M0 RGB4장 | 0.380711 |
| Zero-shot edited: M1 RGB1장+M0 RGB3장 | 0.419275 |
| Vanilla V1 fine-tuning | 0.432650 |
| Original R9 C1 fine-tuning | 0.442193 |

Edited 입력의 개선+3.856pp,그 위 V1 fine-tuning의+1.337pp,C1−V1의+0.954pp를 구분한다. 이는 기존 multi-view 모델도 deformation 입력 충돌에 완벽하지 않다는 증거이며, 현재 C1이 모든 형상을 충분히 복원했다는 뜻은 아니다. Source-only도 undeformed target camera08을 포함한 네 장이고, edited camera08도 depth가 아닌 RGB다.

현재 SS에서 확인된 성공은 원본 R9 C1 전체 경로를 유지하고, native 초기 condition과 complete base+LoRA를 정확히 보존한 상태에서의 학습이다. 동일 budget/execution의 V1 대비 final128×3 이득+0.954pp는 세 sampling noise 모두 양수다. 이 결과가 H3/rejection/RAW query 조합의 개선을 뜻하지 않는다. 원본 head·Top64·temperature·adapter·초기 gate·projection과 현재 데이터의 RGB/camera/latent 계약을 고정했다.

실제 trained C1의 cached/online evidence·condition, BF16, CFG p0/p1/mixed mask도 통과했다. Active projection RMS0.023112로 모듈은 실제로 동작했고 입력/weight fingerprint가 같았다. v1의 같은 infinity 차감에서 생긴 NaN은 이미 통과한 `assert_close` 뒤의 통계 표시 문제였다. Verifier만 수정한 v2가 finite 통계와 동일 mask를 기록했다. 이를 모델 버그나 성능 개선의 원인으로 서술하지 않는다.

다만 C1 전체 이득의 어느 부분이 matcher/fusion이고 어느 부분이 retargeter인지는 **동일 조건 B3 final**이 필요하다. 같은 seed는 training의 모든 random draw를 bit-exact로 짝지었다는 뜻이 아니며, fixed-weight sampling noise 반복을 독립 training seed로 취급하지 않는다. 현재 B3는480 updates를 마치는 중이고 C1/V1 weight를 다시 변경하지 않는다.

## 재현 가능한 코드와 다음 판정

후속 [sampler8개 진단](2026-10-18-ss-sampler-sensitivity-ko.md)도 완료됐다. 같은 네 checkpoint에서50−12 IoU는모두−1.1~−1.4pp였다. 이 모델들의 낮은 효과를 sampling12steps 부족으로 설명할 근거는 없었다. [Confidence 곡선](2026-10-18-matcher-confidence-coverage-ko.md)은 기존 fixed head의 실제 값을 재계산했고 threshold를 다시 선택하지 않았다. H3의 후보 recall93.16%와 Top1 23.71%의 차이, frozen operating point의 precision33.16%/coverage11.41%를 확인해 candidate miss와 reranking/utility bottleneck을 구분했다. 해당 confidence 분모38,044와 SS cache foreground38,366은 합치지 않는다.

원본 Q2C는 `historical_q2c_training.py`/`reproduce_q2c.py`,원본 G2 수식과 endpoint는 `historical_reproduction.py`/`historical_geometry_runtime.py`/`reproduce_geometry_content.py`,현재 physical query는 `geometry_content_adapter.py`/`train_current_geometry_endpoint.py`에 구현돼 있다. Source AST authority·실제 입력/weight SHA·실행 command는 각 runtime proof에 고정돼 있으며 과거 worktree를 실행 중 import하지 않는다. Source archive+patch는 실제 임시 checkout에 적용해 재구성했고 현재 verifier source까지 byte 일치 검사를 남겼다.

1. 완료된 원본 수치 재현과 현재 task 개선을 보존한다. 기존14개와 scalar pilot도 삭제하거나 양의 결과로 다시 표시하지 않는다.
2. 진행 중 B3의 same-final128×3를 마쳐 B3−V1,C1−B3,C1−V1을 분리한다. 중간32 수치로 best epoch를 고르지 않는다.
3. 계획된4개 checkpoint의12/50-step 진단을 끝내 sampler가 생성 경향을 왜곡했는지 확인한다. Primary12-step 비교는 유지한다.
4. Query는 현 상태로 Full에 포함하거나 장기/PBR 학습으로 확대하지 않는다. Source 보존과 같은-information shuffle gate를 통과한 모듈만 다음 후보로 삼는다.
5. SS 최종 선택과 PBR 확대/보류를 통합 리포트에 명시한다. Learned per-view WHERE의 개선은 아직 구현/검증되지 않은 후속이며 GT 진단의 양의 수치로 대체하지 않는다.

Original-network6개의 strict 결과·2,016개 평가 기록·W&B126개 지표/21개 실제 PNG·54개 contrast table/aggregate artifact와 raw checkpoint는 `provenance/regional-network-all-six-evaluations-and-readback-v3-file4.json`에서 actual-file4 SHA256/크기를 확인했다. C1 계약은 `provenance/current-C1-QA-v2-completed-file4.json`을 따른다. 이 문서는 아직 미완료인 B3/sampler/최종 결정을 완료로 표시하지 않는다.
