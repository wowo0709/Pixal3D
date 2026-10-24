# 남은 필수 실험·평가와 논문 작성 계획

최신 결과와 모듈별 대조 정의는 [최종 파이프라인 필수 실험·평가·writing 정리](2026-10-24-final-pipeline-essential-experiments-and-writing-ko.md)에 갱신했다. 아래는 13:35 UTC 시점의 계획 기록이다.

추가 갱신: [Learned cascade32 완료와 동일 panel 진단](2026-10-24-learned-cascade-complete-and-same-panel-diagnosis-ko.md)을 반영한다. 새 training arm보다 기존 R/Fixed/Learned 세 pipeline을 동일 DEV128/noise로 비교하는 평가 확장을 우선 준비하며 기존32개는 재사용한다. 이 확장은 아직 실행 완료가 아니다.

2026-10-24 13:35 UTC 실제 상태 기준. 목적은 개별 요소의 조건부 효과를 확인하고, 효과가 있는 요소로 구성한 실제 파이프라인을 검증하는 것이다. 모든 모듈이 필수라는 결론을 미리 정하지 않는다. 기존 실험을 재사용하고 최종 구조에서 빠진 대조군만 보완한다.

## 1. 현재 진행 중인 필수 비교를 마무리

| 작업 | 남은 실행 | 판단할 질문 |
|---|---|---|
| FULL_VOXEL_Q | 480-step 학습 및 DEV128×3 평가 | 동일 조건에서 voxel query와 ray query의 차이 |
| FULL_MEAN_MATCH | 480-step 학습 및 DEV128×3 평가 | 평균 descriptor 대신 member 기반 후보를 사용하는 효과 |
| E_LEARNED_RAW | 480-step 학습 및 DEV128×3 평가 | 같은 carrier/use/R·identity V에서 learned Q/K·utility의 추가 효과 |
| 기존 core12·followup 집계 | 실제 평가 완료 후 paired 통계 | 점추정과 사전 정의한 contrast-family의 불확실성 구분 |

현재 입력·LoRA 초기 checkpoint·학습 budget·sampler·평가 sample/GT/noise pairing을 유지한다. 완료된11개 SS 모델을 다시 처음부터 학습하지 않는다. 생성 noise3개를 독립 training seed3개로 해석하지 않는다.

## 2. 실제 파이프라인을 보고 최종 후보를 좁히기

V1/R joint와 PBR4개 모델의 고정 mesh DEV32 평가는 완료됐다. D1-cos+R도32개 생성과8개 metric 집계가 완료됐다. D1의 높은 SS IoU가 source LPIPS/DreamSim·CD까지 우위로 이어지지는 않았다. E_FIXED_RAW+R cascade는 진행 중이다.

남은 순서는 다음과 같다.

1. E_FIXED_RAW+R의 DEV32 cascade·8개 metric을 완료한다.
2. 기존 R, D1-cos, E_FIXED_RAW 및 이미 평가한 FULL을 target CD/F-Score와 source LPIPS/DreamSim을 중심으로 비교한다. SS IoU나 source CLIP 하나로 고르지 않는다.
3. E_LEARNED_RAW가 최종 SS 평가에서 유력하면 이 후보의 cascade만 추가한다. 기존 후보에 추가 이득이 없으면 평가 조합을 늘리지 않는다.
4. 최종 SS 후보에서 PBR vanilla와 Top-1 R을 짝비교한다. 현재 PBR4개 비교는 V1 SS geometry에서의 결과이므로 다른 SS에서도 같은 선택이 최적인지 확인해야 한다.

4번은 기존 native Shape·guide·SS prediction을 재사용한다. 필요한 texture replay만 같은 초기 noise로 실행하고, 같은 실제 rendering triangles를 사용한다. 기존 Shape cascade부터 다시 생성하지 않는다. 기존 조합이 이미 이 조건을 충족하면 재평가 없이 사용한다. Native cascade의 기존 texture RNG와 seed-reset replay를 섞어 비교하지 않는다.

현재 DEV32 결과는 Raw PBR의 추가 joint/seed202 확장을 지지하지 않는다. 우선 PBR vanilla/R과 등록된 R 중심 joint 후보로 범위를 유지한다. 최종 pipeline은 stage별 증가량을 더해 추정하지 않고 직접 실행한 결과로 결정한다.

## 3. 최종 구조 안에서 필요한 ablation만 보완

기존 ladder는 해당 설정에서의 효과를 보여준다. 최종 구조가 달라지면 그 효과를 그대로 최종 모델의 필수성으로 일반화하지 않는다. 각 행에 대해 기존 arm이 같은 구조·초기화·budget·입력/평가 조건인지 먼저 확인하고, 같으면 재사용한다. 비교가 빠진 경우에만 SS stage에서 matched 학습을 추가한다. 추가 개수는 최종 구조와 이 재사용 점검 이후 확정한다.

| 최종 문맥에서 검증할 요소 | 대조군 정의 | 재사용할 기존 근거 |
|---|---|---|
| Semantic correspondence | 원래 spatial correspondence로 복귀 | C_SPATIAL/C_RAW |
| Matching decoder | 같은 semantic 후보 수에서 raw descriptor와 학습 descriptor 비교 | C_RAW/C_D1_COS/C_DECODER; descriptor와 scalar score는 분리 |
| Member 기반 후보 구성 | 같은 matching/전달 경로에서 ray의 member 후보를 평균 descriptor로 대체 | FULL_MEAN_MATCH/FULL; 최종 구조에 member 설계를 유지할 때만 필요 |
| Top-K aggregation | 실제 Top-1 correspondence feature 하나만 사용 | C_TOP1_V/C_FIXED_TOPK; 최종 모델과 동일한 경로인지 확인 |
| Ray 단위 query/evidence | 개별 voxel 단위로 변경 | FULL_VOXEL_Q/FULL; 최종 구조가 다르면 필요한 matched counterpart만 보완 |
| Retargeting | R 모듈을 물리적으로 제거하고 나머지 경로 유지 | C_DECODER_NO_R/C_DECODER; 최종 문맥과 같지 않으면 보완 |
| Learned probability refinement | 같은 evidence·identity V에서 refinement만 제거 | E_LEARNED_RAW/E_FIXED_RAW; 이미 진행 중 |

No-score는 uniform attention이 아니라 실제 Top-1이다. No-corr는 무작위 feature가 아니라 원래 spatial correspondence다. 제거한 query/channel을0으로 채워 대체하지 않는다. 최종 모델에 없는 learned head나 deformation-query를 입증하기 위한 arm도 추가하지 않는다.

Matching-aware attention의 효과도 분리해서 쓴다. Top-1 대 score-weighted Top-K는 후보 수와 결합 방식의 **묶음 효과**다. Scalar matching score 자체의 이득을 주장하려면 같은 후보·value·전달 경로에서 score 사용만 다른 대조가 필요하다. 기존 D1-cos/decoder 비교와 최종 경로의 일치 여부를 먼저 점검한다. 같은 Top-K의 uniform 결합을 보완 대조로 사용하더라도 이를 사용자가 정의한 No-score arm으로 부르지 않는다. Frozen matching posterior를 이용한 attention과 추가로 학습하는 Q/K·utility head의 효과도 구별한다. 후자의 최종 성능 이득이 없으면 최종 모델과 핵심 claim에서 제외한다.

공통 posterior의 **일관된 전달 자체가 성능 향상의 원인**이라는 별도 claim을 채택한다면, 같은 후보·가중치·R·V를 유지한 split-evidence 대 shared-evidence 비교가 추가로 필요하다. 이 matched 근거 없이 기존 FULL 대 E_FIXED_RAW의 여러 차이를 하나의 원인으로 설명하지 않는다. 이 claim을 채택하지 않으면 공통 posterior 사용은 구조 설명으로 한정한다.

### Ablation 평가

모든 필요한 SS arm은 동일 DEV128×3의 voxel IoU/precision/recall/F1로 평가한다. 논문에 들어갈 최종 구조의 대조군은 같은 고정 panel에서 frozen Shape512/1024 및 공통 downstream 설정으로 CD/F-Score도 측정한다. Source-preservation claim에 쓰는 대표 비교는 rendering LPIPS/DreamSim까지 측정한다. 원래14개 모델 전체를 모든 PBR 조합으로 확장하지 않는다.

Ray의 효율성까지 주장할 경우 기존 checkpoint로 query unit 외에 precision·batch·FFN chunk·sampler를 맞춘 latency/peak-memory 측정을 한 번 수행한다. 현재 학습 runtime만으로 ray 구조의 순수 속도 향상을 계산하지 않는다.

## 4. 필요한 기전 진단은 작은 고정 panel에서

최종 후보와 관련된 기존 matching cache를 재사용해 raw/learned descriptor의 Recall@1/Recall@K와 correspondence 오차를 확인한다. Top-1이 틀리지만 정답이 Top-K에 있는 경우에 실제 weight가 유효 후보를 활용하는지도 분리해서 본다. 이는 대응 탐색 개선과 최종3D 개선을 연결하기 위한 보조 근거다.

최종 downstream 이득이 없으면 큰 arm을 늘리기 전에 같은 고정 panel에서 score/use를 고정한 correspondence-value 교란 등으로 후보 탐색·가중 결합·전달 중 어디가 병목인지 확인한다. 필요시 oracle은 명시적인 진단 상한으로만 사용하며 deployable method나 TEST 성능으로 보고하지 않는다. 추론 시 모듈을 끄는 진단은 matched 재학습 ablation을 대신하지 않는다.

Target conditioning 경로 불변, 초기 baseline 함수 복원, matching이 없을 때 native source 유지, 동일 SS 재사용 시 실제 render mesh 동일성도 기능 검사로 확인한다. 이는 새 학습 arm이 아니다. In-memory final render adapter는 CPU 검사를 통과했으며 실제 DEV GPU compatibility 확인과 새로운 source-lock closure가 남아 있다. TEST는 이 확인과 최종 freeze 이후에만 실행한다.

## 5. 독립 확인과 최종 TEST

1. 독립 training seed202로 V1/R baseline과 최종 후보를 확인한다. 후보가 baseline과 같으면 중복 학습하지 않는다. 모든 ablation을 여러 seed로 반복하지 않는다.
2. PBR vanilla/R의 필수 seed202 확인을 마친다. 선택되지 않은 Raw/feature PBR의 seed 반복은 추가하지 않는다.
3. 예약 VAL36에서 최종 후보와 필요한 baseline의 fresh-load 재현성·cache/checkpoint 정합을 확인한다.
4. 실제 모델 조합·checkpoint·metric·camera·sampler·통계/claim 기준을 동결한다.
5. TEST508을 source-only zero-shot4, edited-target 포함 zero-shot4, vanilla V1, R baseline, 최종 Ours에 대해 실행한다. Ours가 기존 조합과 같으면4개로 중복을 제거한다. Source-only는 M0의4개 view이고, edited 입력은 M1 target RGB1개+M0 source3개다. Target-only는 추가하지 않는다.

## 6. Metric과 claim의 역할을 사전에 구분

| 평가 목적 | 논문에서 중심으로 볼 지표 | 함께 보고할 지표 |
|---|---|---|
| 3D target alignment | CD↓, surface F-Score↑ | Target PSNR↑, SSIM↑, M-IoU↑ |
| Source appearance preservation | Source LPIPS↓, DreamSim↓ | Source CLIP↑ |
| Global rendering naturalness | FID↓, KID↓ | 최종 전체 panel의 global 값으로 보고 |
| SS 단계 진단 | Macro voxel IoU↑ | Precision/recall/voxel F1 |
| Correspondence 기전 | Recall@K·오차·attention 활용 | 논문 주 결과10개 metric과 별도 |

요청한10개 metric은 최종 main table에서 모두 보고한다. 양의 결과가 나온 metric만 사후에 primary로 바꾸지 않는다. Source view9/10, target view8, 동일한 original material on M1 reference를 유지한다. M-IoU는 재질 opacity도 반영하는2D alpha mask이고, surface F-Score와 voxel F1을 혼동하지 않는다.

CD/F-Score는 같은 normalization·surface sampling·threshold로 측정하고 결과별 ICP를 하지 않는다. Paired 통계는 source cluster를 단위로 수행한다. FID/KID를 asset별 지표처럼 취급하거나 음수 KID를0으로 자르지 않는다. 현재 DEV32의 보정 전 CI를 최종 TEST 통계로 재사용하지 않는다.

## 7. 논문 표·그림·writing 산출물

**Main Table:** 위4/5개 실제 pipeline을10개 metric의 세 목적별 column group으로 비교한다. Zero-shot 두 종류로 편집된 입력의 정보량과 fine-tuning의 필요성을 분리한다.

**Main ablation table 초안:** 최종 모델, spatial 복귀, decoder 제거/raw semantic, Top-1 복귀, voxel query, R 제거를 필요한 행으로 둔다. Member 후보 구성을 최종 설계의 기여로 주장하면 평균 descriptor 행도 포함한다. 최종 모델에 learned refinement가 있으면 E_FIXED_RAW 행을 재사용한다. 핵심 column은 CD/F-Score·LPIPS/DreamSim과 보조 SS IoU다. 수치가 없는 행은 미측정으로 남기고 추정해 채우지 않는다. 기존 조건부 ladder와 부정적인 confidence/V-O 결과도 보조 표·discussion에서 보존한다.

**기전 그림:** 같은 ray의 voxel 집합과 공유 image evidence, semantic Top-K 후보 및 weight, R로의 전달을 실제 중간 결과로 시각화한다. Matching 성공·실패와 얇은 구조/큰 변형 사례를 함께 제시한다.

**정성 결과:** 동일 sample/view에서 GT·zero-shot·V1·R·Ours의 target/source render와3D 차이를 비교한다. Target feature 경로가 불변이라는 구조적 사실을 output RGB의 완전 보존으로 과장하지 않는다.

**Writing 작업 순서:** Method와 평가 protocol은 먼저 작성한다. 결과마다 claim-evidence ledger를 갱신하고, 최종 Table1/ablation/정성 결과와 CI가 준비되면 Results·Discussion·Abstract의 강도를 정한다. 현재 사용 가능한 method 설명 초안은 다음과 같다.

> We investigate how semantic correspondence transfers multi-view evidence from undeformed source images to an edited target. The model retains the native target-conditioning stream and adapts source features through a ray-based retargeting path. We evaluate geometric alignment, source appearance preservation, and rendering naturalness under fixed data, camera, and sampling protocols.

`All modules are necessary`, `learned attention consistently improves performance`, `all metrics improve`는 현재 근거로 쓰지 않는다. Decoder descriptor의 효과와 scalar confidence 효과도 구분한다. 최종 empirical claim은 실제 matched ablation과 TEST 결과가 지지하는 범위로 작성한다.

## 실행 우선순위

현재3개 학습/자동 평가 → E_FIXED_RAW joint와 후보 축소 → 선택 SS에서 PBR 짝비교 → 빠진 최종-context ablation → 필요한 독립 확인 → source/protocol freeze와 TEST508 → 표·그림·최종 writing 순서다. 평가와 문서 갱신은 각 결과가 나올 때 병행한다. Backbone 교체, temperature/gate grid, 전 SS×PBR 조합과 모든 arm의 PBR 재학습은 지금 범위에 넣지 않는다.

후속 실제 실행과 논문 초안은 [필수 계획 실행 기록](2026-10-24-essential-plan-execution-and-paper-draft-ko.md)에 기록한다. 이 문서의 최초 상태와 이후 완료된 결과를 구별한다.
