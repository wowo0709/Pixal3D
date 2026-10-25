# 논문 claim과 필요한 근거

[SS/cascade 표·독립 seed 연결과 실제 두 번째 epoch](../reports/2026-10-25-SS128x3-table-and-selected-seed-continuation-ko.md)를 반영한다. 새 SS family6와 cascade family48을 구분하고 실제 noise0/checkpoint identity를 확인한 표만 사용한다. Same quick32 step192에서 Uniform가 Fixed보다 macro IoU+0.002169이며 첫192 training trace는 정확히 같다. 이는 학습 중 단일-noise 관측으로 최종 score 효과·architecture 선택의 근거를 대신하지 않는다.5epoch/128×3/downstream 완료 전 weighting의 필수성을 주장하지 않는다.

최신 실행 근거: [최종 Fixed 대조의 실제 cascade·metric 연결](../reports/2026-10-25-final-fixed-cascade-metrics-continuation-ko.md). 원래 R/Fixed baseline의116개 평균값과128개 GT 재사용을 실제 검증했고 다섯 own SS 이후 cascade/official8metric connector와 최종7모델 집계를 등록했다.48-comparison DEV family의 보정 구간을 사전 지정했다. 등록은 새 control의 quality 완료가 아니며 해당 결과는 여전히 pending이다.

최신 남은 작업과 산출물: [핵심 실험 행렬·최종 평가·writing](../reports/2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md). 최종 문맥 신규5개 대조는 실제 downstream 평가까지 완료해야 하며, 기대한 모듈별 positive claim을 결과에 앞서 확정하지 않는다.

이 문서는 결과를 본 뒤 유리한 metric으로 주장을 바꾸지 않기 위한 원장이다. 아래 empirical claim은 DEV 단계의 해석이며, 최종 TEST claim으로 승인된 것이 아니다. 현재 두 리포트의 계획을 유지한다.

남은 실행과 표·writing의 최신 통합 계획은 [논문 완성을 위한 핵심 실험 정리](../reports/2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)를 따른다. Matching score 효과를 검증하기 위해 동일 Top-K의 Uniform-K 대조를 핵심 계획에 포함하며 사용자 정의 No-score=Top-1과 구분한다.

현재 실행 상태와 최종 claim·표·writing 완료 조건은 [2026-10-25 핵심 실행 행렬](../reports/2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md)에 갱신했다. 신규5개 대조의 최종 품질은 아직 미측정이다.

| Claim 후보 | 현재 근거 | 미완료 검증 | 현재 허용하는 표현 |
|---|---|---|---|
| V1 기준선이 독립 학습에서도 재현된다 | Seed1010.467355, seed2020.467177; 실제480 trace·startup RNG가 다르고 초기 LoRA·학습 조건·DEV128×3 pairing 일치 | 선택 모델·PBR의 독립 효과와 full-cascade 확인은 별도 | 두 실제 V1 실행의 유사한 점추정; 동등성이나 임의 training seed population의 보장 아님 |
| Top-1 semantic matching+retargeting branch의 V1 대비 효과가 독립 학습에서 재현된다 | Seed101 ΔIoU+0.005245, seed202+0.005169; 각 cluster CI95가 양수. 양쪽480 trace pairing·학습 stream 독립·원본 frozen701 tensors 일치 확인 | 선택 모델이 다르면 그 후보 seed202, PBR 독립 확인, full-cascade/TEST | 두 실제 SS training seed에서 combined branch 개선. Retargeter/decoder 단독 효과나 training seed 모집단의 보장 아님 |
| Semantic correspondence가 spatial보다 유효하다 | C_SPATIAL→C_RAW의 DEV128×3 macro IoU +0.001117 | 원래 finite-family 통계, 최종 구조의 matched 비교 및 downstream | “해당 SS 설정에서 양의 점추정” |
| Matching decoder가 correspondence를 개선한다 | Raw→D1-cos IoU +0.000648; 별도 matching DEV32의 exact@1은1.25%→7.73% | 최종 구조/downstream 연결; 진단32와 SS128을 같은 paired 실험으로 혼동하지 않음 | Descriptor와 scalar score의 효과를 분리 |
| Scalar score가 생성 품질을 개선한다 | D1-cos→decoder IoU −0.000408; 별도 진단의 exact@1은7.73%→8.56% | 동일 Top-K support/value 경로의 score 대조가 필요 | Ranking 개선과 generative gain을 구별; 생성 품질 positive claim 보류 |
| Top-K attention이 Top-1보다 유효하다 | Fixed-K의 SS +0.000527 및 별도 DEV32 mask +0.003717 | 최종 경로의 matched 비교·독립 확인 | 후보 수와 결합의 묶음 효과로 한정 |
| 추가 learned attention이 필요하다 | Anchored SS128×3 IoU0.474794로 Fixed보다+0.001323, CI는0 포함. 실제 cascade128에서도 Learned는 Fixed보다 CD/F-Score 점추정 열위이고 paired CI는0 포함 | 최종 선택에서 추가 이득을 입증할 근거 없음; 현재 working Fixed에는 head 미채택 | SS 최고점이 pipeline 우위를 입증하지 않으며 learned refinement의 필수성·전반적 preservation 개선은 미확정 |
| Ray 단위 처리가 품질을 개선한다 | 원래 FULL의 DEV128×3 완료: ray0.472078, voxel0.472314, Δ−0.000236; ten-contrast CI[−0.000878,+0.000351] | 최종 경로에 유효한 matched 대조 | 원래 FULL에서 양의 품질 효과 미확인. Ray 구조 설명 가능; 필수성·동등성 주장 불가 |
| Member 후보가 품질을 개선한다 | 별도 matching DEV32의 exact@64 50.76%→73.36%; 그러나 완료된 FULL mean 제어 IoU0.472679 대 member FULL0.472078 | 원래 finite-family 통계와 최종 identity-value 경로의 matched 확인·downstream 연결 | 후보 coverage 개선은 확인; 기존 FULL에서 생성 품질의 양의 점추정은 아님 |
| Retargeting이 유효하다 | C_NO_R→C_DECODER SS +0.003929; 실제 V1/R joint 및 combined branch 두 seed 재현 완료 | 최종 weighted 경로의 물리적 R 제거 비교; 단독 R 효과의 독립 seed는 미확인 | 현재 matched 설정의 효과로 한정; V1/R seed 반복은 R 단독 ablation 아님 |
| Shared evidence 전달 자체가 개선 원인이다 | E_FIXED_RAW SS0.473471; FULL과는 복수 차이 | 동일 후보/가중치/R/V의 split-vs-shared 대조가 있어야 인과 주장 | 현재는 구조 설명; 독립 causal claim 보류 |
| E_FIXED_RAW가 R보다 좋은 최종 모델이다 | DEV128 F-Score+0.003963, 탐색용 CI[+0.001011,+0.007314]; CD−0.00002230은 CI0 포함. Source LPIPS+0.000211/DreamSim+0.003100은 각각 CI0 포함 | Working Fixed/RPBR의 matched controls, Fixed seed202, 필수 PBR 확인, TEST | Geometry 중심의 working candidate. Preservation 개선·비열등성·최종 보편적 우위는 미확정 |
| Source 적응과 target 보존을 구조적으로 분리한다 | Native MV4 carrier·target 경로 유지, source map correction만 추가 | 실제 target/source 결과 및 실패 사례 | 입력 target stream 유지 ≠ output RGB 완전 보존 |
| Ray 구조가 더 빠르다 | 현재 실행 시간은 FFN chunk 등이 다름 | 동일 precision/batch/chunk/sampler benchmark | 순수 속도 배수 주장 보류 |
| 최종 파이프라인이 일반화한다 | TEST 미실행 | 선택·protocol freeze 후 TEST508·source-cluster 통계 | 최종 generalization claim 보류 |

## 결과 표의 역할

원래12모델/10개 module contrast/2개 pipeline contrast의 전체 통계가 완료됐다.
모든10개 module-family 보정 CI는0을 포함하며, R 제거의+0.003929도
보정 CI[−0.000525,+0.008987]다. 원래 FULL−V1의 보정 CI는 양수지만
이를 개별 모듈의 필수성으로 해석하지 않는다. Top-1 combined branch의
두 seed 재현과 최종 identity-value 구조의 후속 실험은 각각 별도 근거다.

추가 실행: 새 대조의 cascade binding까지169개 단위/계약 검사가 통과했고,
Uniform-K는 실제 micro96/accum1·3-update capacity를 완료해480-step 과학 학습을
시작했다. 실제 본 학습 결과와 새 대조의 full-cascade 품질은 아직 미측정이다.

DEV128 실제8개 metric 집계가 완료됐다. Working candidate는 Fixed SS+Top-1 R PBR로
기록했으며 TEST freeze는 아니다. 선택 문맥의 물리적 No-R도 source5/optimizer365·
micro96 capacity를 실제 통과해 본 학습을 시작했다. Fixed seed202는 기존 V1seed202의
실제480 trace를 사용하는 own proof→capacity→science→128×3만 등록했다.

최종 대조의 [코드·config 재사용 감사](../reports/2026-10-24-final-context-reuse-and-physical-controls-ko.md)를 반영한다. 기존 matching controls는 Top-1 appearance이며 E의 weighted Top-K와 자동으로 같지 않다. Uniform-K↔Fixed는 weighting, Fixed↔Learned는 refinement, 같은 learned head의 additive-logit-prior 제거는 직접 prior 항의 대조다. Raw→D1-score만으로 descriptor와 scalar score를 따로 입증하지 않는다. Fixed에 없는 learned Q/K의 voxel-query 필요성도 주장하지 않는다. 새 네 대조는 native trainer/evaluator까지 구현했고115개 검사와4개 실제 GPU 초기 함수·backward proof를 통과했다. Micro96 capacity·480 학습·새 대조 cascade binding/실제 평가가 남아 있으며 성능 개선의 근거는 아직 없다. [최신 필수 실행 행렬](../reports/2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)에 선택 모델별 학습 범위와 증거를 기록했다.

- `tables/dev-ss128.tex`: 동일 SS panel의 완료된14개 endpoint와 조건부 비교. 원래 core family와 새 exploratory follow-up은 통계적으로 구분하고 Micro IoU와 섞지 않는다.
- `tables/dev-original-module-effects.tex`: 완료된 원래12모델의10개 module-family 보정 CI. 최종 구조의 missing matched 대조를 대신하지 않는다.
- `tables/dev-joint32.tex`: V1/R/D1-cos/E_FIXED_RAW/E_LEARNED_RAW의 실제5개 조합 비교. V1/R baseline의 per-asset 일치 검증을 거친 결과다.
- `tables/dev-joint128.tex`: 동일128/93 source·noise1의 R/Fixed/Learned 실제3개 cascade·8metric. Working Fixed 선택과 appearance 한계를 함께 보고하며 TEST나 generation-noise3회 평균이 아니다.
- `tables/dev-matching32.tex`: 재사용한 matching 진단의 observable relation coverage. GT visibility는 진단 분모에만 사용하며 모델의 write gate가 아니다.
- `tables/final-main-template.tex`: TEST508, 사용자 지정10개 metric. 미측정은 대시로 남긴다.
- `tables/final-ablation-template.tex`: Matching ladder의 adjacent 비교와 Fixed 기준 aggregation/R 대조를 구분한다. Actual Fixed/Top-1 두 행만 채웠고 새 control은 pending이다. Spatial을 scalar Full의 단일 제거 대조로 쓰지 않는다.

최종 Fixed correspondence3개는 own native recipe/runtime/evaluator/cascade binding까지
구현했다.257개 검사와 실제 checkpoint 기반 생성자 정합이 통과했고21개 gated 작업을
등록했다. GPU proof/capacity/science/실제 quality는 아직 미완료다. 이전 Top-1 ladder의
수치를 이 최종 문맥으로 대체하지 않는다. [구현·실행 기록](../reports/2026-10-24-shared-fixed-correspondence-implementation-and-queue-ko.md)을 따른다.

진행 중인 Fixed 문맥 No-R/Uniform의 첫 epoch quick32 readout에서는 초기 macro IoU가
모두0.458358로 일치한다. Step96 No-R0.463828 대 Fixed0.461789의 양의 점추정이 있어
최종 R 필요성을 가정하지 않는다. 아직5epoch/DEV128×3가 아니며 ablation의 최종
수치로 채우지 않는다. Global occupancy IoU와 macro mean asset IoU도 구별한다.

후속 paired quick32 step288에서 Uniform0.474767 대 Fixed0.471552다.
평균 차이+0.003215와 달리 개선14/하락18·median−0.000181이며 큰 개선3개가
전체 net gain을 주도한다. No-R step192는0.467777 대 Fixed0.463634다.
이후 No-R step288은0.469064로 Fixed−no-R가+0.002488로 뒤집혔다.
Matching decoder의 실제9개 saved state는 동일하고, Fixed/Uniform의28개
gradient entry 모두에서 R/use/fusion이 학습되고 있다. 이 중간 관측은
weighting/R의 최종 필요성이나 gradient 단절·saturation의 causal 증거가 아니다.
최종 표에 옮기지 않으며 [실제 중간 분석](../reports/2026-10-25-ray-mechanism-paper-and-final-readiness-ko.md)의 범위를 유지한다.

## Writing에서 지킬 해석

PBR 독립 seed의 fixed4 stage trend는 같은 seed 안의 V1/R·모든 epoch에서
실제 GT PNG pixel 동일성을 요구한다. 독립 seed 사이에는 입력 RGB·camera·noise·
GT latent·cold prediction latent가 정확히 같아도 rendered GT pixel이 다를 수 있다.
실제 seed101 V1/R의6개 epoch와 seed202 vanilla의0/380/760 prefix에서 이를
확인했다. 기존 producer의 초기 render bound를 재검증하고 seed별 reference와
metric을 그대로 기록하며 common-pixel quality로 pooling하지 않는다.
이는 generated-geometry/source-preservation 확인이 아니며 최종 cascade/TEST의
공통 rendered GT 규칙을 완화하지 않는다. Seed202 R·완료1900 collector는 미완료다.
수정 이유·실제 근거·회귀 검사는 [PBR reference-scope 기록](../reports/2026-10-25-PBR-seed-reference-scope-and-collector-ko.md)을 따른다.

Primary geometry는 CD/F-Score, primary source appearance는 LPIPS/DreamSim이다. CLIP은 semantic 보조 지표이고 FID/KID는 global naturalness다. “All metrics improve”, “all modules are necessary”, “learned attention is essential”은 현재 근거에 없는 문장이다. 원래 비교 family와 이후 탐색적 follow-up, training seed와 generation noise를 구분한다. 음의 결과와 실패 사례도 discussion에 보존한다.

`dev-qualitative.tex`의 실제 render figure는 Fixed 대 R의 F-Score 증가 최대와 source LPIPS 증가 최대 사례다. 결과 조건으로 선택한 설명용 사례이므로 평균 효과나 성공 빈도의 근거로 사용하지 않는다. Full-frame GT pixel 일치와 기존 metric annotations를 검증했으며, appearance 차이의 causal mechanism이나 final Ours/TEST 품질을 입증하지 않는다.

실제 cascade와 같은32 asset/noise에서 SS IoU는 R0.476967·Fixed0.478940·Learned0.477442다. Full128×3의 learned/fixed 순위와 다르다. 서로 다른 panel/noise의 순위 차이만으로 모듈 구현 오류나 loss의 causal failure를 결론내리지 않는다.

학습된 Fixed/Learned의 실제 conditioning을 같은32개에서 관측한 활용 진단이 완료됐다. Native BF16의50,516개 observable relation 중 Top-1 miss/Top-K hit31,408개에서 exact GT mass는 Uniform-K1.6266%, Fixed6.4601%, Learned6.1756%다. Learned−Fixed는−0.2844pp, 탐색용 cluster CI95 [−0.9776,+0.4890]pp로 추가 이득을 확인하지 못했다. Prior 대비 확률 변화 자체는 실제로 관측됐다. 따라서 matching prior의 후보 활용과 learned refinement의 추가 효과를 나누어 서술한다.

과거 CPU FP32 진단50,493개와 actual BF16 관계50,516개는 projection precision의 patch rounding 차이로 분모가 다르며, native GPU 주소의 실제 재구성으로 원인을 확인했다. 기존 CPU core parity를 GPU precision parity로 확대 해석하지 않는다. 저장 관측의 source-use·correction 및 downstream association은 진단이며, diagnostic Uniform-K는 matched 학습 ablation을 대신하지 않는다. 실제 audit/closure와 남은 실험은 [최신 핵심 계획](../reports/2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)의 기전 항목에 연결했다.

`actual-ray-evidence-figure-v323`은 위 실제 저장 관측과 원본 RGBA의 native black-background RGB 전처리를 사용한 기전 그림이다. 사전순 첫 Top-1 miss/Top-K hit·관측 member2개 이상 관계를 선택했으며 예시의 GT 후보는 rank2, probability mass25.64%다. GT는 주석과 진단 subset에만 사용했고 새 NN forward는 없다. 전체 native ray와 저장된 observed member를 구분했다. 실제 PNG를 검수하고 PDF/PNG·선정 근거를 file4에 보존했다. 이 조건부 예시나 pooled mass를 ray/weighting의 최종 품질 우위, 성공 빈도, matched Uniform의 결과 또는 target RGB 완전 보존의 증거로 쓰지 않는다. Manuscript 본문·caption에 이 범위를 명시하고 `paper-draft-actual-ray-evidence-v325`에서 삽입·compile·실제 페이지 검수를 마쳤다. 최종 TEST 정성 결과는 별도 미완료 작업이다.

같은32개에서 Learned−Fixed exact GT mass와 실제 출력 개선을 연결한 Spearman rho는 CD−0.0828, F-Score+0.0623, source LPIPS+0.1683, source DreamSim+0.0400이며 네 탐색용 CI 모두0을 포함한다. 정답 mass refinement가 최종 품질을 개선한다는 연결 근거는 아직 부족하다. 연관성 부재가 causal effect 부재를 증명하는 것도 아니다. [독립 V1·실제 활용/출력 연결 기록](../reports/2026-10-24-independent-V1-readout-and-utilization-output-link-ko.md)에 수치와 실제 audit를 남겼다.

[원래24개 문헌의 primary-source 대조](../reports/2026-10-24-paper-primary-literature-and-claim-scope-ko.md)를 반영해 correspondence-aware attention과 projection conditioning 자체를 신규 기여로 주장하지 않는다. 현재의 입력은 edited-target RGB와 source RGB이며 GT target mesh를 추론 입력으로 서술하지 않는다. 외부 editing/texturing 방법의 입력 차이는 문제 설정 설명이며, 실행하지 않은 비교의 성능 우위 근거가 아니다.

가까운4개 방법의 본문·공식 코드를 추가 대조했다. ShapeUP의 texture 단계도
edited image와 source multi-view를 결합하므로 source-view 조건 자체를
처음 제안했다고 쓰지 않는다. MVDiffusion의 residual zero initialization도
pretrained function을 보존하므로 cold initialization 자체를 신규 기여로
쓰지 않는다. CorrAdapter의 공식 MV-Adapter 구현은 cross-view match의 local
window를 aggregate·blend한다. 이 차이는 우리 global semantic Top-K/ray
전달과의 구조 비교이며 상대 성능 우위의 근거가 아니다. Published Pixal3D의
Direct3D-S2/DINOv2와 supplied native TRELLIS.2/DINOv3 구현도 명시적으로
구분한다. [Method-scope report](../reports/2026-10-25-closest-method-scope-and-writing-ko.md)에 원문 section·공식 immutable code ref와 실제 config 근거를 기록했다.
