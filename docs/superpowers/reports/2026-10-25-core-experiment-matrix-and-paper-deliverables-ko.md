# 최종 파이프라인·모듈 검증·논문 완성을 위한 남은 작업

작성: 2026-10-25 환경 날짜. 실행 상태 관측: 2026-10-24 23:21 UTC.

후속 실제 실행 연결: [다섯 control의 cascade·8metric·논문 집계 등록](2026-10-25-final-fixed-cascade-metrics-continuation-ko.md). 기준선116개 평균값과 GT 재사용을 검증하고, own SS archive에 의존한 connector5개 및 최종 reducer를 등록했다. 아래의 후속 연결 미완료 설명은 해당 실행 기록으로 갱신하며, 실제 신규 cascade 품질은 계속 미완료다.

목표는 각 요소가 실제로 기여하는지 판별하고, 그 근거에 맞는 하나의 파이프라인과 논문을 완성하는 것이다. 모든 요소의 효과가 양수라는 결론을 전제하지 않는다. 이전 결과의 조건이 같으면 재사용하며 지엽적인 구조 탐색은 추가하지 않는다. 이 문서는 남은 작업을 정리한 것이며 새 학습·평가 실행을 뜻하지 않는다.

## 1. 현재 근거와 잠정 선택

현재 working candidate는 **Fixed SS + frozen Shape512/1024 + Top-1 retargeting PBR**다. Fixed는 matching probability로 Top-K value를 가중 결합하는 구조다. 추가 learned Q/K attention head는 없다. 아직 최종 TEST 모델로 동결하지 않았다.

- Top-1 semantic matching+retargeting branch 대 V1의 SS 개선은 두 실제 training seed에서 +0.005245/+0.005169다. 이것은 결합 branch의 근거이며 retargeting MLP 단독 효과가 아니다.
- 공통 DEV128 cascade에서 Fixed의 F-Score는 0.844058, Top-1 R은 0.840095다. 차이 +0.003963의 탐색용 source-cluster CI95는 [+0.001011,+0.007314]다.
- 같은 비교의 source LPIPS는 0.053438 대 0.053227, DreamSim은 0.151909 대 0.148809다. 두 지표 모두 낮을수록 좋으므로 preservation 개선을 주장할 수 없다. 차이의 CI가 0을 포함한다고 비열등성이 입증되는 것도 아니다.
- Learned refinement는 SS 점추정이 더 높아도 cascade geometry 우위가 확인되지 않았다. 원래 ray-vs-voxel 비교도 ray의 품질상 필수성을 입증하지 않았다.

따라서 현재 논문의 중심 질문은 “semantic 후보를 적절히 활용하고 source feature를 적응시키면 최종 geometry와 appearance에 무엇이 달라지는가”다. SS IoU 최고점만으로 Full을 정하지 않는다.

## 2. 남은 핵심 ablation: 기존 2행 + 신규 5행

모든 신규 행은 동일 원본 multi-view LoRA 초기값, DeformNeXt split, SS global batch96, 480 updates/약5epochs, 같은 seed·학습 stream·native conditioning 계약을 사용한다. 아래 matching ladder는 이웃 행끼리 비교한다.

| 행 | Correspondence / aggregation / retargeting | 비교와 질문 | 현재 상태 |
|---|---|---|---|
| M_FIXED_SPATIAL | 원래 spatial3 / Fixed 결합 / R | 다음 raw semantic 행과 비교: semantic 탐색의 효과 | 구현·검사·실행 등록, 실제 학습 대기 |
| M_FIXED_RAW | Raw semantic64 / cosine weighting / R | Spatial→Raw: semantic correspondence의 묶음 효과 | 동일 |
| M_FIXED_D1_COS | Decoder semantic64 / cosine weighting / R | Raw→D1 cosine: contextual decoder의 효과 | 동일 |
| E_FIXED_RAW | Decoder semantic64 / scalar probability weighting / R | D1 cosine→scalar: scalar matching 모듈의 효과 | SS128×3·cascade128×1 완료 |
| 실제 Top-1 R | Decoder / 실제 최상위 feature1개 / R | Fixed와 비교: 여러 후보 활용의 효과 | SS·cascade 완료, 재학습 불필요 |
| A_FIXED_UNIFORM | Fixed와 같은64개 후보·value / uniform / R | Fixed와 비교: matching score weighting의 효과 | GPU7 학습137/480 |
| A_FIXED_NO_R | Fixed와 같은 후보·weight·전달 / R 물리적 제거 | Fixed와 비교: 최종 경로에서 retargeting의 효과 | GPU6 학습111/480 |

**신규 ablation은 다섯 개다.** 이미 학습 중인 Uniform/No-R 두 개와 등록된 matching 세 개이며, 각 arm의 proof/capacity/evaluation 작업을 별개의 모델로 세지 않는다.

해석 경계:

- Spatial3 대 semantic64는 원래 spatial 모듈을 semantic retrieval로 교체하는 효과다. 후보 수까지 고정한 순수 주소 비교가 아니다.
- Raw→D1 cosine→D1 scalar를 나누어 decoder와 score를 한 번에 바꾼 효과를 따로 나누어 주장하는 오류를 피한다. Cosine→scalar는 선택 support도 바뀔 수 있다.
- **같은 support에서 score 활용 효과는 Uniform↔Fixed**로 검증한다. 공통 posterior가 carrier/use/R에 공유되므로 순수 attention-logit 한 위치만의 효과로 부르지 않는다.
- 사용자 정의 No-score는 실제 Top-1이며 Uniform과 다르다. No-corr는 원래 spatial 대응이다. 제거한 retargeter/channel을 0인 가짜 관측으로 대체하지 않는다.
- Ray grouping은 모든 신규 행의 공통 구조다. Working Fixed에 없는 learned Q/K를 제거하는 가짜 arm은 만들지 않는다. Ray의 독립 품질 기여를 핵심 claim으로 삼지 않으면 신규 ray/voxel 학습은 필요 없다. 기존 결과와 한계는 보조 자료에 공개한다.

## 3. 학습 후 반드시 실행할 실제 출력 평가

각 신규 arm의 완료 조건은 **480 학습 → strict checkpoint load → SS DEV128×3 → 공통 Shape/PBR cascade DEV128×1 →8개 metric 및 paired 통계**다. 빠른 epoch32 IoU만으로 ablation을 종료하지 않는다.

1. SS stage: macro per-asset voxel IoU, precision, recall, F1. 학습 seed와 generation noise를 구별하고 global occupancy IoU와 혼합하지 않는다.
2. 실제 cascade: 동일128개/93 source cluster, 동일 noise·camera·GT·frozen Shape·공통 Top-1 R PBR로 신규 다섯 행을 평가한다. 기존 Fixed/Top-1 결과와 공통 GT 렌더는 재사용한다. 신규 다섯 행은 최대640개 cascade case이며 별도의 PBR ablation 재학습은 하지 않는다.
3. 여덟 metric: source LPIPS/DreamSim/CLIP, target PSNR/SSIM/M-IoU, surface CD/F-Score. 최종5개 핵심 열 외의 metric도 보조 표로 공개한다.
4. 인접 matching 대조와 Fixed 대비 Top-1/Uniform/No-R를 사전에 정한 contrast family로 집계한다. 동일 source의 여러 deformation을 독립 표본으로 세지 않고 source-cluster paired CI를 쓴다. DEV32 screening 뒤 확장한 DEV128은 탐색적 결과라고 명시한다.

현재 own checkpoint/evaluator/cascade binding은 구현됐지만, **신규 다섯 arm의 전체 case-plan·metric 후속 연결과 실제 cascade 실행은 남아 있다.** 이를 평가 완료로 표시하지 않는다.

평가 reference는 source camera9/10에서 M1 geometry+원래 material의 GT, target camera8에서 편집 목표 GT를 사용한다. CD는 공통 normalization의20k surface samples, F-Score threshold0.02로 계산하며 결과별 ICP나 scale fitting은 하지 않는다. M-IoU는2D alpha mask다. 현재 입력 camera 평가를 새 시점 일반화로 설명하지 않는다.

## 4. 독립 확인과 파이프라인 선정

남은 독립 학습은 선택 Fixed의 SS seed202 한 개와 PBR vanilla/R seed202다. V1/R SS 두 seed는 완료돼 반복하지 않는다. 모든 ablation의 seed를 늘리지 않는다.

관측 시 GPU0–3은 필수 PBR vanilla seed202의508/1900을 실행 중이다. 이후 R seed202가 의존 작업이다. GPU6/7의 현재 두 SS 대조 뒤에 matching 세 개와 Fixed seed202의 검증·학습·평가가 등록돼 있다. 실제 작업은 승인된 GPU0/1/2/3/6/7만 사용한다.

선정 순서:

1. 최종 문맥의 ablation 및 downstream geometry/appearance를 함께 판독한다.
2. Working Fixed의 독립 seed 결과와 선택 checkpoint의 fresh-load/composition 정합을 확인한다. 실제 seed별 결과와 평균을 분리해 보고한다.
3. CD/F-Score와 LPIPS/DreamSim을 primary로 유지한다. 임의 가중합으로 승자를 만들지 않는다. Appearance 악화와 geometry 개선이 함께 있으면 trade-off를 보고한다.
4. 근거 없는 learned refinement나 개별 요소의 필수성 주장은 제외한다. Fixed가 R보다 추가 이점을 뒷받침하지 못하면 R을 최종 선택할 수 있다. 최종 구조가 바뀌면 그 구조에 유효한 대조만 남긴다.
5. 실제 checkpoint/config/code·split·camera/RGB·sampler·noise·stage composition·metric implementation·통계 family를 동결한다.

## 5. 최종 TEST508: 실제4/5개 pipeline,10개 metric

134 source cluster의 TEST508은 동결 뒤 한 번의 최종 비교로 사용한다. TEST 결과로 architecture를 다시 고르지 않는다.

| Main table 행 | 입력/역할 |
|---|---|
| Zero-shot source-only4 | 변형 전 target camera의 source view를 포함한 원본 RGB4장 |
| Zero-shot edited4 | 편집 target RGB1장+원본 source RGB3장 |
| Vanilla V1 | DeformNeXt vanilla fine-tuning |
| Top-1 R | 강한 semantic+retargeting 기준선 |
| Final Ours | 선택한 실제 조합. R과 같으면 중복 행 제거 |

- Source preservation: LPIPS↓, DreamSim↓, CLIP↑.
- Global naturalness: FID↓, KID↓.
- Target alignment: PSNR↑, SSIM↑, M-IoU↑, CD↓, F-Score↑.

FID/KID는 작은 ablation마다 반복하지 않고 최종 공통 render population에서 측정한다. 현재 protocol은 모델별508×3=1524 images, official Inception2048 FP32/TF32 off, KID100 subsets×1000·seed42017이며 signed KID를 자르지 않는다. FID/KID는 global descriptive 값으로 보고하고,8개 per-asset metric의 selected-vs-V1/R 대비는 사전 고정한 family의 보정 source-cluster interval로 보고한다.

원본/GT/최종 checkpoint는 보존한다. 큰 중간 guide는 downstream 의존성과 실제 file4 복제·hash를 확인한 뒤 소유한 중간물만 rolling 관리한다. 저장 공간 부족으로 평가 표본 수를 조용히 줄이지 않는다.

## 6. 논문 표·그림·writing 산출물

| 산출물 | 구체적인 구성 | 현재와 남은 작업 |
|---|---|---|
| Main table | 위4/5개 모델×10 metric, 목적별3개 column group | TEST 실제 값 전부 남음 |
| Ablation table | Panel A: spatial→raw→decoder cosine→scalar. Panel B: Top-1/Uniform/no-R 대 Fixed | Fixed/Top-1은 실제 값, 신규5행 미완료 |
| Mechanism figure/table | Retrieval coverage→유효 후보 probability mass→같은 사례의 출력 변화 | 기존 관측 재사용, matched Uniform 결과와 연결 |
| Qualitative figure | 같은 asset/view의 GT·두 zero-shot·V1·R·Ours, 성공과 실패 | 최종 모델 render와 명시적인 사례 선정 기준 필요 |
| Supplement | 모든8metric, 기존 ray/member/learned 음의 결과, 두 seed, protocol·실패 사례 | 기존 완료 결과 반영·최종 정합 검수 |

Ablation 본문 열은 `Correspondence / Weighting / R | SS IoU / CD / F-Score / LPIPS / DreamSim`으로 구성한다. 기존12모델 원래 문맥의 수치를 신규 최종 문맥의 행에 옮기지 않는다. Matching-aware weighting과 추가 learned attention을 구분한다.

Writing 순서는 problem/method·protocol 정리 → DEV ablation/기전 → 독립 재현 → TEST main results → discussion → abstract/conclusion 확정이다. 이미12page/24reference compiled 개발 초안이 있으며, 남은 작업은 실제 최종 표·그림·claim을 넣고 PDF를 검수하는 것이다. 현재 contribution의 표현은 다음 범위가 적절하다.

> We aggregate uncertain semantic correspondence evidence at the camera-ray level and adapt source conditioning while retaining the native target-conditioning stream. Matched ablations evaluate retrieval, score-weighted aggregation, and retargeting separately; geometric alignment and source appearance preservation are reported independently.

이는 설계와 검증 범위의 설명이다. “모든 모듈이 필수”, “learned attention이 일관되게 개선”, “모든 metric 우세”, “완전한 target 보존”은 현재 결과로 쓰지 않는다. Metric 개선이 없는 요소는 구조 설명 또는 음의 결과로 남기고 독립 성능 기여에서 제외한다.

## 7. 이번 범위에서 추가하지 않을 것과 종료 기준

새 backbone/head 탐색, gate/temperature grid, 모든 SS×PBR 조합, target-only, clean80k 재학습, 모든 ablation의 다중 seed, 최종 모델에 없는 query 제거는 추가하지 않는다. Matching coverage·native precision 등 완료된 진단을 반복하지 않는다.

**남은 단위는 신규5개 SS ablation + 선택 SS seed202 + 필수 PBR seed202 두 실행의 완료, 실제 downstream 평가, TEST, 논문 마무리다.** Arm 수와 proof/capacity/evaluation job 수를 구별한다. 종료 조건은 기대한 story를 만드는 것이 아니라, 선택된 실제 pipeline의 효과·모듈별 근거·한계와 Main/Ablation/Mechanism/Qualitative 결과 및 논문 문장이 서로 일치하는 것이다.

## 근거

- [완료된 DEV128 실제8metric과 선택 근거](2026-10-24-dev128-complete-fixed-controls-and-independent-confirmation-ko.md)
- [최종 문맥3개 correspondence 구현·등록과 첫 epoch의 한계](2026-10-24-shared-fixed-correspondence-implementation-and-queue-ko.md)
- [상세 필수 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)
- [Claim-evidence 원장](../paper/claim-evidence-ledger.md)
- [실제 ablation 표 소스](../paper/tables/final-ablation-template.tex)
- [논문 개발 초안](../paper/retargeting-evidence-draft.tex)
