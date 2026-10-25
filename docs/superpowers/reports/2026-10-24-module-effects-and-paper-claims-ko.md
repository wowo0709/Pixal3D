# 현재 모듈 효과와 논문에서 주장할 수 있는 범위

2026-10-25 후속 완료: Fixed SS의 두 training seed·DEV128×3를 실제 집계했다. V1 대비 IoU 개선은 seed101/202에서+0.006116/+0.007553이며 각각3대조 보정 CI95가 양수다. Top-1 대비+0.000871/+0.002385는 보정 구간이 둘 다0을 포함한다. Combined Fixed의 SS 재현이며 decoder/score/R 단독 효과나 Top-1 대비 추가 효과 확정은 아니다. Working Fixed seed101의 VAL36 두 fresh-load·72 archive·tail4 QA도 완료했다. Spatial480은 끝났고 Raw semantic/Decoder-cosine 본 학습, 세 correspondence의 SS128×3·신규5행 cascade·PBR 독립 확인·최종 freeze/TEST·writing은 계속 남아 있다. [현재 근거·범위·16쪽 manuscript 반영](2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md)을 우선한다.

2026-10-25 최신 판독: [최종 문맥의7행·남은 실행·논문 산출물](2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md)을 우선한다. Uniform/no-R의480 학습과 DEV128×3는 완료됐다. 실제 완료된 Fixed−Top-1/Uniform/no-R의 세 대비를 원래6개 contrast family로 보정한 IoU CI는 모두0을 포함한다. R 제거 대비의 양의 점추정을 최종 필요성의 증명으로 쓰지 않는다. Fixed는 확률 가중 결합이며 추가 learned Q/K head가 없다. Ray의 원래 matched 품질 우위도 입증되지 않았다. 아래 초기 관측·중간 checkpoint는 해당 조건의 이력이다. 남은 matching3개·공통 cascade·독립 확인·fresh-load·동결 뒤 TEST10metric과 최종 writing을 대신하지 않는다.

2026-10-25 후속: [같은384-step 네 모델 비교](2026-10-25-ray-mechanism-paper-and-final-readiness-ko.md)에서 V1/Fixed/Uniform/No-R의 macro IoU는0.470704/0.476221/0.477081/0.469716이다. 실제 cold/384 training trace가 같고 Fixed−No-R+0.006505의 긍정적 중간 신호를 관측했다.480/128×3/downstream 최종 기여는 아직 미완료다. [최종 정성 사례 protocol](2026-10-25-final-qualitative-selection-protocol-ko.md)은 TEST metadata로 두 사례를 사전 고정했으며 final image나 TEST quality는 만들지 않았다.

2026-10-25 후속: [PBR 독립 seed reference 범위·실제 collector 수정과1140 검증](2026-10-25-PBR-seed-reference-scope-and-collector-ko.md). 같은 seed의 모델/epoch 간 GT pixel은 정확히 유지하고 독립 seed의 renderer reference는 구분한다. 기존 NN을 유지한 채 pending CPU collector 하나를 교체했고107개 검사와 실제 prefix/producer 증거를 확인했다. Seed202 vanilla1140 metric·image는 기존 W&B에 있으며 이는 fixed4 stage trend다. 신규 SS 대조·PBR 최종 독립 확인·TEST 품질 효과는 계속 미완료다.

2026-10-25 후속: [실제 기전 그림의 논문 반영·Fixed VAL36 연결·최종 저장 준비](2026-10-25-ray-mechanism-paper-and-final-readiness-ko.md)를 진행했다. 실제 native 관측 그림을 넣은13쪽/24개 인용 PDF를 compile·검수했으며, 이 진단을 새 모듈의 품질 효과로 해석하지 않는다. 신규 다섯 대조의 최종 결과와 TEST는 여전히 미완료다.

2026-10-25 후속 실행: [SS128×3·실제 cascade 표·선택 후보의 독립 확인 연결](2026-10-25-SS128x3-table-and-selected-seed-continuation-ko.md). 실제 기존 SS6개 평가의8개 평균값을 재현했고,5개 완료 모델의 원본 frozen NN/trace 정합을 확인했다. Own 결과에 의존한 SS/cascade 표·Fixed202 collector3개 CPU 작업을 등록했다. Quick32 step192에서 Uniform0.465803 대 Fixed0.463634이며 score-weighting 우위의 최종 근거는 아직 없다. Intermediate 값을 final128×3와 혼합하지 않는다.

2026-10-24 22:43 UTC 실행 갱신: 최종 Fixed의 spatial/raw/D1-cosine 대조를 [구현·검증·등록](2026-10-24-shared-fixed-correspondence-implementation-and-queue-ko.md)했다. 동일 core10/전달 경로의 own trainer/evaluator/cascade binding,257개 검사, actual checkpoint 생성자 정합까지 통과했다. GPU6/7의 기존 작업 뒤 own proof→capacity→480→SS128×3를21개 작업으로 연결했으며, 새3개 arm의 GPU 품질 근거는 아직 없다. 원래 Top-1 ladder를 새 weighted 문맥의 결과로 대체하지 않는다.

2026-10-24 22:10 UTC 갱신: [실제 DEV128 cascade·선택 문맥 대조·독립 확인](2026-10-24-dev128-complete-fixed-controls-and-independent-confirmation-ko.md)을 완료/연결했다. R/Fixed/Learned의8개 metric 집계에서 Fixed F-Score는0.844058 대 R0.840095, 탐색용 ΔCI[+0.001011,+0.007314]지만 source LPIPS/DreamSim의 개선·비열등성은 미확인이다. Fixed SS+Top-1 R PBR를 working candidate로 기록했다. No-R는 GPU6에서 실제 micro96 capacity를 통과해480 학습을 시작했고 Uniform-K는 GPU7에서 학습한다. Fixed seed202의 own proof/capacity/480/128×3도 등록했다. TEST freeze와 최종 문맥 matching ladder는 아직 남아 있다.

2026-10-24 21:49 UTC 추가 결과: 원래12모델 DEV128×3 통계가 완료됐다. FULL ray0.472078 대 voxel0.472314이며 ray−voxel은−0.000236, ten-contrast CI[−0.000878,+0.000351]이다. 원래10개 module-family CI가 모두0을 포함하므로 개별 모듈의 필수성을 확정하지 않는다. 원래 FULL−V1의 양의 family CI 및 Top-1 combined branch 두 seed 재현과 구별한다. 아래 최초 결과의 “pending” 설명은 해당 시점 기록이며 [완료된 family와 실행 기록](2026-10-24-uniform-score-control-and-independent-paper-results-ko.md)의 후속 결과를 우선한다.

최종 문맥의 [코드·config 재사용 감사와 실제 대조 구현](2026-10-24-final-context-reuse-and-physical-controls-ko.md)을 반영한다. 기존 matching controls는 Top-1 전달이고 새 Fixed/Learned는 weighted shared evidence라 최종 문맥 대조로 자동 대체할 수 없다. Fixed에는 learned Q/K가 없으며 Uniform-K↔Fixed, Fixed↔Learned, 같은 learned head의 additive-prior 제거를 구분한다. 후속 구현은 native trainer/evaluator·4개 실제 GPU 초기 함수/backward 검사·cascade binding을 포함한169개 검사까지 진행했다. Uniform-K는 GPU7에서 기존 voxel 평가 후 micro96 capacity → 조건부480 학습 → DEV128×3 평가를 연결했다. 본 학습과 실제 cascade 품질 평가는 아직 미완료다.

독립 seed 갱신: V1/Top-1 semantic matching+retargeting의 SS 학습·평가가 seed101과202 모두 완료됐다. V1 대비 IoU 개선은 각각+0.005245/+0.005169이며 각 source-cluster CI95는 양수다. 이는 combined branch의 SS 재현성으로, R/decoder 단독 효과나 최종 cascade/TEST의 증명이 아니다. 실제 수치와 다음 작업은 [최신 핵심 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md), 논문 반영·실행 연결은 [후속 기록](2026-10-24-uniform-score-control-and-independent-paper-results-ko.md)을 따른다.

최신 남은 작업 정리: [논문 완성을 위한 핵심 실험·ablation·writing 설계](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md). 최종 문맥의 대조와 기존 조건부 효과를 구별하고, 실제 cascade 평가 후 최종 모델을 결정한다.

최신 후속 결과: E_LEARNED_RAW의480/DEV128×3는 IoU0.474794, 원래 FULL mean 제어는0.472679로 완료됐다. Learned−Fixed의 CI는0을 포함하고, 원래 FULL의 member−mean 점추정은−0.000601이다. 아래의11개 완료/후보 학습 중 설명은 최초 관측 기록이다. 최신 상태와 필요한 matched 대조는 [최종 파이프라인 계획](2026-10-24-final-pipeline-essential-experiments-and-writing-ko.md), 실제 figure·PDF·독립 seed 준비는 [후속 기록](2026-10-24-actual-qualitative-and-independent-seed-preparation-ko.md)을 따른다.

Learned의 실제 cascade32도 완료됐으며 R 대비 source LPIPS/DreamSim/CD의 점추정은 열위, F-Score는 우위지만 해당 CI는 모두0을 포함한다. 같은32/noise에서는 SS IoU도 Fixed가 Learned보다 높아 Full128×3와 순위가 다르다. [최신 cascade 결과와 같은 panel 진단](2026-10-24-learned-cascade-complete-and-same-panel-diagnosis-ko.md)에 따라 새 조합 대신 기존3개 핵심 pipeline의 동일 DEV128 확장을 준비한다.

2026-10-24. 완료된 SS DEV128×3, SS5개 모델의 DEV32 cascade, 고정 mesh의 PBR4개 모델 DEV32, V1/retargeting joint DEV32를 구분해 정리한다. 모든 모듈이 일관되게 성능을 올렸거나 필수라는 결론은 아직 성립하지 않는다. 최종 voxel-query 대조군과 새 learned-evidence 모델, 독립 training seed와 TEST508은 진행 중이다.

## 1. SS에서의 요소별 수치

동일 DEV128 asset·93 source cluster, 생성 noise3개, training seed101 하나,480 update의 **asset별 macro voxel IoU**다. 기존 micro IoU와 직접 비교하지 않는다. 아래 변화는 각각 맞춘 대조군끼리의 점추정이며, 서로 다른 비교의 증가량을 누적해 더하지 않는다. 원래 finite-family 전체 통계와 독립 확인이 완료된 결과도 아니다.

| 요소 | 실제 비교 | IoU 이전 → 이후 | 절대 차이 | 현재 해석 |
|---|---|---:|---:|---|
| Spatial → raw semantic correspondence | C_SPATIAL → C_RAW | 0.471242 → 0.472360 | +0.001117 | Semantic 후보의 개선 신호 |
| 학습된 matching embedding | C_RAW → C_D1_COS | 0.472360 → 0.473008 | +0.000648 | Decoder descriptor의 개선 신호 |
| Decoder scalar matching score | C_D1_COS → C_DECODER | 0.473008 → 0.472600 | −0.000408 | 기존 Top-1 경로에서 score 추가 이득 미확인 |
| Learned V/output transform | C_DECODER → C_TOP1_V | 0.472600 → 0.472218 | −0.000382 | Voxel 지표에서는 개선 없음 |
| 같은 V/O에서 Top-1 → fixed Top-K | C_TOP1_V → C_FIXED_TOPK | 0.472218 → 0.472745 | +0.000527 | 불확실한 후보의 결합에 긍정적 신호 |
| Fixed 결합 → learned Q/K·utility | C_FIXED_TOPK → FULL | 0.472745 → 0.472078 | −0.000667 | 기존 learned attention의 SS IoU 추가 이득 없음 |
| Readout 내부 matching prior | FULL_QK_ONLY → FULL | 0.472446 → 0.472078 | −0.000368 | 해당 readout의 prior 사용 추가 이득 미확인; utility 입력도 바뀌는 기존 대조 |
| Retargeter의 물리적 추가 | C_DECODER_NO_R → C_DECODER | 0.468671 → 0.472600 | +0.003929 | 현재 가장 큰 명확한 matched 점추정 |
| 개별 voxel query → ray query | FULL_VOXEL_Q → FULL | 최종 비교 진행 중 | 미정 | Best에 포함됐다는 사실만으로 독립 기여를 확정하지 않음 |

Matching decoder의 descriptor와 scalar score를 같은 효과로 묶지 않는다. C_D1_COS와 C_DECODER의 차이는 이미 학습된 descriptor 위에서 scalar score를 사용하는 효과다. FULL_QK_ONLY는 동일한 후보·context·V/O·R을 유지하고 attention logit의 matching prior를 제거하는 별도 대조군이며, 사용자가 정의한 No-score=실제 Top-1 arm과 다르다.

현재 완료된11개 SS 모델의 최고는 **E_FIXED_RAW=0.473471**, vanilla V1은0.467355로 전체 차이는 **+0.006116**이다. E_FIXED_RAW는 matching model의 posterior로 원본 후보 feature를 결합하고, carrier·source-use·R이 같은 evidence를 사용한다. 추가로 학습하는 Q/K·V/O가 없다. 이름의 raw는 원본 value 전달을 뜻하며 C_RAW처럼 decoder 이전의 raw semantic descriptor만 쓴다는 의미가 아니다. FULL 대비+0.001393을 하나의 순수 attention 효과로 해석하지 않는다.

같은 전달 경로와 identity V를 유지하고 Q/K·utility7개를 추가한 E_LEARNED_RAW가 학습 중이다.288-update DEV32 macro는 fixed0.471552, learned0.471581로 **+0.000029**에 불과하다. 이 중간 panel은 위128×3 최종 결과와 다르며,5 epoch 및128×3 완료 전 learned attention의 이득을 확정하지 않는다.

## 2. Voxel IoU 외 지표에서 확인된 효과

아래는 별도의 DEV32·training seed101 하나·noise1회에서 각 SS prediction을 원본 Shape512/1024와 같은 학습된 V1 PBR1900으로 연결한 결과다. Source-cluster bootstrap4,000회의 보정 전 탐색용 CI이며, 위128×3와 표본·단계가 다르다.

| 요소 비교 | 지표 변화 | 95% CI | 설명 범위 |
|---|---:|---|---|
| C_TOP1_V − C_DECODER | Surface F-Score +0.005184 | [+0.001497,+0.009446] | V/output 변환이 surface에는 긍정적 신호; voxel IoU와 방향이 다름 |
| C_FIXED_TOPK − C_TOP1_V | Target mask IoU +0.003717 | [+0.000258,+0.007653] | Fixed Top-K의2D silhouette 개선 신호 |
| FULL − C_FIXED_TOPK | Source CLIP +0.006163 | [+0.001412,+0.011157] | Learned Q/K의 선택적인 source semantic similarity 개선 신호 |

마지막 비교에서 source LPIPS는+0.000391, DreamSim은+0.002614로 평균이 악화했고 surface F-Score는+0.000249로 CI가0을 포함했다. Learned attention이 모든 source preservation·geometry 지표를 개선했다고 쓰지 않는다. F-Score는 surface20k samples·threshold0.02이며 voxel F1과 다르다.

## 3. PBR에서는 attention의 추가 효과가 확인되지 않음

PBR4개 모델의128개 실제 case·384개 official view 평가가 완료됐다. 같은 V1 생성 geometry/native guide·초기 texture noise·sampler와 **동일한 실제 render vertices/faces**를 사용했다. 각 case의 독립적인 mesh simplification 교란을 제거한 결과다.

| PBR 모델 | Source LPIPS ↓ | Source DreamSim ↓ | Source CLIP ↑ | Target PSNR ↑ | Target SSIM ↑ |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0.053099 | 0.155692 | 0.942434 | 24.47068 | 0.950488 |
| Top-1 R | 0.052984 | 0.154160 | 0.942141 | 24.54852 | 0.950585 |
| Top-1 R+feature | 0.053019 | 0.154338 | 0.942222 | 24.51148 | 0.950536 |
| Raw Top-K+feature | 0.052986 | 0.154346 | 0.942330 | 24.52061 | 0.950544 |

Top-1 R 대비 feature 추가의 PSNR 변화는−0.037040 dB, 탐색용95% CI[−0.066157,−0.008718]이다. Raw 추가는 feature 대비+0.009127 dB지만 CI[−0.011994,+0.031999]로0을 포함한다. Raw 대 Top-1 R의 LPIPS 차이는+0.00000231, PSNR은−0.027913 dB이며 두 CI 모두0을 포함한다. 작은 GT-shape fixed4 panel에서 Raw가 좋았던 결과를 이32개 generated-geometry 평가의 우위로 일반화할 수 없다.

Top-1 R의 vanilla 대비 DreamSim 차이−0.001532는 탐색용 CI가 약[−0.003120,−0.000014]이지만, LPIPS·PSNR·CLIP까지 일관된 우위가 입증된 것은 아니다. 여러 지표·contrast를 본 뒤의 보정 전 DEV32 신호다. PBR Raw를 선택하거나 별도 joint/seed202 학습으로 확장할 근거로 쓰지 않고, 현재 등록된 R 중심 joint 비교와 필수 V1/R seed202 확인을 유지한다.

PBR의 `top1-plus-feature`에는 이미64후보·4-head learned correction이 있다. Raw mode는 R의 appearance 입력을 `v1 + Δv`에서 같은 posterior로 집계한 `v_raw_weighted + Δv`로 바꾼다. 두 mode 모두 Top-1 carrier/source-use를 유지하고, feature correction은 R만 거친다. 따라서 PBR의 이 비교를 Top-1 대 Top-K의 첫 도입으로 설명하지 않는다.

## 4. 실제 joint pipeline과 논문 구성

V1 SS+V1 PBR 대비 C_DECODER SS+Top-1 R PBR을 실제 연결한 DEV32 결과도 완료됐다. Source LPIPS−0.000995, DreamSim−0.003256, target PSNR+0.103972 dB, SSIM+0.000796, mask IoU+0.004734, CD−0.00003830, surface F-Score+0.002271이다. Source CLIP은−0.001798로 악화했다. 이는 SS/PBR 조합의 실제 결과이며 retargeter 단독 효과가 아니다. 단일 seed의 DEV32로 최종 일반화나 모든 지표의 우위를 주장하지 않는다.

논문의 중심은 **correspondence로 얻은 evidence를 pretrained image-conditioning 경로에 일관되게 전달하는 ray 기반 feature retargeting**이 적절하다.

1. Spatial correspondence의 한계를 raw semantic correspondence와 학습된 matching descriptor의 matched 비교로 제시한다. Descriptor의 이득과 confidence calibration은 분리한다.
2. Top-1의 불확실성 때문에 Top-K 후보를 유지하고 결합하는 동기를 설명한다. Fixed 결합과 learned Q/K·V/O를 별도 design choice로 제시하며, learned attention이 필수라는 주장을 앞세우지 않는다.
3. 동일 입력 카메라 ray의 voxel은 같은 image evidence를 공유한다는 설계 근거로 ray query를 설명한다. 독립적인 품질 향상량은 FULL_VOXEL_Q의 최종 matched 결과 이후 확정한다.
4. Target/native 경로를 보존하고 source feature를 R로 적응시키는 것을 핵심으로 둔다. 같은 posterior가 carrier/use/R에 일관되게 전달되는지와 원본 feature magnitude 보존을 설명한다.
5. Ablation 표는 Spatial→Raw semantic→Learned descriptor, Top-1→Fixed Top-K→Learned refinement, R 유무, query unit, 전달 경로로 구성한다. 코드 이름 FULL을 자동으로 논문의 Ours-full로 채택하지 않는다.

남은 필수 근거는 query/member 대조군과 E_LEARNED_RAW의480/128×3, D1-cos/E_FIXED_RAW joint cascade, 독립 training seed, VAL36 fresh-load/cache QA, 실제 선택을 동결한 뒤 TEST508의10개 metric과 source-cluster 통계다. FID/KID는 아직 이 DEV 비교에 포함되지 않았다.

근거: [SS128×3](2026-10-24-SS-fixed-evidence-completed480-and-final-evaluation-ko.md), [SS cascade와 고정 mesh 평가 프로토콜](2026-10-24-core5-common-PBR-DEV32-eight-metric-results-ko.md), [PBR4 corrected W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/PBR-core4-frozen-render-geometry32-v203), [V1/R joint W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/joint-V1-retarget-SS-PBR-native-guides32-v176).

후속 E_FIXED_RAW joint의 실제 geometry/appearance tradeoff와 남은 실험 실행, 논문 초안은 [필수 계획 실행 기록](2026-10-24-essential-plan-execution-and-paper-draft-ko.md)을 참고한다. 위 표의 최초 관측을 이후 결과로 소급해 바꾸지 않는다.
