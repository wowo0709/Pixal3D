# 논문 완성을 위한 남은 핵심 실험과 ablation·writing 설계

2026-10-24 19:27 UTC 실제 결과/queue 기준 갱신. 목적은 효과가 있는 모듈을 하나의 실제 파이프라인으로 확정하고, 각 기여의 근거와 한계를 같은 평가 조건에서 제시하는 것이다. 아래의 대조군 목록은 필요한 **비교**이며, 전부 신규 학습을 뜻하지 않는다. 기존 결과가 최종 구조와 일치하면 재사용한다. 진행 중인 정상 학습/평가는 유지한다. 이 문서 갱신 자체로 새로운 학습이나 TEST를 실행한 것은 아니다.

## 1. 지금 확정한 사실과 아직 결정하지 못한 것

19:43 UTC 후속 실행: [독립 V1 readout·활용/출력 연결](2026-10-24-independent-V1-readout-and-utilization-output-link-ko.md)을 완료했다. V1 seed202의 동일 DEV128×3 IoU는0.467177이다. 대기 Fixed24건은 GPU1로 실행 위치만 옮겨 첫 사례가 완료됐고, 전체 추가288건 중99건 완료·4건 실행이다. 아래19:27 상태 표의 V1 집계 대기는 해소됐으며 최종 구조·TEST·논문 완성 범위는 유지한다.

SS의 동일 DEV128·93 source cluster·generation noise3회·training seed101·480 updates에서 V1 0.467355, Top-1 R 0.472600, Fixed 0.473471, Learned 0.474794의 macro voxel IoU를 얻었다. Learned−Fixed의 module-family CI95는 [−0.000409,+0.003161]이다. 최고 점추정이 learned attention의 필수성을 증명하지 않는다.

실제 DEV32 cascade에서는 Fixed가 R보다 CD/F-Score에서 좋지만 source LPIPS/DreamSim은 나빴다. Learned도 모든 지표에서 우세하지 않았다. 같은32개/noise의 SS 순위는 Fixed > Learned였으므로, 서로 다른 panel/noise에서 얻은 순위를 연결해 구현 실패나 단계 간 정보 손실의 원인으로 단정하지 않는다.

| 항목 | 실제 상태 | 남은 의미 |
|---|---|---|
| 기존 SS 모델들 | 13개 endpoint의 480/DEV128×3 완료 | 원래 비교 family 통계는 voxel endpoint 이후 마무리 |
| FULL_VOXEL_Q | 429/480 학습 중 | 원래 FULL 문맥의 ray/voxel 대조; 새 최종 구조와의 일치도 별도 점검 |
| V1/R/D1-cos/Fixed/Learned DEV32 cascade | 5개 조합과8개 metric 완료 | 작은 개발 panel이며 최종 TEST 아님 |
| 같은 Fixed SS의 vanilla/R PBR | 실제32개 짝비교 완료 | 같은 geometry에서도 source appearance 문제가 해결됐다고 볼 근거 없음 |
| R/Fixed/Learned DEV128 cascade 확장 | 기존32개씩 재사용; 추가288개 중82개 완료,3개 실행,203개 대기 | 동일128/noise에서 geometry·appearance 비교 필요; 전체 metric 미완료 |
| 독립 SS seed202 V1/R | V1의480 학습 및 DEV128×3 평가 완료, 집계 대기; R은 GPU0 학습 중 | V1의 seed 비교 집계를 연결하고 R 종료 후 모델 간 이득의 독립 재현성 확인 |
| 실제 Fixed/Learned 활용 진단 | 같은32개 모델별 관측·native 분모 대조·집계·file4 보존 완료 | Score prior는 유효 후보에 mass를 배분하지만, 추가 learned refinement의 이득은 미확인 |
| 독립 PBR seed202 vanilla/R | 준비돼 있으나 본 학습 미완료 | 필수 확인 대상; 선택하지 않은 PBR grid는 추가하지 않음 |
| 최종 TEST508 | 품질 평가 미실행 | 모델·protocol freeze 뒤 실행 |

회수 가능한 기존 생성물을 다시 만들지 않는다. DEV128 확장은 GPU2 Fixed, GPU3 R, GPU6 Learned이며 GPU7은 voxel 대조다. 이 실행 관측 수치는 계속 변하며, 완료되지 않은 비교를 논문 결과로 채우지 않는다.

## 2. 남은 작업의 우선순위와 종료 조건

| 순서 | 필수 작업 | 결과물/종료 조건 |
|---|---|---|
| 1 | 진행 중인 voxel 대조와 R/Fixed/Learned DEV128 평가 완료 | 실제 paired8개 metric·source-cluster CI·기존 family 통계. 성능 표에 포함할 최종 SS/PBR 조합 선정 |
| 2 | 최종 구조에서 필요한 matched ablation 완성 | 아래 표의 각 주장에 단일 변경 대조가 존재. 일치하는 기존 결과 재사용, 빠진 경우만 학습 |
| 3 | Matching의 탐색→활용→출력 연결 진단 | 후보 Recall, 유효 후보 attention mass/전달량, 같은 사례의 geometry·appearance 변화 |
| 4 | 독립 training seed와 fresh-load 정합 확인 | V1/R 및 선택 후보만 seed202 확인, 선택 checkpoint의 재현성·데이터 정합 통과 |
| 5 | 최종 모델/metric/통계 동결 후 TEST508 | 중복 제거한4/5개 실제 pipeline의10개 metric·paired 통계·정성 결과 |
| 6 | 표·그림·claim을 실제 결과에 맞춰 작성 | Main/ablation/mechanism 표, Results/Discussion/Abstract/Conclusion, 최종 compiled PDF 검토 |

평가·문서 작성은 결과가 나올 때 병행한다. 마지막에 한꺼번에 metric을 계산하거나, SS IoU 순위만으로 모델을 고르지 않는다. 새 구조를 TEST에서 고르지 않는다.

## 3. 최종 Ours를 어떻게 결정할 것인가

현재의 R, Fixed, Learned 세 pipeline을 동일 DEV128/noise·공통 frozen Shape·공통 Top-1 R PBR로 비교한다. Primary geometry는 CD/F-Score, primary source preservation은 LPIPS/DreamSim이다. CLIP과 SS IoU는 보조 근거다. 임의의 metric 가중합으로 승자를 만들지 않는다.

DEV128에서 Learned의 추가 이득이 뚜렷하지 않으면 Fixed를 우선 검토하고, Fixed가 R 대비 의미 있는 downstream 이점을 제공하지 못하면 R을 강한 기준선으로 유지한다. Geometry와 appearance가 서로 반대면 trade-off를 그대로 보고한다. 최종 모델은 complexity와 관측된 이득을 함께 보고 선택하되, 특정 story를 위해 module을 남기지 않는다. 선택 당시 사용한 metric·paired 차이·불확실성·선택 이유를 기록한다.

Same-SS vanilla/R PBR의 기존 실제 짝비교는 재사용한다. 최종 SS가 달라 해당 비교가 성립하지 않는 경우에만 같은 Shape/native guide/texture noise/render mesh에서 필요한 두 PBR replay를 보완한다. 모든 SS×PBR 조합을 만들지 않는다.

## 4. 논문용 matched ablation 표

최종 모델 한 행과 아래 다섯 대조가 핵심이다. 학습은 원본 MV LoRA checkpoint, DeformNeXt split, 동일 seed·optimizer·480 updates·global batch96을 유지한다. 이름이 같다는 이유로 이전 arm을 재사용하지 않고 config·코드 경로·입출력·학습/평가 계약이 같은지 확인한다.

| 표의 행 | Full에서 바꾸는 요소 | 입증할 질문 | 기존 결과의 재사용 조건 |
|---|---|---|---|
| Final Ours | 없음 | 최종 조합이 실제로 유효한가? | 선택·동결한 실제 조합 |
| Spatial correspondence | Semantic 탐색을 원래 spatial 대응으로 교체 | 변형 상황에서 semantic 대응이 필요한가? | C_SPATIAL이 최종 carrier/use/R 등과 같아야 함 |
| Raw semantic descriptor | Matching decoder descriptor를 raw descriptor로 교체 | Decoder가 retrieval과 생성에 기여하는가? | 후보 budget·scalar score 사용·aggregation·전달 경로를 분리해 통제 |
| Top-1 | 실제 최상위 correspondence feature 하나만 사용 | 복수 후보 결합이 단일 후보보다 유효한가? | 최종 경로의 Top-1↔Fixed 공통 계약은 기존 감사 근거 재사용 |
| Voxel query | 동일 matching/전달 경로에서 개별 voxel query 사용 | Ray 단위 evidence 공유가 유효한가? | 원래 FULL_VOXEL_Q는 원래 FULL의 대조; 최종 E 경로와 다르면 별도 matched 대조 필요 |
| Without retargeting | R 변환을 물리적으로 제거하고 대응 feature 전달 유지 | 정확한 대응 외에 source feature 적응도 필요한가? | C_DECODER_NO_R가 최종 문맥과 같아야 함 |

Raw/decoder 비교에서 candidate support 자체의 변화는 retrieval 설계의 효과로 측정하되 후보 budget과 후속 전달 경로를 같게 한다. Decoder와 scalar confidence를 동시에 바꾼 비교로 각 효과를 따로 주장하지 않는다.

### Claim에 따라 필요한 추가 행

| 추가 행 | 언제 필요한가? | 단일 변경 정의 |
|---|---|---|
| Uniform-K | **Matching score 자체의 효과를 핵심 claim으로 유지할 때 필요** | 같은 Top-K 후보/value/전달 경로에서 posterior weighting만 equal weighting으로 교체 |
| Fixed probability | Learned Q/K·utility를 최종 모델에 채택할 때 | 같은 후보·identity V·carrier/use/R에서 learned refinement만 제거; Fixed/Learned 기존 근거 재사용 |
| Mean-ray descriptor | Member 후보 구성을 독립 기여로 주장할 때 | 최종 경로에서 member 구성만 평균 descriptor로 교체 |

현재 요청처럼 matching score의 사용과 효과까지 검증하려면 **Uniform-K를 이번 핵심 계획에 포함**한다. 따라서 기본 표는 Full와 다섯 대조, Uniform-K의7개 행이다. 이 중 기존 결과를 재사용할 수 있는 행을 뺀 만큼만 신규 학습한다. Learned/Member의 추가 행은 최종 모델에 실제 채택할 때만 더한다. 이 표는 최종 구조 안에서 한 요소씩 바꾸는 표이며, 기존 spatial→raw→decoder의 누적 ladder는 동기·선행 개발 근거로 보조 자료에 남긴다. 서로 다른 구조의 증가량을 더해 full의 효과로 계산하지 않는다.

사용자 정의 **No-score는 실제 Top-1**, **No-corr는 원래 spatial 대응**이다. Uniform-K는 별개의 score 대조다. 제거한 channel을0으로 채우지 않는다. 최종 구조에 deformation query가 없다면 해당 제거 arm을 추가하지 않는다.

Top-1↔Top-K만으로는 matching score 자체의 이득을 분리할 수 없다. Score 사용의 효과는 Uniform-K 비교로, 추가 학습 attention의 효과는 Fixed↔Learned로 나눈다. Shared posterior 전달은 현재 method의 구조로 설명하며 독립 causal claim을 하지 않는다. 그 claim을 채택하지 않으면 split/shared 추가 학습도 필요 없다.

Uniform-K의 구체적인 적용 범위도 실행 전에 고정한다. 후보 주소·valid mask·value·query·retargeter·source-use 정의를 유지하고, 정의한 aggregation posterior만 uniform으로 바꾼다. 같은 posterior가 여러 경로로 공유되는 구조라면 **어느 소비 경로까지 함께 바뀌는지**를 config에 기록한다. 여러 경로가 함께 바뀌면 claim은 공통 score-weighted evidence의 효과로 한정하고, 순수 attention-logit 한 곳의 효과라고 쓰지 않는다. Support가 달라지는 decoder 비교 역시 후보 탐색과 weighting을 모두 바꿨다면 decoder 모듈의 묶음 효과다.

### Ablation의 실제 평가

모든 필요한 SS 대조: 동일 DEV128×3의 macro IoU·precision·recall·voxel F1. 논문 핵심 행: 공통 frozen Shape와 공통 PBR의 고정 downstream panel에서 CD/F-Score·LPIPS/DreamSim. 작은 panel에서 불확실한 핵심 claim은 같은 DEV128로 확장하거나 claim을 낮춘다. Panel이 다른 수치를 같은 조건의 열처럼 섞지 않는다.

권장 표 열은 `Correspondence / Aggregation / Query / Retargeting | CD↓ / F-Score↑ / LPIPS↓ / DreamSim↓ / SS IoU↑`다. 같은 learned descriptor에서 scalar score 사용 여부도 명시한다. 모든 arm에 PBR 재학습을 추가하지 않는다. 추론 때 모듈을 끄는 결과는 진단이며 matched 재학습 결과와 구분한다.

## 5. 필요한 기전 검증은 하나의 고정 panel로 연결

이미 측정한 matching DEV32에서 raw→decoder descriptor의 exact@1은1.25%→7.73%, decoder scalar score 사용 시8.56%였다. Member 후보는 exact@64를50.76%→73.36%로 높였지만 원래 FULL의 생성 IoU는 mean보다 낮았다. 이는 retrieval과 생성 품질을 구별해야 할 실제 근거다.

같은 고정 사례에서 다음을 연결한다.

1. **탐색:** Raw/decoder의 Recall@1/@K·correspondence 오차, Top-1 오답이지만 정답이 Top-K에 포함되는 빈도.
2. **활용:** 그 subset에서 유효 후보의 attention mass와 실제 source correction 크기, Fixed/Uniform-K/Learned의 차이.
3. **출력:** 같은 사례의 surface 오차·source render 오차 변화와 실패 양상.

GT correspondence/visibility는 평가 분모와 진단에만 사용한다. 추론 feature에 GT visibility gate를 넣지 않는다. Native target 경로 유지, 초기 baseline 함수 복원, 무매칭 영역에서 native source fallback, 같은 SS 비교의 mesh 동일성은 기능 검사로 확인한다. 이 기능 검사는 새 학습 arm이 아니다.

### 완료된 실제 활용 진단과 남은 질문

같은32개에서 Fixed/Learned의 실제 native conditioning forward64회를 관측했다. Hook 유무의 conditioning 동일성, 두 모델의 입력·후보·prior 일치와 저장된 관측 재집계가 통과했다. 실제 GPU BF16 projection의 관측 가능 관계는50,516개이고, 과거 CPU FP32 진단의50,493개와는 다르다. CPU/GPU projection precision에 따른 patch rounding 차이를 별도로 재현했으므로 두 분모를 섞지 않는다. 이 차이를 고치기 위해 학습된 모델의 수치 경로나 데이터를 바꾸지 않았다.

| Top-1 오답·Top-K 정답 포함 subset | 정답 후보 exact probability mass |
|---|---:|
| 같은 support의 Uniform-K 진단 | 1.6266% |
| Fixed matching prior | 6.4601% |
| Learned attention | 6.1756% |

분모는 해당 subset의31,408개 관계다. Learned−Fixed는−0.2844 percentage point, 탐색용 source-cluster CI95 [−0.9776,+0.4890]pp다. Learned 확률은 prior에서 실제로 달라지지만, 추가 정답 mass 개선은 확인되지 않았다. Fixed−Uniform은+4.8335pp, CI95 [+4.3219,+5.3534]pp다. 이는 **확률 활용의 기전 근거**이며 Uniform-K matched 학습 대비 최종 생성 성능의 증명이 아니다. Exact patch mass만으로 모든 semantic 후보의 유용성을 설명할 수도 없다.

따라서 남은 기전 작업은 새 matching head 탐색이 아니라, 이 관측을 같은 사례의 surface/source-render 오차와 연결하고 Uniform-K matched 학습의 downstream 결과로 검증하는 것이다. Source-use가 거의1이라는 관측만으로 calibration 실패나 최종 오차의 원인을 확정하지 않는다.

Oracle 학습, backbone 교체, gate/temperature grid, 새로운 head의 대량 탐색은 지금 추가하지 않는다.

## 6. 독립 확인과 최종 TEST

Seed202 V1 SS의480 학습과 DEV128×3 평가는 완료됐고, 기존 seed101과 동일한 macro 집계 readout도 완료했다. V1은 seed1010.467355, seed2020.467177이다. R SS를 완료하고, 최종 후보가 다르면 그 후보 하나만 같은 seed로 확인한다. PBR vanilla/R의 필수 seed202도 완료한다. 모든 ablation을 여러 학습 seed로 반복하지 않는다. 학습 seed2개와 생성 noise3개를 혼동하지 않는다. Quick epoch IoU를 최종 DEV128×3 macro IoU와 비교하지 않는다.

TEST 전 동결 항목은 실제 checkpoint/config/code identity, DeformNeXt split, camera·RGB 입력, native sampler·generation seed, Shape/PBR composition, metric registry와 reference, contrast family 및 source-cluster CI 규칙이다. 현재 VAL36의 V1/R binding 검증과 Learned fresh-load2회 검사는 완료돼 있으며 선택 모델에 맞게 필요한 정합만 보완한다. TEST508 cache/loader QA 통과는 품질 측정 완료를 뜻하지 않는다.

Main table은 TEST508·134 source cluster의 다음 실제4/5개 pipeline으로 구성한다.

| 행 | 입력과 역할 |
|---|---|
| Zero-shot source-only4 | M0의 target-camera view를 포함한 변형 전 RGB4장: 기존 prior의 상태 |
| Zero-shot edited4 | M1 target RGB1장 + M0 source RGB3장: edited 입력 자체의 효과 |
| Vanilla V1 | DeformNeXt fine-tuning의 효과 |
| Top-1 R | Retargeting 중심의 강한 기준선 |
| Final Ours | DEV에서 선택한 실제 조합; R과 같으면 중복 제거 |

Target-only는 추가하지 않는다. Deformation 학습에 clean80k를 섞지 않는다. Test 결과를 본 뒤 architecture/checkpoint를 다시 고르지 않는다. Final-model ablation은 DEV 근거임을 표시하고 최종 일반화 claim은 TEST로 검증한다.

## 7. Metric과 표를 논문 claim에 연결

| 평가 목적 | 핵심 metric | 함께 보고할 metric | 연결할 claim |
|---|---|---|---|
| Target alignment | CD↓, surface F-Score↑ | Target PSNR↑, SSIM↑, M-IoU↑ | 편집 목표 geometry/target-view 정합 |
| Source preservation | Source LPIPS↓, DreamSim↓ | Source CLIP↑ | 원래 appearance의 유지 |
| Global naturalness | FID↓, KID↓ | 전체 panel global 집계 | Render distribution의 자연스러움 |
| Stage 진단 | SS macro voxel IoU↑ | Voxel precision/recall/F1 | Sparse-structure 학습 및 ablation |

Main table은 요청한10개 metric 모두를 세 목적의 column group으로 보고한다. Ablation 본문은 CD/F-Score와 LPIPS/DreamSim·SS IoU 중심으로 읽기 쉽게 구성하고 나머지 측정값은 보조 표에 둔다. FID/KID를 매 epoch나 작은 ablation마다 계산하지 않는다.

Target camera8, source 평가 camera9/10을 유지한다. Source reference는 M1 geometry에 original material을 적용한 같은-view GT다. 변형 전 이미지를 픽셀 단위 GT로 삼아 정당한 geometry 변화를 preservation 실패로 세지 않는다. Source CLIP은 semantic 보조 지표이며 texture 보존 단독 근거가 아니다.

CD/F-Score는 공통 normalization·20k surface sampling·threshold0.02, 결과별 ICP/scale fitting 없이 측정한다. M-IoU는2D alpha mask이며 voxel IoU와 다르다. Global FID/KID는 정해진 전체 render population에서 동일 설정으로 계산하고 signed KID를0으로 자르지 않는다.

Paired 통계 단위는 source cluster다. DEV32 screening 뒤 확장한 DEV128은 탐색적임을 표시한다. TEST contrast/CI 기준은 TEST 전에 확정한다. 불리한 metric을 숨기거나 결과를 본 뒤 primary metric을 교체하지 않는다.

현재 구현의 최종 비교 family는 selected 대 V1/R의 중복 없는 C개 contrast에 대해8개 per-asset metric×C이며 Bonferroni-adjusted source-cluster interval을 사용한다. FID/KID는 별도의 global descriptive 값이다. CI가0을 포함하는 것은 동등성이나 비열등성을 증명하지 않는다. 현재 camera8/9/10은 입력 camera 평가이므로 새 시점 일반화 claim의 근거로 사용하지 않는다. Metric 원논문·실제 preprocessing·표면 거리 정의는 [metric 대조 기록](2026-10-24-metric-definitions-citations-and-independent-V1-ko.md)에 정리했다.

## 8. 논문 산출물과 현재 가능한 writing

| 산출물 | 내용 | 아직 필요한 근거 |
|---|---|---|
| Table1 Main | 실제4/5개 pipeline ×10개 metric | 최종 freeze 뒤 TEST508 |
| Main ablation | Full와 단일 변경 대조, 필요시 Uniform-K/Fixed/Mean | 최종 문맥의 누락된 matched 비교·downstream |
| Mechanism table/figure | Retrieval coverage→attention 활용→실제 출력 변화 | 같은 panel의 활용/전달 진단 |
| Qualitative figure | GT·두 zero-shot·V1·R·Ours의 동일 asset/view, 성공/실패 | 최종 모델의 실제 render; 결과 선택 규칙 명시 |
| Supplement | 기존 ladder·seed 확인·negative results·protocol | 기존 family 통계와 독립 seed 완료 |

Method와 protocol은 이미 작성 중이며 DEV 표·정성 그림을 포함한 compiled 초안이 있다. 그러나 최종 Abstract/Results/Conclusion의 수치와 주장 강도는 확정하지 않았다. 기존 코드 이름 FULL을 논문의 최종 Ours로 자동 채택하지 않는다.

지금부터 writing은 결과와 함께 갱신한다. Method에는 실제 선택 경로와 target/source의 역할을 고정하고, Ablation에는 각 대조의 단일 변경과 재사용 근거를 명시한다. Results는 `edited 입력/FT → 최종 pipeline → 모듈 대조 → 기전` 순서로 쓴다. Discussion에는 geometry/appearance trade-off와 learned refinement의 음의 결과를 남긴다. TEST 전에는 Abstract/Conclusion에 최종 우위 수치를 넣지 않는다. 모든 표의 수치는 실제 audit에서 가져오고, PDF 컴파일·표/그림 clipping·citation을 최종 검수한다.

현재 사용할 수 있는 problem/method 설명:

> Spatial projection can become inconsistent after geometric editing. We study semantic correspondence as a source of multi-view evidence, aggregate uncertain candidates at the camera-ray level, and adapt source conditioning through a retargeting path while retaining the native target-conditioning stream. We evaluate geometric alignment, source appearance preservation, and global rendering naturalness separately.

실증 서술은 다음 순서로 구성한다.

1. 두 zero-shot과 V1로 edited 입력의 정보량과 fine-tuning 효과를 구분한다.
2. R 기준선과 최종 Ours의 실제 pipeline 차이를 geometry/appearance/global metric으로 설명한다.
3. 최종 문맥의 ablation으로 semantic retrieval, decoder, aggregation, ray grouping, retargeting의 기여를 분리한다.
4. 기전 그림으로 좋은 후보가 실제 활용되는지 보여준다.
5. Discussion에서 실패 사례·geometry/appearance trade-off·학습 seed 수·입력 view 평가 범위를 설명한다.

현재 `all modules are necessary`, `matching score improves generation`, `learned attention consistently improves preservation`, `all metrics improve`는 쓰지 않는다. Target-conditioning 경로 유지도 output RGB의 완전 보존으로 표현하지 않는다. 최종 근거가 부족한 요소는 기여에서 제외하거나 검증된 구조적 역할까지만 설명한다.

## 9. 이번 단계에서 하지 않을 실험

전체 SS×PBR 조합, 선택하지 않은 Raw/feature PBR의 seed 반복, 모든 ablation의 다중 seed, 새 backbone/head 탐색, gate/temperature sweep, source/target view 수 grid, 최종 모델에 없는 query 제거, 근거 없는 split/shared causal 비교는 추가하지 않는다. 순수 효율 claim을 채택하지 않으면 별도 latency grid도 추가하지 않는다.

남은 신규 학습 개수는 **최종 구조 선정 → 기존 결과의 계약 일치 확인** 뒤 확정한다. 현재 필요한 대조 행 수를 그대로 신규 arm 수로 세지 않는다. 종료 기준은 사전에 기대한 story의 성립이 아니라 실제 조합·모듈 기여·한계가 재현 가능한 근거로 확정되고 논문의 표와 문장이 그 근거와 일치하는 것이다.

## 근거

- [현재 모듈 효과와 claim 범위](2026-10-24-module-effects-and-paper-claims-ko.md)
- [최종 파이프라인 필수 계획](2026-10-24-final-pipeline-essential-experiments-and-writing-ko.md)
- [같은 panel 진단과 실제5개 DEV32 조합](2026-10-24-learned-cascade-complete-and-same-panel-diagnosis-ko.md)
- [동일 DEV128 실제 실행과 재사용](2026-10-24-paired-DEV128-actual-launch-and-reuse-ko.md)
- [Claim-evidence 원장](../paper/claim-evidence-ledger.md)
- [현재 논문 소스](../paper/retargeting-evidence-draft.tex)

실행 관측: `agent-handoff.json`의 실제 `PBR_NATIVE_SCIENCE_RUNTIME` 아래 `unified-module-pipeline-v1/paired-DEV128-stage-evaluation-v240` index/queue 및 `voxel-query-FFN12-recovery-v188/FULL_VOXEL_Q/science-v1/native/progress.json`. 현재 후속 집계는 `metrics-v257`이며 전체 `summary-v1/audit.json`은 관측 시 없었다. 실제 활용 진단은 `trained-ray-utilization-mechanism-v259/native-reduction-v270/audit.json`과 인접 `file4-closure.json`, geometry 차이 검증은 `address-audit-v269/audit.json`에 보존했다. 이 문서는 계획 정리이며 새 학습 등록이나 TEST 완료를 뜻하지 않는다.
