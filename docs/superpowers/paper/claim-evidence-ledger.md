# 논문 claim과 필요한 근거

이 문서는 결과를 본 뒤 유리한 metric으로 주장을 바꾸지 않기 위한 원장이다. 아래 empirical claim은 DEV 단계의 해석이며, 최종 TEST claim으로 승인된 것이 아니다. 현재 두 리포트의 계획을 유지한다.

남은 실행과 표·writing의 최신 통합 계획은 [논문 완성을 위한 핵심 실험 정리](../reports/2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)를 따른다. Matching score 효과를 검증하기 위해 동일 Top-K의 Uniform-K 대조를 핵심 계획에 포함하며 사용자 정의 No-score=Top-1과 구분한다.

| Claim 후보 | 현재 근거 | 미완료 검증 | 현재 허용하는 표현 |
|---|---|---|---|
| Semantic correspondence가 spatial보다 유효하다 | C_SPATIAL→C_RAW의 DEV128×3 macro IoU +0.001117 | 원래 finite-family 통계, 최종 구조의 matched 비교 및 downstream | “해당 SS 설정에서 양의 점추정” |
| Matching decoder가 correspondence를 개선한다 | Raw→D1-cos IoU +0.000648; 별도 matching DEV32의 exact@1은1.25%→7.73% | 최종 구조/downstream 연결; 진단32와 SS128을 같은 paired 실험으로 혼동하지 않음 | Descriptor와 scalar score의 효과를 분리 |
| Scalar score가 생성 품질을 개선한다 | D1-cos→decoder IoU −0.000408; 별도 진단의 exact@1은7.73%→8.56% | 동일 Top-K support/value 경로의 score 대조가 필요 | Ranking 개선과 generative gain을 구별; 생성 품질 positive claim 보류 |
| Top-K attention이 Top-1보다 유효하다 | Fixed-K의 SS +0.000527 및 별도 DEV32 mask +0.003717 | 최종 경로의 matched 비교·독립 확인 | 후보 수와 결합의 묶음 효과로 한정 |
| 추가 learned attention이 필요하다 | 새 anchored SS128×3 IoU0.474794로 fixed보다+0.001323, CI는0 포함; 완료된 cascade32는 R보다 LPIPS/DreamSim/CD 점추정 열위·F-Score 우위, 모두 CI는0 포함 | 더 넓은 paired cascade와 독립 확인; 최종 문맥의 비교 | SS 최고점이 pipeline 우위를 입증하지 않으며 필수성·전반적 preservation 개선은 미확정 |
| Ray 단위 처리가 품질을 개선한다 | 설계와 실제 구현 확인 | FULL_VOXEL_Q의480/128×3 및 최종 경로 일치 | 원리 설명 가능, 독립 성능 기여 미확정 |
| Member 후보가 품질을 개선한다 | 별도 matching DEV32의 exact@64 50.76%→73.36%; 그러나 완료된 FULL mean 제어 IoU0.472679 대 member FULL0.472078 | 원래 finite-family 통계와 최종 identity-value 경로의 matched 확인·downstream 연결 | 후보 coverage 개선은 확인; 기존 FULL에서 생성 품질의 양의 점추정은 아님 |
| Retargeting이 유효하다 | C_NO_R→C_DECODER SS +0.003929; 실제 V1/R joint 완료 | 독립 학습 seed, 최종 경로의 R 제거 비교 | 현재 matched 설정의 효과로 한정 |
| Shared evidence 전달 자체가 개선 원인이다 | E_FIXED_RAW SS0.473471; FULL과는 복수 차이 | 동일 후보/가중치/R/V의 split-vs-shared 대조가 있어야 인과 주장 | 현재는 구조 설명; 독립 causal claim 보류 |
| E_FIXED_RAW가 R보다 좋은 최종 모델이다 | DEV32 CD/F-Score 개선, source LPIPS/DreamSim 악화 | 선택 SS에서 PBR 짝비교, 독립 확인, TEST | Geometry/appearance tradeoff; 보편적 우위 주장 금지 |
| Source 적응과 target 보존을 구조적으로 분리한다 | Native MV4 carrier·target 경로 유지, source map correction만 추가 | 실제 target/source 결과 및 실패 사례 | 입력 target stream 유지 ≠ output RGB 완전 보존 |
| Ray 구조가 더 빠르다 | 현재 실행 시간은 FFN chunk 등이 다름 | 동일 precision/batch/chunk/sampler benchmark | 순수 속도 배수 주장 보류 |
| 최종 파이프라인이 일반화한다 | TEST 미실행 | 선택·protocol freeze 후 TEST508·source-cluster 통계 | 최종 generalization claim 보류 |

## 결과 표의 역할

- `tables/dev-ss128.tex`: 동일 SS panel의 완료된13개 endpoint와 조건부 비교. 원래 core family와 새 exploratory follow-up은 통계적으로 구분하고 Micro IoU와 섞지 않는다.
- `tables/dev-joint32.tex`: V1/R/D1-cos/E_FIXED_RAW/E_LEARNED_RAW의 실제5개 조합 비교. V1/R baseline의 per-asset 일치 검증을 거친 결과다.
- `tables/dev-matching32.tex`: 재사용한 matching 진단의 observable relation coverage. GT visibility는 진단 분모에만 사용하며 모델의 write gate가 아니다.
- `tables/final-main-template.tex`: TEST508, 사용자 지정10개 metric. 미측정은 대시로 남긴다.
- `tables/final-ablation-template.tex`: 최종 구조에서의 ablation. 기존 조건부 비교로 빈 행을 자동 채우지 않는다.

## Writing에서 지킬 해석

Primary geometry는 CD/F-Score, primary source appearance는 LPIPS/DreamSim이다. CLIP은 semantic 보조 지표이고 FID/KID는 global naturalness다. “All metrics improve”, “all modules are necessary”, “learned attention is essential”은 현재 근거에 없는 문장이다. 원래 비교 family와 이후 탐색적 follow-up, training seed와 generation noise를 구분한다. 음의 결과와 실패 사례도 discussion에 보존한다.

`dev-qualitative.tex`의 실제 render figure는 Fixed 대 R의 F-Score 증가 최대와 source LPIPS 증가 최대 사례다. 결과 조건으로 선택한 설명용 사례이므로 평균 효과나 성공 빈도의 근거로 사용하지 않는다. Full-frame GT pixel 일치와 기존 metric annotations를 검증했으며, appearance 차이의 causal mechanism이나 final Ours/TEST 품질을 입증하지 않는다.

실제 cascade와 같은32 asset/noise에서 SS IoU는 R0.476967·Fixed0.478940·Learned0.477442다. Full128×3의 learned/fixed 순위와 다르다. 서로 다른 panel/noise의 순위 차이만으로 모듈 구현 오류나 loss의 causal failure를 결론내리지 않는다.

학습된 Fixed/Learned의 실제 conditioning을 같은32개에서 관측하는 [활용 진단을 구현·연결했다](../reports/2026-10-24-trained-attention-utilization-preparation-and-paper-ko.md). 실제 GPU 결과는 pending이다. Top-1 miss/Top-K hit subset의 GT mass·source-use·실제 correction과 기존 downstream metric의 association을 읽으며, diagnostic Uniform-K를 matched 학습 ablation으로 대체하지 않는다.
