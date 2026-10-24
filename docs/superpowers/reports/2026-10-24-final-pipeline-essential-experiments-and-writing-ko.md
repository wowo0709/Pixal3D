# 최종 파이프라인 검증에 필요한 실험·평가·논문 작성

2026-10-24 16:23 UTC 실제 결과와 queue 관측 기준. 이 문서는 기존 필수 계획의 우선순위를 최신 결과에 맞게 구체화한다. 새 실험을 등록했다는 뜻은 아니다. 진행 중인 학습과 평가는 유지하고, 최종 구조에서 빠진 비교만 추가한다.

## 판단의 출발점

같은 DEV128·generation noise 3개·training seed101·480-update SS 평가에서 다음 결과를 얻었다. Noise 3개는 독립 학습 seed 3개가 아니다.

| SS 모델 | Macro voxel IoU | 해석 |
|---|---:|---|
| Vanilla V1 | 0.467355 | 같은 panel의 기준선 |
| Top-1 retargeting R | 0.472600 | V1 대비 +0.005245 |
| E_FIXED_RAW | 0.473471 | R 대비 +0.000871 |
| E_LEARNED_RAW | 0.474794 | Fixed 대비 +0.001323; 최종 후보이지 확정 모델은 아님 |

E_LEARNED_RAW−V1의 해당 pipeline-family CI95는 [+0.002585,+0.012504]이지만, learned−fixed의 module-family CI95는 [−0.000409,+0.003161], learned−R의 pipeline-family CI95는 [−0.000521,+0.004981]이다. 새 learned 경로의 양의 점추정과 개별 모듈의 필수성을 구분한다.

기존 FULL의 member 구성은 mean descriptor 대비 IoU가 −0.000601이었다. 반면 별도 matching DEV32에서는 exact correspondence coverage@64가 50.76%에서 73.36%로 증가했다. **후보 탐색 개선과 최종 생성 품질 개선은 같은 결론이 아니다.** Ray-query 성능 대조도 아직 완료되지 않았다.

E_FIXED_RAW+R의 실제 DEV32 cascade는 R baseline보다 CD/F-Score가 좋아지고 source LPIPS/DreamSim은 나빠졌다. 같은 SS/Shape geometry에서 vanilla PBR와 R PBR를 짝비교해도 appearance 문제가 해결됐다고 보기 어렵다. 따라서 IoU만으로 최종 Ours를 정하지 않는다.

## 1. 지금 진행 중인 핵심 비교와 후보 평가를 끝낸다

| 작업 | 현재 상태 | 완료 후 결정 |
|---|---|---|
| 원래 FULL_VOXEL_Q | 학습 중; 이후 128×3 평가 필요 | 원래 FULL 문맥에서 ray/voxel의 조건부 효과와 기존 비교 family 통계 |
| FULL_MEAN_MATCH | 학습·128×3 평가 완료 | member가 생성 성능을 높인다는 주장은 현재 보류; 원래 family 통계에 포함 |
| E_LEARNED_RAW | 학습·128×3 평가 완료 | 추가 SS 재학습 대신 실제 cascade 결과 확인 |
| E_LEARNED_RAW+R DEV32 | Shape 생성 27/32 완료, 2개 실행·3개 대기; PBR replay·최종8개 metric 대기 | 후보의 geometry/appearance trade-off를 R·Fixed와 직접 비교 |
| E_FIXED_RAW의 vanilla/R PBR 짝비교 | 32개 및8개 metric 완료 | 같은 geometry에서 PBR 선택 효과 재사용; 전 조합으로 확대하지 않음 |
| 독립 SS seed202 V1/R | GPU1/0에서 각각 학습 중 | 기존 기준선 증가량의 재현성 확인 |
| VAL36 fresh-load 검사 | V1/R 기존 결과의 binding 일치 확인, learned 후보의2회 독립 load 검사 통과 | SS checkpoint/cache 재현성 근거로 사용; TEST 품질 평가와 구별 |

후보 선택은 target CD/F-Score와 source LPIPS/DreamSim을 함께 본다. 동일 asset별 paired 차이와 불확실성도 보고한다. Geometry 우위와 appearance 열위가 함께 남으면 trade-off로 설명한다. 어떤 metric을 어느 정도 양보할지 결정하지 않은 상태에서 임의의 종합 점수로 우승 모델을 만들지 않는다.

## 2. 최종 구조를 하나 정한 뒤, 그 구조에서 모듈 기여를 분리한다

먼저 완료된 arm의 코드 경로·초기 checkpoint·학습 budget·입력·평가 panel이 최종 모델과 일치하는지 확인한다. 일치하면 그대로 재사용한다. 아래는 **필요한 대조 정의**이며 모두 신규 학습이라는 뜻이 아니다.

| 표의 행 | Full에서 바꾸는 요소 | 검증할 질문 |
|---|---|---|
| Full candidate | 없음 | 조합 전체의 실제 효과 |
| Spatial correspondence | Semantic correspondence 대신 원래 spatial 대응 사용 | Semantic 탐색이 필요한가? |
| Raw semantic correspondence | Learned matching decoder를 제거하고 raw descriptor 사용; 같은 후보 budget·후속 경로 유지 | Decoder 학습의 기여가 있는가? |
| Top-1 | 대응 feature 하나만 사용; R과 나머지 경로 유지 | 복수 후보 결합이 단일 후보보다 유효한가? |
| Voxel query | Ray를 공유 단위로 삼는 대신 개별 voxel 단위로 계산 | Ray 단위 evidence/query가 품질에 기여하는가? |
| Without retargeting | R 변환을 물리적으로 제거하고 대응 feature의 직접 전달 경로 사용 | 대응을 찾는 것 외에 feature 변환도 필요한가? |

최종 모델에 member 구성과 learned probability refinement를 유지한다면 다음 두 행도 필요하다.

| 조건부 필수 행 | 정확한 대조 |
|---|---|
| Mean-ray descriptor | 최종 경로를 유지한 채 member 후보 구성만 mean descriptor 구성으로 대체 |
| Fixed probability | 동일 evidence·identity V·carrier/use/R에서 learned refinement만 제거; E_FIXED_RAW/E_LEARNED_RAW는 이 비교의 기존 근거 |

기존 FULL과 새 identity-value 경로는 다르므로, 기존 FULL의 mean/voxel 결과를 새 Full의 matched ablation으로 자동 대체하지 않는다. 기존 Top-1 R과 Fixed 경로가 같은 carrier/use/R·초기화라는 점은 재사용 검토의 출발점이며, 실제 실행 fingerprint와 비교 계약까지 확인한다.

후속 `final-context-Top1-fixed-reuse-audit-v225`에서 실제 closed config3개의 common numerical contract·dataset·validation·learning rate 일치를 확인했다. Top-1/Fixed의 conditioner config 차이는 ray mode와 aggregation뿐이다. Source AST의 target/posterior/use/mass/numerator가 같고, 두 E 후보의 기존 실제 GPU proof에서 초기 core10이 Top-1과 같음을 확인했다. 따라서 Top-1↔Fixed, Fixed↔Learned 비교의 기존 근거를 재사용한다. 이 감사는 새로운 NN forward가 아니며 mean/voxel/raw/spatial/no-R의 최종 문맥 대조까지 충족하지는 않는다.

사용자 정의에 따라 **No-score는 Top-1**, **No-corr는 원래 spatial 대응**이다. 제거한 feature channel을0으로 채우지 않는다. Deformation query가 최종 모델에 없다면 그 제거 arm도 없다. R 제거 시 native target stream과 source fallback 계약은 유지한다.

### Matching-aware attention의 claim을 분리한다

Top-1 대 score-weighted Top-K는 후보 수와 가중 결합의 묶음 효과다. 논문에서 **matching score 자체의 기여**까지 주장하려면 동일 Top-K 후보·value·전달 경로에서 posterior weighting 대 equal weighting 대조가 필요하다. 이 보완 대조는 `Uniform-K`로 표기하며 사용자 정의 No-score와 구분한다.

추가 learned Q/K·utility의 효과는 fixed 대 learned 행으로 검증한다. Frozen matching probability를 쓰는 attention과 추가 학습 head를 같은 모듈로 뭉뚱그리지 않는다. Shared evidence 전달 자체의 인과적 우위를 별도 기여로 주장하지 않으면 split/shared 경로 실험은 추가하지 않는다.

### 학습·평가 budget

빠진 대조는 우선 SS에서 같은 원본 MV checkpoint·seed·480 updates·global batch96·동일 optimizer와 sampler로 맞춘다. 추론 시 모듈을 꺼 본 결과는 기능 진단이며 matched 재학습 ablation을 대신하지 않는다.

각 필요한 SS arm은 동일 DEV128×3의 IoU/precision/recall/voxel F1을 평가한다. 최종 ablation 표에 들어갈 행은 공통 frozen Shape512/1024와 공통 PBR 설정으로 작은 고정 downstream panel의 CD/F-Score를 측정하고, preservation claim에 사용할 행은 source LPIPS/DreamSim도 측정한다. 모든 SS arm마다 PBR를 다시 학습하거나 전체 SS×PBR 조합을 만들지 않는다. DEV128의 SS 수치와 DEV32의 rendering 수치는 표 머리글에서 panel을 구분한다.

## 3. Matching 개선이 실제 활용되는지 한 번 연결해서 확인한다

이미 확보한 matching DEV32/caches를 먼저 재사용한다. 기존 진단에서 Top-1은 틀리지만 정답이 Top-64에 포함되는 경우가 전체 관측 relation의62.17%였다. 이는 Top-K를 사용할 동기이며 attention의 효과를 입증한 결과는 아니다.

필요한 연결 검증은 다음 세 가지다.

1. Raw/decoder/member의 Recall@1·@K와 대응 오차: 탐색 문제를 얼마나 줄였는가?
2. Top-1 오답·Top-K 정답인 부분에서 유효 후보에 준 attention mass와 실제 전달량: 찾은 정보를 사용하는가?
3. 같은 사례의 SS·surface·source render 변화: 전달이 최종 예측을 개선하는가?

GT 가시성은 이 진단의 평가 분모로만 사용한다. 추론 모듈에 새 visibility write gate를 넣지 않는다. 매칭 불확실성이 큰 영역에서는 native source fallback과 target stream 보존을 기능 검사한다. 새 oracle 학습이나 대규모 head 탐색을 기본 계획으로 늘리지 않는다.

## 4. 독립 확인 후 최종 TEST508을 실행한다

- 진행 중인 SS V1/R의 seed202를 완료하고 최종 후보가 다르면 그 후보만 seed202로 확인한다. 모든 ablation을 여러 training seed로 반복하지 않는다.
- 필수 PBR vanilla/R의 seed202 확인을 완료한다. 최종 파이프라인이 채택하지 않은 Raw/feature PBR의 반복은 늘리지 않는다.
- 최종 checkpoint·dataset split·실제 코드 identity·camera·sampler·metric implementation·통계 contrast를 동결한다. 실제 DEV GPU render adapter 호환 검사는 완료됐으며 최종 freeze에 그 source59 lineage를 반영한다.
- 이후 TEST508에서 아래4/5개 실제 pipeline을 실행한다. Ours가 기존 조합과 같으면 중복 행을 제거한다.

| Main table 행 | 입력/학습 조건 |
|---|---|
| Zero-shot source-only4 | Deform 전 M0의 target-camera RGB를 포함한 source4장 |
| Zero-shot edited4 | M1 target RGB1장 + M0 source RGB3장 |
| Vanilla V1 | DeformNeXt fine-tuning, 새 모듈 없음 |
| Retargeting baseline R | 기존 Top-1 R 조합 |
| Final Ours | DEV에서 선택·동결한 실제 SS/PBR 조합 |

Zero-shot 둘은 편집 입력의 정보량과 fine-tuning의 필요성을 구분한다. Target-only는 실행하지 않는다. Deformation 학습에는 DeformNeXt split을 사용하며, 원래 MV 사전학습의 clean80k를 섞지 않는다. TEST를 보고 구조나 checkpoint를 다시 고르지 않는다.

## 5. Metric·표·claim을 함께 확정한다

| 목적 | 중심 metric | 함께 보고할 metric |
|---|---|---|
| Target 3D alignment | CD↓, surface F-Score↑ | Target PSNR↑, SSIM↑, M-IoU↑ |
| Source preservation | Source LPIPS↓, DreamSim↓ | Source CLIP↑ |
| Global naturalness | FID↓, KID↓ | 전체 TEST panel에서 global 집계 |
| SS 학습/ablation 진단 | Macro voxel IoU↑ | Voxel precision/recall/F1 |

Main table은 사용자 지정10개 metric을 모두 보고한다. Target camera8과 source camera9/10, M1 geometry에 original material을 적용한 reference를 유지한다. CD/F-Score는 공통 normalization·surface sampling·threshold를 사용하고 결과별 ICP를 하지 않는다. FID/KID는 per-asset metric으로 다루지 않으며 음수 KID를0으로 자르지 않는다.

통계 단위는 source cluster다. Paired generation noise와 독립 training seed를 구분하고, 최종 비교 family와 bootstrap/CI 규칙은 TEST 전에 정한다. DEV의 탐색적 CI를 TEST의 확정 근거로 재사용하지 않는다.

**표 구성**

1. Main table: 실제4/5개 pipeline ×10개 metric. 세 목적에 따라 column group을 나눈다.
2. Main ablation: 최종 구조와 위 matched 제거/대체 행. CD/F-Score·LPIPS/DreamSim이 중심이고 SS IoU를 보조로 둔다.
3. Mechanism table/figure: Recall@K, Top-1 실패·Top-K 성공, attention 활용과 downstream 변화.
4. 보조 표: 기존 조건부 ladder, seed 확인, 부정적인 member/score/추가 attention 결과와 실패 사례. 최종 구조에서의 효과와 섞지 않는다.

**Writing 구성**

- Method: native target conditioning 유지 → ray 단위 source evidence → semantic/matching 후보 → Top-K attention → retargeting/source fallback. 최종 후보에 실제 포함된 요소만 method의 핵심 기여로 적는다.
- Results: 입력 정보의 효과, fine-tuning의 효과, 최종 조합의 효과, 모듈별 matched 근거 순서로 쓴다.
- Discussion: 후보 탐색 개선이 생성 품질로 이어지는 조건, geometry/appearance trade-off, 작은 train-seed 수와 입력 view 평가의 범위를 설명한다.
- Figure: GT·두 zero-shot·V1·R·Ours를 같은 asset/view로 비교하고, 큰 변형·얇은 구조·매칭 실패를 포함한다. 실제 ray/member 후보와 attention을 시각화한다.
- Abstract/Conclusion: 최종 TEST와 matched ablation이 나온 후 수치와 주장 강도를 확정한다. 현재 compiled PDF는 개발 결과 초안이며 최종 논문이 아니다.

현 근거로 `모든 모듈이 필수`, `matching score가 생성 품질을 개선`, `learned attention이 일관되게 우수`, `모든 metric 개선`을 쓰지 않는다. 원래 FULL에서의 member 음의 결과도 숨기지 않는다. 후보 탐색 개선은 matching claim, 최종 3D 개선은 downstream claim으로 나누고 둘을 연결하는 실제 근거를 제시한다.

## 실행 순서와 종료 기준

**진행 중인 voxel 대조와 learned cascade 완료 → 최종 후보 선택 → 재사용 점검 후 빠진 matched ablation만 수행 → 독립 seed 확인 → 최종 freeze·TEST508 → 전체 metric/통계·정성 그림 → 최종 표·writing·PDF 검토** 순서다. 평가와 writing은 결과가 나올 때마다 병행한다.

Temperature/gate grid, backbone 교체, 모든 SS×PBR 조합, 채택하지 않은 모듈의 대규모 확장은 제외한다. 최종 구조의 핵심 효과가 재현되지 않으면 그 모듈을 뺀 간결한 모델을 채택하고 claim을 줄인다. 실험의 종료 기준은 미리 정한 스토리를 입증하는 것이 아니라, 실제 효과가 있는 조합과 그 근거를 확정하는 것이다.

## 근거와 후속 기록

- [기존 필수 계획](2026-10-24-essential-remaining-experiments-and-paper-plan-ko.md)
- [실행 기록](2026-10-24-essential-plan-execution-and-paper-draft-ko.md)
- [Learned attention 최종 SS 결과](2026-10-24-anchored-attention-final-SS-and-paper-validation-ko.md)
- [Matching 기전 근거](2026-10-24-matching-mechanism-paper-evidence-ko.md)
- [동일 SS의 PBR 짝비교와 validation 재사용](2026-10-24-fixed-evidence-paired-PBR32-and-validation-reuse-ko.md)
- [Claim-evidence 원장](../paper/claim-evidence-ledger.md), [논문 소스](../paper/retargeting-evidence-draft.tex)

관측에 사용한 실제 audit은 결과 root `unified-module-pipeline-v1` 아래 다음 위치에 있다.

- `consistent-raw-evidence-followup-v177/four-model-summary-v1/audit.json`
- `mean-match-final-descriptive-v221/audit.json`
- `fixed-evidence-SS-paired-PBR-native32-v209/same-SS-pair-summary-v1/audit.json`
- `learned-evidence-candidate-VAL36-v220/parity-v1/audit.json`
- `baseline-VAL36-reuse-verification-v217/audit.json`

이 목록의 완료 결과와 진행 중인 learned cascade/seed202/voxel 대조를 구분한다. TEST508과 최종-context ablation의 미측정 수치를 추정해 채우지 않는다.
