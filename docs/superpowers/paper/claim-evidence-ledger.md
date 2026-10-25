# 논문 claim과 필요한 근거

2026-10-25 현재 근거를 기준으로 정리한 원장이다. 아래는 DEV 결과이며 TEST claim이
아니다. 실험·epoch의 과거 상태를 현재의 미완료 상태로 읽지 않도록 현재 표와
검증 범위를 한곳에 모았다. 이전 원장의 상세 실행 이력은 Git commit `527fe06` 및
실제 file4의 `SS-confirmation-VAL36-PBR1140-continuation-v449/claim-evidence-ledger.md`에
보존한다. [핵심 실행 행렬](../reports/2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md)을 따른다.

## 현재 완료 범위와 남은 작업

- V1/Top-1 R/Fixed의 두 training seed·DEV128×3 집계는 완료됐다. Fixed는 V1 대비
  개선을 두 seed에서 확인했으며 Top-1 대비 추가 이득은 보정 후 미확정이다.
- Working Fixed seed101의 VAL36 두 fresh-load·72 archive·prefix32·tail4 검증은
  완료됐다. 해당 SS checkpoint의 replay QA이며 rendering 품질이나 모델 선택이 아니다.
- Uniform/no-R와 Spatial의480 학습·SS128×3는 완료됐다. Spatial의 strict 집계는
  IoU0.472267이며 v491에서 원본 full-study validator·실제 terminal·GT/noise 정합을 확인했다.
  Raw semantic/D1-cosine은 학습 중이다. 두 correspondence의 최종 SS128×3와
  신규5개 control의 cascade128·8metric·전체 family 통계는 미완료다.
- PBR V1/R×seed101/202의1900과 strict collector는 완료됐다. 실제 frozen700
  tensor×4·LoRA360×4·1900 rank trace·초기 identity·같은 seed의6epoch GT pixel을
  확인했다. Fixed4 target-stage training QA이며 inferential quality나 full-cascade
  seed 확인이 아니다. 최종 조합 선택·freeze·TEST508·최종 writing은 미완료다.
- 최신 검수된 개발 manuscript는 v494의16쪽·24개 참고문헌이다. Spatial SS cell과
  완료 PBR4행 training-QA 표를 원본 JSON에서 반영했고 수정5/10/11/13/14/15쪽을
  시각 검수·file4 보존했다. 실제 warning/미해결참조/overfull/underfull0이며 나머지
  수치 표·그림은 그대로다. 최종 control·cascade·TEST·writing 완료는 아니다.

## Claim별 현재 판정

| Claim 후보 | 현재 근거 | 미완료 검증 | 현재 허용하는 표현 |
|---|---|---|---|
| V1 기준선이 독립 학습에서도 재현된다 | Seed1010.467355, seed2020.467177; 실제480 trace·startup RNG가 다르고 초기 LoRA·학습 조건·DEV128×3 pairing 일치 | 선택 모델·PBR의 독립 효과와 full-cascade 확인은 별도 | 두 실제 V1 실행의 유사한 점추정; 동등성이나 임의 training seed population의 보장 아님 |
| Top-1 semantic matching+retargeting branch의 V1 대비 효과가 독립 학습에서 재현된다 | Seed101 ΔIoU+0.005245, seed202+0.005169; 각 cluster CI95가 양수. 양쪽480 trace pairing·학습 stream 독립·원본 frozen701 tensors 일치 확인 | Fixed seed202 SS와 PBR 두-seed training QA는 완료. Full-cascade/TEST는 별도 | 두 실제 SS training seed에서 combined branch 개선. Retargeter/decoder 단독 효과나 training seed 모집단의 보장 아님 |
| Fixed combined branch의 V1 대비 SS 개선이 두 학습에서 재현된다 | Seed101 ΔIoU+0.006116, seed202+0.007553. 각 seed의3대조 보정 CI95가 양수; 실제480 trace·초기 RNG·6개 checkpoint의 frozen701 tensor·18개 평가 binding 확인 | PBR 두-seed training QA는 완료; full-cascade/TEST는 미완료. Fixed−Top-1의 보정 CI는 두 seed 모두0 포함 | 두 실제 SS training seed에서 combined Fixed 개선. Decoder/score/R 단독 효과나 Top-1 대비 추가 이득 확정은 아님 |
| PBR 두-seed training QA가 완료됐다 | V1/R×seed101/2021900·frozen700×4·실제 rank trace·초기 latent·각 seed의6epoch GT pixel 확인. R−V1 crop LPIPS−0.001268/−0.003592, target PSNR−0.016409/−0.243949dB. 원본16장과4행 표 원격 검증 | 최종 geometry/source를 포함한 실제 cascade/TEST 품질은 별도 | Fixed4 GT-shape target8의 혼합된 점추정. 다른 seed의 quality pooling·inferential quality·full-pipeline 재현 주장 불가 |
| Semantic correspondence가 spatial보다 유효하다 | C_SPATIAL→C_RAW의 DEV128×3 macro IoU +0.001117 | 원래 family는 완료됐으며 모든10개 module CI가0을 포함. 최종 Spatial0.472267은 집계 완료; 최종 문맥 Spatial→Raw 및 downstream 대조는 미완료 | “해당 SS 설정에서 양의 점추정” |
| Matching decoder가 correspondence를 개선한다 | Raw→D1-cos IoU +0.000648; 별도 matching DEV32의 exact@1은1.25%→7.73% | 최종 구조/downstream 연결; 진단32와 SS128을 같은 paired 실험으로 혼동하지 않음 | Descriptor와 scalar score의 효과를 분리 |
| Scalar score가 생성 품질을 개선한다 | D1-cos→decoder IoU −0.000408; 별도 진단의 exact@1은7.73%→8.56% | 동일 Top-K support/value 경로의 score 대조가 필요 | Ranking 개선과 generative gain을 구별; 생성 품질 positive claim 보류 |
| Top-K attention이 Top-1보다 유효하다 | 기존 Fixed-K SS+0.000527. 최종 Fixed−Top-1은+0.000871, 사전 family6 CI[−0.001642,+0.003679] | 최종 경로 cascade·독립 확인. 기존 DEV32 mask는 다른 scope | 후보 수와 결합의 묶음 효과; 현재 최종 SS의 보정 우위 미확인 |
| Matching-aware weighting이 동일 후보를 더 잘 활용한다 | 최종 Fixed−Uniform SS−0.000142, family6 CI[−0.002590,+0.002233]. 별도 native 진단에서 GT mass 증가는 관측 | 동일 support의 실제 cascade8metric 및 기전/출력 연결 | 후보 probability mass와 생성 품질을 구분. SS 우위·무효·동등성 모두 미확정 |
| 추가 learned attention이 필요하다 | Anchored SS128×3 IoU0.474794로 Fixed보다+0.001323, CI는0 포함. 실제 cascade128에서도 Learned는 Fixed보다 CD/F-Score 점추정 열위이고 paired CI는0 포함 | 최종 선택에서 추가 이득을 입증할 근거 없음; 현재 working Fixed에는 head 미채택 | SS 최고점이 pipeline 우위를 입증하지 않으며 learned refinement의 필수성·전반적 preservation 개선은 미확정 |
| Ray 단위 처리가 품질을 개선한다 | 원래 FULL의 DEV128×3 완료: ray0.472078, voxel0.472314, Δ−0.000236; ten-contrast CI[−0.000878,+0.000351] | 최종 경로에 유효한 matched 대조 | 원래 FULL에서 양의 품질 효과 미확인. Ray 구조 설명 가능; 필수성·동등성 주장 불가 |
| Member 후보가 품질을 개선한다 | 별도 matching DEV32의 exact@64 50.76%→73.36%; 그러나 완료된 FULL mean 제어 IoU0.472679 대 member FULL0.472078 | 원래 member−mean family CI[−0.002246,+0.001262]. 최종 identity-value 경로의 독립 효과는 미입증 | 후보 coverage 개선은 확인; 기존 FULL에서 생성 품질의 양의 점추정은 아님 |
| Retargeting이 유효하다 | 기존 Top-1 R SS+0.003929. 최종 Fixed−물리적 no-R SS+0.004639, family6 CI[−0.000811,+0.010633]; 일반 CI는 양수 | 최종 weighted 경로의 cascade 비교; 단독 R 효과의 독립 seed는 미확인 | 긍정적 점추정과 보정 전 구간. 보정 후 필요성은 미확정; V1/R 두 seed는 R 단독 ablation 아님 |
| Shared evidence 전달 자체가 개선 원인이다 | E_FIXED_RAW SS0.473471; FULL과는 복수 차이 | 동일 후보/가중치/R/V의 split-vs-shared 대조가 있어야 인과 주장 | 현재는 구조 설명; 독립 causal claim 보류 |
| E_FIXED_RAW가 R보다 좋은 최종 모델이다 | DEV128 F-Score+0.003963, 탐색용 CI[+0.001011,+0.007314]; CD−0.00002230은 CI0 포함. Source LPIPS+0.000211/DreamSim+0.003100은 각각 CI0 포함 | Working Fixed/RPBR의 matched controls·TEST. PBR 두-seed training QA는 완료. Fixed seed202 SS와 working seed101 SS 재로딩 QA는 완료 | Geometry 중심의 working candidate. Preservation 개선·비열등성·최종 보편적 우위는 미확정 |
| Source 적응과 target 보존을 구조적으로 분리한다 | Native MV4 carrier·target 경로 유지, source map correction만 추가 | 실제 target/source 결과 및 실패 사례 | 입력 target stream 유지 ≠ output RGB 완전 보존 |
| Ray 구조가 더 빠르다 | 현재 실행 시간은 FFN chunk 등이 다름 | 동일 precision/batch/chunk/sampler benchmark | 순수 속도 배수 주장 보류 |
| 최종 파이프라인이 일반화한다 | TEST 미실행 | 선택·protocol freeze 후 TEST508·source-cluster 통계 | 최종 generalization claim 보류 |

## 수치와 통계의 범위

| Training seed | V1 IoU | Top-1 R | Fixed | Fixed−V1 보정 CI95 | Fixed−R 보정 CI95 |
|---|---:|---:|---:|---|---|
| 101 | 0.467355 | 0.472600 | 0.473471 | [+0.001139,+0.011309] | [−0.001356,+0.003387] |
| 202 | 0.467177 | 0.472346 | 0.474730 | [+0.002225,+0.013285] | [−0.000206,+0.005183] |

위 두-seed 확인은 동일128 assets/93 source/3 generation noises이며 각 seed와 두
고정 seed 평균에 각각3대조 family·4,000 source-cluster draws를 사용했다. 두 고정
seed 평균의 Fixed−V1은+0.006835, CI[+0.001834,+0.012196], Fixed−R은
+0.001628, CI[−0.000477,+0.003957]이다.9개 전체 동시 구간이나 training-seed
모집단 불확실성이 아니다. `completed-two-seed-collection-v314/native-result-v1`과
`Fixed-SS-two-seed-paper-and-WandB-v446`의 실제 closure가 근거다.

최종 문맥의 module study는 별도6대조 family다. 완료된3개 대조는 원래 전체6개
보정을 유지한12,000 draws이며 Fixed−Top-1/Uniform/no-R의 보정 IoU CI95가 모두
0을 포함한다(`completed-three-of-six-SS-contrasts-v409`). 신규 matching3개가 끝나면
동일 family를 완성한다. Cascade는 별도48대조 family다. 원래 FULL의10개 module
family, 이후 exploratory learned 비교, 두-seed3대조를 서로 바꾸지 않는다.

Quick32의 단일-noise epoch 값, global occupancy IoU, 최종 DEV128×3 macro per-asset
IoU를 혼합하지 않는다. Raw/D1-cos 첫 epoch96의0.460716/0.461137도 최종 표가
아니다. 초기 per-asset metric·GT/count/seed와 처음96개의 실제 학습 trace는
정확히 같으며 당시 V1은0.463801이다. 예정된480·최종128×3·cascade 비교를 유지한다.

후속 같은 seed101 quick32 epoch192는 V1/Spatial/Raw/D1-cos/Fixed/Top-1이
0.467447/0.465800/0.463297/0.463916/0.463634/0.460882다. Raw−Spatial은
−0.002503, D1-cos−Raw는+0.000619, Fixed−D1-cos는−0.000282다. 전체6개 모델의
같은 cold 기록·GT/count/noise·192 trace를 검증했다. 이는 작은 중간 panel의
기술적 점추정이며 최종 module-family/생성 품질/모델 선택의 근거가 아니다.
`correspondence-first192-paired-readout-v468`과 기존 Raw192 run의 원격 summary를
대조한 `correspondence-first192-paired-WandB-v470`에 실제 file4 근거를 보존한다.

같은 epoch288 quick32에서 V1/Spatial/Raw/D1-cos/Fixed/Top-1은
0.468973/0.468629/0.473207/0.473322/0.471552/0.467991이다. Raw−Spatial은
+0.004578(개선21/하락11), D1-cos−Raw는+0.000114(16/16),
Fixed−D1-cos는−0.001770(15/17)이다. 같은 cold·GT/count/noise·288 trace를
검증했고 `correspondence-paired-epoch288-v482`와 원격 원래 값의 보존을 확인한
`correspondence-paired-epoch288-WandB-v483`에 actual file4로 보존했다.
Semantic 전달의 긍정적 중간 신호를 decoder/scalar의 최종 효과·family·cascade·
TEST나 새 모델 선택으로 확대하지 않는다.

## 표·그림과 writing의 역할

| 산출물 | 사용 근거와 남은 범위 |
|---|---|
| `tables/dev-ss128.tex`, `dev-original-module-effects.tex` | 원래 endpoint와 완료10-module family. 최종 shared-evidence 대조와 구분 |
| `tables/dev-independent-ss.tex` | 실제 V1/R/Fixed 두 seed와 고정 seed 평균; 개별 모듈의 인과 효과 아님 |
| `tables/dev-completed-fixed-contrasts.tex` | 완료3개/원래6대조 family, 남은 결과를 가짜 값으로 채우지 않음 |
| `tables/dev-joint32.tex`, `dev-joint128.tex`, `dev-paired-pbr32.tex` | 해당 panel/noise의 실제 cascade와 같은 geometry의 PBR 비교를 구분 |
| `tables/final-ablation-template.tex` | Spatial→Raw→D1 cosine→scalar ladder와 Top-1/Uniform/no-R 직접 대조. Fixed/Top-1은 SS·cascade, Spatial/Uniform/no-R는 SS만 완료 |
| `tables/dev-independent-pbr.tex` | 완료1900의 V1/R×seed101/2024행. 원격4행·원본16장·actual file4 근거; fixed4 training QA 범위 |
| `tables/final-main-template.tex` | 동결 뒤 TEST508/134 source의4/5개 실제 pipeline·사용자 지정10metric. 모든 품질 값 미측정 |
| `dev-ray-evidence.tex` | 실제 RGB/ray member/후보/posterior. 조건부 예시이며 평균 품질 효과의 근거 아님 |
| 최종 정성 그림 | v355 metadata-only 고정2사례와 conditional lower-median 정책. 실제 TEST 이미지 미완료 |

Ablation 본문은 SS IoU/CD/F-Score/LPIPS/DreamSim을 중심으로 해석하고 전체
8metric은 supplement에 모두 보고한다. Main은 LPIPS/DreamSim/CLIP,
FID/KID, PSNR/SSIM/M-IoU/CD/F-Score의10개 metric이다. Model 선택은 geometry와
appearance를 함께 사용하며 임의 가중합으로 승자를 만들지 않는다. FID/KID는
공통 final render population의 분포 요약이며3D 물리적 타당성의 증거가 아니다.

Geometry 향상과 appearance 손실이 함께 있으면 trade-off를 적는다. CI가0을
포함한다는 사실로 효과 부재·동등성·비열등성·source 보존을 선언하지 않는다.
“All metrics improve”, “all modules are necessary”, “learned attention is essential”은
현재 허용하지 않는다. Final Results/Discussion 뒤 Abstract/Conclusion을 확정한다.

## 구현·reference와 기전의 해석

현재 Fixed는 native MV4 carrier 위에서 source feature만 보정한다. Native target
경로와 global tokens는 유지하지만 output target RGB가 완전히 보존된다는 뜻은 아니다.
Matching probability가 carrier/use/retargeter에 공유되며 별도 learned Q/K head는 없다.
따라서 learned attention 필요성이나 shared routing의 단독 인과 효과를 주장하지 않는다.

실제 사용 matcher는 `PromotedR9Matcher`의 RGB production checkpoint다.
`matcher-training-method-provenance-v439`에서7epochs/62,545 updates,
train9143/validation508·9651 teacher input fingerprint·현재 native RGB SHA를 확인했다.
Teacher 제외6개는 matcher에만 적용하며 SS train9149와 PBR train9139를 혼합하지
않는다. Legacy depth-input production 로그를 현재 RGB 모델의 학습 근거로 쓰지 않는다.
Offline topology/depth teacher는 inference GT visibility gate가 아니다.

Source camera9/10 reference는 edited M1 geometry의 원래 source material이다.
따라서 undeformed M0 RGB와의 픽셀 일치가 아니라 deformation 후 appearance 보존을
평가한다. Target camera8의 편집 목표와 source 역할을 분리하며 입력 camera 평가를
새 시점 일반화로 쓰지 않는다. CD/F-Score에는 결과별 ICP나 scale fitting을 하지 않는다.

현재 학습 config·DEV panel·TEST manifest를 metadata로 다시 확인한 v477에서
train9149/validation508/TEST508의 sample ID·source UID·source cluster 교집합은
모두0이다. Source cluster 수는 각각1790/132/134이며 DEV128은 validation의
128개/93 source subset이다. 원본 local/actual file4 manifest SHA·size는 기존 감사와
같다. 이는 deformation split 사이의 분리이며 이전 mixed385 pretraining과의 source
무교차를 입증하지 않는다. 이 검증에서 TEST 예측·품질·geometry 값을 열지 않았다.

PBR fixed4 stage trend는 같은 training seed 안의 V1/R·모든 epoch에서 GT PNG pixel의
정확한 일치를 요구한다. 독립 seed 사이에는 native 초기-render bound 안에서 GT pixel이
다를 수 있어 각 seed의 reference와 metric을 유지한다. 초기 RGB/camera/noise/GT latent/
cold prediction latent는 같은 계약을 따른다. 이는 final cascade/TEST의 공통 GT 규칙을
완화하지 않으며 generated geometry나 source preservation의 확인도 아니다.

최신 실제 PBR R2021520은 V1/R render PSNR26.675535/26.510421,
crop LPIPS0.088800/0.086774다. R의 LPIPS 점추정은 낮지만 PSNR 격차는1140보다
커진 mixed target-stage 경향이다. 네 rank1520 trace와 같은 seed의 GT pixel을
검증했고 원래8개 panel·원격 W&B summary·actual file4 SHA/size를 보존했다
(`PBR202-R-actual-epoch1520-v413`, `PBR202-epoch1520-WandB-v414`).
PBR1900·두 seed collector·full-cascade/source/TEST나 모듈의 필수성으로 해석하지 않는다.

실제 BF16 관측의50,516 observable relations 중 Top-1 miss/Top-K hit31,408개에서
exact GT probability mass는 diagnostic Uniform1.6266%, Fixed6.4601%, Learned6.1756%다.
Learned−Fixed−0.2844pp의 탐색용 CI95는[−0.9776,+0.4890]pp다. 후보 활용의 진단과
matched Uniform 학습의 최종 생성 품질을 구분한다. 과거 CPU FP32의50,493개와 다른
분모는 native projection rounding으로 확인했으며 GPU/CPU precision parity를 가정하지 않는다.
같은32사례의 GT-mass 변화와 CD/F-Score/LPIPS/DreamSim 출력 변화의 association도
네 탐색용 CI가0을 포함하므로 최종 품질로의 연결은 아직 미입증이다.

실제 ray 그림은 사전순 첫 Top-1 miss/Top-K hit·관측 member2개 이상 사례이며
GT 후보 rank2·mass25.64%는 조건부 예시다. 최종 TEST 그림의 선택 정책과 구분한다.
Learned/head/member/ray의 음의 결과는 supplement와 discussion에 남긴다.

## 최종 TEST 전후 완료 조건

첫 실제 DEV cache 회수의 postcondition은 v488에서 통과했다. 지정 로컬3개 NPZ만
없고 나머지30개 case file이 원래 SHA/size와 같으며 NAS 원본3개도 회수 후 같은
값이다. 이는 storage integration 근거이며 품질 효과나 전체640/TEST 완료가 아니다.

1. 최종 문맥7행 SS·cascade와 paired family 통계를 완료하고 PBR 독립 확인을 마친다.
2. 실제 증거로 SS/Shape/PBR 조합을 선택한다. Fixed 추가 이득이 지지되지 않으면
   Top-1 R을 선택할 수 있다. 새로운 A/M control이 선택되면 실제 binding/export를
   구현하며 Fixed라는 이름으로 바꾸지 않는다. 원래 계획의 선택 후보 독립 확인에
   따라 그 SS 구조의 seed202와 해당 VAL36 QA가 필요하다. 완료된 V1/R202는
   재사용하며 Fixed202의 근거를 다른 구조로 옮기거나 전 ablation의 seed를 늘리지 않는다.
3. Checkpoint/config/code/split/camera/RGB/sampler/noise/composition/metric/statistics를
   동결한다. v424 packet의 qualitative spec과 source map을 기존 numerical pins에
   병합하고 원본 manifest의 actual read-only receipt를 유지한다.
4. 동결 TEST508의10metric·source-cluster 통계·실제 정성 그림을 생성한다.
   Ours가 R과 같은 실제 조합이면 중복 행을 제거하며 TEST로 구조를 다시 고르지 않는다.
5. 같은 frozen result에서 Main/Ablation/Mechanism/Supplement를 내보내고
   Results/Discussion/Abstract/Conclusion의 수치·claim을 대조한다. 최종 PDF를
   실제 compile·시각 검수한 뒤에만 논문 작성을 완료로 표시한다.

[검증한 선행연구와 claim 범위](../reports/2026-10-25-closest-method-scope-and-writing-ko.md)에
따라 projection conditioning, source multi-view 입력, zero initialization,
correspondence-aware attention 자체를 최초 기여로 쓰지 않는다. 선행연구의 공식 코드와
구조 비교는 상대 성능 우위의 근거가 아니다. Published Pixal3D와 supplied native
TRELLIS.2/DINOv3 구현의 lineage도 구분한다. 실행하지 않은 외부 방법보다 우수하다거나
SOTA라는 표현은 현재 비교가 지지하지 않는다.
