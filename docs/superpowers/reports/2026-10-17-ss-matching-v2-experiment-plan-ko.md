# SS 중심 matching v2 실험 계획

작성일: 2026-10-17 KST  
대상: DeformNeXt-10k / multi-view Pixal3D SS64 / VGGT-Omega epoch-7

실행 범위 복원: [SS 전체 모듈 비교와 현재 큐](2026-10-17-ss-full-module-restoration-ko.md)에 원래 필수 Top1/rejection/attention과 adapter/query 다섯 arm을 native projection에서 실행하는 계획을 기록했다. 별도 P1/P2 interface 진단은 이 다섯 비교를 대체하지 않는다. 아래 최초 nearest gate·single-GPU 계획과 구분해 읽는다.

## 1. 결정 사항과 실험 범위

Matching 개선을 먼저 검증하고, 개선이 확인된 구성에 retargeting adapter와 deformation-aware query를 순서대로 추가한다. 새 SS 학습은 **필수 3개, 조건부 2개로 최대 5개**다. Matching head 실험과 기존 checkpoint를 이용하는 진단은 SS denoiser 학습과 별도로 집계한다.

- 새 SS run은 GPU 한 장, global batch 96, 5 epochs를 기본으로 한다. Microbatch는 실제 메모리 측정 후 가능한 최대로 정한다.
- 기존 RGB zero-shot, V1, B1~C1 checkpoint는 재평가에 사용한다. V1은 다시 학습하지 않는다. 진행 중인 기존 C2는 결과가 완성되면 이력 비교에 추가한다.
- 각 새 SS arm은 **동일한 원본 mixed385 base+LoRA checkpoint에서 독립적으로 시작**한다. V1 또는 직전 arm의 학습 weight를 이어받지 않는다. 표의 “추가”는 구조적 차이를 뜻한다.
- 새 matching의 개선이 SS에서 확인되기 전에는 PBR의 새 arm으로 확대하지 않는다. 기존 PBR 결과는 별도로 보존한다.
- 본 문서는 합의한 실험 설계다. 신규 config 구현, 실행 큐 전환 또는 새 실험 완료를 뜻하지 않는다. 기존 학습 큐를 변경한 기록으로도 사용하지 않는다.

핵심 질문은 다음 네 가지다.

1. 정확한 correspondence를 제공하면 현재 SS feature 주입 구조에서 개선 여지가 있는가?
2. 후보 검색과 scorer를 개선한 matching v2가 기존 matching 및 vanilla V1보다 유용한가?
3. 잘못된 matching을 거르는 기능과 score 기반 attention은 각각 추가 이득이 있는가?
4. 개선된 matching 위에서 retargeting adapter와 deformation query가 각각 유용한가?

## 2. 현재 근거와 해결할 문제

### 2.1 기존 SS 결과

아래는 같은 4개 quick-panel asset, 같은 생성 설정에서 얻은 **기존 집계 방식의 voxel IoU**다. Zero-shot은 원래 multi-view 학습을 마친 모델을 DeformNeXt에 추가 학습하지 않은 상태다. 입력은 모두 edited RGB 1장과 source RGB 3장이다.

| 기존 arm | DeformNeXt 추가 학습 | SS IoU | Edited-RGB zero-shot 대비 |
| --- | --- | ---: | ---: |
| Zero-shot | 없음 | 0.473075 | 기준 |
| V1: native spatial multi-view + LoRA | 5 epochs | 0.494769 | +2.169%p |
| B1: RAW cosine Top-1 | 5 epochs | 0.492715 | +1.964%p |
| B2: learned Top-1 | 5 epochs | 0.494333 | +2.126%p |
| B3: matching-aware attention | 5 epochs | 0.494794 | +2.172%p |
| C1: B3 + retargeting adapter | 5 epochs | 0.469843 | -0.323%p |

B1~B3의 V1 대비 추가 이득은 확인되지 않았다. C1은 epoch 3의 IoU 0.499314에서 epoch 5의 0.469843으로 하락했지만 validation flow loss는 개선됐다. Loss 감소를 생성 품질 향상으로 해석할 수 없다. 표본이 4개뿐이므로 이 표로 전체 데이터셋의 모듈 효과를 확정하지 않는다.

별도 32개 패널의 source-only / edited-RGB zero-shot IoU는 0.396067 / 0.415866이다. 이는 입력 효과를 확인한 결과이며, 위 4개 패널 수치와 직접 빼서 학습 개선량을 계산하지 않는다.

### 2.2 Matching cache 진단

기존 RGB matcher를 validation 508개 전체에서 조사한 결과다. 정확한 정답은 topology-visible teacher가 지정한 source patch와의 일치로 정의한다.

| 진단 | 결과 | 의미 |
| --- | ---: | --- |
| RAW Top-1 정답률 | 529 / 22,893 = 2.31% | 원래 descriptor 검색만으로 정확한 주소 선택이 어렵다. |
| Learned Top-1 정답률 | 4,851 / 22,893 = 21.19% | Head는 개선됐지만 정확한 전달에 여전히 한계가 있다. |
| Learned Top-1, 같은 source view의 정답 주변 1 patch 이내 | 50.30% | 정확한 patch 일치의 엄격함을 고려해도 오차가 남는다. |
| Top-64 proposal recall | 14,619 / 22,893 = 63.86% | 가시적 정답의 36.14%는 현재 reranker의 후보에 없다. |
| Foreground 중 teacher가 source 가시 대응점을 확인하지 못한 비율 | 15,473 / 38,366 = 40.33% | 대응점을 거부하는 별도 기능이 필요하다. |
| Temperature 0.005에서 평균 최대 attention weight | 92.71% | 상대 가중치가 정확도의 신뢰도를 뜻하지 않는다. |
| 같은 attention의 평균 실효 후보 수 | 1.26개 | B3가 B2 Top-1과 비슷하게 동작한다. |

Top-1 정답률의 분모는 **가시적 query 전체**이며, 후보에 정답이 있는 query만의 정확도가 아니다. Visibility teacher는 cutout/투명 재질에서 보수적일 수 있으므로 불확실한 query를 곧바로 확정 negative로 사용하지 않는다.

현재 head loss는 정답이 Top-64에 들어온 query만 학습한다. 가시적 정답의 후보 누락과 source에서 관측되지 않는 foreground query를 학습으로 해결하지 못한다. 기존 temperature를 단순히 높여도 Top-1 순위는 바뀌지 않으며, 틀린 feature를 더 넓게 섞을 수 있다.

SS의 기존 semantic gate는 5 epochs 후 약 0.043이다. 이는 semantic 변경량의 주입이 작다는 뜻이지만, matching이 불필요하다는 학습된 결론으로 단정하지 않는다. 480 updates와 gate LR 1e-4 자체의 영향도 있다. C1 residual은 이 작은 semantic gate 밖에서 추가되므로 별도로 진단한다.

## 3. 데이터·모델·입력 고정

### 3.1 데이터와 split

| 항목 | 고정 내용 |
| --- | --- |
| 본 학습 데이터 | 구축된 DeformNeXt-10k RGB/world-v4 population |
| SS train | 9,149개, source UID 3,184개 / source cluster 1,790개 |
| SS validation | 508개, source UID 153개 / source cluster 132개 |
| 재사용 | RGB, 카메라, 정합된 M1 SS latent, geometry, Omega/DINO feature, 검증된 target projection map |
| 사용하지 않는 학습 데이터 | mixed385 약 80k clean object subset을 DeformNeXt 본 학습에 추가하지 않는다. |
| Matching supervision | 인증된 RGB topology-visible teacher만 사용; 인증 실패 사례는 head supervision에서 제외하고 원래 SS population은 유지 |

Matcher 내부 calibration subset은 train의 source UID/cluster를 묶어 분리하고 ID 목록을 고정한다. Deformation variant나 동일 source의 다른 샘플이 head train과 calibration으로 나뉘지 않도록 한다. 정확한 분할 비율과 hash는 실행 전 manifest에 기록한다. 이 분할로 SS 본 학습의 9,149개 population을 바꾸지는 않는다.

Temperature, rejection threshold, 후보 구성 등은 이 train 내부 calibration에서 결정한다. Validation 508개는 고정된 구성의 비교에 사용하며, 별도의 미사용 test 성능으로 표현하지 않는다. 기존 H0는 전체 기존 head-train population으로 학습됐으므로 새 calibration subset에 대한 독립 검증값을 갖는 통제군으로 간주하지 않는다.

새 원본 다운로드나 전체 SS latent 재구축은 필요하지 않다. 새로운 proposal/점수/confidence/deformation sidecar는 별도 버전으로 생성한다. 연속 source 좌표 등 새 teacher 필드가 필요하면 해당 label만 보완하고 SS latent는 유지한다.

### 3.2 초기 checkpoint와 고정 backbone

- SS64 초기값: `dev-jaehyeok` 계열 multi-view LoRA 구조와 다음의 paired config/checkpoint.
  - `/file4/jaehyeok/results/Pixal3D_youngwoo/multiview/train/mixed385-lora-ddp-20261006-151618/ss64/config.json`
  - `/file4/jaehyeok/results/Pixal3D_youngwoo/multiview/train/mixed385-lora-ddp-20261006-151618/ss64/ckpts/denoiser_step0004210.pt`
- Checkpoint에 포함된 frozen base와 LoRA를 strict continuation loader로 함께 읽고, optimizer/sampler는 새 run으로 초기화한다.
- Omega: 기존 clean object RGB로 학습한 epoch-7, completed updates 23,597을 유지한다. Matching v2의 1차 실험에서 Omega backbone 자체를 다시 학습하지 않는다.
- DINO encoder와 SS VAE도 기존 고정 값을 유지한다.
- H 계열에서 선택한 matcher 하나와 동일 hash의 cache를 모든 새 SS arm이 공유한다.

### 3.3 네 카메라 RGB 입력

| 슬롯 | Source-only zero-shot | Edited-RGB zero-shot 및 모든 학습 arm |
| --- | --- | --- |
| 0 | M0 camera 8 RGB | M1 camera 8 RGB |
| 1 | M0 camera 9 RGB | 동일 |
| 2 | M0 camera 10 RGB | 동일 |
| 3 | M0 camera 11 RGB | 동일 |

두 조건 모두 서로 다른 네 카메라의 입력을 사용한다. Target은 depth가 아닌 RGB다. 학습·기본 정렬 평가는 M1 GT를 사용한다. Source-only를 M1 GT와 비교한 수치는 입력 효과를 보여주지만, source-only의 M0 복원 능력을 보여주는 수치는 아니다. 필요 시 동일 source-only 결과를 M0 GT와 비교하는 진단을 별도로 표시한다.

## 4. 단계 A: 추가 SS 학습 없는 원인 진단

| ID | 고정 checkpoint와 변경 조건 | 질문 |
| --- | --- | --- |
| D-GT | 동일 SS checkpoint에서 native / 현재 matching / GT correspondence로 전달한 feature 비교 | 주소가 정확해지면 SS 개선 여지가 생기는가? |
| D-B | 학습된 B3에서 semantic 경로 ON / OFF / wrong-correspondence 비교 | 학습된 모델이 semantic 경로를 실제로, 올바르게 이용하는가? |
| D-C | 학습된 C1에서 adapter ON / adapter OFF 비교; semantic 경로는 유지 | C1의 하락이 adapter 출력과 관련 있는가? |

D-GT에서는 source에 보이는 GT 대응점을 사용하고 관측되지 않는 query는 native 조건으로 되돌린다. GT 주소는 현재 Top-64 밖에서도 가져와 후보 누락과 주입 구조를 분리한다. GT 정보는 진단에만 사용하고 production inference 입력에 넣지 않는다.

초기 gate 0으로 GT 경로까지 꺼진 상태에서 “oracle 효과 없음”을 판정하지 않는다. 동일한 nonzero semantic gate로 current/GT를 비교하고, 주입 크기 점검이 필요하면 train calibration에서 제한된 scale sweep을 먼저 고정한다. Checkpoint, 입력, generation noise는 각 비교에서 동일하게 유지한다.

D-B의 semantic OFF는 **추가 transport 경로만** 끄는 것이며 native multi-view source 조건은 남는다. Target-only arm이 아니다. 학습된 B3의 경로를 끈 결과를 vanilla V1 checkpoint와 동일시하지 않는다. D-C도 같은 C1 weight에서 residual만 제거해 비교한다.

**진행 조건:** GT correspondence에도 개선 근거가 없다면 matching 확대를 바로 SS 본 학습으로 연결하지 않고, 주입 위치·크기·projection/value 표현을 먼저 재검토한다. Pilot 실행 통과와 모듈 효과 입증은 별개다.

## 5. 단계 B: matching head 실험

| Arm | 구성 | 주 비교와 목적 |
| --- | --- | --- |
| H0 | 기존 RAW Top-64 + topology-visible listwise head | 기존 matching의 이력 기준 |
| H1 | 후보 recall을 개선한 proposal + 동일 종류의 ranking supervision | 후보 누락을 줄인 matcher의 효과 |
| H2 | H1 + NULL 및 사용 가능성 confidence 학습 | 대응점 부재/후보 실패를 구분해 거르는 효과 |
| H3 | H2 + transported feature의 유용성을 학습하는 loss | 주소 정확도와 feature 전달 품질을 함께 개선할 수 있는지 |

H1~H3는 동일한 새 head-train/calibration split, 초기화 정책, ranking query budget과 optimizer recipe를 사용한다. NULL auxiliary query와 추가 계산량은 별도 기록한다. 기존 H0는 calibration 분할과 학습 budget이 다를 수 있어 H0→H1을 엄밀한 단일 변수 인과 비교로 표현하지 않는다. 그 비교가 필요하면 작은 head H0만 같은 새 protocol로 replay하고 SS V1은 재학습하지 않는다.

### 5.1 Proposal과 supervision

1. 기존 descriptor로 global Top-64/128/256의 recall·오차·메모리를 먼저 측정한다. Query별 유효 source 수와 view별 분포도 기록한다.
2. 후보 확대만으로 recall이 부족하면 frozen descriptor 위의 작은 Q/K adapter를 전체 source-token retrieval objective로 학습하는 분기를 검토한다. 현재 후보 안의 reranking만 학습해 후보 누락이 해결됐다고 주장하지 않는다.
3. Visibility와 candidate hit를 분리한다. `source에 가시 정답 없음`, `가시 정답은 있으나 이번 후보에 없음`, `후보에 정답 있음`, `teacher 불확실`을 각각 기록한다. Proposal miss를 실제 비가시성의 정답 label로 바꾸지 않는다.
4. Background가 NULL 학습을 압도하지 않도록 foreground query의 상태별 sampling을 고정한다. 불확실한 cutout/투명 재질 visibility는 확정 negative에서 제외하거나 감독 가중치를 낮춘다.
5. Feature loss의 teacher는 GT 대응점으로 얻은 native DINO feature다. 색상·재질 경계와 가시성을 고려한다. 연속 좌표/보간을 사용하더라도 인접 patch를 모두 정답으로 풀어 주지는 않는다.

현 D1/cache contract는 후보 64개를 고정 검사한다. K 확대는 config 숫자만 바꾸는 작업이 아니며 scorer, masking, cache schema, online/offline 경로를 함께 변경·검증해야 한다. 변경된 proposal/head는 새로운 버전으로 식별한다.

### 5.2 Confidence와 attention

- `p_use`는 선택한 후보 집합의 feature를 사용할 만한지에 대한 별도 출력이다. Softmax 최대값을 정확도의 확률로 간주하지 않는다.
- Temperature와 rejection 정책은 train calibration에서 고정한다. 실효 후보 수를 늘리는 것 자체를 성공 기준으로 사용하지 않는다.
- H3를 자동 채택하지 않는다. Ranking, proposal recall, confidence 보정, 거부 후 precision/coverage, feature error와 단계 A의 주입 가능성을 함께 보고 공유 head `H*`를 정한다. Validation을 살펴본 구성 선택이 있으면 그 선택 과정을 기록한다.
- Shared head freeze 후 모든 새 SS arm이 같은 `H*` checkpoint/hash와 candidate/logit/confidence cache를 사용한다.

필수 SS 3개를 모두 실행하려면 공통 matcher에 검증 가능한 `p_use` 출력이 있어야 한다. NULL/confidence 경로의 학습·보정이 준비되지 않았다면 R arm과 B3-v2를 실행 준비 완료로 표시하지 않는다. Softmax 최대값을 임시 confidence로 대체해 다른 실험을 수행하지 않는다.

## 6. 단계 C: SS 학습 arm

아래의 변화는 모두 같은 공유 matcher `H*`와 candidate set 위에서 비교한다. No-score 대조는 **Top-1 correspondence feature 선택**이며 uniform 평균이 아니다. Spatial correspondence 기준은 native V1이다.

| SS arm | Feature 선택 | NULL/confidence 사용 | 추가 모듈 | 주 비교 | 우선순위 |
| --- | --- | --- | --- | --- | --- |
| B2-v2 | Learned score Top-1 | 사용하지 않음 | 공통 semantic fusion | 기존 B2/V1 대비 matching v2 전체 효과 | 필수 |
| B2-v2-R | 동일 Top-1 | 사용 | 없음 | B2-v2 대비 거부·신뢰도 조절 효과 | 필수 |
| B3-v2 | Learned score 기반 candidate attention | 동일하게 사용 | 없음 | B2-v2-R 대비 score-weighted feature fusion 효과 | 필수 |
| C1-v2 | B3-v2와 동일 | 동일하게 사용 | Retargeting adapter | B3-v2 대비 retargeting 효과 | 조건부 |
| C2-v2 | C1-v2와 동일 | 동일하게 사용 | 물리적인 deformation query 채널 추가 | C1-v2 대비 query 효과 | 조건부 |

여기서 B2-v2도 score를 argmax 선택에 사용한다. B2-v2-R은 confidence를 사용하지만 **candidate feature를 score로 가중 평균하지 않는다.** 따라서 B2-v2-R→B3-v2가 matching-aware attention의 추가 효과를 분리한다.

B2-v2의 rejection OFF는 learned NULL/confidence 사용만 끄는 것이다. Background·유효 source 부족 등 모든 arm의 필수 입력 validity 검사는 유지한다. B2-v2-R과 B3-v2는 같은 `p_use`와 rejection 규칙을 적용해, attention과 함께 confidence 정의까지 바뀌지 않도록 한다. NULL을 제외한 candidate 간 조건부 weights와 source 사용 여부를 분리한다.

C1-v2/C2-v2는 대응점을 거부한 query에서 **adapter를 포함한 추가 경로 전체**를 끄고 native 조건을 보존한다. 허용된 query의 residual 구조·bounded output·초기 0 출력은 기존 recipe를 유지한다. C1에는 deformation 입력 채널 자체가 없고, C2에서 RAW FRAME+GLOBAL 2048-D deformation context를 추가한다. C2 context 생성도 공유 score/temperature contract를 사용한다.

Gate LR 확대, 별도 fusion scale 튜닝 또는 adapter 재설계를 위 표에 조용히 섞지 않는다. 필요하면 matching 비교와 분리한 후속 실험으로 정의한다.

**C 계열 진행 조건:** 필수 세 arm을 먼저 평가하고, 가장 나은 matching 구성이 V1보다 유용하다는 근거가 있는지 확인한다. 표의 C1-v2/C2-v2는 B3-v2 기반이다. B3-v2보다 Top-1 구성이 낫다면 기존 표대로 자동 실행하지 말고 공통 부모 구성을 명시적으로 정정한다. C1과 C2는 2×2로 확대하지 않고 adapter 추가, query 추가의 두 arm으로 유지한다. C2는 C1의 동일한 nuisance 설정을 사용해 직접 비교하며, C1의 하락을 C2가 회복하는 결과도 가능하므로 C1 단독 PASS를 필수 조건으로 두지는 않는다.

## 7. SS 공통 학습 조건과 자원

| 항목 | 설정 |
| --- | --- |
| 장치 | Run당 GPU 한 장, 물리 GPU는 실행 명령에 명시 |
| Global batch | 96 |
| Microbatch | 새 matcher/candidate 수로 실측 후 최대치 사용; 96이 안 맞으면 48×2 등 실제 pack 수 기록 |
| 학습 길이 | 5 epochs, 현재 9,149개 기준 96 updates/epoch, 총 480 updates |
| Denoiser 학습 | 원래 base+LoRA continuation; LoRA rank 16 / alpha 16 / dropout 0 |
| LoRA LR | 1e-5 |
| Conditioner LR | 1e-4; 비교 중 gate만 별도 변경하지 않음 |
| Optimizer | 기존 AdamW, weight decay 0.01 / betas (0.9, 0.95) / eps 1e-8 |
| Scheduler | 기존 25-update linear warmup 후 constant LR |
| 기타 | 기존 adaptive gradient clipping, CFG dropout 0.1, EMA 없음 |
| Seed | 1차는 동일한 seed 정책; 실제 seed와 RNG provenance 기록. 근소한 차이는 추가 seed로 확인 |

원본 B 계열 microbatch 96과 C 계열 48×2는 RNG 소비와 timestep/noise 배치가 다를 수 있다. 구조가 다른 arm끼리 같은 seed만으로 동일한 stochastic training trajectory라고 주장하지 않는다. 직접 비교하는 새 arm은 가능하면 같은 microbatch/accumulation을 사용하고, 맞출 수 없으면 차이를 결과에 명시한다.

기존 SS의 약 35~38초/update를 기준으로 480 updates는 순수 step 계산 약 4.7~5.1시간이다. 이는 **기존 구현의 참고치**이며 candidate 확대와 새 confidence 경로의 속도·메모리를 측정하기 전 새 arm의 완료 시간으로 확약하지 않는다. Head 학습·cache 구축·평가·저장 시간은 별도 집계한다.

PBR에서 확보된 GPU를 임의로 새 SS에 재할당하지 않는다. 본 계획의 기본은 한 run을 한 GPU에서 실행할 수 있다는 것이며, 새 arm들의 동시 실행 배치는 별도 실행 계획에 기록한다.

## 8. 평가와 채택 기준

### 8.1 같은 panel과 checkpoint를 비교

- 매 epoch: 고정 32개 SS panel. 최종 비교: 고정 128개 panel. Category·변형 강도·source cluster 분포를 고려해 deterministic하게 선정하고 sample ID/hash를 고정한다.
- 기존 4개 quick-panel 결과는 이력이다. Zero-shot 두 조건, V1, 기존 matching arm과 새 arm 모두를 공통 panel에서 다시 평가한다. 서로 다른 크기의 aggregate를 직접 비교하지 않는다.
- 생성 기본값: 12 sampling steps, guidance 3.0, 고정된 sample별 noise seed. 같은 asset은 panel 위치가 바뀌어도 동일한 noise를 사용하도록 seed map을 저장한다.
- Primary 비교는 모든 arm의 **5-epoch final checkpoint**다. Epoch별 수치와 best epoch는 경향으로 함께 기록하고, final끼리의 비교에 한 arm의 best만 끼워 넣지 않는다.
- 12-step 평가와 flow loss가 엇갈리면 대표 checkpoint에 같은 더 긴 sampling 설정을 적용하는 진단을 추가한다. 특정 arm에만 유리한 sampler 설정을 쓰지 않는다.

### 8.2 SS 지표와 logging

본 단계의 생성 지표는 voxel IoU, F1, precision, recall이다. 기존 global voxel-count aggregate를 명시적으로 유지하고 sample별 지표와 TP/FP/FN도 저장해 집계 방식이 섞이지 않도록 한다. GT/pred voxel projection panel도 같은 asset 순서로 기록한다. SS를 rendering metric이나 PBR 재질 성능으로 해석하지 않는다.

Matching 지표는 다음을 함께 기록한다.

- 가시 query 전체의 proposal recall / Top-1 정확도 / 근접 patch 정확도; 후보 hit query에 조건부인 reranking 정확도는 별도 열로 표시.
- 비가시 foreground, proposal miss, teacher 불확실 query의 수와 분모.
- 사용한 query의 정확도와 coverage 곡선, NULL 판정 및 confidence calibration, per-view 선택 분포.
- Transported feature error, attention entropy/실효 후보 수, native 조건 대비 update RMS.
- Semantic gate, 허용된 query 비율, retarget residual RMS, 처리 시간 및 peak memory.

W&B는 기존 `youngwoojeon-unist/pixal3d-deformnext-retargeting` 프로젝트를 사용하고 run ID에 matching-v2와 arm을 구분한다. Loss·metric·이미지·gate·성능 수치를 함께 남긴다.

### 8.3 판단 기준

| 단계 | 판단 |
| --- | --- |
| 실행 검증 | 초기 native 출력 보존, cache/online parity, finite loss/gradient, 실제 batch memory를 확인 |
| Matching 비교 | 가시 query denominator를 유지해 recall·오차·precision/coverage·feature error의 개선과 tradeoff를 보고 |
| SS 필수 arm 비교 | Zero-shot 및 V1 대비 결과와 B2-v2→B2-v2-R→B3-v2의 paired 차이를 각각 보고 |
| C 계열 진입 | 개선된 matching이 SS에 유용하다는 근거를 확보한 뒤 진행; 작은 평균 차이만으로 full 구조를 확정하지 않음 |
| 최종 채택 | 같은 panel의 paired 차이를 source cluster 단위로 bootstrap하고, 근소한 차이는 추가 seed에서 방향과 크기를 확인 |

모든 수치에 표본 수, sampling 설정, checkpoint step을 붙인다. 동일 source의 여러 변형을 독립 asset처럼 취급해 불확실성을 과소평가하지 않는다. 평균 개선이 작거나 신뢰구간이 0을 포함하면 “효과 확인 안 됨”으로 기록한다. 실행 pilot 통과를 성능 PASS로 부르지 않는다.

## 9. 실행 순서·저장·산출물

1. 고정 32/128 panel과 train 내부 calibration split을 생성하고 hash를 기록한다.
2. 기존 zero-shot/V1/B/C checkpoint를 공통 panel로 재평가하고 단계 A 진단을 수행한다.
3. Proposal recall 비교와 H1~H3 head 실험을 수행한다. 필요 시 H0 head replay를 추가한다.
4. 공유 matcher `H*`, rejection 규칙, temperature, cache schema를 freeze한다.
5. Native-start 및 실제 memory 검증 후 **B2-v2 → B2-v2-R → B3-v2**의 독립 SS 학습을 비교한다.
6. SS matching 개선 근거가 있으면 **C1-v2 → C2-v2**를 독립 학습하고 직접 비교한다.
7. SS에서 채택할 구성과 한계를 정리한 뒤 PBR 확대 여부를 결정한다.

File4 신규 산출물은 기존 결과와 분리한 **제안 namespace** `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/` 아래에 둔다. `panels/`, `head_calibration/`, `heads/`, `matches/`, `diagnostics/`, `runs_ss64/`, `evaluations/`를 구분하고, 실제 resolved config와 provenance를 함께 보관한다. 기존 `rgb_v1`의 checkpoint/cache를 덮어쓰거나 진행 중 arm의 matcher를 중간에 바꾸지 않는다.

완료 보고서는 다음을 포함한다.

- 모든 baseline과 arm의 동일-panel final 성능, epoch 경향 및 추가 seed 결과.
- 각 비교가 분리하는 변경점과 분리하지 못한 변경점.
- Head checkpoint SHA, proposal/attention/NULL/feature-loss contract 및 입력 fingerprint.
- 초기 native 보존, active-path cache parity, dropout/AMP 조건의 검증 근거.
- GPU당 실제 microbatch, accumulation, update 시간, 메모리와 전체 소요 시간.
- 채택/보류 구성 및 PBR로 확장할 근거.

## 10. 근거와 유효 범위

- [RGB·네 카메라 입력 정정 기록](2026-10-16-rgb-input-correction-ko.md): 기존 depth 입력 결과는 이번 RGB 결론에서 제외한다.
- Matching 진단 원본: `/root/dev/.cache/deformnext-retargeting/production-r9-rgb-v1/diagnostics/rgb-arm-cause-analysis/matching-audit.json`.
- 기존 RGB 결과: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_v1/runs_world_v4/ss64/v1/validation.jsonl` 및 `runs_r9_v5/ss64/{b1,b2,b3,c1,c2}/validation.jsonl`.
- 기존 RGB head 이력: 같은 `rgb_v1/q2c_r9_v5/d1-topology-final.jsonl`. Depth head의 높은 정확도를 RGB head 수치로 대체하지 않는다.
- 과거 R9 authority: `anchor-hybrid-prestudy/docs/superpowers/reports_final2/2026-09-16-exg-app-cue-stage-b-r9-four-module-ablation-results-ko.md`. 원래 controlled RGB task의 이득과 이번 SS 생성 이득을 구분한다. 과거에도 exact matching과 task error는 함께 움직이지 않았으므로 Top-1 하나로 새 full model을 결정하지 않는다.

이 계획에서 full model은 미리 C2로 확정하지 않는다. SS에서 실제로 확인된 추가 이득과 재현성을 근거로 최종 구조를 결정한다.

실제 실행과 공통 panel 재평가 결과는 [2026-10-17 실행 기록](2026-10-17-ss-matching-v2-execution-ko.md)에 별도로 기록한다. 위 4개 quick-panel 이력과 새 32/128개 panel의 결과를 구분한다.
