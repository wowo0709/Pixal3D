# 모듈별 효과 검증과 단일 파이프라인 완성을 위한 핵심 실험

기준 시각: 2026-10-23 14:07 UTC. 브랜치: `wowo0709/feature-retargeting`.

실험의 목적은 semantic matching, matching-aware attention, retargeting을 같은 생성 경로에서 조합하고, 각 설계가 실제 생성 품질에 기여하는지 확인하는 것이다. 각 모듈의 성능 향상을 미리 전제하지 않는다. 후보 탐색 정확도 상승, 저장된 모델의 추론 개입, 서로 다른 프로토콜의 IoU 차이를 최종 생성 성능에 대한 모듈별 증거로 대신하지 않는다.

## 1. 실험 범위와 실행 상태

기존 지엽 실험 36개는 대기열의 실행 전 gate를 통해 `HELD` 상태로 보류했다. 실행 중인 정상 학습을 재시작하거나, 기존 결과를 삭제하거나, 완료 상태를 조작하지 않았다. 추가적인 teacher 회귀, feature 치환·순열 sweep, 과거 direct-correction 경로의 진단 실험은 현재 진행하지 않는다.

현재 실행 중인 핵심 실험은 다음과 같다. 아래 step은 시점별 관측치이며 완료 결과가 아니다.

| GPU | 실행 내용 | 관측된 진행 | 목적 |
|---|---|---:|---|
| 6 | SS `C_TOP1_V` | 82/480 | Top-1에 적용한 V/output 변환 효과 분리 |
| 7 | SS `C_FIXED_TOPK` | 80/480 | 같은 V/output에서 Top-K 결합 효과 분리 |
| 0–3 | PBR `top1-plus-feature` DDP | 887/1900 | SS 개선과 별도로 texture-stage 전달 효과 검증 |

SS 두 실험은 각각 GPU 하나, micro/global batch 96, accumulation 1이다. PBR은 global batch 24, GPU당 microbatch 6, accumulation 1이다. GPU 4·5는 사용하지 않는다.

`C_SPATIAL`, `C_RAW`, `C_D1_COS`는 구현 및 관련 CPU 검사 25개를 통과하고, 각자의 GPU 초기 동등성 검사 → full-batch 용량 검사 → 공통 기준과의 데이터·노이즈·초기 가중치 비교 → 본 학습 순으로 연결했다. 아직 이 세 arm의 GPU 검증이나 학습 결과는 없다. 아래 나머지 설계 대조군은 계획이며, 구현·실행 완료로 보고하지 않는다.

## 2. 공통 파이프라인

입력은 변형 후 target RGB 한 장과 변형 전 source RGB 세 장이다. target-first 입력과 기존 native global/target conditioning을 유지한다. source-only zero-shot은 같은 카메라의 변형 전 RGB 네 장을 사용한다.

source camera에서 같은 projection ray를 공유하는 voxel들을 하나의 query 단위로 묶는다. 그 ray에 대응하는 target patch member들에서 semantic 후보를 모으고, matching decoder와 score로 correspondence 후보·prior를 만든다. 선택한 source feature를 Top-1, 고정 Top-K, 또는 학습 attention으로 읽은 뒤 공통 retargeting adapter에 전달한다. correction은 이 경로를 통해 기존 source conditioning에 반영한다. 별도의 attention correction 우회 경로는 두지 않는다. 대응 근거가 부족한 source 영역은 기존 native source conditioning을 유지한다.

target branch를 바꾸지 않는 것은 입력 conditioning 경로의 보존이다. 최종 생성 결과의 target 보존까지 구조만으로 보장한다고 주장하지 않는다. 실제 target alignment와 source preservation을 함께 측정한다. 추론에는 GT depth, GT mesh, GT correspondence를 사용하지 않는다. visibility gate를 새로 추가하지 않는다.

이 단계의 기본 파이프라인에는 deformation-query channel이 물리적으로 없다. 핵심 조합이 검증된 이후에만 retargeting 위에 deformation-aware query를 추가하는 arm 하나를 검토한다. 0으로 채운 가짜 channel이나 2×2 설계로 대체하지 않는다.

## 3. 핵심 대조군: 총 12개 arm, 10개 모듈 비교

수치 프로토콜의 정의는 `experiments/deformnext_retargeting/module_pipeline_protocol_v2.py`이다. 공통 arm은 같은 target-first source-map 전달 경로와 R-only carrier를 사용한다.

| 순서 | 비교 | 분리해서 답할 질문 |
|---|---|---|
| 1 | `C_RAW` − `C_SPATIAL` | 원래 camera projection correspondence를 semantic correspondence로 바꾸면 생성 품질이 좋아지는가? |
| 2 | `C_D1_COS` − `C_RAW` | 같은 RAW64 point proposal에서 matching decoder의 contextual feature가 기여하는가? |
| 3 | `C_DECODER` − `C_D1_COS` | 같은 decoder에서 learned scalar matching score가 cosine보다 기여하는가? |
| 4 | `C_TOP1_V` − `C_DECODER` | Top-1 correspondence 자체에 V/output 변환을 적용한 효과인가? |
| 5 | `C_FIXED_TOPK` − `C_TOP1_V` | 같은 변환에서 Top-K 후보를 결합한 효과가 있는가? |
| 6 | `FULL` − `C_FIXED_TOPK` | 같은 후보·V/output에서 학습 QKV attention이 고정 결합보다 좋은가? |
| 7 | `FULL` − `FULL_QK_ONLY` | attention 내부 matching score prior가 추가로 필요한가? |
| 8 | `FULL` − `FULL_VOXEL_Q` | 같은 후보와 retargeting에서 ray query 단위가 개별 voxel query보다 좋은가? |
| 9 | `C_DECODER` − `C_DECODER_NO_R` | Top-1 경로에서 retargeting transform 자체가 기여하는가? |
| 10 | `FULL` − `FULL_MEAN_MATCH` | 같은 ray query에서 member 기반 후보 탐색이 mean 기반보다 좋은가? |

12개 arm은 위 표에 등장하는 11개 모듈 arm과 native vanilla `V1`이다. 핵심 전체 비교는 `FULL` 대 `V1`, `FULL` 대 강한 단순 기준인 `C_DECODER`이다. 기존 결과는 모델·데이터·학습·평가 계약이 실제로 일치하는 경우에만 재사용한다.

이 표의 해석에는 다음 제약이 있다.

- `C_SPATIAL`은 원래 3-view projection 후보를 쓴다. semantic RAW64와 비교한 결과는 correspondence 모듈 교체 효과이며, 동일 후보 수에서 주소 선택만 바꾼 효과로 표현하지 않는다.
- Top-1은 선택된 correspondence feature 하나만 실제 사용한다. uniform confidence/attention을 Top-1 ablation으로 부르지 않는다.
- RAW arm에는 decoder와 scalar head가 없다. D1-cosine arm에는 frozen decoder만 있고 scalar head는 없다. 이 둘을 분리해야 decoder 효과와 score 효과를 혼동하지 않는다.
- `C_FIXED_TOPK`는 frozen matching prior의 비균일 가중치를 쓴다. 학습 attention과 V/output 초기값을 공유한다.
- `FULL_QK_ONLY`는 attention에 더하는 prior만 제거한다. 후보 탐색, Top-1 carrier, decoder 문맥을 같이 바꾸지 않는다.
- voxel-query 대조군도 같은 ray 후보·문맥을 사용한다. readout의 raw aggregate와 transformed feature를 ray 단위로 모은 다음 동일한 retargeting에 전달한다.
- retargeting을 제거하면 attention까지 끊기는 비교를 피하기 위해 R 효과는 Top-1 경로에서 측정한다. 이것만으로 full attention과 R 사이의 모든 상호작용까지 증명했다고 주장하지 않는다.

## 4. 공정한 학습과 채택 기준

SS는 동일한 multi-view LoRA checkpoint step 4,210에서 시작한다. DeformNeXt train 9,149개, 5 epoch/480 update, seed 101, global/microbatch 96, accumulation 1, LoRA rank/alpha 16, denoiser LR `1e-5`, source-module LR `1e-4`를 공통으로 유지한다. 80k multi-view 사전학습 데이터를 deformation 학습 데이터로 다시 사용하지 않는다.

각 arm은 첫 학습 전 native velocity/loss, LoRA 초기값, 학습 asset 순서, noise/timestep trace를 기준 arm과 확인한다. 모듈이 없는 arm은 실제 parameter와 optimizer에서 해당 부분을 제거한다. 모델 초기화를 맞췄다는 선언만으로 GPU 검증을 통과한 것으로 간주하지 않는다.

각 epoch에는 같은 32개 개발 샘플을 사용하고, 마지막에는 128개 × 생성 noise 3개를 같은 asset·GT·seed로 비교한다. primary는 asset별 세 noise 평균을 낸 mean voxel IoU다. pooled TP/FP/FN IoU는 이름을 분리한다. source-cluster 단위 paired bootstrap을 사용하고, 모듈 비교군 10개와 전체 비교군 2개를 사전에 구분한다. 생성 noise 세 번은 독립 학습 seed 세 번이 아니다.

모듈의 기여는 대응하는 matched 비교에서 판단한다. 이후 선택한 최종 조합과 V1·강한 단순 기준을 독립 학습 seed 202로 확인한다. 모든 12개 arm을 무조건 추가 seed로 반복하거나, 수치가 나빠진 arm을 근거 없이 더 오래 돌리지 않는다. 부정적인 결과가 나온 모듈은 논문 스토리에 맞추기 위해 필수라고 선언하지 않는다.

## 5. 단일 완성 파이프라인 검증

SS IoU는 구조 선택의 빠른 지표다. 완성된 파이프라인의 향상 여부는 생성한 SS → Shape512 → Shape1024 → PBR1024를 실제로 연결해서 확인한다.

1. SS 모듈 비교에서는 각 모델이 생성한 SS를 사용하고, 이후 Shape와 PBR checkpoint는 공통으로 고정한다. GT geometry로 SS 차이를 지우지 않는다.
2. PBR 모듈 비교에서는 같은 생성 geometry와 원래 native guide를 고정해서 texture 전달 효과를 분리한다. GT-shape-conditioned stage 평가 결과는 별도로 표시한다.
3. 같은 개발 샘플 32개에서 target alignment, source preservation을 함께 확인한다. source metric이 좋아지고 target이 나빠지는 조합도 한쪽 수치만으로 채택하지 않는다.
4. 최종 구조·checkpoint·seed·평가 계약을 동결한 다음 TEST 508개를 평가한다. TEST 결과를 구조 탐색에 되먹이지 않는다.

| 평가 목적 | Metric |
|---|---|
| Source preservation | source view LPIPS, DreamSim, CLIP |
| Target alignment | target view PSNR, SSIM, 2D mask M-IoU, 3D CD, F-Score |
| Global naturalness | FID, KID |

매 epoch에 이 전체 평가를 실행하지 않는다. SS의 빠른 voxel 평가와 PBR의 소규모 rendering 평가로 경향을 보고, 후보가 정해지면 생성 geometry를 포함한 개발 평가와 최종 평가를 수행한다. target crop LPIPS를 source-preservation LPIPS로 바꿔 부르지 않는다.

현재 공통 PBR을 사용하는 새 SS arm들의 full-cascade 연결과 최종 TEST 실행은 아직 완료되지 않았다. raw/spatial/D1-cosine의 GPU 검증·학습, 남은 네 설계 대조군의 구현·검증, 공통 cascade 평가가 남아 있다. 완성 파이프라인의 성능 향상을 이미 입증했다고 보고하지 않는다.

## 6. 산출물과 종료 조건

최종 산출물은 (1) 10개 모듈 비교와 불확실성을 담은 표, (2) V1·강한 단순 기준·선택된 full 조합의 동일 조건 비교, (3) 실제 end-to-end 생성 결과와 10개 평가 metric, (4) 재현 가능한 config·checkpoint·입출력 provenance다.

최종 조합은 이 증거에 따라 결정한다. 기존의 `.44` pooled IoU와 `.47` mean IoU 차이를 모듈이 만든 3pp 상승으로 설명하지 않는다. 후보 탐색 개선만 관찰된 모듈은 그 범위로 주장하고, 생성 성능 기여까지 확인된 모듈과 구분한다.
