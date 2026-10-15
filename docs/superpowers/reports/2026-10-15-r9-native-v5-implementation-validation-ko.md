# R9 승격 코드 복원 및 native 초기 성능 보존 검증

현재 branch는 `feature-retargeting`이고 backbone/LoRA continuation은 `dev-jaehyeok` 기반이다. 이전 semantic 구현은 승격한 R9 D1과 달랐다. 이번 변경은 matching core를 원본과 일치시키고, generation 통합부는 native V1에서 정확히 시작하도록 수정했다. **초기 하락 방지는 검증됐지만, 학습 후 V1 대비 향상은 아직 검증하지 않았다.** 기존 world-v4 semantic 학습은 계속 중단 상태다.

## 원본 구현과의 수치 일치

Authority는 `anchor-hybrid-prestudy/pixal3d/experiments/exg_app_cue_b_r9_ladder.py`이며 SHA256은 `e7be96c47803d16a769fcb9e9cf0026b314c4983c6b40f9d3c21604b15d973d2`다. 관련 결과는 `reports_final2/2026-09-16-exg-app-cue-stage-b-r9-four-module-ablation-results-ko.md`다.

`pixal3d/modules/r9_matching.py`에 원본 scorer의 선택된 정의를 그대로 복사했다. 런타임에 이전 worktree를 import하지 않는다. AST hash 회귀 검사와 실제 원본을 import하는 독립 비교 도구를 함께 둔다.

| 항목 | 이번 구현 |
| --- | --- |
| Matching descriptor | L12 FRAME RAW 1024-D; 기존 concat cache의 첫 1024채널 |
| Proposal / projector / layer | GLOBAL64 / P0-RAW / M0-L12 |
| Decoder | D1 query-only contextual attention, 1 block, 4 heads × 32-D |
| Scalar head | 원본 LayerNorm/64-hidden/bounded residual head |
| Parameter count | 928,453; 원본과 동일 |
| Seed / state 비교 | 42017/42018/42019 × 초기 및 nonzero learned-gate state |
| CPU 및 CUDA 결과 | 초기 가중치, 직접 forward, parameter gradient 차이 모두 0 |
| Query chunk 처리 | 최대 차이 1.1921e-7; GEMM batch shape rounding 범위 |

기존 2048-D pairwise checkpoint는 새 모델에 strict load할 수 없다. Historical audit만 명시적인 `legacy-v4`로 실행한다. Matching은 FRAME 1024-D이고, C2의 deformation context는 별도의 RAW FRAME+GLOBAL 2048-D로 유지했다. C1에는 그 채널 자체가 없다.

원본 R9의 성능 근거는 **controlled APP-CUE RGB task**다. 이번 topology-visible listwise warm-up은 DeformNeXt의 새 supervision이다. Token domain도 multi-view image patches다. 따라서 scorer 수치 재현을 이전 controlled-task 실험 전체 또는 SS/PBR 성능 향상 재현으로 표현하지 않는다.

## 초기 native identity와 학습 경로

기존 source 교체식으로 시작하지 않는다. 새 semantic arm의 projection은 다음과 같다.

`C = C_native + tanh(g) * valid * delta + 3/4 * valid * R`

`delta = (C_anchor + 3 * E) / 4 - C_native`, `g=0`, retargeter의 최종 출력 `R=0`으로 초기화한다. Retargeter는 zero semantic gate의 바깥에 연결하므로 gate와 retargeter 출력층 모두 첫 update부터 gradient를 받는다. 두 zero 연산을 곱해 학습을 막지 않는다. PBR semantic delta는 DINO LR 채널에만 적용하고 native NAF HR 경로는 보존한다. Retargeter의 HR residual도 처음에는 0이다.

Matching-aware attention은 global categorical softmax다. Query별 logit에 상수를 더해도 evidence/view mass가 바뀌지 않는다. View mass는 후보 확률의 marginal이며 별도의 uncalibrated sigmoid confidence를 곱하지 않는다. Candidate weight의 의미는 전체 후보에 대한 확률이다. Proposal 및 D1은 AMP 내부에서도 FP32로 계산하고 cache/online 경로가 같은 함수를 사용한다.

Geometry-only 입력 anchor의 alpha와 source validity로 사용 불가능한 query는 native fallback한다. 유효 source site가 64개 미만이면 모든 semantic arm에서 fallback한다. 전체 scan 결과 train 9,149개 중 130개(최소 35 site), validation 508개 중 0개(최소 75 site)다. 이 130개도 native LoRA 학습에는 참여한다. 실제 no-match/visibility calibration head를 완성했다는 뜻은 아니다.

## 실제 GPU 검증 결과

원래 mixed385 SS step 4210 / PBR step 16850의 base+LoRA를 strict continuation했다. 별도로 topology teacher 8 train / 8 validation에서 8 update한 **검증용** D1을 frozen load하여, 초기 head뿐 아니라 학습된 matching score에서도 확인했다. Denoiser optimizer update는 0이다.

| 검사 | SS | PBR |
| --- | ---: | ---: |
| Category-stratified condition/denoiser 비교 asset | 32 | 16 |
| V1/B1/B2/B3/C1/C2 condition 최대 오차 | 0 | 0 |
| 고정 noise/t=500 denoiser 최대 오차 | 0 | 0 |
| 대표 1 asset, 전 arm 12-step Euler/CFG 3 생성 오차 | 0 | 0 |
| 대표 1 asset에서 전 arm 동일 지표 | voxel IoU 0.4371839751 | FG PSNR 23.6931934323 dB |
| 최초 backward | LoRA/gate/adapter gradient > 0 | LoRA/gate/adapter gradient > 0 |

마지막 metric 행은 **대표 1 asset의 equality 회귀 검사**다. 32/16개 전체의 생성 품질 평균이나 기존 4/2개 패널 평균과 비교하는 수치가 아니다. Training/inference public entrypoint도 대표 asset의 전 arm에서 native condition과 정확히 일치했다.

PBR renderer는 ambient-occlusion ray를 랜덤 샘플링한다. 동일 latent의 shaded PSNR도 이 RNG 때문에 흔들렸다. 비교 대상에 동일 ray seed를 적용하고 `fork_rng`로 training RNG를 보존했다. 최종 PBR shaded/base-color/alpha image 차이도 0이다.

추가 검증:

- 실제 8 asset × SS/PBR에서 BF16 AMP로 cache/online을 비교했다. C2 retargeter 출력층을 nonzero로 활성화한 상태에서도 evidence 차이는 0이고 score/weight/residual 검사도 통과했다.
- GPU 2/3의 2-rank DDP에서 5개 semantic arm × 1024/2048 value width를 각각 2 update했다. Gate/adapter가 학습되고 rank 간 parameter가 정확히 일치했다.
- 관련 CPU 회귀 테스트 **111개 통과**. Native camera/latent/LoRA 계약, zero-start 및 첫 gradient, offset invariance, split/teacher/proposal miss, cache hash/version rejection 등을 포함한다.

## Teacher와 cache 수정 범위

`topology_labels.py`는 M1 triangle barycentrics로 M0 point를 구한 뒤 source z-buffer depth tolerance 0.005 및 실제 source alpha를 확인한다. 기존처럼 X1을 그대로 M0 camera에 투영한 spatial index를 teacher로 쓰지 않는다. 입력 content hash와 split/role을 저장하고 training에서 재검사한다. Label은 conditioning loader가 읽지 않는다.

투명 안경은 opaque geometry raster보다 실제 RGBA silhouette이 작다. Source image의 관측 alpha가 geometry silhouette 안에 들어오는지 검사하고, target geometry anchor는 별도로 높은 mask IoU를 요구한다. Opaque z-buffer는 cutout 뒤 표면을 과도하게 occlude할 수 있어 visibility는 보수적이다. Material-aware ray tracing과 같다고 주장하지 않는다.

Warm-up loss는 visible proposal miss와 null/unobservable query를 분리하고, nonempty positive가 없는 update를 건너뛴다. Validation accuracy denominator에는 visible proposal miss를 포함한다. Checkpoint는 원본 protocol처럼 fixed final update를 저장한다. 검증용 8 update에서 loss가 3.8207643→3.8193796으로 변했고 정상적인 checkpoint/cache 생성 및 strict load를 확인했다. 이 작은 표본의 수치로 matcher를 승격하지 않는다.

현재 생성한 것은 **teacher 40개(8 train + 32 validation), 검증용 match 8개**다. 전체 teacher와 production matching head/cache는 아직 구축하지 않았다. 기존 world-v4 geometry/Shape/PBR/SS latent는 이번 수정 때문에 다시 만들지 않는다.

## 본 학습 재개 조건과 저장 위치

새 recipe는 `r9-native-v5`, `deformnext-match-r9-v5`, weight hash와 proposal/aggregation policy를 검사한다. 출력은 기존 결과와 분리된 `runs_r9_v5`다. 기본 semantic release는 `audit-hold`이며, exact frozen matcher의 초기 검증 및 pilot 완료 없이 campaign이 실행되지 않는다. 기존 `runs_world_v4/SEMANTIC_AUDIT_HOLD.json`도 유지한다.

다음 단계는 전체 teacher 구축 → fixed shared D1 warm-up/cache → 실제 frozen head의 초기 검증 → production microbatch profile → 짧은 native 대비 pilot이다. Gate/adapter가 열린 뒤에도 좋아지는지는 pilot의 endpoint metric으로 판단한다. SS global 96 / PBR global 24 recipe는 유지하되, 변경된 v5 모듈의 최대 microbatch/메모리는 새로 profile해야 한다. 과거 v4의 87.3/64.7 GiB를 v5 측정으로 표시하지 않는다.

검증 artifact는 file4의 `/file4/youngwoo/deformation/experiments/retargeting-v1/r9-native-v5-audit-20261015/`에 보관한다. `acceptance-summary.json`, `authority-{cpu,cuda}.json`, `{ss64,pbr1024}-learned/verification.json`, `cache-online-parity.json`, `conditioner-ddp.json`이 근거다. 검증용 weight는 production checkpoint로 취급하지 않는다.
