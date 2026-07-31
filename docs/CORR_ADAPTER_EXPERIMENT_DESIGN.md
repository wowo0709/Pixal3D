# Multi-view Correspondence-aware Routing/Transport 실험 설계

> 상태: 사용자 승인 완료 — Stage 1 feature-level foundation 구현 완료
> 작성일: 2026-07-31
> 대상 브랜치: `feature/multiview-correspondence`
> 구현 대상: multi-view Shape512, Shape1024, PBR1024
> intervention 제외: SS64
> 최초 조건: `B=1`, `K=4`, seed 42

## 1. 연구 질문과 주장 범위

검증할 질문은 독립적으로 생성된 geometry-conditioned multi-view 이미지에
view inconsistency와 hallucination이 있을 때, correspondence-aware
routing/transport가 올바른 evidence를 보존하면서 잘못된 evidence를 억제하는
유용한 inductive bias인가이다.

이번 실험의 최대 주장은 위 inductive bias의 유용성 검증이다.
“multi-view inconsistency를 해결했다”는 주장은 하지 않는다.

이번 범위에서 제외한다.

- Pixal3D/flow/DINO/NAF weight update
- denoiser architecture 또는 ProjectAttention input 변경
- mesh proxy와 mesh-guided signal
- foreground hard masking을 기본 aggregation으로 사용
- actual VLM output에 oracle mask/correspondence 부여
- source/generated image의 local-window matching을 reliability로 사용
- 공개 single-view checkpoint를 multi-view 결과 대신 사용

## 2. 설계 불변조건

1. option이 없으면 기존 equal mean 경로와 수치적으로 동일해야 한다.
2. `z_global=[B,5,1024]`, `z_proj=[B,R^3,2048]`를 유지한다.
3. L/H 순서는 `[low 1024, high 1024]`로 유지한다.
4. 한 view의 동일 weight를 L/H 양쪽에 적용한다.
5. Stage 1에서는 projection 위치를 이동시키지 않는다.
6. global token은 S0–S4에서 arithmetic mean을 유지한다.
7. valid mask를 S0/S1의 hidden visibility prior로 사용하지 않는다.
8. K=1, identical K views, `alpha=0`은 baseline과 일치해야 한다.
9. non-finite/degenerate score에서는 uniform fallback을 사용한다.
10. 모든 arm에서 checkpoint, upstream artifact, noise, sampler, CFG,
    render camera를 고정한다.

## 3. 권장 architecture

### 3.1 선택안: explicit policy hook + CPU cache/chunk

`DinoV3ProjFeatureExtractor._forward_multiview()`의 view별 feature 생성과
기존 online mean 사이에 명시적 aggregation policy를 둔다.

```text
if policy == "equal_mean":
    return existing_online_mean(view_generator)

per_view = sequential_extract_and_cpu_cache(view_generator)
return chunked_policy_aggregate(per_view)
```

이 구조를 선택하는 이유는 다음과 같다.

- default path를 코드 수준에서 그대로 유지할 수 있다.
- denoiser와 checkpoint interface를 건드리지 않는다.
- S1–S5의 per-view weight/confidence를 기록할 수 있다.
- naive R64/K4 GPU stack의 4 GiB projection 상주를 피한다.
- chunked 구현을 작은 naive reference와 직접 비교할 수 있다.

초기 correctness 구현 후 측정 결과에 따라 two-pass recomputation 또는
active-only 최적화를 별도 성능 변경으로 검토한다.

### 3.2 정책과 diagnostics 분리

aggregation 계산은 순수 tensor utility로 두고, conditioner는 policy를 호출하는
얇은 integration layer로 둔다. diagnostics는 optional sidecar로 반환하거나
collector에 기록하되 denoiser condition dict에는 넣지 않는다.

필수 diagnostics:

- per-voxel/per-view score와 weight
- weight entropy와 uniform deviation
- branch별 agreement
- anchor/non-anchor 평균
- active coordinate mask
- fallback count

## 4. Stage 1 방법

view `i`, voxel `q`에서 다음처럼 분리한다.

```text
F_i(q) = concat(L_i(q), H_i(q))
l_i(q) = normalize(L_i(q))
h_i(q) = normalize(H_i(q))
```

leave-one-out prototype은 normalized feature의 다른-view 합을 다시
normalize해 계산한다.

```text
p^L_-i(q) = normalize(sum_{j != i} l_j(q))
p^H_-i(q) = normalize(sum_{j != i} h_j(q))

s_i(q) =
    0.5 * cosine(l_i(q), p^L_-i(q))
  + 0.5 * cosine(h_i(q), p^H_-i(q))

r_i(q) = softmax_i(s_i(q) / temperature)
w_i(q) = (1 - alpha) / K + alpha * r_i(q)
F_new(q) = sum_i w_i(q) * F_i(q)
```

zero norm, non-finite score, 또는 전 view가 unusable인 voxel은
`w_i=1/K`로 되돌린다.

### 4.1 비교 arm

| Arm | projection | global | 목적 |
|---|---|---|---|
| S0 | 기존 equal mean | equal mean | 학습 분포 baseline |
| S1 | residual consensus | equal mean | primary 방법 |
| S2 | pure consensus (`alpha=1`) | equal mean | stress/distribution-shift 진단 |
| S3 | oracle-mask residual | equal mean | deploy 불가능한 headroom |
| S4 | oracle-mask reject + renorm | equal mean | controlled ceiling |
| S5 | S1 + scalar global routing | 같은 view reliability로 weighted mean | global bottleneck 진단 |

S5의 scalar reliability는 downstream이 실제 소비하는 active voxel에서 S1
weight를 축약해 만든다. global token을 `[B,5K,1024]`로 concat하지 않는다.

### 4.2 hyperparameter 동결 절차

temperature와 S1 alpha는 3D output을 생성하기 전에 작은 feature-only
calibration subset에서 한 번만 선택한다.

1. calibration object/corruption ID를 evaluation set과 분리해 manifest에
   먼저 고정한다.
2. 평가할 temperature/alpha 후보와 selection rule을 run 전에 manifest에
   기록한다.
3. clean preservation과 corruption recovery를 동시에 보고한다.
4. 선택값을 고정한 뒤 Gate B/C 3D 결과를 보고 변경하지 않는다.

현재 데이터와 checkpoint feature distribution을 보지 않은 상태에서 특정
temperature/alpha 숫자를 확정하지 않는다. 이 선택은 3D quality가 아니라
feature-level 사전 기준만 사용한다.

## 5. Controlled corruption

Track A는 clean calibrated K=4 multi-view object에서 non-anchor 한 view만
corrupt하는 조건으로 시작한다. anchor corruption은 별도 stress arm이다.

| ID | corruption | 저장해야 할 oracle |
|---|---|---|
| C1 | foreground local hue/saturation/brightness/material change | affected mask, parameters |
| C2 | logo/decal/donor texture alpha composite | alpha mask, donor/hash |
| C3 | local deletion/background-like 또는 inpaint content | deletion mask, fill metadata |
| C4 | local affine translation/rotation/scale/shear | forward matrix, mask, inverse grid, invalid/hole |
| C5 | bounded smooth random field warp | forward field, inverse grid, mask, invalid/hole |

모든 corruption은 `(object_id, view_id, corruption_id, seed)`로 deterministic
해야 한다. 원본/결과/mask/field hash를 저장하고 contact sheet 검수를 통과한
case만 evaluation에 포함한다. 실패도 이유와 함께 manifest에 남긴다.

Oracle mask는 동일 projection coordinate에서 mask를 bilinear sample하여
`a_i(q)=1-m_i(q)`로 만든다. out-of-bounds mask sample은 corrupted라고
간주하지 않으며, projection validity는 별도 diagnostic으로 기록한다. 이를
visibility rejection과 섞지 않는다.

## 6. Feature-level Gate A

clean calibrated equal-mean feature를 reference로 사용한다. primary 계산은
stage별 active coordinate에서 하고 dense grid는 보조로 보고한다.

필수 metric:

- aggregated feature cosine error
- L2 drift와 L/H branch norm
- corrupted-mask projected voxel의 corrupt-view weight mass
- mask 밖 clean-region preservation
- correct unique-view retention
- equal mean → consensus → oracle gap
- entropy, per-view 평균, anchor/non-anchor 평균
- clean/corrupt weight histogram
- all-clean uniform deviation

Gate A 결과는 다음을 모두 포함해야 한다.

1. S0 대비 S1의 corrupted feature recovery
2. all-clean에서 S1의 손상
3. S3/S4 oracle headroom
4. correct unique-view rejection failure
5. K=4 2-vs-2 conflict와 low-overlap 진단

Stage 1에서 consensus 또는 oracle 어느 쪽도 의미 있는 headroom을 보이지
않으면 Stage 2를 구현하지 않고 negative result를 보고한다.

## 7. Stage-local 3D isolation

체크포인트가 제공된 뒤 equal-mean regression을 먼저 통과해야 한다.

### A. Shape512

- SS64는 intervention하지 않는다.
- 동일 SS64 coordinate, Shape512 noise, sampler/CFG를 모든 arm에서 사용한다.
- Shape512 aggregation만 바꾼다.

### B. Shape1024

- 동일 SS64 coordinate를 기록한다.
- clean/equal-mean Shape512에서 얻은 동일 SLat과 upsample/quantized coordinate를
  artifact로 고정한다.
- 동일 Shape1024 noise를 사용하고 Shape1024 aggregation만 바꾼다.

### C. PBR1024

- 동일 Shape1024 SLat/coordinate와 PBR noise를 사용한다.
- PBR1024 aggregation만 바꾼다.

### D. Cumulative

stage-local 결과로 방법을 선택한 뒤에만
Shape512 → Shape1024 → PBR1024 전체에 누적 적용한다.

기존 pipeline이 매번 전체 cascade를 실행하므로, upstream latent/coordinate와
noise를 save/load할 수 있는 별도 resumable stage runner를 만든다.

## 8. Progressive execution gates

| Gate | 범위 | 진입 조건 | 산출물 |
|---|---|---|---|
| 0 | 코드/저장소 감사 | 없음 | audit, baseline test |
| A | feature-only, 전체 S arm | utility/unit test 완료 | metric table, weight/confidence 시각화 |
| B | 대표 3–5 object stage-local 3D | 체크포인트 등록 + equal-mean regression + Gate A headroom | stage-local render/latent comparison |
| C | 10–20 object, S0/best/oracle | Gate B에서 feature 개선이 3D에 연결 | controlled-track summary |
| D | real VLM track | controlled track에서 유효성 확인 | 별도 real-data report |
| 2 | transport | Stage 1 consensus 또는 oracle headroom 확인 | T0–T4 controlled experiment |

한 seed 42 결과는 mechanism prototype으로만 해석한다.

## 9. Stage 2 진입 후의 범위

Stage 2는 Gate A/B 결과 승인 후 별도 상세 설계를 확정한다. 최초 prototype은
같은 normalized image-space offset을 L/H branch에 공유한다.

```text
F_i_transport(q) = sample(feature_map_i, p_i(q) + delta_i(q))
L_total = L_data + lambda_smooth L_smooth + lambda_magnitude L_magnitude
```

허용되는 update는 5–10회로 고정한 offset optimization/smoothing뿐이다.
flow/DINO/NAF parameter는 update하지 않는다.

비교 순서는 T0 weight-only, T1 unregularized bounded transport,
T2 smooth transport, T3 confidence-residual transport, T4 controlled C4/C5
oracle transport다. actual VLM view에는 T4를 적용하지 않는다.

Stage 2 단위 테스트는 identity/translation marker 이동, pixel↔normalized-grid
변환, out-of-bound/hole mask, smooth warp convention, neighbor TV를 먼저
통과해야 한다.

## 10. 3D 평가와 시각화

가능한 3D metric:

- conditioning-view silhouette IoU, SSIM, LPIPS, foreground RGB error
- DINO similarity와 held-out/novel-view perceptual consistency
- 공식 적용 가능성이 검증된 경우에만 MEt3R
- stage output latent drift
- mesh vertex/face/component 통계는 변화량 diagnostic으로만 사용

ground-truth 3D가 없으면 baseline 대비 Chamfer/normal/mesh 통계를 정확도
지표로 해석하지 않는다.

최종 report에는 source/generated/corrupted grid, mask overlay, warp field,
projected-pixel overlay, confidence/weight map, feature cosine/PCA, stage-local
render, cumulative render, transport offset/smoothness, failure contact sheet를
상대 경로로 직접 삽입한다.

## 11. 구현 work package와 검증

설계 승인 후 다음 순서로 수행한다.

1. **Pure aggregation utilities**
   - naive reference, chunked residual weighting, fallback/diagnostics
   - normalization, sum-to-one, outlier, identical view, dtype/device/shape test
2. **Conditioner integration**
   - explicit policy config와 exact default bypass
   - K=1, repeated K, alpha=0, default equal-mean regression test
3. **Projection/oracle diagnostics**
   - coordinate/valid-mask query와 mask projection
   - baseline feature 계산에는 valid mask를 적용하지 않음
4. **Controlled corruption package**
   - C1–C5 deterministic generation, mask/field metadata, contact sheet
   - identity/translation/smooth-warp correspondence test
5. **Feature evaluator와 manifest**
   - active-coordinate metric, oracle gap, resumable status와 hashes
6. **Checkpoint registry와 fail-closed loader**
   - path/hash/step/raw-EMA/config/strict result 기록
   - public single-view fallback 방지
7. **Stage-local runner**
   - upstream coords/SLat/noise 저장·재사용과 hash 검증
8. **Gate A → B → C**
   - feature 결과를 먼저 승인하고 순차적으로 GPU 범위를 확대
9. **Stage 2 go/no-go**
   - headroom이 확인된 경우에만 상세 transport 구현 계획을 확정
10. **보고서**
    - negative/failure 결과를 포함해 prototype report와 그림 완성

각 package는 작은 단위 테스트를 먼저 작성하고, 기존
`tests/multiview` 전체 회귀를 통과시킨 뒤 다음 package로 넘어간다.

## 12. Artifact와 manifest

run 단위 manifest에 다음을 저장한다.

- git commit/code snapshot
- checkpoint absolute path, SHA-256, step, raw/EMA, config, load result
- object/view ID와 input/mask/warp hash
- K, seed, camera/FOV/distance/transform/mesh_scale
- stage, arm, alpha, temperature
- sampler/CFG/noise hash
- fixed upstream coordinate/SLat hash
- metric/visualization/output 경로
- started/completed/failed 상태와 실패 이유

동일 run ID 재실행 시 완료 artifact는 hash가 맞으면 재사용하고, 불일치하면
조용히 overwrite하지 않고 새 run 또는 명시적 invalidation을 요구한다.

## 13. 실행 전 필요한 외부 입력

설계 승인 후 utility 구현은 checkpoint 없이 진행할 수 있다. GPU Gate B
이전에는 다음이 반드시 필요하다.

- Shape512/Shape1024/PBR1024 multi-view checkpoint의 정확한 경로와 raw/EMA 정보
- multi-view SS64 checkpoint 또는 고정된 SS64 sparse coordinate artifact
- Track A clean calibrated multi-view data root/manifest
- Track B를 실행할 단계에서 source/generated pair와 camera metadata 위치

현재 checkpoint가 없으므로 실제 3D generation, 모델별 결과 비교, GPU
성능 실험은 계획하지 않는다.
