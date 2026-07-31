# CorrAdapter-style Multi-view Pixal3D 코드 감사

> 상태: 사용자 승인 완료 — Stage 0 코드 감사
> 작성일: 2026-07-31
> 실험 저장소: `Pixal3D`의 linked worktree
> `/root/dev/Pixal3D/.worktrees/multiview-model-extension`
> 작업 브랜치: `feature/multiview-correspondence`
> 기준 커밋: `3ff6fca5fa65246e9b85b90d4c86a0aebb488d4a`

## 1. 저장소와 근거 범위

실제 구현 대상은 multi-view 확장 코드가 있는 위 linked worktree다.
요청서에 적힌 `/root/dev/Pixal3D-multiview` 디렉터리는 현재 노드에 존재하지
않았으며, 임의로 저장소를 새로 만들지 않았다. 작업 브랜치는
`feature/multiview-model-extension`의 현재 커밋에서 분기했다.

`/root/dev/Pixal3D`의 `master` checkout은 single-view 선행 실험 참고
저장소일 뿐 이번 구현 대상이 아니다. 또한 다음 두 선행 문서는 현재 로컬
파일시스템과 로컬 git ref에서 찾을 수 없었다.

- `.worktrees/projection-feature-ablation/docs/experiments/2026-07-28-projection-conditioning-causal-ablation.md`
- `docs/superpowers/specs/2026-07-27-projection-feature-ablation-design.md`

따라서 선행 실험의 수치는 사용자가 제공한 source of truth로 인용하되,
로컬 산출물을 재검증했다고 기록하지 않는다. 반면 multi-view 구조에 관한
아래 결론은 현재 브랜치의 실제 코드와 config를 근거로 한다.

## 2. 현재 conditioning 경로

### 2.1 입력과 anchor-relative camera

`DinoV3ProjFeatureExtractor._forward_multiview()`는 다음 입력을 요구한다.

| 입력 | shape |
|---|---:|
| `image` | `[B, K, 3, H, W]` |
| `camera_angle_x` | `[B, K]` |
| `distance` | `[B, K]` |
| `mesh_scale` | `[B]` |
| `transform_matrix` | `[B, K, 4, 4]` |

근거:
[`image_conditioned_proj.py`](../pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py)
689–716행.

첫 view가 anchor다.
`compute_multiview_projection_matrices()`는 FP32에서 다음을 계산한다.

```text
relative_0 = I
relative_i = inverse(T_anchor) @ T_i
fixed_anchor[:, 1, 3] = -distance[:, 0]
projection_0 = fixed_anchor
projection_i = fixed_anchor @ relative_i
```

근거: 같은 파일 225–256행. 각 non-anchor의 `distance_i`는 projection
translation을 직접 결정하지 않고 anchor distance만 사용된다. 이는 현재
anchor-relative 설계와 일치한다.

### 2.2 view별 feature 생성과 L/H concat

각 view는 동일한 `_forward_single_view()`를 순차적으로 통과한다.

| feature | 생성 방식 | view별 shape |
|---|---|---:|
| `z_global_i` | DINOv3 CLS 1개 + register token 4개 | `[B, 5, 1024]` |
| `z_proj_lr_i` | native DINO patch field back-projection | `[B, R^3, 1024]` |
| `z_proj_hr_i` | RGB-guided NAF field back-projection | `[B, R^3, 1024]` |
| `z_proj_i` | `cat([z_proj_lr_i, z_proj_hr_i], dim=-1)` | `[B, R^3, 2048]` |

근거: 같은 파일 585–687행. L/H concat 순서는 676–677행에서 고정되어
있다. 이 순서는 denoiser의 서로 다른 `proj_linear` column에 대응하므로
변경하면 안 된다.

### 2.3 view mean이 수행되는 위치

`_forward_multiview()`는 generator로 view별
`(z_global_i, z_proj_i)`를 만들고 즉시
`_online_mean_tensor_groups()`에 넘긴다.

```text
view_features = (_forward_single_view(view_i) for i in range(K))
return _online_mean_tensor_groups(view_features)
```

근거: 같은 파일 689–727행. `_online_mean_tensor_groups()`는
fp16/bf16 입력을 FP32 누산한 뒤 원 dtype으로 되돌린다(35–70행).

따라서 현재 equal-mean 직후 per-view feature는 사라진다. 가장 안전한
aggregation hook은 view generator가 생성된 뒤 기존 online mean을 호출하기
직전이다.

최종 conditioner interface는 다음과 같다.

| 출력 | shape |
|---|---:|
| `z_global` | `[B, 5, 1024]` |
| `z_proj` | `[B, R^3, 2048]` |

### 2.4 denoiser가 condition을 소비하는 방식

Dense와 sparse `ProjectAttention` 모두 다음 계산을 한다.

```text
global_out = CrossAttention(x, context["global"])
proj_out = proj_linear(context["proj"])
output = global_out + proj_out
```

근거:
[`pixal3d/modules/attention/proj_attention.py`](../pixal3d/modules/attention/proj_attention.py)
33–48행과
[`pixal3d/modules/sparse/attention/proj_attention.py`](../pixal3d/modules/sparse/attention/proj_attention.py)
21–47행.

세 대상 config는 모두 `image_attn_mode="proj"`이고
`proj_in_channels=2048`이다. 따라서 parameter-free aggregation만
conditioner 내부에 추가하고 최종 shape와 L/H 순서를 유지하면 denoiser
architecture와 flow-model state dict는 바뀌지 않는다.

## 3. Projection coordinate와 valid mask

`ProjGrid.forward()`는 dense R³ grid를 `mesh_scale`로 정렬한 다음
`project_points_to_image_batch()`로 다음 값을 만든다.

- image pixel coordinate `[B, R^3, 2]`
- depth `[B, R^3]`
- `valid_mask` `[B, R^3]`

근거: `image_conditioned_proj.py` 297–347행.

그러나 `valid_mask`는 sampled feature에 곱해지거나 반환되지 않는다.
sampling은 `padding_mode="border"`를 사용하는 `grid_sample`이고
(194–218행), `ProjGrid.forward()`는 sampled feature만 반환한다.

즉 현재 equal-mean baseline은 invalid projection을 hard reject하지 않는다.
새 방법에서 valid mask를 조용히 weight에 포함하면 confidence routing뿐 아니라
visibility 정책도 동시에 바뀐다. 최초 S0/S1에는 적용하지 않고, 사용할 경우
명시적인 별도 ablation으로 분리해야 한다.

추가로 기존 `ProjGrid.visualize_projection()`은 explicit
`transform_matrix`가 들어오면 assert로 중단된다(351–388행). 필수
multi-view projected-pixel 시각화를 위해서는 모델 계산을 바꾸지 않는
diagnostics-only projection query가 필요하다.

## 4. Dense projection에서 sparse active coordinate로의 변환

Conditioner는 view 평균까지 dense `[B, R^3, 2048]`에서 수행한다.
그 뒤 sparse stage가 실제로 사용할 coordinate만 indexing한다.

- trainer path:
  `encode_image_proj()`가 conditioner 호출 후 `coords`로 dense grid를
  indexing한다. 근거: `image_conditioned_proj.py` 1460–1527행.
- inference path:
  `get_proj_cond_shape()`가 dense result를 reshape하고 `coords`로
  indexing해 `SparseTensor`를 만든다. 근거:
  [`pixal3d_image_to_3d.py`](../pixal3d/pipelines/pixal3d_image_to_3d.py)
  307–394행.

따라서 현재 의미론을 가장 정확히 보존하는 첫 프로토타입은 dense
aggregation hook이다. 다만 feature-level primary metric은 flow model이
실제로 소비하는 active sparse coordinate에서 계산하고, dense 전체 평가는
보조 진단으로 둔다.

## 5. Stage별 차이와 sparse selection

| Stage | intervention | grid | image/NAF | active coordinate의 출처 |
|---|---|---:|---|---|
| SS64 | 제외 | 16 | 512 / NAF 없음 | SS flow output을 decode한 occupancy |
| Shape512 | 대상 | 32 | 512 / 512 | 고정된 SS64 coordinate |
| Shape1024 | 대상 | 64 | 1024 / 512 | 고정 Shape512 SLat을 decoder로 upsample 후 R64에 quantize/unique |
| PBR1024 | 대상 | 64 | 1024 / 1024 | 고정 Shape1024 SLat의 exact coordinate |

`sample_sparse_structure()`는 SS latent를 decode하고 occupancy의
`argwhere`로 coordinate를 만든다(`pixal3d_image_to_3d.py` 400–447행).
Shape512는 이를 그대로 사용한다(791–804행). Shape1024는 Shape512 SLat을
upsample하고 target grid에 quantize/unique한다(806–845행). PBR1024는 최종
Shape1024 SLat coordinate를 사용한다(866–879행).

현재 `run()`은 전체 cascade를 매번 연속 실행한다. 따라서 stage-local
causal isolation을 위해서는 다음 frozen artifact를 명시적으로 주입하는
별도 experiment runner가 필요하다.

- Shape512: 동일 SS64 coordinate와 동일 noise
- Shape1024: 동일 SS64 coordinate, 동일 baseline Shape512 SLat,
  동일 Shape1024 coordinate와 noise
- PBR1024: 동일 Shape1024 SLat/coordinate와 동일 PBR noise

seed만 같게 두는 것보다 artifact 자체를 저장하고 hash로 묶는 편이
intervention 사이 RNG 소비 차이를 차단한다.

## 6. 메모리 감사

`B=1`, `K=4`, `C=2048`일 때 projection tensor 자체의 이론 메모리는
다음과 같다.

| grid | N | view 1개 bf16 | K=4 bf16 stack | K=4 fp32 stack |
|---|---:|---:|---:|---:|
| R32 | 32,768 | 128 MiB | 512 MiB | 1 GiB |
| R64 | 262,144 | 1 GiB | 4 GiB | 8 GiB |

Global tensor `[1,4,5,1024]`는 bf16 약 40 KiB에 불과하다. 반면 R64
equal-mean의 FP32 projection accumulator만 약 2 GiB이며, cast/add 중
temporary와 현재 view의 DINO/NAF activation은 별도다. batch가 증가하면
표의 값은 `B`에 비례한다.

### 비교한 구현 전략

1. **Naive GPU stack**
   - 장점: 가장 단순하며 작은 tensor reference 구현에 적합하다.
   - 단점: R64 projection만 4 GiB이고 feature extraction activation과
     flow model memory를 포함하지 않는다.
   - 결정: 단위 테스트용 reference 외에는 사용하지 않는다.

2. **CPU/off-device bf16 cache + voxel chunk aggregation**
   - view를 순차 생성해 CPU bf16 cache에 보관한다.
   - R64/K4 host cache는 약 4 GiB다.
   - 예를 들어 voxel chunk 4,096개는 raw K-view projection이 약 64 MiB라
     GPU working set을 제한할 수 있다.
   - view weight/confidence 같은 diagnostics는 상대적으로 작다
     (`K*R^3` fp32 약 4 MiB).
   - 결정: 정확성, 관측 가능성, 재현성을 우선한 최초 Stage 1 권장안이다.

3. **Online/two-pass consensus with feature recomputation**
   - 1차 pass에서 normalized branch sum을 계산하고, 2차 pass에서 LOO
     score와 stable online softmax numerator/denominator를 누산할 수 있다.
   - host cache는 줄지만 DINO/NAF 계산을 반복하고 구현 복잡도가 증가한다.
   - 결정: CPU cache나 wall-clock이 병목으로 확인된 뒤의 최적화안이다.

4. **Sparse-active-only aggregation**
   - dense result를 모두 유지하지 않아 메모리는 가장 작을 수 있다.
   - 그러나 현재 conditioner 반환 contract와 dense diagnostics 경로를
     바꾸고 trainer/pipeline 두 경로에 훅이 분산된다.
   - 결정: 최초 정확성 프로토타입 이후에만 검토한다.

Equal mean과 `alpha=0`은 수학적으로 같은 새 weighted kernel을 거치지 않고
기존 `_online_mean_tensor_groups()`를 그대로 호출해야 수치적 회귀를 가장
강하게 보장할 수 있다.

## 7. Default compatibility와 checkpoint 안전성

새 option이 없을 때는 현재 generator와 online mean 경로를 그대로 사용한다.
새 aggregation은 parameter-free이고 최종 condition key/shape를 유지하므로
denoiser state dict에는 새 key가 생기지 않는다.

다만 현재 일반 inference 진입점은 기본 model path가
`TencentARC/Pixal3D`이고 flow checkpoint override가 선택적이다
([`inference.py`](../inference.py) 28–29행, 79–117행, 481–503행).
이는 실험에서 multi-view checkpoint가 빠졌을 때 공개 single-view weight로
조용히 실행될 위험이 있다.

실험 runner는 다음을 fail-closed로 강제해야 한다.

- Shape512/Shape1024/PBR1024 multi-view checkpoint 절대 경로 필수
- 각 파일 SHA-256, step, raw/EMA, config 기록
- 허용된 예외(`rope_phases`) 외 strict-compatible load 확인
- 모든 arm의 checkpoint hash 일치 확인
- SS64는 intervention에서 제외하되 multi-view SS checkpoint 또는 사전 저장된
  SS coordinate artifact를 필수 입력으로 요구

체크포인트가 제공되기 전에는 flow model을 이용한 3D generation을 실행하거나
결과를 주장하지 않는다.

## 8. 확인된 위험과 설계 보정

1. 요청서의 multi-view 저장소 경로는 현재 노드와 다르다. canonical linked
   worktree를 사용했고 임의 `git init`은 하지 않았다.
2. 선행 single-view 보고서 두 파일은 로컬에 없다. 사용자 제공 수치와
   multi-view 코드 감사 근거를 구분한다.
3. valid mask는 계산되지만 baseline에 적용되지 않는다. S1에 visibility
   weighting을 섞지 않는다.
4. 기존 projection 시각화는 explicit transform을 거부한다. diagnostics-only
   utility가 필요하다.
5. 기존 pipeline은 stage-local upstream artifact 고정을 지원하지 않는다.
   별도 runner가 필요하다.
6. 기존 inference의 optional checkpoint override는 single-view fallback
   위험이 있다. 실험 runner는 checkpoint 부재 시 중단해야 한다.
7. K=2의 leave-one-out agreement는 두 view가 대칭이어서 어느 쪽이
   잘못됐는지 식별할 수 없다. 최초 K=4 고정은 필수적이며, 2-vs-2 conflict는
   여전히 명시적 failure case다.
8. downstream이 sparse coordinate만 소비하므로 dense 전체 평균으로만
   feature 개선을 주장하면 안 된다. active-coordinate metric을 primary로 둔다.
9. `get_proj_cond_shape()`의 임시 grid override 복구는 `try/finally`가 아니다.
   stage runner가 override를 사용할 경우 예외 뒤 conditioner 상태 오염을
   방지하는 scoped context가 필요하다.

## 9. 현재 검증 상태

브랜치 생성 직후 GPU를 사용하지 않는 기존 multi-view test suite를 실행했다.

```text
780 passed, 16 skipped, 6 warnings in 45.72s
```

실행 환경:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  /opt/conda/envs/pixal3d/bin/python -m pytest tests/multiview -q
```

현재 감사 단계에서는 모델 코드나 inference 동작을 변경하지 않았다.
