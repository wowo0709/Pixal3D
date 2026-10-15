# DeformNeXt semantic arm 성능 하락 원인 감사

2026-10-15. 결론은 **epoch 부족만으로 설명할 수 없으며, 이전 리포트의 구조를 재현하지 못한 구현과 native conditioning을 크게 바꾸는 fusion을 먼저 수정해야 한다**는 것이다. 현재 결과만으로 semantic matching이나 retargeting 개념 자체를 폐기하지 않는다.

## 실행 상태와 해석 범위

- V1 SS/PBR, SS B1/B2, PBR B1은 5 epoch 완료했다. SS B3도 480 step으로 완료했고 최종 quick IoU는 0.429815, 최고는 epoch 4의 0.435721이다.
- 원인 분석 중 SS/PBR 자동 arm queue를 종료했다. PBR B2는 step 362에서 정상 checkpoint-stop했다. C1/C2는 학습하지 않았다.
- `runs_world_v4/SEMANTIC_AUDIT_HOLD.json`은 기존 semantic arm의 자동 재실행을 막는다. V1에는 적용하지 않는다. 기존 checkpoint와 metric은 삭제하지 않는다.
- world-v4 geometry cache의 등록·좌표 계약은 이번 감사에서 변경하지 않았다. 수정 대상은 descriptor/matcher supervision, feature carrier, fusion이다.

이 문서는 정해진 새 설계의 성공을 선언하지 않는다. 아래 generation counterfactual은 SS 4개/PBR 2개 고정 panel이며, topology 진단은 32 category-stratified validation asset이다. 결과를 본 뒤 고른 blend 계수는 진단용이며 새로운 main ablation hyperparameter나 최종 성능으로 승격하지 않는다.

## 1. 초기 하락은 새 파라미터의 미학습 때문인가?

현재 B1에는 학습 가능한 추가 parameter가 없다. B2/B3도 Q2C checkpoint가 frozen이고 matching score/cache를 소비한다. 학습되는 것은 공통 denoiser LoRA다. 따라서 B1/B2/B3의 step-zero 하락을 새 모듈의 random initialization 탓으로 설명하면 안 된다.

C1/C2는 추가 retargeter를 갖지만 최종 residual layer는 zero-init이다. 실제 tensor audit에서 초기 C1/C2는 B3와 최대 약 1e-5의 CUDA scatter 순서 차이만 있었다. 기존 backward 검증은 adapter gradient를 확인했다. 다만 이 초기화가 보존하는 것은 **B3**이고 V1이 아니다. C1/C2 학습 전 성능이나 학습 효과는 아직 측정하지 않았다.

V1의 projection은 `(A + S1 + S2 + S3) / 4`다. 현재 semantic arm은 이를 `(A + 3 * transported_source) / 4`로 교체한다. Native global feature mean은 보존하지만 spatial source contribution 전체를 처음부터 교체한다. B1은 세 view 전체 중 Top-1 하나를 세 source의 기여량으로 사용한다. 이는 native mean과 다른 feature 분포와 view 정보를 만든다.

## 2. 이전에 승격한 구조와 다른 구현

이전 근거는 `anchor-hybrid-prestudy/docs/superpowers/reports_final2/2026-09-16-exg-app-cue-stage-b-r9-four-module-ablation-results-ko.md` 및 `pixal3d/experiments/exg_app_cue_b_r9_ladder.py`다.

| 항목 | 이전 승격 구조 | 현재 retargeting 구현 |
|---|---|---|
| Omega representation | L12 FRAME RAW, 1024-D | aggregator의 FRAME+GLOBAL concat, 2048-D |
| Q2C | D1 query-only contextual attention, 1 layer, 4 heads × 32-D | `[target, candidate, cosine]` pairwise MLP |
| 후보 집합 문맥 | query가 64개 후보 전체를 attend | 각 후보 score가 다른 후보와 독립 |
| Q2C 근거의 endpoint | GT surface 위 controlled APP-CUE RGB task | 실제 SS/PBR generation conditioning |

현재 `DeformationAwareConditioner.q2c`에는 attention block이 없다. 리포트에서 비교 대상으로 남긴 pairwise head와 성격이 유사하며, 승격한 D1을 구현했다고 볼 수 없다. 이 차이는 통합 구현의 누락이다.

32개 topology 진단에서 FRAME-only로 바꾸면 현재 concat 대비 Top-1이 visible query의 14.54%에서 달라졌다. Exact-patch Top-1은 concat 29.29%, FRAME-only 30.90%, Top-64 recall은 86.51%/87.21%였다. 이 표본에서 FRAME-only 개선은 작다. 따라서 branch 불일치만으로 generation 하락 전부를 설명하지 않는다.

또한 이전 리포트는 controlled RGB carrier의 모듈 순위를 보여 줬으며, 실제 Shape/deformation endpoint 이득은 별도 검증해야 한다고 명시한다. 이전 proxy-task gain을 새 SS/PBR endpoint의 확정 이득으로 옮겨 해석하지 않는다.

## 3. correspondence supervision은 topology-exact가 아니다

`precompute_transport.spatial_indices()`는 M1 depth를 world point `X1`으로 unproject하고, **동일한 X1**을 M0 camera에 project한다. `train_q2c.loss_for_cache()`는 이 spatial index를 positive label로 사용한다. M1의 triangle/barycentric correspondence를 통해 M0 point `X0`를 구하지 않는다. 따라서 함수 설명의 topology-exact supervision과 실제 구현이 다르다. Visibility도 같은 surface point의 source z-buffer visibility가 아니라 coarse source foreground patch 유효성에 의존한다.

독립 evaluator는 M1 mesh에 M0 vertex positions를 attribute로 붙여 rasterize했다. Target patch center의 barycentric source point를 M0 source view에 project하고, M0 z-buffer 차이 0.005 이하인 관측만 비교했다. 32개 asset의 target mask raster IoU 최저는 0.97825였다. Pixel/patch 경계의 raster 차이가 있으므로 exact-patch metric의 한계를 유지한다.

- Source-visible target query: 2,291개, visible query-view pair: 2,981개.
- 기존 spatial positive와 topology positive가 같은 patch인 pair: **65.15%**.
- 1 patch보다 큰 차이가 나는 pair: **12.41%**. 평균 차이 **0.4723 patch**.
- 같은 evaluator에서 current raw Top-1 **29.29%**, current pairwise reranker Top-1 **55.74%**, Top-64 recall **86.51%**.

즉 supervision은 수정해야 하지만, 현재 reranker가 raw matching보다 실제 대응 정확도를 개선하기도 한다. **B2의 generation 열세를 단순히 matcher 정확도 저하로 설명하는 것은 반증된다.** Value 선택·view 처리·fusion이 별도 bottleneck이다. 기존 Q2C log의 8.2%는 background를 포함한 다른 denominator와 spatial pseudo-label 기준이라 이 수치와 직접 비교하지 않는다.

## 4. confidence와 validity 계약

현재 warm-up loss는 listwise logsumexp 차이다. 한 query의 모든 score에 상수 `c`를 더해도 ranking과 loss가 변하지 않는다. 반면 B3 `_attention()`은 view confidence를 `max(sigmoid(logit))`로 만들고 view 간 normalize한다. 학습에서 식별하지 않는 absolute score offset이 fusion을 결정한다.

실제 cache 4개 category × 두 stage에서 모든 logit을 +10 이동한 probe:

- View 내부 candidate weight 변화는 약 1e-6 이하로 작았다.
- View weight의 평균 L1 변화는 **0.3215**.
- Source evidence의 평균 relative RMSE 변화는 **약 0.5909**.

이는 실제 모델의 오차가 59%라는 뜻이 아니라, 같은 ranking/listwise objective를 유지해도 fusion이 크게 바뀔 수 있는 calibration 결함을 보여 준다. 음의 shift에서는 `clamp_min(1e-8)` 때문에 confidence mass 자체가 줄어드는 경우도 있어 별도 주의가 필요하다.

또한 32개 진단에서 target background query의 100%에 source candidate가 배정됐다. 현재 matching cache에는 target no-match/NULL 판정이 없으며, source foreground validity만으로 target query의 관측 가능성을 판정하지 못한다. Semantic branch를 적용할 수 없는 query에는 native path fallback이 필요하다.

## 5. 값의 해상도와 fusion을 분리한 실측

`audit_transport_features.py`는 native와 cache를 **동일한 spatial projection**으로 비교한 뒤 semantic routing만 추가했다. Source/anchor image, camera, world-v4 scale은 동일하다.

| Native projection 대비 feature relative RMSE, 4 asset 평균 | SS | PBR |
|---|---:|---:|
| Cached spatial control | 0.0030 | 0.0458 |
| Current B1 | 0.3139 | 0.1634 |
| Current B2 | 0.3104 | 0.1528 |
| Current B3 | 0.3278 | 0.1228 |

SS의 cached spatial control은 native에 가깝다. PBR cache는 64×64 DINO 및 1024×1024 NAF를 32×32 값으로 평균해 저장하므로 source-value carrier가 달라진다. Omega matching이 틀리지 않아도 세밀한 appearance 정보를 잃을 수 있다.

다음은 **원래 mixed385 checkpoint, 동일 noise·12-step Euler·CFG 3, optimizer update 0**의 generation counterfactual이다. `alpha`는 `C = C_native + alpha * (C_B3 - C_native)`이며, alpha 1이 현재 B3다.

| 조건 | SS voxel IoU, 4 asset | PBR foreground PSNR, 2 asset |
|---|---:|---:|
| Native V1 | 0.451906 | 22.3945 dB |
| Cached values, spatial projection 유지 | 0.449264 | 18.9754 dB |
| Current B1 | 0.385256 | 16.0800 dB |
| Current B2 | 0.378752 | 15.1942 dB |
| Current B3, alpha 1 | 0.393490 | 11.9590 dB |
| Native + 0.1 × B3 변화량 | 0.454693 | 23.3145 dB |
| Native + 0.25 × B3 변화량 | 0.450080 | 23.8333 dB |
| Native + 0.5 × B3 변화량 | 0.419137 | 23.9901 dB |

Native SS와 B1/B2는 저장된 step-zero metric을 재현한다. B3에는 CUDA weighted scatter의 작은 차이가 있다. PBR native shaded PSNR은 기존 baseline 22.3738 dB와 약 0.021 dB 차이이며 native base-color PSNR 19.6681 dB는 재현한다. Shaded render의 작은 수치 차이를 허용해도 3–10 dB의 carrier/fusion 차이를 설명하지는 못한다.

PBR alpha 0.25의 base-color PSNR은 19.1319 dB로 native 19.6681보다 낮다. 따라서 shaded PSNR 상승을 모든 appearance 지표 개선으로 읽으면 안 된다. 계수의 최적값도 stage별로 다르다. 이 실험은 **native path를 보존하면 초기 하락을 상당 부분 피할 수 있음**을 보여 주는 diagnostic이며 새 모델 승격 근거가 아니다.

## 6. 더 학습할지, 무엇을 고칠지

SS B1/B2는 5 epoch 동안 plateau이고, B3도 최고 0.4357로 V1 최고 0.5005 아래다. PBR B1은 5 epoch까지 개선했지만 FG PSNR 19.39 dB로 V1의 학습 전 22.37 dB에도 못 미친다. 추가 epoch가 LoRA의 적응을 도울 가능성은 있지만, frozen proposal·잘못된 teacher·absolute-score calibration·coarse carrier 자체는 고치지 않는다. 현재 구현에 긴 학습 예산을 더 투입하는 것을 우선하지 않는다.

전면적인 backbone 교체 대신 다음 순서로 작은 실험을 열어야 한다.

1. **재현 계약 수정:** FRAME-only L12와 실제 D1 contextual decoder를 별도 schema로 구현하고 기존 pairwise 결과와 명확히 구분한다. 기존 feature cache에서 FRAME slice를 재사용할 수 있다.
2. **Teacher/visibility 수정:** topology transfer와 source z-buffer visibility로 correspondence를 만들고 proposal miss/NULL을 분리한다. GT는 학습 label·oracle 진단에만 사용하며 inference condition에 넣지 않는다.
3. **Fusion 수정:** native contribution을 보존하고 새 residual을 zero-start한다. Zero gate와 zero output layer를 동시에 둬 gradient를 막지 않는다. Rank/listwise score에는 shift-invariant normalization을 쓰고 absolute confidence가 필요하면 no-match/visibility calibration을 별도로 학습한다.
4. **PBR carrier 수정:** 32×32 pooling된 값으로 native 고해상도 appearance 경로 전체를 대체하지 않는다. Native HR path 유지와 matched HR sampling을 분리해 검사한다.
5. **Oracle와 짧은 학습으로 순서 결정:** native / predicted-corr / topology-oracle-corr / wrong-corr / source-off를 같은 held-out cohort에서 비교한다. Oracle도 V1을 못 넘으면 carrier/fusion/retargeting을 먼저 고치고 matcher 학습을 늘리지 않는다. Oracle은 좋고 predicted만 나쁘면 proposal/decoder/visibility를 개선한다.
6. **확대 평가 후 본 학습:** 다른 32–64개 validation asset에서 초기 native identity, 1 epoch pilot, endpoint utility를 확인한다. 현재 진단 panel에서 고른 계수를 그대로 승격하지 않는다. Source-off/wrong-corr, gate 사용량, target alignment와 source preservation을 함께 확인한다.

현재 근거는 semantic 방법 전체의 실패가 아니라 **구현 재현성 문제와 conditioning 교체 방식의 실패**를 지지한다. v4 geometry latent를 전부 다시 만드는 작업은 이번 원인 수정에 요구되지 않는다. 먼저 matcher/teacher/fusion과 PBR value sampling을 수정하고, 해당 match sidecar와 관련 checkpoint만 새 버전으로 분리해야 한다.

## 재현 artifact

모든 diagnostic은 `experiments/deformnext_retargeting/audit_*.py`에 있고 학습 update를 수행하지 않는다.

- `/root/dev/.cache/deformnext-retargeting/audit-semantic-topology-frame/report.json`
- `/root/dev/.cache/deformnext-retargeting/audit-semantic-features/report.json`
- `/root/dev/.cache/deformnext-retargeting/audit-semantic-confidence/report.json`
- `/root/dev/.cache/deformnext-retargeting/audit-semantic-generation/{ss64,pbr1024}/report.json`

Native optimizer/model continuation, geometry registration, LoRA/adapter gradient 검증이 통과해도 matcher의 학습 라벨·promoted architecture·endpoint utility를 검증한 것은 아니다. 이번 누락을 실제 구조·teacher·feature contract 및 causal control 검증으로 보완해야 한다.
