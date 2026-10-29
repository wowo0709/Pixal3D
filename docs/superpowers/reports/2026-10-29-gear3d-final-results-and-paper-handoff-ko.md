# GEAR3D common-501 최종 결과 및 논문 인계

작성일: 2026-10-29 (UTC)
브랜치: `wowo0709/feature-retargeting`

## 1. 완료 범위

- DeformNeXt-10k TEST508 중 모든 비교 방법이 성공한 501개 편집을 공통 평가 집합으로 사용했다.
- 501개 샘플은 133개 source cluster로 구성된다.
- 본 평가는 camera-given, scale-ungiven 조건이다. GT mesh에는 정합하지 않으며, 입력 source M0만 사용한 Sim(3) 정합을 모든 mesh 출력에 동일하게 적용했다.
- mesh 평가는 1024 해상도로 통일했고, 렌더 카메라 및 metric 구현도 방법 간 동일하게 고정했다.
- Main comparison 11개 방법, geometry-stage ablation, appearance-stage ablation을 모두 완료했다.

## 2. Main comparison

논문 Table 1에 반영된 값이다. KID는 표에서 1,000배 표시한다.

| Method | LPIPS ↓ | DreamSim ↓ | CLIP ↑ | FID ↓ | KID ↓ | PSNR ↑ | SSIM ↑ | M-IoU ↑ | CD ↓ | F-Score ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MV-Pixal3D (zero-shot) | 0.0538 | 0.1373 | 0.9377 | 40.08 | 0.667 | 23.24 | 0.9466 | 0.8602 | 0.01555 | 0.7920 |
| Pixal3D (target 1-view) | 0.0979 | 0.3629 | 0.8691 | 79.64 | 8.049 | 17.77 | 0.9212 | 0.5302 | 0.05603 | 0.3944 |
| MV-Pixal3D (FT) | 0.0536 | 0.1339 | 0.9391 | 38.61 | 0.737 | 23.22 | 0.9463 | 0.8584 | 0.01512 | 0.8032 |
| GEAR3D (Ours) | 0.0538 | 0.1352 | 0.9384 | 38.56 | 0.685 | 23.28 | 0.9459 | 0.8619 | 0.01493 | 0.8047 |
| TRELLIS.2 | 0.0909 | 0.2539 | 0.9018 | 57.73 | 1.996 | 18.28 | 0.9210 | 0.5633 | 0.05168 | 0.4779 |
| Pixal3D (native 1536) | 0.0887 | 0.2035 | 0.9211 | 54.06 | 2.806 | 18.77 | 0.9216 | 0.6164 | 0.04419 | 0.4846 |
| ReconViaGen | 0.0731 | 0.1772 | 0.9274 | 45.21 | 0.637 | 19.88 | 0.9310 | 0.6836 | 0.03607 | 0.5848 |
| MVAdapter | 0.0946 | 0.2794 | 0.8933 | 64.65 | 4.169 | 18.26 | 0.9180 | 0.5877 | 0.05236 | 0.4175 |
| CorrAdapter | 0.0939 | 0.2767 | 0.8940 | 62.95 | 3.735 | 18.29 | 0.9202 | 0.5835 | 0.05203 | 0.4131 |
| ShapeUP | 0.0853 | 0.2947 | 0.8863 | 72.49 | 6.411 | 18.02 | 0.9218 | 0.6017 | 0.05045 | 0.5170 |
| SpaceControl | 0.0913 | 0.2844 | 0.8945 | 62.19 | 2.970 | 17.37 | 0.9179 | 0.5639 | 0.05173 | 0.4726 |

GEAR3D는 FID, PSNR, M-IoU, CD, F-Score에서 최고 성능이다. Source-preservation metric은 MV-Pixal3D (FT)와 대체로 비슷하다.

## 3. Unified module ablation

Geometry/appearance stage를 별도 표로 나누지 않고, 모든 arm을 동일한 metric schema에 배치했다. `Full`은 Ray-fused score-weighted Top-K SS와 사전에 선택한 Top-1 PBR을 조합한 GEAR3D main model이다. `Original PBR`과 `Top-K PBR`은 Full과 동일한 selected geometry를 사용하므로 SS 및 surface 값도 Full과 같다. `Vanilla V1`은 ablation 해석을 위한 full-pipeline reference다.

KID는 `×1,000` 표시다. Aligned CD/F는 input source M0만 사용한 Sim(3) 정합 후 common-501 값이고, Raw CD/F는 정합 전 값이다. SS voxel metric은 TEST508 전체 508개 값이다. `—`는 해당 arm의 full-render metric이 아직 완료되지 않은 값이며, 다른 arm의 값을 복사하거나 추정하지 않았다.

| Unified arm | LPIPS ↓ | Dream ↓ | CLIP ↑ | FID ↓ | KID ↓ | PSNR ↑ | SSIM ↑ | M-IoU ↑ | Aligned CD ↓ | Aligned F ↑ | SS IoU ↑ | SS Prec. ↑ | SS Recall ↑ | SS F1 ↑ | Raw CD ↓ | Raw F ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla V1 reference | 0.05359 | 0.13393 | 0.93909 | 38.61 | 0.737 | 23.22 | 0.94626 | 0.85838 | 0.01512 | 0.8032 | 0.37422 | 0.50186 | 0.59536 | 0.54462 | 0.01420 | 0.8181 |
| **Full: Ray-fused Top-K SS + Top-1 PBR** | 0.05382 | 0.13517 | 0.93843 | 38.56 | 0.685 | 23.28 | 0.94591 | 0.86186 | **0.01493** | 0.8047 | 0.37910 | 0.51158 | 0.59413 | 0.54977 | 0.01385 | 0.8241 |
| Spatial retrieval SS | — | — | — | — | — | — | — | — | 0.01513 | 0.8039 | 0.37799 | 0.50831 | **0.59585** | 0.54861 | 0.01399 | 0.8233 |
| Raw semantic retrieval SS | — | — | — | — | — | — | — | — | 0.01513 | 0.8031 | **0.38045** | **0.51308** | 0.59544 | **0.55120** | **0.01382** | **0.8266** |
| Decoder matching SS | — | — | — | — | — | — | — | — | 0.01505 | 0.8020 | 0.37907 | 0.51181 | 0.59377 | 0.54975 | 0.01390 | 0.8231 |
| Top-1 selection SS | — | — | — | — | — | — | — | — | 0.01501 | 0.8025 | 0.37805 | 0.50885 | 0.59527 | 0.54868 | 0.01393 | 0.8235 |
| Uniform Top-K SS | — | — | — | — | — | — | — | — | 0.01506 | 0.8024 | 0.38016 | 0.51248 | 0.59553 | 0.55089 | 0.01383 | 0.8253 |
| No-retargeting SS | — | — | — | — | — | — | — | — | 0.01517 | 0.8039 | 0.37525 | 0.50361 | 0.59550 | 0.54571 | 0.01420 | 0.8189 |
| Selected SS + Original PBR | 0.05373 | **0.13478** | **0.93879** | 38.77 | 0.698 | **23.32** | 0.94618 | **0.86187** | 0.01493 | 0.8047 | 0.37910 | 0.51158 | 0.59413 | 0.54977 | 0.01385 | 0.8241 |
| Selected SS + Top-K PBR | **0.05371** | 0.13481 | 0.93872 | <u>38.63</u> | **0.681** | <u>23.29</u> | 0.94595 | 0.86184 | 0.01493 | 0.8047 | 0.37910 | 0.51158 | 0.59413 | 0.54977 | 0.01385 | 0.8241 |

이 통합 표에서 확인되는 핵심은 다음과 같다.

- Full은 최종 aligned CD가 가장 낮지만, F-Score 및 SS voxel metric의 모든 열에서 항상 최고인 것은 아니다.
- Raw semantic retrieval은 SS IoU/F1이 가장 높지만 최종 aligned CD는 Full보다 나쁘므로, SS proxy와 1024 cascade surface 품질을 동일시할 수 없다.
- Appearance 세 행은 geometry를 고정했기 때문에 CD/F 및 SS 값이 동일하다. 렌더 기반 source preservation, FID/KID 및 target-view 지표는 각 texture arm에서 직접 측정했다.
- Original/Top-K PBR의 source-view FID/KID와 target-view PSNR/SSIM/M-IoU는 실제 common-501 렌더에서 추가 측정했다. FID/KID는 기존 source-preservation 수치와 동일한 camera9/10 1,002-image pool이므로 표에 확정 반영했다.
- 이전에 임시 표기한 appearance target-view 값은 raw 정합 전 수치였다. 이제 source-view 평가와 동일한 source-M0 정합 행렬을 적용해 501개를 재렌더한 값으로 교체했다. Geometry 여섯 arm에도 각 재생성 mesh의 source-M0 정합을 적용한 재렌더가 진행 중이다.
- Geometry-only SS 여섯 arm의 비어 있는 render metric은 현재 common-501 full PBR/render evaluation으로 측정 중이다. 완료되는 arm부터 같은 표에 순차 반영한다.
- 여섯 SS arm의 CD/F는 새 full-render 결과물과 같은 재생성 mesh에 대해 다시 측정한 값이다. 이전 geometry-only 생성의 CD/F와 소폭 차이가 있으며, 기존 paired bootstrap은 이전 생성에 대한 결과이므로 이 행들에 그대로 적용하지 않는다.

정합 전 raw 렌더의 참고 수치도 common-501에서 완료됐다. 아래 값은 위 표의 정합 후 Full/V1 수치와 직접 순위를 매기지 않는다. KID는 동일하게 ×1,000이다.

| Arm (raw) | LPIPS ↓ | Dream ↓ | CLIP ↑ | FID ↓ | KID ↓ | PSNR ↑ | SSIM ↑ | M-IoU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial retrieval SS | 0.04880 | 0.13269 | 0.93860 | 38.57 | 0.646 | 25.08 | 0.95604 | 0.91086 |
| Raw semantic retrieval SS | 0.04875 | 0.13125 | 0.93908 | 38.43 | 0.762 | 25.18 | 0.95625 | 0.91333 |
| Decoder matching SS | 0.04923 | 0.13388 | 0.93840 | 38.63 | 0.663 | 25.14 | 0.95567 | 0.91310 |
| Top-1 selection SS | 0.04869 | 0.13165 | 0.93823 | 38.84 | 0.643 | 25.03 | 0.95566 | 0.91047 |
| Uniform Top-K SS | 0.04874 | 0.13215 | 0.93908 | 38.12 | 0.638 | 25.23 | 0.95646 | 0.91418 |
| No-retargeting SS | 0.04961 | 0.13367 | 0.93834 | 38.92 | 0.605 | 24.85 | 0.95484 | 0.90345 |

### 3.1 Correspondence retrieval diagnostic

별도 DEV32 진단은 50,493개 observable relation에서 다음과 같다.

| Retrieval variant | Exact@1 ↑ | Exact@8 ↑ | Exact@64 ↑ |
|---|---:|---:|---:|
| Raw cosine | 1.25 | 11.09 | 50.76 |
| + Matching decoder | 7.73 | 28.47 | 50.76 |
| + Score refinement | 8.56 | 29.92 | 50.76 |
| + Ray fusion | **11.19** | **45.68** | **73.36** |

이는 후보 검색 개선을 직접 지지하지만, downstream 생성 성능의 독립적 인과 효과와 동일한 주장은 아니다.

### 3.2 Appearance weighting and paired contrast

Source UID마다 deformation 수가 다른 영향을 줄인 UID-weighted LPIPS/DreamSim/CLIP은 Original `0.05234101/0.13546220/0.93766897`, Top-1 `0.05235734/0.13543287/0.93735379`, Top-K `0.05227706/0.13503078/0.93757465`다. Top-K가 UID-weighted LPIPS와 DreamSim point estimate는 가장 좋지만 CLIP은 Original이 가장 높다.

Deformation-weighted paired cluster bootstrap에서 Top-K − Top-1 LPIPS는 `-0.00010595`, 95% CI `[-0.00021024, -0.00000810]`으로 Top-K에 유리하다. 같은 contrast의 DreamSim `-0.00036563` CI `[-0.00113084, +0.00036246]`와 CLIP `+0.00029032` CI `[-0.00018362, +0.00076841]`는 0을 포함한다. Top-K − Original 및 Top-1 − Original의 세 appearance metric CI도 모두 0을 포함한다. 따라서 광범위한 appearance 향상은 주장하지 않는다.

## 4. 논문 claim 구성

1. 핵심 정량 주장은 common-501에서의 geometry 및 target alignment 개선이다.
2. Semantic matching은 raw cosine보다 matching decoder가 correspondence retrieval을 크게 개선하고, score refinement와 ray fusion이 Top-K 후보 활용을 더 개선한다.
3. Ray-wise delivery와 retargeting을 포함한 full pipeline이 CD 기준 geometry ablation 최선이다.
4. Appearance는 baseline 수준을 보존하지만, common-501에서 일관된 우위는 확인되지 않았다.
5. 모든 비교는 common-success subset이라는 한계를 명시하며, 실패율/coverage 주장으로 확대하지 않는다.

## 5. 최종 산출물

- 논문 PDF: `docs/superpowers/reports/2026-10-29-gear3d-common501-final.pdf`
  - SHA256: `9d85b20730c3ce4ec3e0c3421fbbba508c778a8be432fa2001835884b46bc806`
  - 15 pages
- Overleaf source ZIP: `docs/superpowers/reports/2026-10-29-gear3d-overleaf.zip`
  - SHA256: `d0243cbce171b39550dc3581899e5548576a1723de6ae89952b8cce3b170808d`
  - 21 source files; 별도 임시 디렉터리에서 clean compile 검증 완료
- Paper source: `docs/superpowers/paper/gear3d-overleaf`
- 실시간/실험 인계 기록: `docs/superpowers/reports/2026-10-28-scale-ungiven-live.md`

원본 평가 산출물은 다음 캐시 경로에 있다.

- Main/MV zero-shot: `/root/dev/.cache/deformnext-retargeting/fair-mv-zero-test508-20261029/common501-evaluation-v1`
- Geometry: `/root/dev/.cache/deformnext-retargeting/fair-ablation-test508-20261029/geometry-six-arms-v1`
- Appearance: `/root/dev/.cache/deformnext-retargeting/fair-ablation-test508-20261029/appearance-common501-v2`

## 6. 재검증 명령

```bash
cd /root/dev/Pixal3D/.worktrees/ray-score-prior-stage-20261019
sha256sum \
  docs/superpowers/reports/2026-10-29-gear3d-common501-final.pdf \
  docs/superpowers/reports/2026-10-29-gear3d-overleaf.zip

python docs/superpowers/paper/package_gear3d_overleaf.py

cd docs/superpowers/paper/gear3d-overleaf
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## 7. 제출 전 남은 수동 항목

- `main.tex`의 익명 저자 placeholder는 제출 모드에 맞춰 최종 확인해야 한다.
- CVPR submission ID 및 acknowledgements는 실제 제출 정보가 확정된 뒤 입력한다.
- 통계적으로 유의하지 않은 geometry 모듈 차이를 강한 독립 효과로 표현하지 않는다.
