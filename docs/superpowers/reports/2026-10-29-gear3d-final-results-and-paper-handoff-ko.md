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

## 3. Geometry-stage ablation

CD와 F-Score는 각각 `×10^4`, `×10^2` 표시다.

| Variant | CD ↓ | F-Score ↑ |
|---|---:|---:|
| Ours (Ray-fused Top-K) | **149.28** | 80.47 |
| Spatial retrieval | 151.25 | 80.37 |
| Semantic retrieval | 151.60 | 80.22 |
| Decoder matching (cosine) | 149.89 | 80.24 |
| Top-1 selection | 150.40 | 80.17 |
| Uniform Top-K weighting | 150.37 | 80.38 |
| Without retargeting | 151.36 | **80.50** |

Full model은 CD 기준 가장 좋으며, decoder matching·score-aware Top-K·ray fusion·retargeting을 조합한 파이프라인의 geometry 경향을 지지한다. 다만 133개 source cluster paired bootstrap 및 12-endpoint Bonferroni CI에서는 차이가 대체로 0을 포함하므로, 개별 모듈에 대한 통계적 유의성을 과장하지 않는다.

별도 correspondence 진단(DEV32, 50,493 observable relations)에서는 Raw cosine → matching decoder → score refinement → ray fusion 순으로 Exact@1이 1.25 → 7.73 → 8.56 → 11.19, Exact@8이 11.09 → 28.47 → 29.92 → 45.68로 개선됐다. 이는 후보 검색 개선을 직접 지지하지만, downstream 생성 성능의 독립적 인과 효과와 동일한 주장은 아니다.

## 4. Appearance-stage ablation

동일한 selected geometry, paired PBR noise, source cameras 9/10을 사용했다. LPIPS와 DreamSim은 `×10^3` 표시다.

| Variant | LPIPS ↓ | DreamSim ↓ | CLIP ↑ |
|---|---:|---:|---:|
| Original conditioning | 53.73 | **134.78** | **0.9388** |
| Top-1 retargeting (main) | 53.82 | 135.17 | 0.9384 |
| Top-K retargeting | **53.71** | 134.81 | 0.9387 |

Top-K는 Top-1보다 LPIPS가 `-0.00010595` 개선되었고 95% paired cluster bootstrap CI `[-0.00021024, -0.00000810]`은 0을 제외한다. 그러나 Top-K 대 original conditioning의 LPIPS/DreamSim/CLIP CI는 모두 0을 포함한다. 따라서 논문의 주장은 “appearance가 유지되는 가운데 geometry가 개선된다”로 제한하며, 광범위한 appearance 향상을 주장하지 않는다. Main comparison은 test 결과를 보고 checkpoint를 교체하지 않고 사전에 선택된 Top-1 checkpoint를 유지했다.

## 5. 논문 claim 구성

1. 핵심 정량 주장은 common-501에서의 geometry 및 target alignment 개선이다.
2. Semantic matching은 raw cosine보다 matching decoder가 correspondence retrieval을 크게 개선하고, score refinement와 ray fusion이 Top-K 후보 활용을 더 개선한다.
3. Ray-wise delivery와 retargeting을 포함한 full pipeline이 CD 기준 geometry ablation 최선이다.
4. Appearance는 baseline 수준을 보존하지만, common-501에서 일관된 우위는 확인되지 않았다.
5. 모든 비교는 common-success subset이라는 한계를 명시하며, 실패율/coverage 주장으로 확대하지 않는다.

## 6. 최종 산출물

- 논문 PDF: `docs/superpowers/reports/2026-10-29-gear3d-common501-final.pdf`
  - SHA256: `97dc00f73b3d07fd9ba46ed677769d7dbab27ee822e5c0345556b60203879c83`
  - 15 pages
- Overleaf source ZIP: `docs/superpowers/reports/2026-10-29-gear3d-overleaf.zip`
  - SHA256: `be1a23760346a96d86cedf24082b18831160c9e89a8d2793824b64d4f67d5b2b`
  - 21 source files; 별도 임시 디렉터리에서 clean compile 검증 완료
- Paper source: `docs/superpowers/paper/gear3d-overleaf`
- 실시간/실험 인계 기록: `docs/superpowers/reports/2026-10-28-scale-ungiven-live.md`

원본 평가 산출물은 다음 캐시 경로에 있다.

- Main/MV zero-shot: `/root/dev/.cache/deformnext-retargeting/fair-mv-zero-test508-20261029/common501-evaluation-v1`
- Geometry: `/root/dev/.cache/deformnext-retargeting/fair-ablation-test508-20261029/geometry-six-arms-v1`
- Appearance: `/root/dev/.cache/deformnext-retargeting/fair-ablation-test508-20261029/appearance-common501-v2`

## 7. 재검증 명령

```bash
cd /root/dev/Pixal3D/.worktrees/ray-score-prior-stage-20261019
sha256sum \
  docs/superpowers/reports/2026-10-29-gear3d-common501-final.pdf \
  docs/superpowers/reports/2026-10-29-gear3d-overleaf.zip

python docs/superpowers/paper/package_gear3d_overleaf.py

cd docs/superpowers/paper/gear3d-overleaf
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## 8. 제출 전 남은 수동 항목

- `main.tex`의 익명 저자 placeholder는 제출 모드에 맞춰 최종 확인해야 한다.
- CVPR submission ID 및 acknowledgements는 실제 제출 정보가 확정된 뒤 입력한다.
- 통계적으로 유의하지 않은 geometry 모듈 차이를 강한 독립 효과로 표현하지 않는다.
