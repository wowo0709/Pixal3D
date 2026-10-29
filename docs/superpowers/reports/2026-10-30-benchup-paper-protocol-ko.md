# BenchUp 평가 기록 및 논문 기술 초안

이 문서는 현재 실행을 기술하는 초안이다. 실험 완료와 카메라·렌더 검증 전에는 수치나 우열을 논문 주장으로 확정하지 않는다. 최초 계획서 `experiments/benchup/protocol.json`은 draft이며, 아래는 실제 runner와 평가 코드에 근거한다.

## 평가 모집단과 입력

- 공개 BenchUp revision `a8b87439281c7f64788231f0935b6d6aedc3ba85`의 편집 이미지 109건과 원본 물체 23개를 사용한다. ShapeUP 논문에서 기술한 100건·24개와 공개본의 관계는 확인되지 않아 논문의 숫자와 직접 비교하지 않는다.
- Pixal3D 계열은 편집 target RGB 1장과 원본 source RGB 3장 또는 target RGB 1장만 받는다. 다른 모델은 각 방법의 구현에 맞는 입력을 사용한다. ShapeUP은 원본 mesh와 source views, SpaceControl은 원본 RGB와 예측된 target proxy mesh를 사용한다. 입력이 동일하지 않으므로 비교표에 이를 명시한다.
- 공개 데이터에서 target GT mesh와 명시적 카메라 행렬을 찾지 못했다. 카메라는 source mesh와 source RGB만으로 추정했으며, 편집 target과 같은 인덱스의 source 카메라가 일치한다는 가정은 검증되지 않았다. 이를 GT camera 또는 검증된 camera-given이라고 부르지 않는다.
- Pixal3D에 target GT scale을 주지 않는다. 현재 입력 manifest의 `mesh_scale=1.0`은 고정 단위 값이며, source mesh는 카메라 추정과 보조 정합에 쓰인다. 서로 다른 방법이 source mesh에 접근하는 정도가 다르다는 점을 밝힌다.
- seed는 전 방법에 동일하지 않다. 현재 Pixal3D cascade/SS runner는 frozen manifest의 사례 인덱스 `i`에 대해 `42+i`, ReconViaGen 및 ShapeUP texture runner는 `0`을 사용한다. 각 예측 receipt의 seed와 입력 hash가 개별 기록이므로 이를 확인하고, 모든 방법에 동일한 seed를 썼다고 쓰지 않는다.

## 점수와 보조 정합

- **Raw**와 **source-M0 Sim(3)**을 각각 보고한다. 후자는 입력 source mesh만을 사용한 사후 정합이며 편집 target mesh는 사용하지 않는다. source 형태를 향한 정합은 의도된 변형을 벌점화할 수 있다. Raw를 함께 제시한다.
- 렌더는 추정된 target 카메라 1개와 source 카메라 2개를 사용한다. 이미지는 RGBA를 512×512로 Lanczos 리사이즈하고 흰 배경에 합성한다. 렌더 조명과 원본 이미지 조명이 다를 수 있다.
- **Target image alignment:** 편집 입력 target RGB와 target 카메라 렌더 사이의 PSNR, SSIM과 alpha threshold 0.5의 mask IoU. 이는 입력 뷰에 대한 일치이며 독립된 편집 GT나 3D geometry의 정답 정확도가 아니다. SSIM은 11×11 Gaussian, σ=1.5의 선언된 구현을 쓴다.
- **Source preservation proxy:** 두 원본 source RGB와 각 카메라 렌더 사이 LPIPS, DreamSim, CLIP 이미지 유사도의 평균. 원본 source는 편집 전 상태이므로 의도된 편집까지 오차로 집계될 수 있다. 이를 변형된 target의 GT 이미지로 표현하지 않는다.
- **FID/KID:** 동일 성공 집합의 target 렌더와 편집 target 입력 이미지 사이 Inception 특징 분포를 비교한다. 이미지가 동일 물체에 여러 번 포함되므로 독립 표본이 아니며, 특히 공통 성공 수가 100보다 작으면 불안정하다. 이를 다른 reference pool로 계산한 DeformNeXt 수치와 같은 값으로 취급하지 않는다.
- 편집 target의 GT mesh가 없으므로 target CD와 F-score는 계산하지 않는다. 사례별 성공 수/전체 109 및 전 모델 공통 성공 집합을 함께 보고한다. 차이의 불확실성은 물체 단위 paired bootstrap으로 계산하되, 공통 물체가 적을 때 신뢰구간도 잠정적이다.
- TRELLIS.2 계열과 SpaceControl의 네이티브 축은 2026-10-30에 확인한 앞뒤 변환을 적용한다. 수정 전 결과를 수정 후 비교와 혼합하지 않는다. 관련 시각 검증과 추가 한계는 `2026-10-30-benchup-native-orientation-correction-ko.md`에 기록했다.

## 영어 서술 초안 — 결과 입력 전

> We evaluate on the 109 edited images associated with 23 source objects in the public BenchUp release. Because target meshes and calibrated target cameras are not provided, we estimate cameras using source observations and the source mesh, and report image-space comparisons under the stated same-index view assumption. We report unaligned outputs and a supplementary source-mesh Sim(3) alignment, together with per-method coverage and scores on the shared successful cases. Target-view image agreement is measured against the edited input; source-view similarity is a preservation proxy against unedited inputs. Consequently, these metrics do not establish target 3D surface accuracy, and source-view penalties can include intentional edits.

이 문구는 결과가 완료된 후 실제 manifest, receipt와 전체 검증 결과에 맞춰 다시 점검한다. 공통 사례가 적은 중간 수치에서 모델의 우열이나 일반화 성공을 주장하지 않는다.
