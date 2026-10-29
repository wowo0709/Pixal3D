# BenchUp 네이티브 출력 방향 수정

## 발견과 검증

여러 물체의 공통 사례를 확인한 결과 TRELLIS.2, ReconViaGen, MVAdapter+Recon, CorrAdapter+Recon 및 SpaceControl의 기존 raw 렌더에서 입력 이미지의 뒷면이 나타났다. 기존 변환은 TRELLIS 네이티브 `+Y`를 앞쪽으로 취급했다. 고정 변환을 네이티브 Z축 기준 180° 회전했을 때 Batman의 얼굴, 카메라의 렌즈, 고양이의 얼굴이 각각 앞면으로 나타났다. SpaceControl Batman에서도 같은 현상을 확인했다. 이 검사는 개별 사례에 따라 회전을 선택한 것이 아니라 모든 대상 방법에 적용할 고정 축 변환을 검증한 것이다.

- 보정 전후 이미지: `/root/dev/.cache/benchup-runtime/orientation-pilot-180-v1/compare.jpg`, `compare2.jpg`; file4 `benchup/reports/`에도 보관했다.
- 변경 행렬은 네이티브 `-Y` 앞, `+Z` 위를 Pixal anchor의 `+Z` 앞, `+Y` 위로 매핑하는 행렬식 +1의 회전이다.
- 원본 생성 결과와 텍스처 mesh는 재사용한다. source mesh만 사용하는 M0 정합은 새 방향에서 다시 계산하며, 편집 target 3D GT를 사용하지 않는다.
- 이전 방향으로 계산한 TRELLIS.2 계열 및 SpaceControl의 렌더·지표는 폐기 대상으로 분리했다. 수정된 출력은 file4 `benchup/orientation-correct-v2/`에 저장한다. 다른 모델의 결과와 수정 전 수치를 섞지 않는다.

이 파일은 고정 축의 앞뒤 오류를 기록한다. 일부 샘플의 육안 확인은 모든 물체의 출력 방향과 추정 target 카메라 정확성을 증명하지 않는다. 공통 평가에서는 여전히 이 한계를 명시한다.

## 중단된 단일 mesh 복구

CorrAdapter `helmet__render_0006`의 기존 GLB는 완성됐지만 중단 시점에 bake 검증 영수증이 없었다. 원본 예측의 SHA-256, GLB 로딩, UV 및 PBR 이미지, 원본과 GLB 표면의 vertex-distance q99 0.00273을 확인한 후 검증 영수증을 복구했다. GLB와 영수증은 file4에 재보관했고 SHA-256을 확인했다. 수정된 방향의 source 정합과 렌더링은 다른 사례와 동일하게 다시 수행한다.
