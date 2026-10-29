# BenchUp 추정 카메라 점검

`unit-scale-anchor-manifest-v1.jsonl`의 공개 편집 109건에 기록된 source mesh 재투영 실루엣 IoU를 집계했다. target과 같은 인덱스의 source 카메라 및 조건 source 3장에 대한 값이며, 표본은 객체당 편집 수에 따라 중복된다.

| Source 뷰 | 최소 | 중앙값 | IoU < 0.7인 편집 건수 |
|---|---:|---:|---:|
| Target 인덱스의 source 뷰 | 0.661 | 0.878 | 5 |
| 조건 source 1 | 0.661 | 0.882 | 2 |
| 조건 source 2 | 0.661 | 0.864 | 3 |
| 조건 source 3 | 0.681 | 0.881 | 7 |

네 뷰에서 IoU 0.5 미만은 없다. 최저 사례들은 주로 stormtrooper에서 발생한다. 이 수치는 **원본 source mesh와 원본 source 이미지**의 실루엣 일치에 대한 진단이며, 편집 target 이미지가 해당 source 뷰와 동일한 카메라를 유지했다는 증거가 아니다. 카메라·조명 및 모델별 입력 차이가 해소되기 전에는 진행 중인 결과를 잠정 평가로 표시한다.

집계 입력: `/root/dev/.cache/benchup-runtime/unit-scale-anchor-manifest-v1.jsonl`; 원본 후보 및 재투영 자료: `/root/dev/.cache/benchup-camera-v1/render-qa`와 `recovery-render-v1`.
