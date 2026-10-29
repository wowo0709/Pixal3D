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

## 편집 이미지와 source 뷰 유사도 진단

고정된 OpenAI CLIP ViT-B/16 이미지 특징으로 편집 target 이미지와 같은 물체의 원본 source 26장 사이의 유사도를 계산했다. 109건 중 이름상 같은 인덱스의 source가 1위인 경우는 34건, 상위 3위는 58건, 상위 5위는 67건이었다. 순위 중앙값은 3위이며, 가장 높은 유사도와 같은 인덱스의 유사도 차이 중앙값은 0.00187이다. 같은 인덱스가 10위 밖인 경우도 24건이다.

**이는 카메라 정답 검증이 아니다.** CLIP 유사도는 편집 내용, 외형과 가려짐의 영향을 받고 인접 카메라들을 구별하지 못할 수 있다. 결과만으로 같은 인덱스가 같은 카메라라고 확정하거나 다르다고 단정할 수 없다. 이 진단으로 평가 카메라나 입력 뷰는 선택·변경하지 않았다. 사례별 순위와 가중치·manifest 해시는 file4 `benchup/reports/target-view-retrieval-qa-v1.json`에 보관했다.
