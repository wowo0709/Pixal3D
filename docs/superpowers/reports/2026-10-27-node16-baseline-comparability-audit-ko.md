# 노드16 외부 baseline과 ours TEST508 비교 가능성 감사

2026-10-27 현재 실제 실행 파일과 점수 산출물을 대조했다. 결론은 **같은 편집 예제를 사용했지만 평가 프로토콜이 달라 노드16 수치를 현재 논문 Table 1에 그대로 합칠 수 없다**는 것이다. 현재 Table 1의 여섯 행은 자체 공통 평가 아래의 내부 비교로 유지할 수 있다. 그 첫 행은 공식 Pixal3D baseline이 아니라 원본 single-view denoiser 가중치를 우리 프로젝트의 calibrated 1024-cascade 경로로 실행한 참고 결과다.

## 확인한 산출물

- 입력 계약: 이 저장소의 `2026-10-20-deformnext-baseline-agent-handoff-ko.md`와 로컬 `/root/dev/pixal3d-inspection/baseline-benchmark-preparation-v3-20261020`.
- 노드16 실행: `/home/seunghyun/editing_reproduction/setup/run-pixal3d.py`, Pixal3D clone commit `f7cf38429b0bd264f1995f0f8743a88b1c728b94` (`master`).
- 노드16 확정 프로토콜: `/home/seunghyun/editing_reproduction/outputs/deformnext-baselines/2026-10-24-frozen-test/metrics/pixal3d-full-test/protocol.json`, SHA256 `5abddf2ff8a51af360e39a1ff7f5dda61d3b54269afa846b3fd14713a6e0fbb1`.
- 노드16 실제 점수: 같은 경로의 `summary.json` 및 `all-methods-full_test.csv`. Pixal3D는 508/508 생성 및 평가 성공, 146 UID, 134 source cluster.
- 우리 RGB1 실행: `experiments/deformnext_retargeting/evaluate_pixal3d_singleview_baseline.py`; 점수는 `docs/superpowers/paper/final-results/singleview-row-audit.json`; 내부 표는 `docs/superpowers/paper/final-results/main-table.tex`.

준비 inventory와 우리 RGB1 assignment를 대조하면 508/508 sample ID, target RGB 경로·SHA256, source RGB 경로, 저장된 카메라 행렬이 일치했다. **편집 예제 자체의 불일치가 확인된 숫자 차이의 원인은 아니다.** 다만 이를 근거로 후속 생성·채점 경로까지 같다고 해석할 수는 없다.

## 실제 설정 차이

| 항목 | 노드16 외부 baseline | 현재 논문 내부 Table 1 |
|---|---|---|
| 평가 대상 | TEST508 508 edit, 134 source cluster | 동일 TEST508 508 edit, 134 source cluster |
| Pixal3D RGB1 생성 | 공식 `inference.py`, 공식 이미지 전처리, 입력 RGB에서 MoGe 카메라 추정, **1536 cascade**, seed 0 | 원본 single-view stage 가중치 + 프로젝트의 projective DINO extractor/`prepared_views`, 저장된 카메라, **1024 cascade**, 자산별 frozen seed |
| 입력 | M1 RGB@8 한 장 | RGB1 행은 M1 RGB@8 한 장; 다른 5행은 M0 또는 M1@8과 M0@9/10/11 네 장 |
| 이미지 평가 뷰 | 카메라 9/10만; RGB1에는 미관측 뷰 | source LPIPS/DreamSim/CLIP은 9/10, target PSNR/SSIM/M-IoU는 8; RGB1의 9/10만 미관측 |
| 렌더·참조 | Blender 4.2.3 Cycles, 512px, 64 samples, 3 area lights, 흰 배경; GT와 예측 모두 공통 재렌더 | Pixal3D native PBR renderer와 환경 맵, 검은 배경; 공통 original-material M1 참조 |
| FID/KID | 1,016장(508×2, 뷰 9/10), KID 100×100, seed 20261020 | 1,524장(508×3, 뷰 8/9/10), KID 100×1,000, seed 42017 |
| 표면 | 20k area-weighted samples, source-defined world, F@0.02; 노드16 고정 샘플링 seed | 20k area-weighted samples, source-defined normalized frame, F@0.02; 자산별 frozen seed; ICP 없음 |

Surface 거리 단위는 이름만으로 다르다고 단정할 수 없다. 확인한 508 source extent가 모두 1.0이어서 우리 정규화는 수치상 평행 이동만 한다. 그러나 생성 mesh, 표면 샘플, 랜덤 seed가 달라 CD/F-Score도 직접 비교할 수 없다. 이미지 지표는 특히 렌더러, 배경, 카메라 풀 및 참조 픽셀이 달라 단위만 같아도 같은 점수가 아니다. 동일한 이름의 KID도 subset 크기·seed·구현 RNG가 다르다.

| Pixal3D RGB1 결과 | LPIPS ↓ | DreamSim ↓ | CLIP ↑ | FID ↓ | KID ↓ | PSNR ↑ | SSIM ↑ | M-IoU ↑ | CD ↓ | F-Score ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 노드16 공식 경로 + 외부 프로토콜 | 0.107570 | 0.268119 | 0.906773 | 68.560777 | 0.005599 | 16.907946 | 0.915219 | 0.407389 | 0.103517 | 0.153147 |
| 우리 calibrated 경로 + 내부 프로토콜 | 0.060898 | 0.162024 | 0.940617 | 34.147430 | 0.001681 | 24.542797 | 0.951388 | 0.927251 | 0.017891 | 0.743113 |

이 표는 **프로토콜 불일치 진단용**이다. 행간 우열이나 성능 향상량을 읽으면 안 된다. `actual-pixal3d-singleview-TEST508-v1/cases/*/shared/render-evaluation.json` 508개를 다시 집계하면, 같은 우리 RGB1 예측·native renderer에서도 target PSNR/M-IoU가 카메라8의 24.543/0.927에서 카메라9/10 평균 23.248/0.802로 바뀐다. 이는 차이의 한 원인을 확인한 것이며, 남은 차이를 MoGe·cascade·렌더러 중 하나에 귀속하는 대조 실험은 아직 없다.

## 논문 결정과 필요한 공통 평가

1. 현재 여섯 행은 **내부 native 평가 표**로만 유지한다. 첫 행 이름을 `Pixal3D RGB1 (calibrated)`로 바꾸고, 공식 Pixal3D zero-shot baseline이라고 부르지 않는다. 해당 행의 9/10은 미관측 뷰이며 다른 다섯 행의 9/10은 입력 뷰임을 캡션에 명시한다.
2. 노드16 외부 baseline 수치는 이 표에 붙이지 않는다. 별도 표를 만들더라도 지금은 우리 모델의 노드16 공통 프로토콜 재평가가 없어 정량적 우위 주장을 쓰지 않는다.
3. 외부법과 ours를 같은 표에서 비교하려면 **고정된 우리 예측 mesh/material**을 노드16 평가 입력 형식으로 변환한 뒤, 같은 source-defined world, Blender/Cycles, 뷰 9/10, 원본 material M1 GT, 공통 20k surface sampling, 동일 FID/KID 설정으로 508개 모두 재평가한다. 변환 후 mesh 좌표 및 material 검사와 성공 분모 508 확인이 선행되어야 한다. 필요하면 공식 Pixal3D 출력도 같은 공통 evaluator에서 재평가한다.
4. 입력 조건은 별도 표 열로 선언한다. 우리 네 뷰 모델은 source 9/10을 보지만 대부분의 target-only baseline은 보지 않는다. 공통 evaluator를 통과해도 이를 같은 입력 예산의 직접적인 공정 비교로 설명할 수는 없다. 특히 9/10 source preservation 점수는 입력 권한을 함께 읽어야 한다.

노드16 `all-methods-full_test.csv`에서는 MVAdapter 504/508, CorrAdapter 502/508이고 FID/KID가 비어 있다. GeoStyle full run은 취소되었다. 외부 통합 표를 완성할 때 실패 분모 처리와 미완성 행 처리도 같은 규칙으로 고정해야 한다.

## 2026-10-27 공통 평가 진행 후 확인

위의 ‘필요한 공통 평가’는 현재 진행 중이다. [통합 평가 중간 표](2026-10-27-unified-test508-live.md)는 모든 방법이 성공한 동일한 502개 편집을 최종 분모로 사용한다. [QA11](2026-10-27-early-common-qa11-ko.md)은 먼저 완료된 11개 편집·3 source cluster에 두 Pixal3D 출력을 **같은 노드16 Blender/Cycles 평가기**로 채점한 진단이다. 그 결과 공식 경로와 보정 경로는 각각 target M-IoU 0.4531/0.9549, CD 0.06200/0.02045, source LPIPS 0.1068/0.0674로 여전히 크게 다르다. 따라서 기존 절대 점수 차이를 평가기 차이만으로 설명할 수 없다. 다만 QA11은 선택 편향이 있어 전체 502개의 효과 크기나 우열을 대표하지 않는다.

두 경로는 동일한 편집 예제와 공통 평가기를 사용해 통합 **과제 비교** 표에 별도 행으로 넣을 수 있다. 그러나 공식 Pixal3D는 RGB에서 MoGe 카메라를 추정하는 native 1536 cascade이고, 보정 RGB1은 제공된 source-defined 카메라와 projective 1024 cascade를 사용하는 별도 생성 경로다. 후자를 공식 Pixal3D의 재현 결과로 부르거나 둘의 차이를 fine-tuning만의 효과로 귀속하면 안 된다. 통합 표 입력 열의 `C`는 제공된 카메라 메타데이터를 뜻한다. 모듈 및 학습 효과 주장은 같은 입력·카메라·생성 스택의 내부 arm끼리 비교한다.
