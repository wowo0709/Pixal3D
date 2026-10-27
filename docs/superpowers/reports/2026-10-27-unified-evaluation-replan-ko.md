# 외부 baseline과 내부 ablation을 연결하는 평가 재설계

2026-10-27. 근거는 `2026-10-27-node16-baseline-comparability-audit-ko.md`, 노드16의 frozen protocol/실제 평가 산출물, TEST508 내부 원고와 file4 raw prediction archive다. 이 문서는 **평가 계획**이며 새로운 공통 평가 결과나 우위 주장이 아니다.

## 판단

노드16 설정의 장점은 모든 방법의 출력 mesh를 같은 source-defined world에 놓고, 같은 Blender/Cycles 조명·카메라·GT material로 다시 그리며 표면을 공통 규칙으로 채점한다는 점이다. 외부법과 ours의 출력을 비교할 때 이쪽이 적합하다. Cycles와 흰 스튜디오 배경 자체가 실제 촬영 장면을 대표하지는 않는다. 노드16의 9/10번 뷰는 target-only 방법에는 미관측 뷰지만, ours와 ReconViaGen 같은 source-conditioned 방법에는 **입력 source 뷰**다. 따라서 9/10을 모든 방법에 대해 `novel view`라고 부르면 안 된다.

가장 빠르고 검증 가능한 통합은 **기존 노드16 9/10 공통 프로토콜을 유지하고 ours의 frozen 출력을 그 evaluator로 재채점**하는 것이다. 동시에 target8과 모든 방법에 미관측인 추가 카메라를 분리해 측정하면 편집 목표와 3D 일반화를 설명할 수 있다. 각 뷰 집합의 숫자를 섞어 하나의 같은 metric 열이라고 표시하지 않는다.

## 세 평가 축과 표 구성

| 축 | 뷰·참조 | 해석 | 필요한 작업 |
|---|---|---|---|
| 공통 외부 비교 | 현 노드16 protocol의 9/10; original-material M1 GT; CD/F@0.02는 공통 world | 동일 출력 채점. Target-only에는 novel, ours에는 입력 source 뷰 | ours의 기존 mesh/material을 export해 노드16 evaluator로 평가. 노드16의 기존 완성 baseline 9/10 점수 재사용 |
| Target edit fidelity | 입력 M1@8; 같은 Cycles 설정·GT | 편집된 target 관측과 geometry 정합 | **모든** 방법의 기존 출력을 camera8에서 새로 렌더·채점. 새 protocol/hash, 별도 열 또는 표 |
| 진짜 미관측 3D 뷰 | 8/9/10/11과 겹치지 않는 미리 정한 2개 카메라, 예: 기존 orbit의 12/13이 유효하면 사용 | 모든 방법에 미관측인 면의 geometry·appearance | VAL에서 카메라/GT material QA 후 고정. 모든 방법의 같은 mesh와 GT를 추가 렌더; TEST 점수를 보고 뷰를 고르지 않음 |

Source appearance는 9/10 M1-GT 대비 LPIPS/DreamSim/CLIP을 `source-view target fidelity` 또는 `source-view appearance`로 표기한다. 원래 M0 pixel 보존 그 자체를 측정한다고 쓰지 않는다. Foreground 또는 GT-mask crop 지표를 보조로 함께 보고, 흰 배경의 넓은 빈 영역이 PSNR/SSIM/LPIPS를 지배하는지 확인한다. 주 지표는 기존 full-frame과 공통 CD/F-Score로 유지한다. FID/KID는 동일 9/10 이미지 풀과 동일 구현·subset으로 다시 계산하고, 작은 상관 표본의 분포 요약으로만 해석한다.

논문에는 (1) **공통 evaluator 외부 방법 표**와 (2) **동일 backbone·입력·학습 예산을 맞춘 내부 ablation 표**를 둔다. 외부 표에는 입력 조건, 알려진 카메라 사용, 성공/508, 9/10이 각 방법의 입력인지도 같이 적는다. 공식 Pixal3D MoGe/1536은 공식 baseline으로 유지한다. 내부 `Pixal3D RGB1 (calibrated)` 1024는 별도 bridge/control로 표시해 공식 baseline과 합치지 않는다. 외부 표에서는 selected ours와 V1을 최소한 같이 넣어 성능과 fine-tuning 효과를 구별한다. 내부 표에는 RGB1 calibrated, source-only RGB4, edited RGB4, V1, Top-1 R, Selected를 같은 새 evaluator에서 다시 비교한다. 서로 다른 입력의 zero-shot 차이를 모듈 효과라고 해석하지 않는다.

## 무엇을 다시 실행할지

1. **먼저 VAL export smoke (예: 고정 16개).** File4 정본의 prediction NPZ에서 raw surface와 PBR material을 공통 GLB/표면 NPZ로 변환한다. native `o_voxel` export·축 변환·source-defined world를 한 규약으로 고정한다. GLB만 만들면서 생기는 remesh/decimation은 surface CD 입력에 쓰지 않는다. GT를 향한 ICP, bbox 재정규화, 개별 output 수동 정렬을 금지한다. Raw mesh와 export mesh의 bounds·axis·camera8/9 mask·재질 존재·해시를 검사한다.
2. **기존 TEST 예측을 공통 9/10으로 재채점.** 5개 frozen four-view pipeline ×508와 calibrated RGB1 ×508을 대상으로 한다. 생성 seed나 모델 선택은 바꾸지 않는다. File4 raw archive 중 선택 모델 예시의 `prediction.npz` 126,874,539 bytes가 실제로 읽히는 것을 확인했다. 전체 2,540개와 RGB1 508개의 정본 존재·SHA closure를 목록화한 뒤 export한다. 로컬 raw 복사본은 완료 후 폐기된 사례가 있으므로 로컬 `render-geometry.npz`만으로 PBR을 복원한다고 가정하지 않는다.
3. **기존 외부 baseline은 먼저 생성 재실행하지 않는다.** 노드16에서 TRELLIS.2, 공식 Pixal3D, ReconViaGen, ShapeUp, SpaceControl 다섯 방법은 508/508 평가 완료다. MVAdapter 504/508과 CorrAdapter 502/508은 성공률과 조건부 평균을 별도 표시하고 FID/KID를 빈칸으로 둔다. GeoStyle full TEST는 취소 상태여서 완성 행처럼 쓰지 않는다.
4. **Target8을 추가할 때는 모든 방법을 재렌더.** 외부법과 ours 모두 저장 mesh에서 camera8의 같은 GT와 동일 Cycles 설정으로 PSNR/SSIM/M-IoU, 필요하면 perceptual metric을 측정한다. 기존 노드16의 9/10 PSNR을 camera8 PSNR 열에 복사하지 않는다. 새 protocol/hash로 보존한다.
5. **공통 미관측 뷰를 추가한다면 모든 방법에 적용.** GT M1 mesh/material이 있으므로 새 입력 생성이나 학습 없이 렌더링이 가능하나, 카메라 orbit 정의와 좌표 일치, 가려진 물체 비율을 VAL에서 먼저 검사한다. TEST 결과를 보고 유리한 시점을 선택하지 않는다.
6. **내부 모듈 ablation을 같은 주석 체계로 연결.** 기존 DEV128 SS arm은 동일 표면 sampling의 CD/F-Score로, DEV32 PBR arm은 동일 generated geometry의 Cycles appearance로 재채점한다. Raw output이 없으면 해당 arm의 *추론/렌더만* 다시 실행한다. 현재의 matched training과 frozen TEST를 재학습하지 않는다. R→Selected의 TEST 차이는 기존 다중비교 보정에서 확정적이지 않았으므로, 새 공통 렌더 결과만으로 matching-aware attention의 독립 효과를 확정하지 않는다.

## 주장과 통계의 경계

- `selected vs V1`: full pipeline의 end-to-end 효용. 동일 evaluator의 paired source-cluster CI와 508 성공 분모를 제시한다.
- `Top-1 R vs V1`: retargeting 경로. 같은 입력·generation budget·common scorer에서 비교한다.
- `Selected vs Top-1 R`: weighted Top-K SS 경로의 **증분**. 유리한 점추정과 유의한 개선을 구별한다.
- Spatial→semantic→decoder와 ray-wise attention은 기존 DEV의 matched SS geometry 및 retrieval 진단으로 설명한다. 외부 baseline 격차로 개별 모듈의 인과성을 주장하지 않는다.
- 외부 baseline은 방법별 입력 권한과 해상도가 다르므로 task-level 비교다. 카메라 메타데이터가 필요한 ours와 MoGe를 쓰는 공식 Pixal3D의 생성 스택 차이를 명시한다. 필요하면 공식 Pixal3D를 제공 camera로 실행한 별도 **camera-controlled** bridge를 추가하되, 508개 재생성과 별도 라벨이 필요한 추가 실험으로 취급한다.
- 기존 TEST508 결과를 이미 본 상태다. 새로운 camera8/미관측 뷰를 TEST 점수에 따라 선택하거나 통계 family를 좁히면 confirmatory claim이 안 된다. 프로토콜과 primary endpoint는 VAL/과제 정의만으로 동결한다. 재평가의 새 결과는 명시적으로 protocol-revision 검증으로 보고하고, 강한 새 선택 주장은 독립 TEST에서 확인한다.

## 우선순위

**필수:** export/축·재질 QA → ours 6행을 노드16 9/10으로 채점 → 내부/외부 표를 공통 evaluator로 작성 → camera8을 전 방법에서 채점. **권장:** 고정 두 미관측 뷰와 foreground 보조 지표. **조건부:** 공식 Pixal3D camera-controlled 생성, DEV arm의 부족한 추론 재생성. 현재 근거로는 새로운 SS/PBR 학습부터 다시 할 필요가 없다. 가장 큰 실행 위험은 모델 성능보다 raw PBR material의 GLB export/좌표 규약이며, 이것을 VAL smoke에서 먼저 끝내야 한다.
