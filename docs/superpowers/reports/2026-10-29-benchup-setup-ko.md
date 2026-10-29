# BenchUp 벤치마크 준비 기록

## 대상과 출처

- [ShapeUP 논문](https://arxiv.org/html/2602.05676)은 BenchUp을 24개 source mesh와 100개 edit condition으로 기술한다. [공개 데이터셋](https://huggingface.co/datasets/Inbar2344/BenchUp)의 revision `a8b87439281c7f64788231f0935b6d6aedc3ba85`에는 23개 source mesh, 편집 PNG 109개, source multi-view WEBP 598개, source point cloud 23개가 있다. 논문 100개와 공개본 109개의 일치 여부는 확인되지 않았으므로 공개본 109 전체를 대상으로 별도 기록하고 논문 수치를 그대로 같은 모집단 비교에 넣지 않는다.
- 저장: 파일서버4 `/file4/youngwoo/deformation/experiments/retargeting-v1/benchup/raw-a8b8743`. 753개 파일 모두 HF LFS SHA-256을 검증했고 `download-manifest.json`을 생성했다. 검증된 109개 sample JSONL은 `/file4/youngwoo/deformation/experiments/retargeting-v1/benchup/index/benchup-a8b8743.jsonl`이며 SHA-256은 `ec0bbb345f945e6a19e26157307283f81beafea06331d913d6a536ba8a401709`이다.
- 공개본에 edited target GT mesh 및 명시적인 camera extrinsic/intrinsic 파일은 발견되지 않았다. `pc.npz`는 source point data다. 파일명 같은 source/target view index가 동일 카메라라는 가정은 검증 필요하다.
- 기존 Pixal3D 입력 경로에 맞춰 source WEBP 598개를 RGBA PNG로 변환해 file4 `benchup/source-png-a8b8743`에 저장했다. 원본 및 PNG 해시를 전부 확인했다. 109개 편집 모두 같은 index의 source 이미지가 존재하지만, 이것만으로 두 카메라가 같다고 결론 내릴 수 없다.
- 공개 VGGT-Omega 기본 checkpoint로 backhoe source view 4장에 대한 camera/depth 추정 smoke test를 실행했고 finite 출력이 생성됐다. 결과는 file4 `benchup/camera-diagnostics-v1/backhoe.npz`와 JSON receipt에 있다. 이는 우리 epoch-7 모델의 검증이 아니며, 아직 mesh 좌표 정합과 reprojection/occlusion 확인이 되지 않은 진단 결과다.

## 평가 조건

`experiments/benchup/protocol.json`은 직전 GEAR3D Table 1을 바탕으로 하는 **초안**이다. 11개 행을 보존한다: MV-Pixal3D zero-shot, Pixal3D target 1-view, MV-Pixal3D FT, GEAR3D, TRELLIS.2, Pixal3D native, ReconViaGen, MVAdapter, CorrAdapter, ShapeUP, SpaceControl. 기존 checkpoint와 모델별 native 입력을 재사용하고 1024/1536 cascade를 그대로 구분한다. Ours와 MV-Pixal3D에는 target RGB 1장과 동일 source RGB 3장을 준다. 사용자 지정에 따라 node16 실행은 하지 않는다.

직전 실험처럼 raw와 input source mesh M0만 사용한 Sim(3) 정합 후 결과를 모두 유지한다. 다만 BenchUp에는 정확한 카메라 행렬이 공개되지 않았으므로 source 이미지와 mesh로 카메라 추정·검증이 먼저 필요하다. 추정/정합 카메라를 GT 또는 camera-given으로 부르지 않는다. 파일명만으로 세 source view를 확정하지 않는다. 편집 이미지의 시점이 source와 동일한지도 검증한다. 모든 생성법과 렌더 평가는 이 절차에서 고정된 동일 데이터와 렌더 카메라를 사용하되 각 모델이 받는 입력은 표에 명시한다.

Target RGB에 대한 PSNR/SSIM/M-IoU와 LPIPS/CLIP/DINO, source view에 대한 LPIPS/DreamSim/CLIP/DINO, source render pool에 대한 FID/KID를 우선 대상으로 한다. source reference는 원본 geometry이므로 의도된 편집까지 불일치로 계산할 수 있다. 흰 배경/alpha, crop, mask 생성, front/back 평가 뷰, source-only registration의 변형 영향을 고정·검증해야 한다. ShapeUP의 occluded-region metric은 논문에서 정의하나 공개 evaluator 및 visibility/occlusion mask 프로토콜 일치가 확인되기 전에는 논문 수치와 직접 비교하지 않는다. 실제 target mesh 없는 상태에서 CD/F-Score는 산출하지 않는다.

각 방법의 성공 수와 전체 109 기준 coverage를 보고, 동일 성공 교집합에 대한 paired metric도 별도 보고한다. BenchUp 23개 source 단위로 통계 불확실성을 계산한다. 테스트 성능으로 checkpoint, view 또는 seed를 고르지 않는다.

## 재현 순서와 미완 사항

1. 공개 파일의 revision 및 SHA-256 검증, sample manifest 작성.
2. source mesh 렌더와 source 이미지의 상대 camera pose/FOV 추정 후 색상·실루엣·reprojection 검증. source/target 같은 index의 view 일치 검증. 실패를 숨기지 않고 기록.
3. 고정된 source 3장과 카메라, target image, source mesh를 이용해 이전 Table 1의 11개 arm 파일/환경/체크포인트를 검증하고 각 arm을 소규모 smoke test한다. 외부법 runner는 node16 실행을 중단한 상태이므로 로컬에 재구성·의존성 검증이 필요하다.
4. 모든 방법을 같은 109건에 실행하고 raw 및 source-M0 정합 양쪽으로 렌더·평가한다. baseline별 결측과 전 방법 공통집합을 동시에 기록한다.
5. 논문 주표에는 데이터셋·카메라·GT mesh 유무와 입력 차이를 기재하고, ShapeUP 원논문의 100건 수치와 측정 프로토콜이 다르면 별도 출처로만 제시한다.

현재 상태: 원본·sample 인덱스 및 source PNG 준비 완료, 1개 object의 카메라 추정 smoke test 완료. 카메라와 mesh의 정합 검증, 전체 runner 연결, 각 모델 smoke test 및 본평가는 아직 완료되지 않았다.
