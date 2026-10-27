# TEST508 공정 비교 및 논문 표 완성 실행 계획

> **2026-10-28 사용자 결정으로 주 평가 계획 변경:** 카메라 제공을 주 조건으로 사용하며 camera-free 2×2 전체 실험은 필수가 아니다. 또한 이 문서의 `source-defined mesh_scale` 기술은 잘못됐다. 실제 TEST manifest의 사례별 scale은 target M1 geometry에서 계산된다. 수정된 계획과 필수 scale 입력 감사는 [카메라 제공 주 평가 계획](2026-10-28-camera-given-main-and-scale-audit-ko.md)을 따른다. 아래 내용은 이전 2×2 계획의 실행 이력으로 보존한다.

## 2026-10-28 00:24 KST 실측 진행·예상 시간

- 기존 여섯 Ours 예측은 생성 완료. 노드16 공통 evaluator용 GLB 변환은 384/3,048, Ours Cycles camera8/9/10 렌더는 1,149/9,144, 외부 baseline 및 GT의 camera8 추가 렌더는 4,054/4,054 완료. 기존 외부 일곱 방법의 공통 성공 502개 source/geometry/FID/KID와 target8 점수는 확보됐다. 아직 Ours 공통 502개 전체 점수는 없다.
- 최근 GLB 변환 완료량은 약 31개/15분, 191개/60분이다. 남은 약 2,664개를 같은 속도로 처리하면 약 14~22시간이며, 렌더는 변환 직후 따라오는 상태다. 점수 집계·502개 동일 분모 감사에 추가 2~4시간을 잡아 **기존 13행 진단표 마감은 약 16~26시간**으로 추정한다. CPU 부하, file4 I/O 및 난이도별 mesh 복잡도에 따라 연장될 수 있다.
- 노드16 rank2는 index380의 렌더용 mesh decimation으로 원래 경계와 GLB 경계가 0.0144만큼 달라 임계값 0.01에서 멈췄다. 검사에서 decimation 차이와 GLB roundtrip 오류를 분리했다. 동일 사례 재실행의 원래 경계 차이는 0.014387, GLB roundtrip 경계 오류는 0이며 성공 receipt가 생성됐다. rank2를 재개했고, geometry metric은 여전히 원본 raw surface로 계산한다.
- 공식 Pixal3D 1024 카메라 추정/제공은 각각 한 사례 feasibility만 완료. TEST508의 두 행은 아직 batch runner·508쌍 생성·공통 렌더/평가가 남았다. Ours 카메라 미제공 경로는 상대 pose/FOV/scale 추정과 stage cache 재구축이 아직 구현되지 않아 전체 종료일의 주 불확실성이다.
- 공식 Pixal3D 1024 한 사례 생성은 115~126초였다. 1,016개 생성의 단순 GPU 점유 하한은 120초×1,016/4 GPU≈8.5시간, 2 GPU≈17시간이다. 실제 배치 실행, 변환·렌더·실패 재시도 포함 **1~2일**로 잡는다. Ours 카메라 미제공 경로 구현·VAL 검증·508개 생성 및 평가에는 **2~4일**의 조건부 예산을 둔다. 카메라 분포 변화로 재학습이 필요하면 추가 기간을 새 실측으로 산정한다. 표·통계·원고/PDF 최종 교체는 모든 공통 점수 이후 약 **0.5~1일**이다. 이 기간들은 병렬 실행 가능하므로 단순 합산하지 않는다.

2026-10-28. 목적은 **같은 Ours Full 결과를 베이스라인 비교표와 ablation 표에 사용**하면서 카메라 입력, Pixal3D cascade 해상도, 평가 모집단의 차이를 분리하는 것이다. 현재 13행 공통 재평가는 계속하지만 입력이 다른 방법을 나열한 진단표이며 아직 논문 주 베이스라인 표가 아니다.

## 확정 조건과 현재 증거

- 데이터: DeformNeXt TEST508. 모든 방법에 같은 edited M1 RGB@8과 허용된 source 입력을 사용한다. 성공 수의 분모는 508이다.
- 평가: 같은 node16 source-defined world, Blender/Cycles 512px, camera9/10 source 지표, camera8 target 지표, raw surface 20,000점 CD/F@0.02. 기존 7개 외부법은 native 9/10 평가가 완료됐고, camera8 추가 렌더 4,054/4,054가 완료됐다. 기존 내부 6 arm의 공통 evaluator 변환·렌더·채점은 진행 중이다. [중간 표](2026-10-27-unified-test508-live.md)는 진단용이다.
- 카메라 권한: 노드16 외부 runner에는 데이터셋의 정확한 camera-to-world/FOV/scale 묶음을 전달하지 않았다. Pixal3D는 MoGe를 사용한다. ReconViaGen은 M0 source RGB3, ShapeUp은 M0 source mesh, SpaceControl은 source anchor/proxy를 별도로 받으므로 전체 외부법이 같은 RGB1 입력은 아니다.
- [카메라 감사](2026-10-27-camera-input-audit.json): TEST508의 4-view 촬영 rig/FOV60도/target 거리1.8은 고정, 사례별 source-defined scale은 0.645–1.240이다. 공식 Pixal3D의 추정 FOV 중앙값은 35.53도이고 scale은 1.0이다. 이는 입력 차이의 증거이며 성능 차이의 인과 분해 결과는 아니다.
- 공식 Pixal3D 1024의 MoGe/제공 카메라 한 사례 실제 생성이 모두 통과했다. [MoGe receipt](2026-10-28-pixal3d-1024-predicted-pilot.json), [제공 카메라 receipt](2026-10-28-pixal3d-1024-given-pilot.json). 1024 전체 점수는 아직 없다. 기존 official 1536 한 사례에서는 제공 카메라가 CD를 개선하지 않았으므로 카메라 효과 방향을 가정하지 않는다.

## 논문 주 카메라·해상도 대조

| 카메라 조건 | 공식 Pixal3D | Ours Selected |
|---|---|---|
| 이미지에서 추정 | **P−**: T1, 원본 checkpoint/native 1024, MoGe camera, scale1.0. 한 사례 생성 완료; TEST508 미실행 | **O−**: T1+S3, 동일 Ours 1024 checkpoint. RGB에서 상대 pose/FOV/scale을 얻는 입력 경로와 새 cache가 필요. 미구현 |
| 제공 | **P+**: T1, 동일 Pixal3D checkpoint/native 1024, source-defined camera/scale. 한 사례 생성 완료; TEST508 미실행 | **O+**: T1+S3, 현재 Ours Selected 1024의 저장 예측. 생성 완료, 공통 evaluator 진행 중 |

P−/P+는 원본 Pixal3D 가중치, 동일 target RGB 파일, 공식 전처리, sampler, seed0, `1024_cascade`, max tokens32768을 고정하고 camera_params만 다르게 한다. O−/O+의 **순수 카메라 sensitivity 대조**도 가중치, source/target RGB, 샘플링, 1024 cascade를 고정한다. O−는 TEST 성능을 보고 post-hoc으로 카메라 방법을 바꾸지 않는다. 공식 Pixal3D의 기존 **1536 native default**는 참고 행/보조 표에 남기고 1024 대조 수치와 섞지 않는다.

`1024`는 Pixal3D/TRELLIS 계열의 voxel cascade 제어다. TRELLIS.2, ReconViaGen 및 MVAdapter/CorrAdapter의 공통 ReconViaGen 복원기는 이미 1024 cascade다. ShapeUp·SpaceControl의 mesh를 Pixal3D의 1024와 동등하게 강제할 설정은 없다. 다른 외부법을 전부 재생성하지 않는다. 최종 triangle 수는 모델 출력으로 기록하고 raw geometry를 임의 decimation하지 않는다.

## 실행 순서와 통과 조건

1. **현재 공통 평가 닫기.** 기존 6 Ours arm의 3,048 GLB와 9,144 Cycles 렌더, 공통 source/target score를 끝낸다. 이 결과는 O+와 내부 C+ ablation, evaluator parity의 정본이다. 기존 native-renderer 점수를 node16 점수와 섞지 않는다.
2. **Pixal3D P−/P+ full508.** 한 사례 1024 생성이 통과했으므로, 먼저 VAL에서 raw mesh의 공식 축/scale, PBR volume, GLB material, camera8/9 렌더를 검사한다. 그다음 체크포인트를 한 번 로드하는 batch runner로 508쌍을 생성한다. 독립 sample 성공/실패 receipt, input/output SHA, camera_params, 해상도, wall time을 남긴다. 두 카메라 arm에는 동일 postprocess 및 evaluator를 쓴다.
3. **O− 카메라 추정 경로.** 먼저 TRAIN/VAL에서 RGB-only 추정기(VGGT-Omega camera/depth 또는 사전 고정된 다른 estimator)의 상대 4-view pose, FOV, metric scale, 좌표계 정합을 검증한다. 고정 rig prior를 TRAIN에서 얻어 쓰는 경로는 완전한 camera-free 경로와 별도 이름으로 보고한다. TEST의 저장 camera/scale, GT depth, GT target mesh, `spatial_indices`를 O− 생성·cache에 넣지 않는다. 추정 camera로 SS ray query와 matching, Shape/PBR projection·cache를 다시 만들어 32개 이상 VAL 생성에서 finite/좌표/렌더 parity를 확인한다. 분포 변화가 크면 V1/Retarget/Selected를 모두 같은 추정 camera 조건으로 재평가한다. 카메라 추정 조건으로 **재학습**한 변형이 필요하다면 별도 모델 행으로 보고하고, 동일 checkpoint의 순수 O−↔O+ 카메라 효과라고 부르지 않는다. O− Full 한 행만 바꾸어 기존 C+ ablation에 연결하지 않는다.
4. **공통 TEST 평가.** 새 P−/P+/O− mesh/PBR을 기존 node16 evaluator로 camera8/9/10에 렌더한다. Pixal3D와 Ours 직접 대조는 가능한 동일한 texture baking, remesh, render용 face cap으로 출력하고, 그 변환의 geometry/material parity를 검사한다. CD/F는 후처리 전 raw surface로 계산한다. 나머지 외부법은 native 결과를 유지한다.
5. **평가 모집단 다시 동결.** 기존 공통 성공 502개에 P−/P+/O−가 모두 성공하면 그 502개를 유지한다. 새 실패가 있으면 **점수를 보기 전에** 모든 주 표 방법의 성공 교집합을 계산하고 모든 10개 지표, 특히 FID/KID의 참조/예측 이미지를 그 동일 집합으로 다시 집계한다. 각 행의 `n/508` 성공 수와 제외 ID를 함께 공개한다. 부가로 성공 가능한 전체 행 평균을 별도 보고하되 주 표와 혼합하지 않는다.
6. **통계와 논문.** 주요 geometry CD/F@0.02, target8 M-IoU, appearance source9/10 LPIPS/DreamSim을 paired source-cluster 간격과 함께 보고한다. PSNR/SSIM/CLIP/FID/KID도 동일 모집단에 넣는다. Source9/10은 Ours의 입력 뷰이므로 해당 appearance 점수를 미관측 novel-view 성능이라고 부르지 않는다. `P−↔P+`와 **동일 checkpoint** `O−↔O+`가 각 방법의 camera sensitivity, 같은 카메라 조건의 `V1↔Retarget↔Selected`가 모듈 효과다. `P↔O` 차이는 source RGB3 추가 및 모델 구조까지 포함하는 system 비교로만 설명한다. Main baseline 표에는 입력 열과 camera 조건을 명시하고, ablation에는 해당 Ours Selected의 동일 예측/숫자를 그대로 사용한다. 대표 qualitative panel도 두 카메라 조건에서 같은 sample ID로 고정한다.

## 완료 기준

- P−/P+/O−/O+가 실제 1024 예측·공통 renderer 점수·성공 receipt를 모두 가진다.
- 주 비교표와 ablation 표의 Ours Selected 행이 같은 파일 해시와 점수 JSON에서 생성된다.
- 카메라 권한, source RGB/mesh 입력 권한, 1536 참고 결과, native mesh face/texture/export 설정이 caption 또는 보조 표에서 구분된다.
- O−가 검증되지 않으면 카메라 미제공에서 Ours 우위 주장은 쓰지 않는다. 현재 C+ 통합 표는 진단/내부 ablation으로만 남긴다.
