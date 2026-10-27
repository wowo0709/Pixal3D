# 노드16 공통 평가 기반 통합 Table 1 실행 기록

2026-10-27 현재. **새 공통 점수는 아직 생성 중이며, 아래는 실행·검증 현황이다.** 기존 숫자를 다른 프로토콜의 빈 칸에 복사하지 않는다. 사용자 요청에 따른 `2026-10-27-unified-test508-table-amendment-common502.json`이 아래 평가 모집단 규칙을 수정했다.

카메라·1024 해상도 공정성을 반영한 후속 실행 순서는 [2026-10-28 계획](2026-10-28-fair-camera-resolution-benchmark-plan-ko.md)을 따른다. 이 문서 아래의 현재 13행 표는 진단용이며 주 베이스라인 우위표가 아니다.

## 확정된 표

`2026-10-27-unified-test508-table-contract.json`을 숫자 재채점 전에 고정했다. TEST508의 외부 baseline 7행(TRELLIS.2, 공식 Pixal3D, ReconViaGen, MVAdapter, CorrAdapter, ShapeUp, SpaceControl)과 내부 6행(calibrated Pixal3D RGB1, source-only RGB4, edited RGB4, V1, Top-1 R, Selected)을 하나의 표에 배치한다. GeoStyle full TEST는 취소됐으므로 완성된 행처럼 넣지 않는다. DEV128 SS와 DEV32 PBR 모듈 arm은 TEST508과 표본·stage가 다르며 이번 통합 표에 넣지 않는다.

**통합 원칙 정정:** 기존 내부 native-renderer 점수를 이 표에 가져오지 않는다. 여섯 내부 행의 저장된 예측을 노드16 형식으로 변환해 외부 일곱 행과 동일한 평가기로 다시 렌더·채점 중이다. 이 재평가는 평가 조건을 맞추는 작업이며 생성 경로를 같게 만드는 작업은 아니다. 공식 Pixal3D는 RGB에서 MoGe로 카메라를 추정하고 native 1536 cascade를 사용한다. 내부 보정 RGB1은 원본 single-view 가중치를 제공된 source-defined 카메라(`C`)와 projective 1024 cascade에서 사용한다. 따라서 이 둘은 별도 행이고, 후자를 공식 Pixal3D baseline의 재현 수치로 해석하지 않는다. 모듈 효과는 같은 입력·카메라·생성 스택의 V1/Top-1 R/Selected끼리 판단한다.

- **9/10 source-view M1 fidelity:** LPIPS, DreamSim, CLIP. 9/10은 ours의 입력 source 뷰이며 target-only baseline에는 미관측 뷰임을 입력 열에 표시한다.
- **9/10 global distribution:** FID, KID. 모든 방법이 성공한 502개 편집의 같은 1,004 이미지 풀, subset100×100, seed20261020. 캐시된 공식 Inception feature를 공통 ID 순서로 잘라 전 방법에서 다시 계산한다.
- **8 target edit:** PSNR, SSIM, 2D mask IoU. 모든 방법의 저장 출력과 original-material M1 GT를 같은 노드16 Blender 4.2.3 Cycles 설정에서 추가 렌더한다.
- **World geometry:** raw predicted surface의 20,000점 CD, F@0.02. 렌더용 단순화 GLB를 geometry 지표에 쓰지 않는다.

표에는 입력 T1/T1+S3/S4/T1+A1과 제공 카메라 메타데이터 사용을 뜻하는 `C`, 성공 사례 수 `n/508`, 10개 지표를 표시한다. **모든 지표 열은 공통 성공 502개 편집·133 source cluster에서 계산**한다. MVAdapter(504)와 CorrAdapter(502)의 누락 사례도 성공 수에서 드러난다. 평균은 deformation weighting이며 source-cluster CI는 본문/보조 표에 둔다. 새 평가를 본 뒤 지표나 뷰를 바꾸지 않는다.

논문의 주장 우선순위는 geometry에서 CD/F@0.02와 target8 M-IoU, appearance에서 source9/10 LPIPS/DreamSim이다. CLIP은 의미적 일치의 보조 지표, PSNR/SSIM은 camera8의 저수준 영상 일치, FID/KID는 상관된 1,004장 풀의 기술적 분포 요약으로 읽는다. 서로 다른 입력 권한을 가진 외부법의 task-level 격차와, 같은 입력·backbone·예산을 맞춘 V1/R/Selected 증분은 별도 문장으로 설명한다.

아래는 **공통 성공 502개 사례**의 외부법 9/10 + world geometry 결과다. 새 target8 지표와 ours 공통 렌더 지표를 채우기 전의 부분 결과이며, 이것만으로 통합 방법 순위를 결론짓지 않는다. 이전 방법별 전체 성공 사례 점수는 노드16 원본 summary에 보존한다.

| 외부법 | 성공 | LPIPS↓ | DreamSim↓ | CLIP↑ | FID↓ | KID↓ | CD↓ | F@0.02↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRELLIS.2 | 508/508 | .0942 | .2625 | .9028 | 61.21 | .003015 | .08827 | .2969 |
| Pixal3D official | 508/508 | .1064 | .2671 | .9070 | 68.39 | .005995 | .10279 | .1535 |
| ReconViaGen | 508/508 | .0810 | .1911 | .9232 | 45.92 | .000554 | .06183 | .3891 |
| MVAdapter+ReconViaGen | 504/508 | .1036 | .2881 | .8905 | 66.56 | .004588 | .08095 | .2359 |
| CorrAdapter+ReconViaGen | 502/508 | .1030 | .2868 | .8914 | 65.68 | .004493 | .08278 | .2317 |
| ShapeUp | 508/508 | .0854 | .3021 | .8826 | 74.37 | .007286 | .07781 | .4577 |
| TRELLIS.2+SpaceControl | 508/508 | .0969 | .2898 | .8975 | 64.78 | .003508 | .08691 | .2955 |

### ReconViaGen KID 확인

ReconViaGen의 공통 502개 KID **0.000554**는 빈칸 대체나 subset 계산 오류로 보이지 않는다. 외부 7개 방법의 feature cache는 같은 protocol/extractor를 사용하고, 502개×camera9/10의 GT Inception feature는 method 간 bitwise 동일했다. ReconViaGen의 전체 508개 원래 KID도 약 **0.00049**로 비슷하며, 같은 공통 502개에서 seed 20261020–20261024로 KID subset만 바꾸면 **0.000554, 0.000307, 0.000500, 0.000457, 0.000566**이다. 비교용 TRELLIS.2는 같은 seed에서 **0.002879–0.003050**, CorrAdapter는 **0.004316–0.004493**이다. Camera9/10을 각각 따로 계산해도 ReconViaGen은 **0.000464/0.000841**, TRELLIS.2는 **0.003413/0.003431**, CorrAdapter는 **0.004766/0.004525**로 두 뷰 모두 낮다. 이는 RNG 한 번이나 특정 뷰 한쪽만의 우연으로 격차가 생긴다는 해석을 약화하지만 통계적 유의성 검정은 아니다.

노드16 `run-reconviagen.py`는 직접 ReconViaGen에 `[M1@8, M0@9, M0@10, M0@11]` 네 실제 이미지를 넣는다. 평가 camera9/10은 그 입력 source 뷰와 겹친다. MVAdapter·CorrAdapter 경로는 source 원본 3장을 추가하지 않고 target에서 만든 멀티뷰를 ReconViaGen 복원기에 넣는다. 따라서 ReconViaGen의 낮은 KID는 source 뷰 입력 권한의 도움을 받았을 가능성이 크다. 같은 모집단에서 FID 45.92, LPIPS .0810, DreamSim .1911, CLIP .9232도 외부법 중 가장 좋으므로 KID만 동떨어진 결과는 아니다. KID는 전체 이미지 feature 분포 거리이며 target edit 보존이나 방법 고유 모듈의 인과 효과를 직접 증명하지 않는다.

## 실제 검증

- 기존 최종 5개 pipeline의 277개 재사용 + 231개 신규 그룹과 calibrated RGB1을 결합해 **6×508=3,048** 고정 행을 생성했다. 외부 TEST508 sample ID와 일치하며, 노드16에서 prediction/render geometry/anchor/generation 네 파일이 전부 존재함을 확인했다.
- file4 raw prediction을 로컬 디스크로 대량 복사하지 않는다. 노드16 `/file4` 마운트에서 읽는다.
- Selected 한 사례의 GLB export에서 native geometry와 노드16 reader의 world bounds 오차 **0**, raw surface와 render mesh의 역할 분리 확인. 같은 GLB의 Cycles 렌더를 GT 및 기존 native prediction과 육안 대조했다.
- 4096 vs 2048 texture export의 같은 사례·카메라 Cycles RGBA 평균 절대 차이는 **0.000096**, mask 차이는 **0 pixel**. 2048을 export 설정으로 고정했다. 이는 export fidelity 검사이며 모델 품질 비교가 아니다.
- RETARGET 한 사례의 노드16 numeric evaluator는 surface·카메라·receipt를 통과했고, 공식 LPIPS/DreamSim/CLIP/Inception 고정 가중치의 full evaluator도 실패 0으로 통과했다.
- GT camera8과 외부 baseline camera8의 Cycles render smoke가 통과했다. camera8 추가 job은 GT 508 + 외부 baseline 성공 3,546 = **4,054**개다.
- 처음 표 composer를 TEST 파일과 분리된 임시 합성 점수로 검사했다. 이후 사용자 요청으로 공통 502개 방식을 채택해 13개 행×13개 열의 새 composer로 수정했다. 임시 합성 점수는 실제 성능 결과가 아니다.
- 외부 7개 방법의 공식 feature cache 교집합을 검사해 정확히 **502개 편집·1,004 이미지·133 source cluster**를 동결했다. 노드16 공통 subset 파일은 `common-success-502.json`이며, 7개 방법의 참조 Inception feature는 공통 ID에서 bitwise 동일하다. 이 subset으로 외부 전 방법의 FID/KID 재계산을 완료해 `common-source-external-502.json`에 보존했다.

공통 502개 기반 외부 FID/KID 원 단위 결과는 TRELLIS.2 61.206/.003015, Pixal3D official 68.386/.005995, ReconViaGen 45.920/.000554, **MVAdapter 66.559/.004588**, **CorrAdapter 65.682/.004493**, ShapeUp 74.368/.007286, SpaceControl 64.776/.003508이다. 위 표와 최종 통합 표에서는 전 행에 공통 502개 결과만 쓴다.
- camera8의 실제 Cycles 출력 한 쌍으로 노드16의 `image_metrics` 호출을 검증했다(PSNR/SSIM/M-IoU 계산 성공). 이는 전체 TEST 평균이 아니다.

## 실행 위치와 완료 조건

노드16 작업 디렉터리: `/home/youngwoo/retargeting-unified-test508`. 다른 baseline 출력은 변경하지 않는다.

2026-10-27 13:30 UTC의 3-worker 초기 외삽은 약 35시간이었다. 14시경 로컬 GPU 0/1/2/3/6/7을 사용하는 export worker 6개를 추가하고 노드16 GPU 4/5/7 worker 3개를 서로 다른 9-way shard로 재배치했다. 로컬 pilot GLB를 노드16 Cycles camera9에서 렌더한 결과 기존 node16 GLB 대비 RGB MAE **0.0000254**, alpha mask 차이 **0 pixel**이었다. 로컬 파일은 file4에서 사례별 스트리밍해 변환 후 지우고, 노드16에 GLB와 원본 파일 해시가 있는 receipt를 검증 후 게시한다. 현재 작업은 GPU보다 CPU 변환이 병목이어서 GPU util은 낮아도 각 GPU에 worker를 배정해 총 처리량을 높인다. 14:05 UTC까지 export 141/3048; 초기 9-worker 처리량을 단순 외삽하면 export에 약 10~12시간, 전체 scorer·표 검증·리포트까지 **10월 28일 저녁~10월 29일 새벽 KST**가 예상되지만 후반 사례 복잡도에 따라 달라질 수 있다. GT·외부 target8 렌더는 먼저 끝날 것으로 보여 별도 watcher가 공통 502개 PSNR/SSIM/M-IoU를 계산해 중간 표에 반영한다.

14:12 UTC에 두 호스트 모두 CPU 실행 대기열이 포화된 것을 확인했다(로컬 load 131/96 logical CPU, 노드16 load 195/96). 메모리 available은 각각 약 400GiB, 405GiB이고 순간 swap-in/out은 거의 없었다. 로컬 export 6개 중 4개 process group은 출력/임시 파일을 유지한 채 `SIGSTOP`으로 안전하게 일시정지했고, `control_local_export_cpu.py`가 load를 60초마다 보고 최소 2개가 계속 진행되도록 하고 부하가 떨어지면 일시정지한 shard를 순차 재개한다. 노드16의 우리 exporter/renderer 프로세스 nice를 +10으로 낮춰 다른 작업을 우선시했다. **앞의 9-worker ETA는 이 CPU 보호 조치 이후 더 이상 유효하지 않으며**, 다음 처리량 구간을 보고 다시 추정한다.

- `export-manifest-v2.jsonl`: 3,048개 고정 예측. 노드16 `worker-nine-rank{0,1,2}.log`는 GPU 4/5/7, 로컬 `/tmp/retargeting-common-local-rank{3..8}.log`는 GPU 0/1/2/3/6/7의 서로 다른 9-way shard export다. 기존 `worker-rank{0,1,2}.log`는 3-way 작업의 과거 로그이며 작업자는 중단했다.
- `ours-render-rank{0,1,2,3}.log`: 기존 예측을 camera8/9/10에서 Cycles로 렌더.
- `target8-render-rank{0,1,2,3}.log`: 노드16 외부 baseline과 공통 GT의 target8 추가 렌더.
- `completion-progress.json`: 예상 export 3,048 / ours render 9,144 / target8 GT+외부 render 4,054 대비 현황.
- 로컬 `2026-10-27-unified-test508-live.md`는 120초 간격으로 노드16 완료 산출물만 읽어 갱신한다. 현재 공통 502개 외부 source/geometry/FID/KID 수치가 입력됐고, GT·외부 target8 전체 렌더가 닫히면 해당 세 지표가 다음으로 채워진다. Ours는 전 502개 사례의 공통 평가가 완료되기 전까지 점수 칸을 비워 잠정 순위를 만들지 않는다. 완료 시 같은 watcher가 `2026-10-27-unified-test508-table.tex`와 수치 중심 `2026-10-27-unified-test508-results-ko.md`를 로컬 reports에 저장한다. 논문 원고의 주장과 통계적 유의성은 그 뒤 별도 확인이 필요하다.
- 자동 완료 시 `source910-ours-evaluation/summary.json`, `common-source-all13-502.json`, `target8-scores.json`, **`unified-test508-table.tex`**, `unified-completion.json`을 생성한다. 이 다섯 산출물이 모두 없으면 통합 수치 표는 완료 상태가 아니다.
- 기존 4개 ours Cycles worker에는 초기 실행 시 30시간 export 대기 한도가 있었다. 별도 supervisor 4개를 띄워 이 한도에만 걸릴 경우 완료 receipt를 재사용해 각 shard를 재개하도록 했다. 수정된 worker는 마지막 진전 시각부터 72시간을 센다. 진짜 렌더 오류는 자동으로 성공 처리하지 않고 supervisor 실패 기록을 남긴다.
- 같은 렌더러·같은 11개 초기 완료 사례(3 source cluster)의 13방법 짝지은 QA는 `2026-10-27-early-common-qa11-ko.md`에 따로 보존했다. 전체 TEST나 공통 502개 통합 수치의 대용으로 쓰지 않는다.

구현 스크립트는 `experiments/deformnext_retargeting/`의 `build_common_benchmark_manifest.py`, `export_common_benchmark_mesh.py`, `run_common_benchmark_export_worker.py`, `run_local_common_export_worker.py`, `build_common_target8_jobs.py`, `run_common_benchmark_ours_render_worker.py`, `bind_common_benchmark_ours.py`, `score_common_success_source.py`, `score_common_target8.py`, `score_early_paired_qa.py`, `compose_unified_test508_table.py`, `complete_unified_test508_evaluation.py`, `update_unified_live_table.py`다. 각 산출물은 기존 정본과 별도 디렉터리에 쓰며, 완료 후 원고 주 표를 새 수치로 교체하기 전에 성공 분모와 source-cluster 비교를 재검토한다.

## 논문 표 분리와 동일 Ours 결과 재사용

13행 `unified-test508-table.tex`은 모든 외부·내부 방법을 대조하는 **감사용 원표**다. 완료 시 같은 공통 502개 점수 JSON에서 `heterogeneous-input-diagnostic-test508-table.tex`(외부 7법 + Selected 한 행)과 `ablation-test508-table.tex`(내부 6 arm)를 자동 생성한다. **전자는 카메라가 맞지 않는 진단표이며 논문 주 베이스라인 비교표로 사용하지 않는다.** 두 표의 Selected 행은 동일 checkpoint·동일 저장 예측·동일 평가 결과를 재사용한다. 테스트 점수에 맞춰 Selected를 다른 arm으로 바꾸지 않는다. 생성기는 `compose_paper_test508_tables.py`이며 각 표의 굵게/밑줄 순위는 해당 표 안에서 다시 매긴다.

현재 Ours 결과의 입력은 T1+S3+C, 생성은 projective 1024 cascade다. 공식 Pixal3D baseline은 T1 입력, MoGe 카메라 추정, native 1536 cascade다. 노드16의 외부 실행 스크립트를 감사하면 데이터셋의 정확한 camera-to-world/FOV/scale 묶음을 생성 입력으로 넘기는 방법은 없다. 일부는 자체 예측 또는 고정 생성 뷰를 사용하고 ShapeUp은 별도 source mesh를 받는다. 따라서 **제공 카메라 C와 추가 source RGB3이라는 입력 권한 차이가 있어 현재 외부 비교를 공정한 주 성능표로 해석하지 않는다.** 이는 evaluator parity와 카메라 민감도 진단 자료다. Matching/retargeting의 독립 효과는 같은 T1+S3+C를 쓰는 V1→Top-1 R→Selected와 대응 모듈 arm의 matched ablation에서 판단한다.

Ours Full 한 행만 추정 카메라·1536 cascade로 다시 생성해 비교표에 넣고 현재 1024+C ablation의 Full 행을 그대로 두는 방식은 채택하지 않는다. 그러면 두 표의 ‘Ours’가 달라져 모듈 기여를 연결할 수 없다. 카메라 미제공 RGB-only 조건을 별도 claim으로 내세우려면 모든 내부 비교 arm을 그 조건에서 다시 생성·검증하는 **별도 실험 프로토콜**이 필요하다. 이번 공통 평가의 score를 그 새 조건의 결과처럼 사용할 수 없다.

## 카메라 입력 공정성 수정: 주 비교표의 선행 조건

노드16 실제 생성 스크립트 확인 결과, 공식 Pixal3D는 M1 RGB 하나에서 MoGe 카메라를 추정한다. TRELLIS.2, ReconViaGen, MVAdapter/CorrAdapter, ShapeUp, SpaceControl에도 데이터셋의 정답 camera-to-world/FOV/scale 묶음이 생성 입력으로 제공되지 않는다. 다만 이들은 동일 입력 방법이 아니다. ReconViaGen은 M0 RGB3장을 추가로 보고, ShapeUp은 source M0 textured mesh를 받으며, SpaceControl은 TRELLIS.2 proxy와 M0 anchor RGB를 받는다. 따라서 ‘전 외부법이 같은 camera-ungiven RGB1’이라고 표현해도 안 된다.

이번 기존 Ours의 `C`는 target/source view의 저장된 camera-to-world, FOV, 거리 및 source-defined mesh scale이다. 이는 외부법의 MoGe 또는 자체 pose 추정보다 강한 메타데이터 권한이다. 현재 13행 공통 재평가는 평가기 일치와 카메라 민감도 진단에 계속 사용하지만, **카메라 입력이 맞는 2×2 검증 없이 이를 논문 주 베이스라인 우위표로 승격하지 않는다.**

TEST508 RGB manifest 508행을 추가 점검하면 FOV는 전부 60도, target camera 거리는 전부 1.8이고 4-view camera-to-world 행렬은 소수점 12자리에서 하나의 고정 rig다. 반면 source-defined `mesh_scale`은 사례별로 0.645–1.240이다. 즉 `C`에는 고정 촬영 rig뿐 아니라 **사례별 scale**이 포함된다. 노드16 공식 Pixal3D 생성 receipt **508개 전부**에서 MoGe 추정 FOV 중앙값은 35.53도(범위 18.67–58.73도), 생성 `mesh_scale`은 모두 1.0이다. 정본 수치는 [camera 입력 감사 JSON](2026-10-27-camera-input-audit.json)에 있다. 이는 공식/보정 RGB1 격차의 유력한 카메라·scale 원인을 드러내지만, 결과 격차 중 몇 %를 설명하는지는 Pixal3D native1536에서 camera_params만 바꾸는 대조 생성 전에는 단정하지 않는다.

| 생성 시 카메라 조건 | 공식 Pixal3D | Ours Selected |
|---|---|---|
| 제공 안 함 | MoGe + native **1024**, 새 생성 진행 중; 기존 1536만 완료 | RGB만으로 joint multi-view pose/FOV/scale 추정 후 projective 1024, 미구현·미평가 |
| 제공함 | 동일 공식 checkpoint/native **1024**에 저장 FOV·거리·scale을 입력하는 새 실행, 진행 중; 1536 한 사례 feasibility 완료 | 현재 저장 카메라/scale + projective 1024, 완료 |

해상도 1536/1024는 각 방법의 생성 경로이며 camera-given 여부와 혼동하지 않는다. **새 주 1024 대조 안에서** Pixal3D given/ungiven은 이미지·checkpoint·sampler·seed·1024 cascade를 같게 두고 camera_params만 바꾼다. 기존 1536 대조는 별도 참고 결과다. Ours ungiven은 현재 `C`를 단순 삭제할 수 없다. source 3뷰와 target 1뷰의 상대 pose 및 공통 scale을 RGB에서 추정하고, 그 결과로 ray projection·matching/PBR 주소 cache를 새로 만들고, 카메라 좌표계와 출력 world frame을 GT 없이 고정해야 한다. 기존 GT-depth `spatial_indices`를 이 경로에 쓰지 않는다. 먼저 validation 사례에서 pose/registration과 출력 정합을 검증한 뒤 TEST508을 고정 설정으로 재실행한다. 카메라 추정 오차로 conditioner 분포가 달라지면 V1/Retarget/Selected를 같은 추정 카메라 조건에서 재생성하고, 필요 시 해당 조건으로 학습까지 다시 맞춘다.

**해상도 통제 추가:** 주 카메라 2×2에는 공식 Pixal3D도 지원되는 `1024_cascade`(max tokens 32768)로 새로 생성해 넣는다. 공식 Pixal3D의 기존 1536 결과는 native-default 참고/보조 표로 유지한다. 공식 Pixal3D 1024의 제공/미제공 조건은 같은 checkpoint·전처리·sampler·seed0·cascade를 쓰고 camera_params만 다르게 한다. Ours 제공/미제공 역시 같은 학습된 1024 checkpoint와 sampling 설정을 유지한다. 이는 **voxel/cascade 해상도** 통제이며 최종 mesh의 triangle 수까지 동일하게 만든다는 뜻이 아니다. Geometry는 후처리 전 raw surface로 측정하고, image 지표의 렌더 export는 가능한 동일한 texture size·remesh·decimation 규칙과 동일 Blender/Cycles를 사용해 postprocess 차이를 별도로 감사한다. 기존 1536 official 수치를 1024 칸에 대입하지 않는다.

다른 외부법은 일괄 `1024` 재생성 대상이 아니다. 노드16 runner를 확인하면 TRELLIS.2는 이미 `1024_cascade`, ReconViaGen 및 MVAdapter/CorrAdapter의 공통 복원기도 `1024_cascade`다. ShapeUp의 `adapter-1024-550k` 명칭과 SpaceControl의 proxy/texture 해상도는 Pixal3D의 voxel cascade 해상도와 같은 제어 변수가 아니다. 이들을 임의로 바꾸면 공개 방법의 native 설정도 바뀐다. 방법 간 raw mesh triangle 수를 강제로 같게 하지 않고, 공통 표면 샘플20,000점/F@0.02와 공통 512px 렌더 평가를 유지한다. 필요하면 렌더 전용 mesh 단순화와 texture baking 설정 차이를 별도 sensitivity로 측정하되 raw geometry 지표에는 적용하지 않는다.

주 베이스라인 표에는 최소한 Pixal3D·Ours의 각 카메라 조건을 별도 행으로 보여 주고, Ours에 M0 RGB3장이 추가된다는 사실을 입력 열에 표시한다. 같은 카메라·1024 조건에서도 이미지 입력 예산이 다르므로 직접적인 모듈 인과 효과는 동일 T1+S3 조건의 V1/Retarget/Selected ablation에서만 주장한다. 카메라-미제공 Ours가 검증되기 전까지 기존 주 비교표는 미완료다.

2026-10-27 공식 Pixal3D `camera given` **한 사례 생성 feasibility**를 노드16에서 실행했다. 원본 Pixal3D checkpoint, 전처리, sampler/seed0, native1536을 유지하고 MoGe의 camera_params 대신 저장 FOV60도·거리1.8·source-defined scale0.7837을 넣었다. 실제 1536 출력 raw surface가 완성됐고 9,284,437 vertices/18,803,872 faces, 생성 156.6초였다. [receipt](2026-10-27-pixal3d-given-camera-pilot.json)에 RGB SHA와 raw surface SHA를 고정했다. 생성 성공은 전체 508 성능이나 카메라 개선 효과의 증거가 아니다.

이후 같은 공식 고정 export 회전, 같은 GT, 같은 20,000점 표면 sampler/seed에서 한 사례 CD/F@0.02를 계산했다. 추정 카메라 CD 0.09698/F 0.16320, 제공 카메라 CD 0.09932/F 0.16519였다. [한 사례 진단 JSON](2026-10-27-pixal3d-camera-surface-pilot.json). 즉 **FOV 추정이 실제 60도와 다르다는 사실만으로 제공 카메라가 결과를 개선한다고 결론 낼 수 없다.** 두 수치 방향도 지표별로 다르다. 카메라와 scale 효과는 표본을 늘리고 동일 렌더까지 마쳐 paired 비교해야 한다.

2026-10-28에는 같은 샘플에서 공식 Pixal3D **1024 cascade**를 카메라 미제공/제공 두 조건 모두 실행해 실제 1024 raw mesh와 PBR volume을 얻었다. 원본 이미지 SHA, 원본 checkpoint·sampler·seed0는 두 실행에서 같고, `pipeline_type=1024_cascade`, `max_num_tokens=32768`만 기존 1536 실행과 다르다. [추정 카메라 receipt](2026-10-28-pixal3d-1024-predicted-pilot.json) · [제공 카메라 receipt](2026-10-28-pixal3d-1024-given-pilot.json). 이는 실행 가능성 확인이며 아직 1024 조건의 공통 렌더/metric이나 508개 평균은 없다.
