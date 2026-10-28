# 기존 체크포인트를 활용하는 최소 완료 경로

2026-10-28 사용자 지시 반영. **재학습·새 모듈 탐색·새 대규모 ablation은 진행하지 않는다.** 이미 학습한 SS/PBR·Shape checkpoint를 고정하고, 비교표에 필요한 생성·평가와 원고 정리에 집중한다.

## 이미 확보한 결과의 역할

1. 기존 6행 TEST508 내부 표와 DEV128 SS/PBR ablation은 **기존 M1 유래 객체 scale을 생성에 사용한 설정**으로 명시해 내부 분석 표에 보존한다. 효과가 있었던 입력 및 모듈 비교를 설명하되, 외부 baseline과 동일 조건의 주 비교표로 옮기지 않는다.
2. 새 source-disjoint VAL31은 카메라 제공·객체별 GT scale 미제공·image-only scale 조건에서 Selected/Retarget의 decoded mesh F@0.02 개선이 확정되지 않았고, PBR Top-1의 DEV8 appearance 개선도 확정되지 않았음을 보여준다. 이를 숨기거나 추가 학습 전제의 확정 full claim으로 바꾸지 않는다.
3. 외부 7법 raw/+M0 surface와 공통 Cycles 렌더, 공식 RGB1 native-scale 생성은 이미 실행 중이다. 공식 생성 실패는 성공 교집합 크기에 반영하고, 같은 seed의 실패를 임의로 성공 처리하지 않는다.

## 남은 필수 생성·평가

| 우선순위 | 작업 | 범위 | 종료 증거 |
|---|---|---|---|
| 1 | 현재 외부 7법과 공식 RGB1 완료 | 외부 common502 렌더·평가, 공식 TEST508 시도 및 성공 교집합의 raw/+M0 | 사례별 SHA/receipt, 공통 evaluator의 N·10 metric |
| 2 | Ours 주 비교를 최소 생성 | 고정 checkpoint의 **Vanilla V1과 Selected 2행만** 객체별 GT scale 없이 TEST508 생성. 동일 frozen 4-view RGB/A scale head, camera, seed, Shape512/1024/PBR을 공유 | 사례별 generation receipt·scale provenance·world-frame mesh/GLB |
| 3 | 공통 표 완성 | 외부 7법·공식 RGB1·Ours V1/Selected의 **최종 성공 교집합**에서 raw/+source-M0의 source9/10, target8, CD/F 및 FID/KID 재집계 | 동일 sample ID 목록, 10 metric, source-cluster CI, 실패/508 분모 |
| 4 | 논문 정리 | 주 비교표와 oracle-scale 내부 ablation 표를 명확히 분리; 기존 teaser/overall/module figure와 claim을 실측에 맞게 수정; PDF 빌드·시각 검수 | 최종 TeX, PDF, 표 원자료와 재현 경로 |

Retarget/Top-1·두 zero-shot의 GT-scale-free TEST508 신규 생성은 **필수 범위에서 제외**한다. 이미 있는 내부 표에 남겨 디자인 요소의 개발 결과로 제시한다. 최종 TEST 주 표의 V1↔Selected 차이는 모듈 묶음의 end-to-end 차이이며, 개별 matching/attention의 독립 인과 효과라고 해석하지 않는다.

## 공정성 표기와 부담 제한

- 주 표: 카메라 pose/FOV/거리 제공, 객체별 GT mesh scale 미제공. Ours는 입력 RGB/A와 camera에서 학습된 scale을 내부 추정한다. 공식 Pixal3D RGB1은 T1+C와 native scale 1.0을 사용한다. 모델별 이용 가능한 source RGB·camera 사용 여부와 성공/508을 명시한다.
- +M0는 생성 **후에만** 원본 source mesh로 Sim(3) 정합하고, 모든 방법을 같은 교집합에서 다시 렌더·채점한다. Raw와 +M0를 같이 싣고, +M0가 editing target에 항상 유리하다고 가정하지 않는다.
- oracle-scale 내부 표는 기존 결과를 재사용한다. M1 유래 scale은 생성 입력에 포함됐다고 명시하며, 주 표의 GT-scale-free 수치와 같은 열에서 순위 비교하지 않는다.
- 공통 FID/KID 때문에 기존 502 전체와 최종 N이 다르면 저장된 출력의 **평가만 다시 실행**한다. 메쉬 생성이나 학습을 새로 하지 않는다.

이 경로도 Ours V1/Selected의 GT-scale-free TEST 생성과 렌더가 남아 있다. 따라서 실측 없이 최종 논문 완료 시간을 확정하지 않는다. 추가 학습 시간은 계획에 포함하지 않는다.

## 04:10 KST 실행 연결

- 외부 7법 raw/+M0 공통 502사례 Cycles 캠페인은 32청크 중 22청크를 검증·보관했고 다음 청크를 처리 중이다. node16의 공식 Pixal3D RGB1은 현재 약 265/508 시도까지 진행했다. native sparse 실패 1건은 실패로 기록하고 최종 성공 교집합에서 제외한다.
- Ours의 고정 V1/Selected TEST508 SS 실행은 외부 렌더 완료를 기다리며 자동 연결됐다. 이어서 **두 arm만** Shape512→Shape1024→PBR1024를 생성한다. 첫 동일 사례의 두 arm을 먼저 완주·보관 검증하고, 통과하면 GPU 0–3에서 나머지를 처리한다. 학습 optimizer update는 0이다.
- 공식 RGB1의 후속 렌더는 GPU 6–7에만 배정해 Ours GPU 0–3과 겹치지 않게 했다. 모든 대용량 Ours 결과는 checksum 확인 후 file4에 보관한다. 배치별 오류나 공통 성공 표본 축소가 생기면 원인을 먼저 기록하고, 새 학습이나 arm 확대 없이 남은 평가 범위만 조정한다.

## 04:26 KST 재점검

- 외부 7법 공통502 raw/+M0 렌더는 **26/32청크를 file4에 검증 보관**했다. 공식 Pixal3D RGB1 1024는 동일 seed의 TEST508 중 **약 293개 시도**까지 진행했으며, 확인된 native empty-sparse 실패 1건은 기록된 실패로 유지한다. 완료 워처와 평가 워처는 모두 살아 있다.
- Ours V1/Selected TEST508 SS와 후속 full cascade는 외부 렌더 GPU가 풀릴 때 시작하도록 연결돼 있다. 현재 생성 결과나 주 비교 metric은 아직 없다. Shape·PBR checkpoint와 508개 Top-1 replay 필수 경로를 재확인했다. 오래된 템플릿에 남은 사용하지 않는 `native_guides` 경로는 새 plan에서 제거했다.
- Ours의 최종 mesh export는 생성 직후 같은 GPU에서 **native 1M-face budget, 2048 texture, 1024 cascade**로 수행하고 raw NPZ와 GLB를 file4에 checksum 보관하도록 worker를 연결했다. 첫 paired 사례의 실제 생성·export smoke가 통과하기 전까지 완료 소요 시간을 실측 상한으로 주장하지 않는다.
- [로컬 공간 정리](2026-10-28-local-disk-reclaim-ko.md)로 24개 중복 캐시 약 1.10 TB를 제거해 현재 약 1.0 TiB 여유를 확보했다. 선택된 SS 가중치와 현재 평가 입력은 유지했다.

**평가 해석:** 외부 7법과 Ours의 입력 가용성은 다르므로 표에 T1/S3/카메라 사용을 표시한다. 공식 Pixal3D와 Ours의 camera-given·객체 GT scale 미제공·1024 비교를 주된 직접 대조로 삼고, 외부법은 동일 evaluator 및 공통 성공 집합의 참조 baseline으로 제시한다. 기존 oracle-scale 6행 및 DEV ablation은 별도 내부 증거로 남긴다. VAL31의 Selected/Retarget F@0.02 개선은 신뢰구간이 0을 포함하므로 TEST 결과 전에는 확정 성능 주장으로 쓰지 않는다.

## 04:37 KST 중간 점검

- 외부 7법 공통502 raw/+M0 Cycles 렌더는 **29/32청크**가 사례별 결과와 함께 file4에 검증 보관됐다. 30번째 청크가 staging 중이며, 최근 완료 청크는 약 3분 내외였다. 렌더 orchestrator와 node16 평가 watcher는 살아 있다.
- 공식 Pixal3D RGB1 1024는 TEST508의 **index 310까지 PASS**를 로그에서 확인했다. native empty-sparse 실패 1건은 별도로 기록돼 있으며, 마지막 성공 교집합은 아직 확정되지 않았다. node16 생성 GPU7, 후속 surface/export watcher와 통합 평가 watcher 모두 살아 있다.
- 고정 V1/Selected SS, full cascade, Ours surface, 렌더, 최종 four-condition metric watcher가 차례로 연결돼 있다. 현재 SS와 Ours TEST 생성은 외부 렌더 GPU 반환을 기다리고 있다. 공식 후속 렌더는 GPU6–7, Ours cascade는 GPU0–3을 사용한다. **Ours의 공통 조건 metric은 아직 없다.**
- 첫 paired V1/Selected 사례에서 실제 1024 GLB export까지 통과해야 후속 처리시간을 신뢰할 수 있다. 이 smoke가 완료되기 전까지 논문 완료 시각을 확정하지 않는다. 기존 oracle-scale 내부 표와 새 GT-scale-free 공통 표의 수치를 분리한다.
- 13조건 중 새 Ours 4조건의 10-metric 평가가 닫히면, node16의 기존 9조건 평가를 재사용해 동일한 성공 ID·protocol·N을 검증한 뒤 [공통 조건 주 표](../paper/final-results/scale-ungiven-main-table.tex)를 자동 작성하도록 연결했다. 현재 table watcher는 살아 있으며 평가 완료를 기다린다. 표의 해석 문장과 PDF 반영은 실측 수치를 검토한 후 진행한다.

## 04:47 KST 전환·복구 점검

- 외부 7법 공통502의 **32/32청크, 10,542개 렌더 작업**이 file4 checksum 보관을 마쳤다. node16 외부 평가 v5는 바인딩 스크립트의 `PYTHONPATH` 누락으로 생성된 결과를 읽기 전 종료했다. 평가 환경의 benchmark 모듈 경로를 확인하고 v6를 새 namespace에서 재시작했다. 9방법 통합 워처 v4도 새 PID/경로로 이어 붙였으며 기존 렌더·surface 결과는 재생성하지 않는다. 후속 Ours 바인딩/target8 평가에서도 같은 모듈 경로를 명시하도록 고친 Ours 4조건 워처 v3와 최종 표 워처 v3가 대기 중이다.
- 로컬 V1·Selected SS가 GPU0/1에서 시작됐다. V1은 진행 중이다. Selected의 첫 실행은 동결된 훈련 config의 `projection_root`가 train/validation cache를 가리켜 TEST508 첫 샘플을 찾지 못하고 모델 계산 전에 종료했다. 이미 있던 TEST 전용 projection cache 508개를 file4와 전체 `rsync --checksum --dry-run`으로 대조해 byte 차이 **0개**를 확인했다. TEST split에서 이 경로만 선택하도록 evaluator를 고쳤으며 Selected-only 재실행이 GPU1에서 진행 중이다.
- 원래 SS 워처는 V1 완료 직후 첫 Selected 실패를 보고 종료할 예정이므로, 상태를 일시 정지해 후속 watcher의 PID 연결을 보존했다. 복구 워처가 두 arm의 508개 artifact, 동일 freeze·seed·image-only scale·optimizer update 0을 검증한 뒤 원래 위치에 atomic completion receipt를 작성하고 원 워처를 재개한다. 이 과정은 checkpoint 재학습이나 V1 재실행을 포함하지 않는다.
- 최종 공통 표에 앞서 V1↔Selected의 source 3지표, target 3지표, CD/F-Score를 source-cluster paired bootstrap과 8-endpoint 보정 구간으로 계산하는 분석 코드를 준비·smoke 검증했다. 실제 TEST 수치를 얻기 전에는 개선 여부를 주장하지 않는다.
- node16 GPU7에서 공식 RGB1 생성 종료 직후 외부 7법의 학습형 지표와 공식 GLB export가 동시에 시작되는 충돌을 발견했다. 공식 export watcher PID를 일시 정지하고, 외부 7법의 full502 10-metric completion receipt를 검증하면 자동 재개하는 gate를 실행했다. 외부 평가가 오류 종료하면 gate가 export를 재개해 GPU를 무기한 묶지 않는다. 공식 생성 자체는 계속 진행한다.

## 05:06 KST 생성·내보내기 병목 점검

- Ours SS는 고정 V1 **496/508**, Selected **416/508** 완료를 실시간 로그에서 확인했다. node16 공식 RGB1은 **373/508**을 통과했고, 외부 7법의 렌더는 여전히 32/32청크 완료 상태다. 외부 평가 v6는 7방법×502사례 바인딩을 통과하고 target8 점수를 계산 중이다. 최종 10지표와 공통 성공 N은 아직 확정되지 않았다.
- 실제 VAL 1024 mesh(원본 7,227,722 faces)의 1M-face GLB bake는 약 9분이 걸렸다. 동일 생성물의 250k-face bake는 **28.19초**, 원본 표면과 내보낸 GLB의 world-frame q99 최근접점 오차 **0.00620**이다. 공통 Cycles 설정으로 뽑은 두 시점의 1M/250k 이미지 비교는 각각 **PSNR 52.62/54.04 dB**, mask IoU **0.99971/0.99962**였다. 이는 동일 사례의 렌더 bake 근사도이며 GT 대비 품질 결과는 아니다.
- Ours TEST worker는 geometry·CD/F용 원본 NPZ를 그대로 보관하면서 **렌더용 GLB만 250k faces, 2048 texture**로 bake하도록 전환했다. 실제 TEST 첫 paired 사례에서 q99 등록과 GLB·archive를 재검증한다. 생성 해상도 1024, 원본 surface 평가, image-only scale은 바뀌지 않는다. 논문에는 렌더용 face budget을 명시한다. 공식 RGB1 export는 현재 1M-face 계약이므로 성능과 시간은 실제 첫 사례를 확인한 후 보고한다.

## 05:09 KST 공식 RGB1 렌더 병목 조정

- 공식 RGB1의 실제 TEST mesh(원본 13,383,476 faces)를 local GPU3에 복사해 250k 렌더 bake를 검증했다. **q99 world 0.00614**, 공통 Cycles 1M/250k 시점8·9 비교 **PSNR 43.74/40.73 dB**, mask IoU **0.99928/0.99943**이었다. 이 실험은 동일 prediction의 렌더 근사도만 측정하며 GT 품질 차이를 뜻하지 않는다.
- node16의 기존 1M 공식 exporter와 GPU7 직렬 gate를 명시적으로 일시 정지했다. 새 250k exporter는 외부 7방법 10지표와 공식 TEST508 생성 완료 뒤 GPU7에서 시작하도록 bridge PID **801703**으로 연결했다. 사례별 budget·checksum·완료 개수를 검증하고 나서만 기존 공식 export 경로를 새 완료 namespace에 연결한다. 새 경로가 실패하면 정지해 둔 원래 1M gate를 재개한다. 공식 RGB1과 Ours 모두 **1024 생성·원본 surface metric 유지, 렌더용 250k/2048 bake**로 맞춘다.

## 05:10 KST 동결 SS 완료

- V1/Selected 모두 TEST508의 **508/508 SS artifact와 scale 계약을 통과**했다. 복구 워처가 동일 frozen sample ID·checkpoint·image-only scale·TEST projection bytes·학습 update 0을 확인해 후속 cascade의 완료 영수증을 작성했다.
- 같은 GT-scale-free TEST508의 SS voxel IoU는 **V1 0.374215, Selected 0.379096, 차이 +0.004880**이다. 이는 SS 단계의 paired 중간 결과로서, 예전 M1 유래 scale을 사용한 0.43/0.47 수치와 직접 비교하지 않는다. 최종 mesh CD/F·source/target 이미지 metric의 개선 여부는 아직 모른다.
- 첫 paired full-cascade pilot이 GPU0에서 시작됐다. 렌더용 250k faces가 원본 표면과 q99 world 0.01 이내로 등록되지 않는 드문 사례는 500k→1M budget을 순서대로 재시도하도록 후속 worker와 공식 RGB1 exporter를 보강했다. 실패한 GLB는 삭제하고 각 사례의 실제 렌더 budget을 영수증에 기록한다. node16 bridge는 갱신된 코드의 PID **908808**으로 교체돼 대기 중이다.

## 05:14 KST 첫 full-cascade pilot 복구

- 첫 V1 TEST 사례의 Shape/PBR 생성은 완료됐지만 250k 렌더용 GLB의 q99 world가 **0.010854**로 사전 정의한 0.01 한계를 소폭 넘었다. export가 이를 거부했고 원본 geometry·생성 결과는 남아 있다. 첫 pilot 프로세스는 이 이벤트에서 종료돼 후속 surface/render/eval watcher가 중단됐다.
- 250k→500k→1M 순서의 **동일 원본 prediction 렌더 bake fallback**을 실제 worker에 연결하고, 실패한 250k 영수증을 재사용해 500k로 pilot을 다시 실행 중이다. 후속 watcher는 새 PID/namespace로 연결할 예정이다. GT나 학습 checkpoint를 바꾸지 않았고 optimizer update는 없다.
- node16 GPU7에서는 공식 RGB1 생성과 외부 7방법의 학습형 metric 평가가 겹칠 수 있어, 외부 평가 부모 PID **190839**를 target8 CPU 점수 계산이 진행되는 동안 일시 정지했다. 공식 TEST508 시도 완료 영수증이 나오면 gate PID **988475**가 이를 재개한다. 이어서 외부 10지표 완료 후 250k 공식 GLB export bridge PID **908808**이 시작한다. 이 순서로 GPU7 메모리 중복 사용을 피한다.
- 파일서버4에 보관할 때 node16의 `/file4` 경유 rsync가 첫 195 MB prediction에 약 2분 걸렸다. 이미 인증된 직접 file4 SSH control socket으로 별도의 131 MB prediction 전송을 시험하니 **1.45초**였다. 임시 전송 파일은 삭제했다. 후속 Ours worker가 파일서버4에 직접 체크섬 보관하도록 전송 경로를 변경했다. 사용 데이터·원격 보관 위치·영수증 검증 규칙은 동일하다.

## 05:20 KST paired pilot 및 4-GPU 재개

- 동일 첫 TEST 사례의 V1 Shape와 Selected Shape/Top-1 PBR이 모두 생성·렌더용 GLB 정합·파일서버4 checksum 보관을 통과했다. V1 render bake는 250k q99 0.010854로 거부돼 500k **0.006370**으로 통과했고, Selected는 250k **0.008846**으로 통과했다. 두 raw geometry는 렌더 bake와 별도로 보관됐다.
- 새 cascade PID **586063**은 pilot을 재검증하고 GPU0–3 worker PID **586071–586074**를 시작했다. `PIXAL3D_FILE4_CONTROL_SOCKET=/tmp/file4-ctl-v2`로 같은 파일서버4 출력 경로에 직접 전송한다. 새 surface/render/four-condition metric/table watcher PID **586064–586067**과 paired-bootstrap watcher PID **588783**도 살아 있다. 이전 실패 watcher namespace는 평가에 사용하지 않는다.
- 최종 Table 1은 평가한 13조건의 동일 성공 ID·protocol·reference hash·10지표를 확인하고, 그중 +M0 **10행**을 표로 작성한다. source-cluster paired 8-endpoint 분석도 자동 연결했다. 현재 아직 **full-cascade TEST 전체 결과·공통 최종 N·논문 최종 claim은 미완료**다.
- image-only scale estimator `.joblib`의 `training_manifest`는 `rgb_v1/manifests_world_v4/train.jsonl`, `training_sample_ids`는 **9,149개**로 확인했다. 현재 TEST508 manifest와 대조하면 sample ID **0/508**, source UID **0**, source cluster **0**의 교집합이다(훈련 cluster 1,790개, TEST cluster 134개). 따라서 이 추정기는 TEST 샘플·원본 source의 label을 학습에 쓰지 않았다. feature schema는 `RGBA128-mask-bbox-area-camera-v1`이다.

## 05:23 KST 4-GPU 첫 진행 증거

- 4개 worker PID **586071–586074**가 모두 살아 있고, GPU2는 분산 실행의 첫 paired V1/Selected 사례를 새 direct-file4 경로로 끝냈다. 나머지 GPU의 첫 사례도 V1 archive를 통과하고 Selected를 처리 중이다. archive receipt는 pilot 3개를 포함해 **10개**다. 이 숫자는 사례 수가 아니라 V1 Shape·Selected Shape·Selected PBR 단계별 보관 영수증 수다.
- node16 공식 RGB1은 약 **408/508** 시도까지 진행했고, external target8 CPU scorer도 살아 있다. 외부 learned metric 부모는 공식 RGB1 GPU7 생성 완료까지 일시 정지돼 있다. 최종 10지표·공통 N은 아직 없다.

## 05:25 KST 유휴 GPU6·7 활용

- 주 cascade GPU0–3가 앞쪽 TEST 사례를 처리하는 동안, 뒤쪽 index **507부터 400까지**는 GPU6·7이 서로 겹치지 않게 한 사례씩 처리하도록 supervisor PID **589798**을 시작했다. 단일 사례 worker의 `world_size=508`, `rank=해당 index`를 이용해 앞쪽 worker와 동일한 frozen plan·checkpoint·file4 archive 규칙을 유지한다.
- 공식 RGB1 렌더 watcher PID **543994**는 그 입력 GLB가 아직 준비되지 않아 일시 정지했다. node16 공식 export 완료가 확인되면 tail supervisor는 새 사례 배정을 중단하고 현재 두 사례가 끝나자마자 watcher를 자동 재개한다. 따라서 GPU6·7을 공식 Cycles 렌더에 돌려준다. GPU4·5의 다른 작업은 건드리지 않는다.
- 05:26 KST 기준 주 GPU0–3은 첫 분산 paired 완료 영수증을 순차 작성 중이며, GPU6의 뒤쪽 index507은 V1 Shape를 checksum 보관했다. 여섯 GPU 모두 각자 고유 case index를 맡고 있으며 현 시점 archive receipt는 **25개**다. 이 상태만으로 전체 ETA나 final metric을 확정하지 않는다.

## 05:28 KST 매우 초기 geometry sanity check

- 완전히 보관된 **첫 8개 paired TEST 사례**의 생성 직후 raw mesh metric을 동일 sample ID로 비교했다. Selected−V1 F@0.02 평균 **+0.0260**, 중앙값 **+0.0278**, 8개 중 6개 개선이다. 반면 CD 평균 변화는 **+0.000136**(악화), 중앙값은 **−0.000139**, 8개 중 4개 개선으로 방향이 불안정하다. 한 사례의 CD 악화가 평균을 크게 움직인다.
- 이 수치는 초반 8/508에서 나온 순서 의존 표본이며, source-M0 정합 전 생성물 기준이다. 논문 claim이나 최종 공통 표와 섞지 않는다. 최종 N에서 paired cluster CI와 source9/10·target8 2D 지표가 닫혀야 성능 향상을 판단한다.

## 05:36 KST 렌더 mesh 충실도 예외와 복구

- TEST508 index 11의 V1 1024 raw prediction은 정상 생성됐지만, 렌더 전용 GLB bake의 원본 표면 대비 world-frame q99가 native 250k/500k/1M에서 각각 **0.03277/0.03874/0.03695**로 허용치 0.01을 넘었다. GPU OOM이나 학습 실패가 아니라 `o_voxel.postprocess.to_glb`의 추가 topology cleanup·간소화에서 생긴 형상 차이이다. 이 사례를 최종 성공 집합에서 임의 제외하지 않는다.
- 원래 4-GPU cascade 부모 PID 586063을 멈춰 후속 surface/render/eval watcher의 연결을 보존했다. GPU0–2 worker는 계속 진행하고, GPU3 대체 worker PID 594305는 index 11만 건너뛰고 다른 담당 사례를 이어서 처리한다. 별도 복구 watcher PID 594073은 전체 508쌍의 생성·보관 영수증을 확인한 뒤에만 완료 신호를 낸다.
- 문제 사례의 raw mesh를 별도 간소화기로 1M faces까지 줄이면 bake **전** 최근접점 q99는 0.00285였으나, 표준 GLB bake **후**에는 0.03487이었다. 따라서 renderer 내부의 추가 mesh 변경을 피하면서 동일 PBR baker를 사용하는 fallback을 시험 중이다. 이는 검증 중이며 아직 통과한 해결책이 아니다.
- 05:36 KST 기준 file4 checksum archive는 V1 Shape **23/508**, Selected Shape **19/508**, Selected Top-1 PBR **19/508**이다. GPU6·7 tail supervisor는 507, 506, 505, 504, 502번 사례를 각각 처리했다. node16 공식 RGB1 생성은 index 434까지 통과했고 외부 7법 렌더 32/32청크는 완료됐다. 공통 10지표·최종 N·논문 주 claim은 여전히 미완료다.

## 05:40 KST 예외 사례 형상 보존 통과

- index 11 V1의 별도 1M-face 간소화 결과를 표준 PBR/UV bake에 넣되, bake 내부의 두 번째 간소화와 작은 component 제거 등을 건너뛰어 **원본 대비 q99 world 0.002845**로 검사를 통과했다. raw 4,775,408-face prediction은 그대로 surface metric에 사용하고, 이 1M-face GLB는 렌더에만 사용한다. GLB·prediction SHA와 실제 simplifier/cleanup 방식은 사례 receipt에 기록됐다.
- 해당 V1 Shape는 file4 checksum 보관을 마쳤다. 동일 index의 Selected Shape/Top-1 PBR을 단일 사례 worker에서 복구 중이며, 나머지 GPU3 대체 worker는 다음 사례를 계속 진행한다. 모든 508쌍 영수증이 닫히기 전에는 cascade 완료를 선언하지 않는다.
- node16 공식 RGB1 exporter에도 **250k→500k→1M** 재시도 중 마지막 1M에서만 같은 형상 보존 fallback을 쓸 수 있게 코드를 배치했다. node16의 Python 3.10 전용 `fast-simplification 0.2.0`을 사용자 쓰기 가능 폴더에 설치하고 import/구문 검사를 통과했다. 공식 exporter는 아직 시작하지 않았으므로 실제 공식 TEST 사례의 fallback 성공을 주장하지 않는다.
- 05:42 KST 진행 스냅샷: V1 Shape **29/508**, Selected Shape **26/508**, Selected Top-1 PBR **26/508**이 file4 checksum 보관을 마쳤다. 공식 RGB1은 index **444/507**까지 시도했다. 예외 index 11의 V1 Shape 보관은 완료됐고 Selected의 250k 렌더 bake가 q99 기준을 넘어서 500k fallback을 실행 중이다. GPU6·7 뒤쪽 작업은 index 500, 501까지 완료됐다.

## 05:45 KST index 11 양쪽 arm 복구 완료

- 문제 사례의 Selected Top-1 PBR 렌더 mesh도 250k·500k bake가 형상 기준을 넘었으나, 1M-face 형상 보존 경로에서 **q99 world 0.002348**로 통과했다. V1의 q99 0.002845와 함께 각 Shape/PBR 결과를 file4에 SHA 검증 보관했다. 단일 사례 worker는 `PASS_TEST_CASE_TWO_ARMS_ARCHIVED`와 `PASS_TEST_SHARD_TWO_ARMS_ARCHIVED` 영수증을 기록하고 정상 종료했다.
- 원래 GPU3 shard worker의 종료 원인이었던 사례는 이제 전체 508쌍에 포함된다. GPU0–2, 대체 GPU3, GPU6·7의 다른 사례는 계속 진행하고, 복구 watcher가 **전 사례**의 생성·GLB·archive receipt를 검증할 때까지 후속 완료 신호를 내지 않는다.
- 외부 7법의 원래 공통502에 대한 **target camera 8 raw↔source-M0 정합 paired 평가**도 끝났다. M0 정합 후 PSNR 평균 차이는 TRELLIS.2 +0.599 dB, 원래 Pixal3D(estimated C) +2.329 dB, ReconViaGen +0.994 dB, MVAdapter +1.223 dB, CorrAdapter +1.242 dB, ShapeUp +0.256 dB, SpaceControl +0.680 dB다. 이 값은 502개 외부법 자체의 진단치이며 공식 RGB1의 실패 사례를 반영한 **최종 13조건 성공 교집합** 수치는 아니다. 최종 표에는 공통 N으로 다시 계산한 값만 넣는다.

## 논문 반영 시 필수 확인

- 새 13조건 공통 성공 집합에서 +M0 10행·10지표 표와 V1↔Selected source-cluster 8지표 보정 구간이 모두 닫힌 다음, 이를 본문의 주 비교로 올린다. 기존 M1 유래 scale 6행 TEST 표는 조건이 다른 **내부 diagnostic/ablation**으로 명시해 본문 주 순위에 섞지 않는다.
- 현재 원고의 `sections/experiments.tex`와 `final-results/results.tex`는 기존 M1-scale 조건을 주로 설명한다. 새 평가가 닫힌 뒤 `unified-evaluation-protocol-draft.tex`의 camera-given·scale 미제공·source-M0 계약을 본문에 통합하고, 숫자·N·input access·raw/+M0 설명을 실제 영수증과 대조해 수정한다.
- 현재 teaser는 기존 M1-scale TEST archive의 선정 사례에서 왔다. 최종 주 비교의 정성 그림은 새 공통 Cycles 렌더와 동일 성공 사례에서 다시 선정·출처를 기록한다. Teaser 1개, stage overview 1개, 매칭·retargeting 상세 각 1개를 PDF 크기에서 검수한다. 현재 stage overview·상세 도식은 임시 그림 수준이므로 최종 시각 검수 없이 완료라고 선언하지 않는다.
- 새 teaser 사례 선택은 최종 공통 N의 V1↔Selected per-sample target mask IoU와 source LPIPS 차이가 모두 유리한 후보 중 **중앙에 가까운 사례**를 우선하고, 해당 조건을 만족하지 않으면 전체 paired 변화의 중앙 사례를 사용한다. 사례 ID, 두 지표의 차이, 렌더 receipt SHA, 입력 이미지 SHA를 그림 provenance에 기록한다. 극단적 최상위 성능 사례만 뽑아 전체 성능처럼 보이게 하지 않는다.
- 이 규칙을 `select_scale_ungiven_teaser_case.py`로 구현했다. 최종 evaluator의 V1/Selected source LPIPS와 동일 공통 N의 target mask IoU를 결합하고, source-cluster/seed·full-metric 영수증을 확인한 뒤 선택 ID와 수치·입력 파일 해시를 기록한다. 실제 Cycles 렌더와 입력 SHA를 검증해 그림을 만드는 단계는 최종 평가 후 수행한다.
- 주 +M0 10행·10지표 표와 함께, **같은 최종 N**의 10방법에 대해 target-camera PSNR/SSIM/M-IoU의 raw, +M0, paired delta를 나란히 보이는 보조 표를 자동 생성하도록 표 작성 코드를 연결했다. 각 +M0 평균이 주 표의 target metric과 정확히 같은지 검증한다. 외부 7법의 원래 502개 raw 진단치를 최종 N의 수치인 것처럼 재사용하지 않는다.
- 공식 RGB1의 현재 확인된 native empty-sparse 실패 1건은 외부 7법 성공 교집합 502개 **안에** 있다. 추가 실패가 없더라도 최종 13조건 교집합은 최대 **501개**이므로 기존 외부 raw502 수치를 주 표와 같은 N으로 표기할 수 없다. 최종 N은 508개 공식 시도가 모두 닫힌 뒤 확정한다.
- source-M0 정합 코드를 다시 대조했다. 외부 7법과 Ours surface scorer는 동일한 `register_m0` 구현, 동일 sample-ID 기반 4096점 source/prediction 추출 seed와 20,000점 target metric seed를 사용한다. 정합 함수 입력은 예측 surface와 원본 M0 source surface뿐이고, M1 target은 정합 후 CD/F-Score 평가에만 전달된다. 외부 scorer의 오래된 docstring이 M1 파일을 여는 시점까지 잘못 단정하던 문구는 실제 데이터 흐름에 맞게 고쳤다.

## 06:04 KST Ours source-M0 surface 경로 선행 실행

- 전체 cascade 완료를 기다리는 surface scorer를 node16의 실제 file4 archive와 TEST manifest로 **index 11 한 사례**에 먼저 실행했다. V1·Selected 두 mesh의 archive/prediction/export SHA와 image-only scale provenance, Top-1 PBR geometry byte identity 검사를 통과했고, `PASS_OURS_V1_SELECTED_IMAGE_SCALE_SOURCE_M0_SURFACE` 1사례·2행으로 정상 종료했다. 영수증은 node16의 `/home/youngwoo/retargeting-unified-test508/ours-source-m0-surface-pilot-index11.jsonl`에 있다. 실행은 약 10초 이내였으며 한 사례의 metric은 최종 주장을 위한 표본이 아니다.
- 이 사례에서 raw→source-M0 F@0.02는 V1 **0.74929→0.82060**, Selected **0.76322→0.81975**였다. 동일 source mesh 정합이 target metric에 양쪽 모두 영향을 줄 수 있음을 확인했다. 최종 공통 N, 4-shard 전체 실행, 10지표 평가를 대체하지 않는다.

## 05:53 KST 초기 43쌍 raw mesh 경향 수정

- V1 Shape **46/508**, Selected Shape/Top-1 PBR **43/508**이 file4 checksum 보관됐다. 공식 RGB1은 index 463까지 시도했고 GPU7에서 계산이 계속된다. 여섯 로컬 worker도 살아 있으며 file4 직접 SSH 전송이 응답한다.
- 양쪽 arm이 모두 보관된 43개 사례의 `generation.json` raw surface metric을 같은 sample ID로 비교하면 Selected−V1 F@0.02 평균 **+0.001691**, 중앙값 **+0.001397**, 개선 **24/43**이다. CD(L1)는 평균 **+0.000246**, 중앙값 **+0.000186**으로 악화 방향이며 개선 **20/43**이다. 처음 8개에서 관찰된 F-Score +0.0260은 이 43개에서는 재현되지 않는다.
- 이 역시 순서 의존 43/508의 생성 직후 raw mesh 수치이며 source-M0 정합과 최종 공식 성공 교집합을 반영하지 않는다. 따라서 geometry 개선 claim 또는 final method selection의 근거로 올리지 않는다. 최종 paired cluster CI와 target/source 2D 지표가 닫힌 뒤 해석한다.
- 이어서 보관된 **44개 쌍**의 SS per-asset 평가와 raw mesh를 연결해 확인했다. 이 초기 부분집합의 Selected−V1 SS IoU 평균은 **−0.001445**(개선 23/44)로, 전체 TEST508 SS 차이 **+0.004880**과 방향이 다르다. 같은 44개에서 F-Score 차이는 +0.001344, IoU 차이와 F-Score 차이의 사례별 Pearson 상관은 약 **0.65**다. 즉 이 초반 raw mesh 경향만으로 downstream Shape가 SS 개선을 지웠다고 결론 내릴 수 없다. 표본 순서 편향과 최종 공통 성공 집합을 고려해야 한다.
- 이 44쌍에서 `scale_used_in_generation`은 V1과 Selected가 사례별로 동일했다. 따라서 위 paired 차이는 두 arm에 다른 image-only scale을 준 결과가 아니다. 전체 TEST에서도 동일 scale 계약을 후속 audit가 검증해야 한다.
- archive 완료 시점 47개 V1·45개 Selected의 Shape/PBR 생성 시간 중앙값은 각각 **37.9초/39.1초**, 90백분위 **61.6초/62.8초**다. 렌더 GLB budget은 V1 250k/500k/1M **43/3/1**, Selected **43/1/1**로 대부분 250k에서 형상 검사를 통과했다. 이는 bake·전송·렌더·평가를 제외한 stage 생성 시간이며 전체 campaign ETA로 오용하지 않는다.

## 06:03 KST 실행 경로 및 병목 재점검

- Ours의 실제 index 11 V1/Selected GLB를 file4 직접 SSH로 staging하고, 동일 Blender/Cycles 평가 경로에서 raw/+source-M0의 12개 렌더 작업을 **44.43초**에 완료했다. 305 MB 입력 staging, 렌더 결과 checksum 보관, 로컬 임시 파일 회수를 모두 통과했다. `view08`의 두 결과를 직접 열어 모델 형상과 텍스처가 정상 출력되는 것을 확인했다. 이는 렌더 파이프라인 QA이며 성능 metric은 아니다.
- 06:01 KST 기준 file4 checksum 보관은 V1 Shape **59/508**, Selected Shape/Top-1 PBR **55/508**이다. node16 공식 RGB1은 약 **485/508** 생성 영수증이 있고 GPU7에서 실행 중이다. 외부 7방법 렌더 32/32청크와 target8 raw/+M0 paired 점수는 완료; learned metric 및 최종 13조건 공통 평가는 공식 생성 종료를 기다린다.
- GPU6 뒤쪽 index 495의 27,263,088-face 예측은 250k/500k 렌더 GLB의 형상 q99 검사를 실패했고, 1M fast simplifier가 선언 budget을 도달하지 못해 archive 전 중단됐다. 원본 예측과 anchor 평가 파일은 보존됐고 학습/OOM 문제가 아니다. GPU7 뒤쪽 worker와 GPU0–3 본 작업은 계속한다. 이 예외 사례를 복구하거나 공식 실패와 같은 방식으로 문서화한 공통 성공 교집합에서 제외하기 전에는 508 전체 완료를 선언할 수 없다.
- 06:05 KST 재확인: V1 **63/508**, Selected Shape/PBR **58/508**이 file4 checksum 보관을 통과했고 공식 RGB1은 **494개 generation receipt**를 기록했다. 두 방법의 full-cascade 주 비교 지표, 최종 공통 N, 논문 표는 아직 미완료다. index 495의 간소화 목표 도달 여부를 별도 CPU 진단 중이다.

## 06:08 KST 대형 mesh 렌더 복구

- index 495의 원본은 **27,263,088 faces**다. 별도 `fast-simplification` 진단에서 목표 1,000,000 faces를 요청해도 **4,228,688 faces**가 남았다(310.8초). 따라서 기존 1M fallback 실패는 이 mesh가 알고리즘의 목표 face 수에 도달하지 못한 결과로 확인됐다. 원본 예측의 생성 또는 GPU OOM 증거는 아니다.
- 드문 대형 사례용 렌더 budget 2M/5M을 순차 fallback에 추가했고, 동일 원본 대비 q99 world $\leq 0.01$ 검사를 유지했다. 5M budget으로 index 495의 V1 GLB를 GPU6에서 직접 시험 중이며, 아직 성공 결과로 기록하지 않는다. 1024 voxel cascade, raw geometry metric, frozen checkpoint는 변경하지 않았다.
- node16 공식 RGB1 exporter에도 같은 fallback 코드를 배치하고 로컬/원격 SHA를 대조했다. 공식 exporter는 생성 및 외부 7방법의 학습형 metric 종료를 기다린다. 예외가 반복될 때 렌더 mesh의 실제 face 수와 q99를 사례 영수증에 남긴다.
- 추가로 file4 checksum 보관이 끝난 **62개 paired V1/Selected**를 같은 sample ID·같은 image-only scale로 연결했다. 정합 전 raw mesh의 Selected−V1 F@0.02는 평균 **+0.005674**, 중앙값 **+0.001325**, 개선 **34/62**이고 CD(L1)는 평균 **+0.0000444**, 중앙값 **+0.0000106**, 개선 **30/62**다. 완료 순서에 따른 비무작위 부분집합이고 source-M0 정합 전 수치이므로 최종 성능 claim에는 쓰지 않는다.

## 06:11 KST 공식 RGB1 생성 종료

- node16의 공식 RGB1은 TEST508 시도를 모두 닫았다: **507 성공, 1 native empty-sparse 실패**. 실패 ID는 `shard002_6f4509a0f0714891ab41b08cd2848315_d2_fast300fpssupplyv1_balanced_JN_L_target08`이고 시도 완료 영수증에 남았다. 외부 7방법의 GPU7 학습형 10지표 evaluator PID 190839가 gate에 의해 재개됐고 실제 자식 metric PID 2492093이 실행 중이다. 공식+외부+Ours 최종 성공 교집합은 여전히 산출 전이다.
- 로컬 GPU7은 tail supervisor 종료 뒤 빈 시간을 사용해 겹치지 않는 index 486 단일 사례를 처리 중이다. GPU6은 index 495 5M-face 렌더 GLB 복구에 사용한다. 공식 render watcher는 node16의 공식 export 완료 후에만 로컬 GPU6/7 렌더 작업을 시작한다.

## 06:24 KST 생성·평가 워처 점검

- Ours file4 checksum 보관은 V1 **80/508**, Selected Shape/Top-1 PBR **77/508**로 증가했다. 원래 GPU0–3 worker는 살아 있고, 복구 watcher는 전체 영수증 전에는 완료 신호를 내지 않는다. GPU7의 별도 index 486·484는 두 arm 모두 보관됐고 482를 처리 중이다. GPU6에서는 대형 index 495의 5M GLB 작업과 별도 index 491 작업이 겹치지 않는 사례로 진행 중이다.
- node16 외부 7방법의 10지표 evaluator 자식 PID 2492093은 공식 생성 종료 게이트에서 정상 재개됐으며 약 13분째 실행 중이다. 아직 평가 completion/최종 13조건 성공 교집합/10지표 표는 없다. 공식 RGB1은 507/508 성공으로 닫혔다.
- index 495의 5M 렌더 작업은 원본 27M-face mesh에서 16분 이상 CPU 계산 중이고 CPU 사용 및 RSS가 변화한다. 성공 또는 실패 영수증 전까지 이 사례를 완료로 계산하지 않는다. 1024 cascade와 raw mesh 점수 파일은 그대로다.

## 06:35 KST 공식 export와 외부 metric 병행

- node16에서 공식 RGB1의 실제 성공 사례 1개를 250k/2048 GLB로 bake했다. 2,529,484-face 원본 → 236,204-face 렌더 GLB, world-frame q99 **0.002475**로 사전 정의한 0.01 이내였다. 외부 7방법 learned metric 프로세스와 동시 실행해도 GPU7 메모리 사용은 약 4GB 수준으로 관측됐다. 이 파일은 독립 pilot 경로에 있으며 최종 평가에 슬쩍 합치지 않는다.
- 이 실측을 근거로 bridge를 재시작해 공식 TEST 공통 성공 집합의 250k 우선 GLB export를 **외부 7방법 10지표 평가와 병행**했다. 새 bridge PID **3135838**이 공식 export 자식을 시작했으며 원래 1M watcher/gate PID **3016321/569292**는 일시 정지 상태를 유지한다. 외부 10지표 완료 및 공식 export 완료를 둘 다 확인한 뒤에만 기존 경로를 새 결과로 연결하고 로컬 공식 렌더를 열도록 했다.
- 공식 exporter에는 500k/1M/2M/5M 형상 fidelity fallback과 사례별 실제 face budget 영수증을 유지한다. bridge의 완료 검사는 5M까지의 declared budget을 인정하도록 수정했고 로컬/원격 코드를 동기화했다. 이 겹침은 평가 세팅·체크포인트·분모를 바꾸지 않는다.

## 06:36 KST 대형 mesh V1 렌더 복구 통과

- index 495의 V1 27,263,088-face raw mesh는 5M fast simplifier + topology-preserving 표준 PBR/UV bake에서 **4,999,995-face 입력 GLB**, **410 MB 출력**, 원본 대비 world-frame q99 **0.002810**으로 통과했다. 250k/500k에서 형상이 어긋나고 1M에서는 simplify 자체가 4,228,688 faces 이하로 내려가지 않던 사례다. 실제 render budget·SHA·q99는 `benchmark.json`에 기록됐다.
- 이 pass는 texture/GLB 검증이며 최종 2D quality 결과가 아니다. GPU6 rank495 worker를 다시 시작해 이미 검증한 V1 GLB를 중복 생성하지 않고 file4 checksum 보관한 다음 Selected Shape/PBR을 이어 처리한다. 이 worker가 두 arm 모두 보관하기 전에는 index495 완료로 계산하지 않는다.
- 실제 rank495 복구 worker는 V1 raw prediction·410 MB GLB를 file4 checksum 검증 후 보관했고, 현재 Selected Shape 생성을 수행 중이다. node16 공식 RGB1 source-M0 surface scorer도 **501개 공통 성공 사례**의 raw/+M0 점수와 `complete.json`을 작성했다. 공식 export bridge는 4 worker로 첫 **4/501** GLB를 통과했고 외부 7방법 metric 자식과 동시에 살아 있다. 공식 10지표 및 최종 Ours 공통 표는 여전히 미완료다.
- 로컬 disjoint-tail 감독을 새 v2 경로에서 재개했다. GPU6은 index 489부터, GPU7은 index 478부터 원래 GPU0–3 작업과 겹치지 않는 뒤쪽 사례를 처리하며, 공식 export가 닫히면 공식 로컬 렌더 watcher를 자동 재개한다. GPU6의 index495 Selected 복구는 별도 worker로 진행 중이다. node16 공식 export는 **8/501**, GPU7 사용 메모리 약 **11.4 GB/97.9 GB**였고 외부 metric 자식도 살아 있다.
- 최종 주 비교표 자동 생성기는 사용자 요청에 맞춰 새 Ours Selected·V1·공식 RGB1 행을 **표 상단**에, 기존 외부 7방법을 그 아래에 두도록 정리했다. 10개 metric 각각 전체 방법 중 최고값 `\textbf{}`와 2위 `\underline{}` 표시 로직은 그대로 유지한다. 값은 최종 공통 N의 실제 evaluator 결과에서만 채운다.

## 06:41 KST 계획·실행 재점검

- 사용자 승인 범위는 계속 **재학습 없음, frozen V1·Selected만 TEST508 신규 생성, 동일 최종 성공 교집합에서 외부 7법·공식 RGB1과 10지표 비교**이다. 기존 M1-scale 내부 ablation과 새 GT-scale-free 주 비교를 혼합하지 않는다. 최종 비교의 원본 source M0 Sim(3)은 생성 이후에만 적용하며, 정답 target mesh는 정합 입력에 사용하지 않는다.
- Ours 파일서버4 checksum archive는 V1 Shape **99/508**, Selected Shape·Top-1 PBR **94/508**이다. GPU0–3의 4개 shard worker와 GPU6·7의 겹치지 않는 tail worker가 실제 Shape/PBR/export 자식을 실행 중이다. GPU6의 index495 별도 복구 worker는 V1 5M render GLB 보관을 마쳤고 Selected 250k·500k 실패 후 1M fallback을 처리 중이다. 이 사례의 Selected 보관 전에는 전체 완료로 세지 않는다.
- 노드16 공식 RGB1은 TEST508 **507 성공·1 native 실패**로 생성 시도를 닫았다. 공식 source-M0 surface 501사례는 완료됐고, 250k 우선 GLB 내보내기는 **48/501** 결과 파일을 생성했다. 외부 7법 10지표 evaluator는 살아 있으며 현재 첫 방법 결과 1개를 기록했다. 공식 export와 외부 learned metric은 GPU7에서 병행되고 있다.
- 주 cascade와 노드16 exporter의 최신 worker 로그에서 새 traceback/OOM은 발견되지 않았다. 로컬 `/root/dev` 가용 공간은 약 **1.1 TiB**, node17 GPU4·5의 다른 사용자 작업은 그대로 두었다. 이 시점에는 최종 공통 N·10지표 표·paired CI·새 teaser/PDF가 없으므로 논문 성능 claim을 갱신하지 않는다.
- 현재 병목은 Ours 409여 쌍의 남은 Shape/PBR/export와 뒤따를 공통 렌더·learned metric이다. 초반 생성 속도를 단순 연장하면 cascade에 수 시간이 더 들지만, 대형 mesh fallback과 공식 렌더로 GPU6·7 배정이 바뀌므로 **확정 종료 시각은 아직 제시하지 않는다**. 다음 완료 기준은 index495 양 arm archive, 공식 501 export, 외부 7법 평가, Ours 508쌍 archive 순서다.

## 06:46 KST 원고 조건 분리 및 조판 검증

- 원고 `sections/experiments.tex`의 기존 TEST 프로토콜을 **internal oracle-scale** 범위로 명시하고, `sections/unified-evaluation-protocol-draft.tex`의 camera-given·GT scale 미제공·source-M0-only Sim(3) 공통 벤치마크 계약을 본문에 실제 연결했다. 기존 표의 native renderer/FID pool은 새 Cycles 표와 비교 불가함을 명시했다.
- `sections/results.tex`는 새 공통 표와 raw/+M0 paired target8 표가 생성되면 **주 비교로 먼저** 포함하고, 이전 M1-derived-scale TEST 표는 뒤의 internal subsection에 둔다. 아직 결과 파일이 없을 때는 보류 문단을 표시하며 oracle-scale 수치를 대체하지 않는다. 초록·결론도 공통 결과 문장 파일이 마련되기 전에는 cross-method gain을 주장하지 않도록 잠갔다.
- 최신 TeX source를 기존 검증된 figure/typeset snapshot에 덮어 씌워 Tectonic으로 실제 조판했다. **25쪽 PDF 생성 성공**, 미해결 인용/참조·Overfull·fatal error 0개, 1쪽에 `Unified evaluation in progress`가 표시되고 unified protocol/result section은 7쪽에 등장한다. 이 파일은 내부 preview이고 최종 PDF로 배포하지 않는다. 기존 `figures/deformnext-construction.pdf`는 provenance가 있는 이전 build source에서만 preview에 복사했다.
- 실행 상태 재확인: Ours V1 **107/508**, Selected **103/508** archive. 대형 index495의 Selected 250k·500k·1M·2M 렌더 충실도 검사는 실패해 5M fallback 전 단계까지 확인됐다. node16 공식 export **81/501**, 외부 learned 10지표 evaluator는 첫 방법 결과 1개 후 계속 실행 중이다. 현재 최종 공통 지표는 없다.

## 06:49 KST overview figure 교체

- 기존 추상 사각형 중심 overview를 실제 SHA 검증된 target RGB 1장·source RGB 3장의 같은 crop과, source-camera ray가 공유하는 SS voxel posterior 도식, frozen Shape와 별도 PBR 경로로 다시 그렸다. 입력 이미지는 구 oracle-scale teaser의 **입력 사진만** 재사용하며 예측 결과나 점수는 가져오지 않았다. [재생성 코드](../paper/draw_pipeline_overview_v2.py)와 입력 SHA provenance를 남겼다.
- 새 `paper-overall.pdf`를 원고에 넣어 Tectonic cached build 25쪽을 다시 조판했고, 3쪽 전체 페이지 렌더를 육안 검수했다. 그림/캡션 잘림, 미해결 참조, Overfull, fatal error는 없었다. 최종 teaser는 여전히 새 공통 성공 집합의 선정·렌더를 기다린다.
- 실행 스냅샷: Ours V1 **110/508**, Selected **106/508** checksum archive, node16 공식 RGB1 렌더 mesh export **95/501**. 외부 learned evaluator PID 2492093은 계속 실행 중이다. index495 Selected의 5M fallback은 아직 완료 영수증이 없다.

## 06:50 KST 새 조건 실시간 ledger

- [GT-scale-free TEST508 실시간 진행표](2026-10-28-scale-ungiven-live.md)를 `watch_scale_ungiven_live.py`로 연결했다. 120초 간격으로 Ours V1·Selected·Top-1 PBR의 실제 file4 archive receipt, paired 사례 수, node16 공식 export·외부 learned result 개수, 후속 surface/render/metric/table/paired-CI 완료 영수증을 갱신한다. 관측 실패를 학습 실패로 표시하지 않으며, paired-CI 영수증까지 실행한다. 현재 watcher session 86298이 살아 있고 첫 poll은 V1 **112**, Selected **107**, 공식 export **106/501**, 외부 learned result **1/7**을 기록했다.

## 06:52 KST 세부 모듈 그림 재구성

- 이전 큰 사각형·흐름도형 matching/retargeting 그림을 각 **원 컬럼**에 맞는 새 Figure 3/4로 교체했다. Figure 3은 SHA 검증된 실제 target/source RGB를 배경으로 쓰되, 점·posterior 행렬은 **개념 도식**이라고 그림과 캡션에 명시한다. Figure 4는 한 ray의 SS grid 공유 posterior와 native DINO 값→source contrast→bounded correction→MV4 carrier 경로를 보여준다. [재생성 코드](../paper/draw_detail_figures_v2.py)를 추가했다.
- 수정 원고를 cached Tectonic으로 재조판했고 25쪽이며, 3·4쪽을 실제 PDF 크기로 열어 세 그림/캡션의 배치·글자·클리핑을 확인했다. 미해결 인용/참조, overfull, fatal error가 없다. 최종 논문용 teaser만 공통 새 렌더 이후 교체해야 한다.

- 그림 재생성 입력이 나중에 교체할 구 teaser provenance를 참조하지 않도록, 실제 입력 PNG 4장(합계 약 0.5 MB)을 `paper/figures/overview-inputs/`에 SHA 그대로 묶고 `overview-input-provenance.json`을 별도로 만들었다. 두 그림 생성 스크립트가 상대 경로와 절대 경로를 모두 SHA 확인하도록 수정해 새 공통 teaser를 교체해도 Figure 2–4를 같은 픽셀로 재생성할 수 있다.

## 06:55 KST 공정 teaser 생성기 준비

- 최종 공통 N의 V1/Selected LPIPS·target M-IoU에서 사전 고정한 선택 규칙으로 sample ID를 정하는 `select_scale_ungiven_teaser_case.py`에 이어, [실제 fair-protocol teaser 생성기](../../../experiments/deformnext_retargeting/build_scale_ungiven_teaser.py)를 작성했다. 두 arm/공식 RGB1의 동일 ID·동일 Cycles camera8/9 `source_m0` 렌더와 M1 GT·입력 RGB를 모으고, 각 렌더 영수증의 `output_sha256`·protocol SHA를 검증한다. 결과는 같은 display crop을 쓰는 target/source 두 줄과 선택·입력·렌더 provenance JSON으로 나온다.
- index11 pilot 이미지로 **레이아웃만** 실제 PNG를 사용해 시험하고 시각 검수했다. 이 smoke에서 GT/공식 자리에는 임시 V1 이미지를 둔 것이므로 논문 결과가 아니며, 새 최종 teaser/PDF에 복사하지 않았다. 최종 13조건 공통 성공 집합·10지표·선정 영수증이 나올 때만 생성기를 본 실행한다.
- 06:55 KST 진행률은 Ours V1 **117/508**, Selected **111/508**, 공식 RGB1 render mesh **132/501**, 외부 learned evaluator 결과 **1/7**이다. index495 Selected 5M fallback은 살아 있는 CPU 작업으로 계속 처리 중이다.

- Teaser 생성기는 선택 영수증뿐 아니라 **최종 common-manifest SHA, V1/Selected source 평가 결과 두 파일 SHA, target8 평가 SHA**도 요구하도록 강화했다. 결과 선정과 다른 모집단·점수 파일을 뒤섞어 그림을 만들 수 없게 했다. node16의 실제 index11 M1 GT camera8와 camera9 원본 PNG·render receipt를 읽기 전용으로 가져와 protocol/output SHA 대조를 통과했다. 예측·공식 final render는 아직 생성 전이므로 본 teaser는 미완료다.

## 06:57 KST 전체 TEST 입력·scale 계약 감사

- [전체 계획·현재 생성 audit](2026-10-28-test508-image-scale-contract-audit.json)를 추가했다. TEST508 전 사례의 V1/Selected plan 쌍과 508개 image-only scale prediction을 대조했다. 두 arm은 같은 네 RGB/A 입력·camera ID·conditioning input SHA와 같은 예측 scale을 사용하며, `M1_derived_scale_supplied_to_generation=False`다. 현재 file4 archive가 닫힌 V1 **119**, Selected **114**의 실제 `generation.json`도 예측 scale SHA·값·policy를 일치시켰다. Selected Top-1 PBR plan 508개도 같은 image-only scale provenance와 RGB 경로를 참조한다. 전체 완료 플래그는 **false**로 유지하고 나머지 생성 후 같은 감사기를 재실행한다.

## 06:59 KST 계획·실행 중간 재점검

- 새 조건 주 비교에는 고정된 **V1·Selected 두 arm만** TEST508에서 다시 생성한다. 508개 입력 계획의 camera·네 RGB/A·image-only scale 동일성이 전수 확인됐다. 외부 7법, 공식 Pixal3D RGB1, 두 Ours arm을 최종 동일 성공 교집합과 동일 renderer/metric에서 평가하고, source-M0-only Sim(3)은 생성 후에만 적용한다. 기존 M1 유래 scale 표와 DEV ablation 표는 별도 내부 근거로 둔다. 재학습 계획은 없다.
- 실시간 ledger 기준 file4 SHA 보관 완료는 V1 **121/508**, Selected **117/508**, 두 arm 동일 사례 **117/508**이다. 공식 RGB1 source-M0 surface는 **501/501**, 렌더 GLB export는 **155/501**이다. 외부 7법 learned metric 결과는 **1/7**이고 두 번째 방법 evaluator 자식은 CPU·I/O 작업을 계속한다. 따라서 공통 10지표, paired CI, 최종 teaser/PDF는 아직 없다.
- 로컬 GPU 0–3과 disjoint GPU 6–7 worker는 실행 중이다. 일부 순간 GPU 사용률이 0인 것은 GPU 추론이 끝난 뒤 CPU에서 GLB 단순화·bake와 file4 checksum 전송을 처리하기 때문이다. index495 Selected의 5M-face fallback은 CPU 사용률 약 740%, RSS 약 14.8 GB로 계속 동작 중이며 완료로 세지 않는다. node16 공식 exporter 4 worker와 외부 metric evaluator도 실행 중이다. 양쪽 실패 영수증이나 OOM은 새로 확인되지 않았다.
- 최근 paired archive 증가율을 단순 외삽하면 **남은 Ours cascade만 약 6–9시간**이지만, 대형 mesh 및 이후 공통 렌더·평가에는 별도 시간이 든다. 이 범위는 현재 처리량에 근거한 계획용 추정이며 최종 논문 종료 시각이 아니다. 공식 export는 현재 처리량이면 약 1시간 더, 외부 learned evaluator는 결과 1/7만 나와 종료시간을 아직 신뢰성 있게 추정할 수 없다. 다음 gate는 508쌍 SHA 보관, 공식 501 export, 외부 7법 metric, 전체 공통 렌더/10지표, paired CI, 실제 teaser와 최종 PDF 순서다.
- 공정 비교 표와 raw/+M0 표를 받을 원고 section, 입력 사진 기반 overview·one-column 모듈 그림, 최종 teaser 생성기, 공통 결과가 모두 있어야 조판되는 `build_fair_manuscript.py`를 준비했다. 현재 25쪽 preview는 protocol·그림·레이아웃 검사용이며 최종 성능 PDF로 배포하지 않는다.

## 07:01 KST teaser 캡션 조건 교정

- 원고의 teaser 그림 자리에 최종 공정 렌더를 교체하더라도 기존 `M1-derived-scale` 내부 teaser 설명이 그대로 남는 결함을 확인했다. `build_scale_ungiven_teaser.py`가 같은 최종 공통 N·camera8/9·source-M0 정합·선정 규칙을 설명하는 `paper-teaser-caption.tex`를 실제 teaser PDF/PNG·SHA provenance와 함께 생성하도록 수정했다. 원고는 이 캡션이 있을 때만 새 설명을 사용하고, 최종 `build_fair_manuscript.py`는 공통 N과 캡션 내용이 맞는지 필수 gate로 검사한다.
- Python 구문·`git diff --check`를 통과했다. 공정 캡션 분기의 Tectonic 조판은 **임시 smoke 디렉터리**에서 성공했다. 임시 캡션과 이전 teaser 이미지를 섞은 smoke PDF는 논문 산출물이 아니며 paper/reports에 복사하지 않았다. 최종 figure/caption은 13조건 공통 결과와 실제 렌더 SHA가 닫힌 뒤 생성한다.
- 새 `draw()`로 임시 501사례 캡션·PDF를 실제 생성하고, 캡션의 공통 N/정합 문구/개행을 확인한 뒤 그 **생성된 캡션**을 임시 원고에 넣어 Tectonic 조판했다. 25쪽 생성과 PDF 텍스트 내 새 캡션 존재를 확인했다. 이 테스트의 임시 이미지는 실험 결과로 사용하지 않는다.

## 07:03 KST 후속 평가 바인딩 점검

- node16의 실제 index11 한 사례 surface pilot 2행과 file4에 보관된 Ours 12개 raw/+M0·camera8/9/10 렌더 job/receipt를 대조했다. 각 작업의 `metric_mesh_sha256`은 해당 V1/Selected surface 점수의 원본 prediction SHA와 같고, `protocol_sha256` 및 렌더 PNG의 실제 SHA도 영수증과 일치했다(**12/12, 오류 0**). 이는 후속 공통 평가 입력 경로의 기계적 점검이며 최종 성능 표본은 아니다.
- 후속 evaluator는 Ours surface·렌더를 기존 external common502에서 생성한 뒤, 공식 RGB1이 성공한 subset을 받아 **최종 공통 N(현재 예상 501)**으로 V1/Selected 네 조건과 외부 7법·공식 2조건을 다시 바인딩한다. 코드의 `--expected-samples`와 실제 final subset 경로를 확인했다. FID/KID와 source 지표는 이 최종 교집합에서 산출하므로 502 전체의 기존 점수를 섞지 않는다.

## 07:04 KST 대형 사례 복구와 이전 큐 정리

- index495의 Selected Top-1 PBR 18,849,746-face 원본은 250k/500k/1M/2M 렌더용 bake가 기하 fidelity gate를 통과하지 못했으나, **5M faces**에서 원본 대비 world-frame q99 **0.001874**로 0.01 기준을 통과했다. `benchmark.json`과 V1·Selected 파일서버 SHA archive 영수증, 두 arm 사례 완료 영수증을 확인했다. 로컬 대형 GLB는 보관 후 공간 절약 정책에 따라 제거됐으며 file4 보관본이 정본이다. 대형 사례 495를 이제 완료 개수에 포함할 수 있다.
- 4일 전 semantic-topk 실험의 유휴 큐 PID 498146(GPU6/7)·509860(GPU0–3)는 각각 자식 작업 0개, `running` 기록 0개였고 현재 frozen TEST508 watcher와 무관하다. 예전 학습 중단·추가 재학습 금지 지시와 GPU 충돌 방지에 맞춰 두 큐에 SIGTERM을 보내 종료를 확인했다. 현재 Ours 4+2 GPU worker와 node16 official exporter/metric evaluator는 계속 살아 있으며 현재 캠페인의 결과 파일은 건드리지 않았다.

## 07:06 KST 공식 RGB1 직접 대조 통계 연결

- 새 주 비교표의 점추정 순위만으로 Ours가 공식 Pixal3D RGB1보다 안정적으로 나은지 판정할 수 없으므로, **동일 최종 공통 사례**에서 공식 given-camera RGB1 대비 V1 및 Selected의 source-cluster paired bootstrap을 추가했다. 각 대조는 LPIPS/DreamSim/CLIP/CD/F-Score와 target8 PSNR/SSIM/M-IoU 8개이며, 두 대조×8지표 **16-endpoint Bonferroni 95% family**를 사전에 명시한다. FID/KID는 분포 점추정으로만 둔다. 이 직접 대조는 공식 T1+C와 Ours T1+S3+C의 완성 시스템 차이이며 개별 모듈 효과로 해석하지 않는다.
- [계산기](../../../experiments/deformnext_retargeting/score_scale_ungiven_official_contrasts.py)는 공식·Ours evaluator 결과 3개, raw/+M0 target8 결과, 최종 13조건 공통 manifest의 ID·source-cluster·seed0·평가 완료 여부를 대조한다. 4개 synthetic 사례의 smoke에서 두 대조×8지표·보정 구간 계산을 통과했다. 실제 TEST 수치가 아니다.
- [후속 watcher](../../../experiments/deformnext_retargeting/orchestrate_scale_ungiven_official_contrasts.py) PID **661757**을 시작했다. 기존 V1–Selected paired CI 영수증이 닫히면 node16의 9방법 원본 결과와 로컬 Ours 4방법 결과를 가져와 직접 대조를 계산하고, **각 endpoint의 평균이 이미 작성된 10방법 주 표의 셀과 일치하는지** 확인한 뒤에만 완료 영수증을 남긴다. 현재 `WAIT_OURS_COMMON_PAIRED_GATE`, 상위 PID588783 생존을 확인했다.
- 직접 대조 16행을 원고 Supplement에 자동 반영하도록 TeX composer와 조건부 입력을 추가했다. 최종 manuscript builder는 공식 직접 대조 JSON·16행 표를 필수로 요구하며, 그 평균을 주 표 셀과 다시 확인한다. 임시 4사례 JSON으로 TeX composer를 실행하고 별도 smoke 원고에서 25쪽 Tectonic 조판을 통과했다. 실제 결과 아님을 분리했다. 대기만 하던 이전 watcher PID661757을 종료하고 **PID662987**의 v2 watcher로 교체했으며, 최종 paired CI 후 실제 16행 TeX까지 자동 작성한다. `WAIT_OURS_COMMON_PAIRED_GATE`와 상위 PID588783 생존을 확인했다.
- node16 evaluator 결과 JSON을 후속 watcher가 사용하는 정확한 `rsync` wildcard 형식으로 실제 CorrAdapter 1개 파일을 임시 캐시에 가져와 패턴 확장·전송을 확인했다. 최종 `nine` 결과는 아직 없으므로 현재 파일을 공식 직접 대조에 사용하지 않는다. 07:07 KST 실시간 진행표는 Ours V1 **134/508**, Selected **133/508**, 공식 render mesh **195/501**, 외부 learned 결과 **1/7**이다.

## 07:09 KST 최종 teaser 산출 경로 연결

- 최종 공통 paired CI·주 표·공식 501 렌더·Ours 502 렌더 영수증이 모두 닫힌 뒤에만 사례를 고르는 [teaser watcher](../../../experiments/deformnext_retargeting/orchestrate_scale_ungiven_final_teaser.py)를 추가하고 PID **664545**로 시작했다. 현재 `WAIT_FINAL_COMMON_PAIRED_FOR_TEASER`이며 선행 PID588783 생존을 확인했다.
- 선정은 이미 고정한 source LPIPS/target mask-IoU paired median 규칙으로 수행한다. 선택 ID가 최종 13조건 공통 manifest에 속하는지, Ours 결과 SHA·target8 결과 SHA·실제 파일서버 렌더 8장의 PNG/receipt/protocol SHA와 최종 teaser PDF SHA를 검사한 뒤 그림·캡션·provenance 네 파일을 원고에 복사한다. 이전 내부 oracle-scale teaser 네 파일은 watcher 전용 캐시에 보관한다. 현재 로컬의 508×4 입력 RGB 경로 **2032/2032** 존재와 file4 SSH control master 생존을 확인했다.
- 아직 10지표/선정 결과가 없으므로 teaser watcher가 만든 실제 그림은 없고, 기존 내부 그림도 손대지 않았다. 현 상태 07:09 KST Ours V1 **140/508**, Selected **135/508**, 공식 렌더 GLB **204/501**, 외부 learned evaluator 1/7이며 결과 프로세스는 살아 있다.
- file4 직접 SSH control master는 현재 살아 있지만 2시간 유휴 만료 설정이 있어 최종 렌더 이후 teaser 전송 시점에 끊길 수 있다. node16에서 동일 file4의 실제 Ours pilot 렌더 PNG/receipt를 읽고 로컬로 rsync한 뒤 protocol/PNG SHA 검사를 통과했다. 그림 watcher가 node16의 `/file4` mount를 통해 실제 file4 결과를 가져오도록 바꾸고, 아직 대기 중이던 PID664545를 종료해 **PID666141, `fair-final-teaser-v2`**로 재시작했다. Ours/공식 렌더 byte 자체는 바뀌지 않는다.

## 07:11 KST 생성 입력·scale 증분 재감사

- [새 전수 계획·현재 산출물 감사](2026-10-28-test508-image-scale-contract-audit-0711.json)를 같은 고정 캠페인에 실행했다. 508개 paired plan의 동일 RGB/카메라/image-only scale, M0/M1 mesh·GT scale 추론 미사용을 다시 확인했다. 새로 보관된 V1 **144**, Selected **140** 실제 generation 기록 284개와 Selected Top-1 archive 140개가 해당 scale provenance를 모두 통과했다. `all_cascade_results_complete=false`로 남은 사례의 품질·완료는 주장하지 않는다.
- 07:12 KST 이미 성공한 **Ours 렌더 GLB 285개**의 budget/q99를 전수 집계했다. 250k/500k/1M/5M face가 각각 **258/15/10/2개**, 원본 표면과의 world-frame q99 최대 **0.009776**, 0.01 초과 **0개**다. node16 공식 RGB1 GLB **222개**는 250k/500k/1M/2M이 **217/3/1/1개**, q99 최대 **0.009843**이다. 이 값은 렌더용 근사 fidelity 검사이며 GT 대비 모델 품질 지표가 아니다. 대형 fallback 2개만으로 전체 face budget이나 cascade 해상도를 변경하지 않는다.

## 07:13 KST 계획·실행 재점검

- 계획은 변경하지 않는다. 동결 V1·Selected의 TEST508 추론, 공식 RGB1 및 외부 7법의 공통 성공 사례 source-M0 정합, 동일 렌더·10지표 평가, paired CI, 최종 teaser·논문 순서다. 새 학습은 없다.
- 실시간 file4 SHA 영수증은 V1 **145/508**, Selected **141/508**, 동일 사례 paired **141/508**이다. 공식 RGB1 source-M0 surface **501/501**, render GLB export는 조회 순간 **232/501**이었다. 외부 7법 learned metric 완료는 **1/7**이고 다음 evaluator PID2492093은 CPU 실행 및 읽기 바이트 증가를 확인했다. unified nine-method와 Ours final metric은 선행 단계 대기 중이다.
- 로컬 0–3은 cascade/mesh export, 6–7은 겹치지 않는 뒤쪽 사례를 맡는다. GPU 사용률 0%인 순간은 CPU mesh export·SHA 전송 단계와 일치한다. node16 GPU7의 공식 export 및 evaluator도 실행 중이다. 현재 새 OOM/실패 영수증은 관측되지 않았다. 공통 N, 10지표 우열, 최종 논문 PDF는 아직 확정하지 않는다.

## 07:14 KST GPU3 worker 복구

- 재점검에서 GPU3 대체 worker PID594305가 종료된 것을 발견했다. 원인은 index111 V1의 렌더용 fast simplifier가 목표 100만 face에 대해 실제 1,441,854 face를 내어 budget 검사를 통과하지 못한 것이다. Shape 생성 자체가 실패한 것이 아니고, 기존 완료 사례와 파일서버4 archive는 유지됐다.
- 해당 worker는 06:26 KST의 200만/500만 face fallback 코드 수정 전에 시작돼 구버전 예외 처리로 종료했다. 현재 소스는 이미 저장된 25만/50만/100만 실패 로그를 재사용해 200만 face부터 다시 시도한다. 같은 rank3을 세션 32381, PID669224로 재시작했으며 index11의 별도 복구 완료 사례는 명시적으로 건너뛴다. 첫 재실행에서 index3–107의 기존 archive를 재확인하고 index111의 200만 face export 자식 PID669227이 실행 중임을 확인했다. 해당 사례의 최종 SHA 영수증은 아직 대기 중이다.

## 07:18 KST worker 생존 감시·평가 진행

- 실시간 진행표가 archive 수만 보여 주어 GPU3의 중단을 늦게 발견한 문제를 수정했다. `watch_scale_ungiven_live.py`가 `/proc`의 실제 비좀비 worker 명령줄에서 primary rank 0–3의 PID를 표시하고, cascade 미완료 중 rank가 없으면 주의 문구를 쓴다. 단발 smoke에서 **0:586071, 1:586072, 2:586073, 3:669224**를 확인했다. 오래된 감시 PID648732을 종료하고 같은 120초 주기로 세션 **42001**을 시작했다.
- 07:18 KST ledger는 V1 **150/508**, Selected **149/508**, paired **149/508**, 공식 RGB1 render mesh **264/501**이다. node16 외부 learned evaluator가 두 번째 방법 결과를 완료해 **2/7**로 증가했다. GPU3 index111의 200만 face export는 PID669227에서 계속 실행 중이며, 이 사례의 완료로 아직 세지 않는다.

## 07:22 KST index111 V1 복구 검증

- GPU3 index111 V1의 200만 face export가 통과했다. 원본 표면 **14,559,544 faces**, 렌더용 **2,000,000 faces**, world-frame registration q99 **0.0039505**로 사전 한도 0.01 이내다. 원본 metric surface는 건드리지 않았고 `benchmark.json` 및 file4 checksum `archive-receipt.json`의 PASS를 직접 읽었다. rank3 PID669224는 같은 index의 Selected Shape 생성 자식으로 넘어갔으며, 두 arm paired 완료는 아직 기다린다.

- 07:25 KST 같은 index의 Selected Shape와 Top-1 PBR 재생은 통과했다. Top-1 렌더용 GLB의 25만·50만 face 시도는 q99 geometry 한도 때문에 재시도했고, worker가 100만 face 시도를 실행 중이다. 이 단계에서 원본 metric용 surface와 Top-1 PBR geometry identity는 보존한다. 완료 영수증이 닫히기 전까지 이 index를 paired 완료로 세지 않는다.

## 07:27 KST GPU2 worker 복구

- worker 생존 감시가 rank2 PID586073의 zombie 상태를 잡았다. index126 V1의 fast simplifier가 100만 목표에 **1,157,682 faces**를 내어 구버전 worker가 종료한 것이 원인이다. 원본 Shape NPZ는 남아 있고 partial `benchmark.glb/json`은 없어 재시도 가능한 상태를 확인했다.
- 현재 fallback 소스로 rank2를 세션 **68264**, PID**676710**에서 재시작했다. index2–122의 기존 두-arm file4 영수증을 재확인하고, index126의 실패한 25만/50만/100만 예산 로그를 건너뛰어 200만 face export 자식 PID676713으로 진입했다. primary rank 0–3의 `/proc` 실체는 다시 모두 확인됐다. GPU3 index111 Selected의 추가 렌더 budget 재시도는 별도로 계속 중이다.

## 07:31 KST GPU0 worker 복구

- 동일 감시가 rank0 PID586071의 zombie 상태를 잡았다. index112 Selected Top-1 렌더 GLB에서 fast simplifier가 100만 목표 대비 **1,408,566 faces**를 내어 구버전 worker가 종료했다. Top-1 생성과 이전 archive는 보존됐고 partial GLB/report가 없는 것을 확인했다.
- rank0을 현재 fallback 소스로 세션 **3250**, PID**678576**에서 다시 실행했다. index0–108의 기존 paired archive를 재확인하고 index112의 200만 face export 자식 PID678579로 진입했다. 이 시점 primary 0–3은 모두 실제 비좀비 프로세스로 확인됐다. GPU1은 아직 원래 worker이며 같은 종료 증상이 나오는지 감시한다.

- 07:34 KST rank2 index126 V1의 200만 face 렌더 GLB가 PASS했다. 원본 **14,279,878 faces**, 렌더용 **2,000,000 faces**, world-frame q99 **0.0034417**로 0.01 이내이며 file4 SHA 영수증이 닫혔다. 같은 index의 Selected는 아직 후속 중이다. rank0 index112 Selected와 rank3 index111 Selected의 200만 face export는 별도 자식에서 계속 실행 중이다.

- 07:37 KST rank2 index126은 Selected Top-1 PBR 및 Shape까지 file4 SHA 영수증이 모두 닫혀 **두 arm paired 완료**가 됐다. Selected는 원본 **10,838,086 faces**에서 렌더용 **250,000 faces**, q99 **0.0097573**으로 통과했다. 같은 사례 V1은 200만 face가 필요했으므로 렌더 budget을 arm 사이에 강제로 동일하게 하지 않고 각각의 원본 대비 fidelity 기준 0.01을 적용한다. geometry metric은 두 경우 모두 원본 surface를 사용한다. rank2는 다음 index 생성으로 넘어갔다.

## 07:38 KST 대형 mesh 실행 제한 조정

- rank3 index111 Selected의 200만 face export가 11분 넘게 CPU 계산을 계속하고, rank0 index112 Selected도 6분 넘게 계산 중이다. 둘 다 PID/RSS/CPU 생존을 확인했고 완료 영수증은 아직 없다. 200만 예산의 기존 20분 subprocess 제한은 큰 mesh의 정상 처리를 잘라낼 수 있어, **앞으로 시작하는 worker**의 200만·500만 예산 제한을 60분으로 통일했다. `py_compile`을 통과했다. 이미 실행 중인 자식의 제한은 바뀌지 않으므로 두 사례는 기존 20분 안에 끝나는지 계속 감시한다. 품질 한도와 render face budget 순서는 변경하지 않았다.

- 07:40 KST rank3 index111 Selected의 200만 face export도 완료됐다. 원본 **14,754,960 faces**, 렌더용 **2,000,000 faces**, q99 **0.0036792**로 한도 안이며 Selected Top-1 PBR·Shape의 file4 SHA 영수증이 모두 PASS했다. 따라서 index111은 V1/Selected paired 완료로 확정했다. rank3은 다음 사례로 계속 진행한다.

- 실시간 진행표의 종료 조건을 Ours paired CI 단독에서 **Ours paired CI + 공식 RGB1 paired CI + 공정 teaser** 세 영수증으로 연장하고, 마지막 두 gate를 표에 추가했다. 단발 실행에서 세 행과 rank0–3 생존 판별을 확인한 뒤 감시 프로세스를 세션 **99709**로 교체했다. 07:40 KST ledger는 V1 **167/508**, Selected **165/508**, 공식 render mesh **384/501**, 외부 learned 결과 **2/7**이다.

- 07:41 KST rank0 index112 Selected도 200만 face export와 file4 SHA 보관을 끝냈다. 원본 **26,603,938 faces**, 렌더용 **2,000,000 faces**, q99 **0.0047466**이며 worker가 `PASS_TEST_CASE_TWO_ARMS_ARCHIVED`를 출력했다. rank0은 다음 사례로 계속 진행한다. 3개 rank의 old-code fast simplifier 종료 원인과 재실행은 이제 각 문제 사례의 paired 완료로 닫혔다.

## 07:42 KST 후속 평가 연결 점검

- node16 공식 export bridge PID3135838의 실제 코드를 다시 확인했다. 501사례 export와 외부 7법 metric 완료를 둘 다 확인한 뒤에만 검증된 250k 우선 출력으로 기존 경로를 연결한다. 로컬 GPU6/7 tail supervisor PID638206은 그 연결을 보고 작업을 마친 뒤 멈춰 둔 공식 렌더 watcher PID543994를 재개한다. 현재 공식 렌더의 `T` 상태는 이 의도된 GPU 인계 gate이며 deadlock으로 볼 증거가 없다.
- 120초 ledger의 새 poll은 V1 **171/508**, Selected **166/508**, paired **166/508**, 공식 render mesh **392/501**, 외부 learned **2/7**이다. primary 0–3은 모두 실제 프로세스가 살아 있다. 10지표·공통 N 및 최종 PDF는 선행 작업이 끝나지 않아 아직 주장하지 않는다.

## 07:43 KST 최종 원고의 결과 해석 gate

- 기존 unified Results에는 표 두 개만 자동 삽입하고 결과 해석 문단의 필수 조건이 없어, 수치가 나와도 주장을 검토하지 않은 원고가 최종 조판될 수 있었다. `sections/results.tex`에 `unified-results.tex`, `sections/discussion.tex`에 `unified-discussion.tex` 입력 자리를 추가하고, 최종 빌더가 두 파일과 기존 unified Abstract·Conclusion을 모두 요구하도록 했다. 최종 수치를 보기 전이라 네 결과 문장은 아직 만들지 않는다.
- 현 초안의 조건부 입력이 깨지지 않는지 별도 새 snapshot에서 cached Tectonic으로 25쪽 preview를 재조판했다. 종료 코드 0, fatal/overfull/undefined 인용·참조 0이었다. 이 preview는 결과 미완료 원고이며 최종 PDF가 아니다. 조립 runbook에도 새 파일 네 개의 작성 요구를 기록했다.

## 07:46 KST 마지막 구버전 GPU1 worker 복구

- rank1 PID586072가 index117 Selected Top-1 렌더용 fast simplifier에서 100만 목표 대비 **1,502,797 faces**를 내어 종료했다. 원본 생성·이전 archive는 남아 있고 해당 사례의 partial GLB/report가 없는 것을 확인했다. 같은 원인은 앞서 rank0/2/3에서 복구한 100만 예산 실패의 구버전 예외 처리다.
- rank1을 현재 200만/500만 재시도 및 60분 제한 소스로 세션 **79910**, PID**687195**에서 재시작했다. index1–113 기존 paired 영수증을 재확인하고 실패한 25만/50만/100만 로그를 재사용해 index117의 200만 export 자식 PID687208로 진입했다. `/proc` 캠페인/PID 검사에서 primary 0–3은 모두 살아 있고 200만/500만 fallback을 로드했다. 60분 제한은 새 rank1과 이후 시작하는 worker에 적용되며, 먼저 재시작한 rank0/2/3에는 기존 20분 제한이 남아 있다. index117 paired 완료는 아직 기다린다.

## 07:48 KST 외부 metric 생존 감시

- 외부 learned 결과가 2/7에서 수십 분 동안 머무르는 상황을 구분할 수 있도록 실시간 ledger에 node16의 정확한 `M0Sim3-metrics` 평가 자식 PID/상태/누적 읽기 바이트를 추가했다. node16의 특정 `source-sim3-final-evaluation-v6/evaluation` 출력 인자와 실행 파일 이름을 대조하므로 질의 프로세스를 잘못 세지 않는다. 단발 실행은 PID**2492093**, `R`, **64.7 GiB read**와 primary rank 0–3 실체를 확인했다. 평가 자식이 없고 완료 영수증도 없으면 주의 문구를 낸다.
- 감시 프로세스를 새 코드로 세션 **78239**에 재시작했다. 07:48 KST 검증 영수증은 V1 **176/508**, Selected **173/508**, paired **173/508**, 공식 render mesh **430/501**, 외부 learned **2/7**이다. 이 I/O 수치는 평가 진행의 생존 증거이지 성능 수치가 아니다.

## 07:52 KST 공통 지표·표 결합 재감사

- 최종 주 표 composer와 Ours evaluator 연결을 다시 확인했다. 9방법 평가 완료 후 Ours 4조건을 같은 `binding-common-thirteen`에 결합하며, 주 표는 두 evaluator의 `protocol_sha256`·`reference_manifest_sha256`·공통 사례 수 일치, 각 방법의 성공 예측 수 (N), `full_metric_computation`, FID/KID에 쓰인 이미지 수 (2N)을 요구한다. target8 raw/+M0 파일도 모두 같은 (N)이어야 한다. 따라서 현재 외부 502사례 임시 결과의 FID/KID를 공식 501 또는 최종 공통집합 표에 그대로 복사하는 경로는 없다.
- node16 외부 evaluator는 CorrAdapter·MVAdapter 두 결과를 작성했고, 다음 작업 중 PID2492093의 누적 읽기 바이트가 07:48 KST 약 69.6 GB에서 07:49 KST 약 70.1 GB로 늘었다. GPU1 index117의 200만 export 자식도 계속 CPU를 사용한다. 최종 공통 분모와 metric 점수는 아직 없다.

## 07:53 KST 공식 RGB1 렌더 mesh 증분 감사

- node16의 공식 RGB1 exporter가 작성한 **449개 `prediction.json`**을 직접 집계했다. 모두 `PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_RENDER_EXPORT`, 실제 render face budget은 25만 **440**, 50만 **6**, 100만 **2**, 200만 **1**개다. 원본 표면 대비 world-frame q99 최대 **0.0099208**로 사전 한도 0.01 초과는 없다. 이는 렌더용 mesh fidelity 검사이며 GT 성능 점수가 아니다. 남은 52개와 최종 completion 영수증은 미완료다.

- 07:55 KST rank1 index117 Selected도 200만 face export와 Top-1 PBR·Shape file4 SHA 보관을 완료해 V1/Selected paired 영수증을 닫았다. 원본 **15,064,208 faces**, 렌더용 **2,000,000 faces**, q99 **0.0039580**이다. 구버전 100만 simplifier 예외로 멈췄던 primary rank0/1/2/3의 네 문제 사례가 모두 현재 fallback으로 복구·검증됐으며, 각 worker는 다음 사례로 계속 진행한다.

## 08:09 KST 계획·진행 재점검 및 공식 렌더 export 복구

- 계획은 계속 **추가 학습 없이** 동결 V1·Selected TEST508 생성 → 공식 RGB1/외부 7법과 공통 성공 사례 확정 → source-M0-only Sim(3) → 동일 10지표, paired CI, teaser, 최종 논문 순서다. Ours 두 arm의 입력 scale은 훈련셋에서 적합한 image-only 추정이고 GT M1/mesh-derived scale은 추론에 쓰지 않는다. 기존 M1-scale 숫자는 최종 공정 비교에 사용하지 않는다.
- 08:08 KST live SHA ledger는 Ours V1 **201/508**, Selected **198/508**, paired **198/508**이다. primary rank 0–3 네 작업자와 GPU6/7 tail 작업자가 생존 중이다. node16 외부 learned evaluator PID2492093은 **2/7** 결과를 썼고 누적 읽기 약 **81.6 GiB**로 계속 실행 중이다. unified nine-method 및 Ours 최종 10지표는 아직 선행 자료 대기다.
- 공식 RGB1은 source-M0 surface **501/501**이지만 render mesh export가 처음 **498/501**에서 멈췄다. 세 사례의 25만 face CuMesh `connectivity.cu` CUDA Error9는 모델 생성 실패가 아닌 native mesh cleanup 실패다. 기존 성공 498건을 SHA로 재사용하고 이 에러만 명시적으로 higher render-face-budget retry 대상으로 추가했다. 50만 face로 **2건이 추가 PASS**했고 q99 world-frame 오차는 각각 **0.0028403**, **0.0038497**로 사전 0.01 한도를 통과했다. 마지막 1건은 50만 face native export가 CPU 실행 중이며 현재 **500/501**, 최종 complete 영수증은 대기다.
- 복구한 501건이 모두 PASS하면 수정한 bridge의 `--adopt-existing` 경로가 외부 7법 completion까지 기다린 뒤 구 100만 face 출력 경로를 검증된 새 export로 연결한다. 구 exporter/gate PID3016321/569292는 의도적으로 `T` 상태를 유지한다. 최종 렌더와 공통표는 그 인계 이후 자동 실행한다. 지금 수치는 완료 성능을 뜻하지 않는다.

- 08:10 KST 마지막 CuMesh 사례도 50만 face 예산으로 완료됐고 q99 **0.0027536**이다. `complete.json`의 상태는 `PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_EXPORT`, **501/501**이다. 전수 재집계에서 성공 영수증 501건, budget 25만/50만/100만/200만 face **488/9/2/2**, q99 최대 **0.0099208**, 사전 한도 0.01 초과 **0건**을 확인했다. 수정된 bridge는 `ADOPT_REPAIRED_OFFICIAL_RENDER_BAKE`를 출력하고 외부 7법 metric completion을 기다리는 세션 **63373**에서 실행 중이다. 이제 공식 export 복구 자체는 완료됐고 남은 관문은 평가기와 Ours 생성·렌더다.
- 08:10 KST 즉시 갱신한 live ledger는 V1 **204/508**, Selected **201/508**, paired **201/508**, 공식 surface/render **501/501** 및 외부 learned **2/7**이다. rank0–3 모두 같은 PID로 생존했고 node16 evaluator PID2492093의 누적 읽기 바이트는 **89.7 GB**로 증가했다. 따라서 평가기의 정지는 관측되지 않았지만, 최종 10지표와 논문 결과 문장은 아직 비어 있다.

## 08:12 KST 후속 gate 연결·생존 감시 보강

- 공식 Cycles 렌더 watcher가 요구하는 로컬 동결 SS TEST508 완료 영수증과 외부 baseline **32/32** 청크 완료를 다시 확인했다. node16 공식 surface 영수증도 존재한다. 남은 직접 선행 조건은 외부 7법 learned metric 완료와 그 직후 bridge의 verified-export 경로 인계다.
- read-only live watcher에 node16 bridge의 실제 프로세스 PID/receipt를 추가하고, `--once` 실제 node16 질의로 bridge PID**1552286** 및 외부 평가 PID**2492093**을 확인했다. 구 watcher를 종료하고 새 코드 PID**704295**로 재시작했다. 08:12 KST paired cascade **203/508**, 공식 export **501/501**, 외부 learned **2/7**이며, bridge는 `waiting on external metrics`로 명시된다. 외부 metric 완료 후 bridge 프로세스가 사라졌는데 영수증이 없으면 live ledger에 주의 문구를 출력한다.
- 08:14 KST 외부 learned evaluator가 세 번째 결과 파일을 작성해 **3/7**이 됐다. 동일 PID2492093은 `R` 상태이며 누적 실제 읽기 **93.3 GB**로 증가했다. Ours file4 SHA 영수증은 V1 **209/508**, Selected 및 paired **205/508**이다. 공통표 watcher의 입력 경로와 방법 수·공통 분모 검사, 최종 Ours 평가 watcher가 official 501사례에 맞춰 502사례 Ours 렌더를 subset하는 연결을 확인했다.

## 08:19 KST 실행 코드 보존·scale 계약 재검증

- 실행·평가 코드와 현재 결과 독립 논문 초안 154파일을 `wowo0709/feature-retargeting` 브랜치 commit **`fa6a71e`**로 저장하고 GitHub `origin`의 같은 브랜치에 push했다. 전체 새 Python 모듈의 `compileall`, staged whitespace check 및 원격 HEAD 일치를 확인했다. 계속 쓰이는 live ledger는 이후 업데이트될 수 있다.
- [TEST508 이미지 추정 scale 계약 재감사](2026-10-28-test508-image-scale-contract-audit.json)를 같은 고정 캠페인에 다시 실행했다. 검사 시점의 실제 파일서버 SHA 보관은 V1 **217**, Selected **213**, paired **213**사례이며 generation 기록 **430건**에서 image-only estimator SHA·scale 값·RGB/camera equality 및 M1/M0 mesh/GT-scale 추론 미사용을 모두 통과했다. 아직 paired 508 전체가 완료되지 않았으므로 `all_cascade_results_complete=false`다.
- 08:20 KST 실시간 표에 GPU6/7의 실제 tail worker PID와 disjoint case rank를 추가했다. 새 watcher PID**710721**의 node16/local 실측은 V1 **219/508**, Selected/paired **216/508**, GPU6 **708633@index451**, GPU7 **710187@index430**, 외부 learned **3/7**이다. 원래 primary rank0–3도 모두 생존한다. 이 행은 GPU 점유율 순간값이 아니라 현재 할당된 실제 작업자를 보여 준다.

## 08:24 KST GPU6 tail의 CuMesh 렌더 예외 복구

- `disjoint-tail-gpu6-7-v2` 관리자가 종료되고 공식 렌더 watcher가 `T`에서 `S`로 풀린 것을 즉시 확인했다. node16 공식 export 구 경로에는 여전히 completion 영수증/bridge 링크가 없으므로 정상 인계가 아니었다. GPU6 index449 V1의 `benchmark-export` 로그에 `CuMesh connectivity.cu` CUDA Error9가 확인됐다. 모델의 `prediction.npz`·`generation.json`·geometry evaluation은 남아 있고 `benchmark.glb`/report는 없어 render-only 단계 실패다. GPU7 index430은 그 전에 file4 SHA archive를 정상 완료했다.
- 공식 RGB1 export에서 검증한 것과 동일하게 **정확한 CuMesh Error9 invalid configuration**만 더 높은 render-face budget 재시도 대상으로 worker 코드에 추가했다. 실제 실패 로그와 무관한 예외를 넣어 분류를 확인했고 `py_compile`/`git diff --check`를 통과했다. 공식 렌더 watcher PID543994를 다시 `T`로 두고, 완료된 tail 사례를 재사용하는 새 관리자를 PID**713405**, GPU6 rank**449**/GPU7 rank**428**로 재시작했다. GPU6은 기존 25만 face 실패 로그를 재사용하고 50만 face export 자식 PID713414를 실행 중이다. GPU7은 다음 disjoint 사례의 Shape 생성 중이다. 이 사례의 최종 보관 영수증은 아직 대기다.
- 08:24 KST 새 live 표는 V1 **223/508**, Selected/paired **220/508**, 외부 learned **3/7**이며 primary 0–3 및 tail 6/7의 실제 PID를 다시 확인했다. 미완료 GPU6 case449의 최종 결과와 관리자 생존을 계속 추적한다.
- 같은 GPU6 index449의 V1 렌더용 GLB는 50만 face 예산으로 성공했고, 원본 표면 대비 q99 world-frame **0.0046204**로 사전 0.01 한도를 통과해 file4 SHA 보관을 완료했다. Selected Shape 생성 자식이 계속 실행 중이므로 이 사례의 paired 완료는 아직 주장하지 않는다. GPU7 index428도 정상 생성 중이며 node16 외부 learned evaluator의 읽기 바이트는 계속 증가한다.
- 08:29 KST GPU6 index449는 Selected Top-1 PBR 및 Shape까지 file4 SHA 보관을 끝내 `PASS_TEST_CASE_TWO_ARMS_ARCHIVED`로 닫혔다. GPU7 index428도 같은 paired 영수증을 완료했다. 새 tail 관리자는 GPU6 rank**447**/GPU7 rank**426**으로 계속 진행 중이다. live ledger는 V1 **229/508**, Selected/paired **226/508**, 외부 learned **3/7**이다. 공식 렌더 watcher PID543994는 인계 전까지 다시 `T`이고 출력 캠페인 디렉터리는 아직 만들어지지 않았다.
- 08:37 KST GPU7 index426도 양 arm file4 SHA 보관을 닫았다. V1 render budget 50만 face/q99 **0.0056249**, Selected는 25만·50만 실패 후 100만 face/q99 **0.0042194**로 같은 0.01 한도를 통과했다. GPU7은 index424로 내려가며 계속 실행한다.

## 08:30 KST 누적 렌더 근사 fidelity 감사

- 새 [누적 render-only mesh 감사](2026-10-28-test508-render-fidelity-incremental.json)는 현재 file4 SHA 보관된 V1 **230건**, Selected **228건**의 `benchmark.json`·archive receipt를 전수 대조했다. 실제 render face budget 25만/50만/100만/200만/500만 분포는 **410/24/15/7/2**, 원본 표면 대비 q99 world-frame 최대 **0.0098481**로 사전 0.01 한도 초과 **0건**이다. 각 영수증의 GLB·prediction·report SHA, image-only scale, GT fitting 미사용, untouched surface metric 입력도 확인했다. 이는 렌더용 근사와 provenance 감사이며 모델의 TEST 성능 지표가 아니다. 전체 508×2가 아직 아니므로 상태는 `PASS_PARTIAL_RENDER_FIDELITY_AUDIT`이다.

## 08:41 KST 계획·실행 중간 점검

- 실시간 file4 SHA 영수증은 V1 **247/508**, Selected 및 동일 사례 두 arm **243/508**이다. primary GPU 0–3 네 작업자와 disjoint GPU 6/7 tail 작업자 두 개 및 관리자 PID**713405**가 실행 중이다. 새 live watcher는 tail 관리자 생존도 표시하고, 공식 render 인계나 cascade 완료 전에 관리자가 종료되면 경고하도록 보강했다. `py_compile`, `git diff --check`, node16 포함 `--once` 실측을 통과한 새 watcher를 재시작했다.
- 공식 Pixal3D RGB1의 source-M0 surface 및 렌더 GLB는 **501/501**로 완료됐다. node16 외부 7방법 learned 결과는 **4/7**이며 완료 방법은 Pixal3D, MVAdapter, ReconViaGen, CorrAdapter다. 평가 자식 PID**2492093**의 누적 실제 읽기 바이트는 **112.8 GB**로 증가했고 bridge PID**1552286**은 외부 7방법 completion을 기다린다. 공식 Cycles render watcher는 이 인계 전까지 의도적으로 정지해 GPU 6/7의 Ours tail과 충돌하지 않는다.
- 계획은 동결 V1·Selected만 GT 객체 scale 없이 TEST508 생성, 동일 최종 성공 교집합의 source-M0-only Sim(3)·공통 10지표·paired CI, 실제 teaser·최종 원고/PDF 순서로 유지한다. 현재 unified 9방법과 Ours 최종 4조건 학습형 metric, 주 비교표, paired CI는 미완료다. 외부 502 전체 수치를 공식 RGB1 성공 501과 섞지 않는다.
- 외부 평가에서 먼저 닫힌 Pixal3D·MVAdapter·ReconViaGen·CorrAdapter **4개 원본 결과 파일**을 직접 열어 검사했다. 각 파일은 동일한 **502개 sample ID**, 1,004개 source view, 실패 0건, 완전 평가 플래그, 사례별 8개 유한 지표 및 유한한 FID/KID를 가졌다. 이는 원본 common502 평가의 무결성 확인이며, 공식 RGB1을 포함한 최종 공통 **501 이하** 모집단의 FID/KID 재집계를 대신하지 않는다.

## 08:44 KST GPU 6·7 연속 활용 준비

- 실시간 두 arm 동일 사례 보관은 **247/508**, V1 단독은 **251/508**로 증가했다. 현재 완료 index는 0–99 전부, 100–199 중 73, 400–449 중 18, 450–507 전부이며 **350–399는 0건**이다. 따라서 primary와 GPU6/7의 다음 disjoint 구간 사이에 충분한 간격이 있다.
- 외부 evaluator는 4/7 결과를 냈고 PID2492093의 실제 읽기 바이트가 **113.0 GB**로 증가했다. 공식 export bridge는 외부 completion 대기 중이다. 현재 GPU6/7 tail 관리자가 index **401/400**까지 정상 완료했는데도 공식 export가 준비되지 않으면, index **399/398부터 350까지** 이어서 처리하는 [연속 구간 관리자](../../../experiments/deformnext_retargeting/orchestrate_disjoint_tail_extension.py)를 대기 세션 **82248**로 연결했다. 선행 영수증의 연속 index와 두 GPU parity를 검사하고, 공식 export 또는 전체 cascade가 먼저 닫히면 새 작업을 건너뛴다.
- tail 관리자의 후속 버전은 render mesh 예외로 종료될 때 공식 export가 아직 준비되지 않았다면 공식 렌더 watcher를 성급히 재개하지 않도록 수정했다. 현재 실행 중인 v3 프로세스에는 이 변경이 소급 적용되지 않으므로 현행 실시간 감시도 유지한다. 새 연속 관리자 코드의 compile·영수증 연속성 smoke와 whitespace 검사를 통과했다.

## 08:47 KST 부분 생성 결과와 논문 gate 점검

- 실제 shape/PBR SHA 보관이 끝난 동일 **254사례 쌍**의 generation 기록을 다시 읽었다. V1·Selected는 모든 쌍에서 같은 image-only scale을 사용했고, M1 유래 scale·source M0 mesh·GT geometry를 생성에 쓰지 않았다는 각 생성 영수증도 254쌍 전부 통과했다. Source-M0 정합 전 raw surface에서 Selected−V1 F@0.02 평균은 **+0.003426**, CD(L1) 평균은 **−0.0002255**이고, 개선 사례는 각각 **144/254**, **139/254**다. 완료 사례는 index 0–199의 177건과 400–507의 77건에 몰려 있어 비무작위 부분집합이다. 이 수치는 최종 공통 성공 집합·source-M0 정합·Cycles 렌더·source-cluster CI를 반영하지 않으므로 논문 결과나 모델 선정 근거로 사용하지 않는다.
- 최종 원고 빌더 `build_fair_manuscript.py`의 필수 입력을 확인했다. 공통 10방법·10지표 표, V1/Selected 8지표 paired CI, 공식 RGB1 대비 16개 보정 대조, 최종 공통 manifest, 공정 teaser의 실제 렌더 SHA와 캡션이 모두 있어야 PDF를 조판한다. 기존 M1-derived-scale 표와 그림을 새 주 비교 수치로 잘못 승격하지 않는 gate다. 아직 이 입력들이 닫히지 않았으므로 최종 PDF는 생성하지 않는다.
- live watcher는 GPU6/7 tail v3와 뒤따를 v4의 관리자·대기 프로세스를 함께 표시하고, node16 상태 질의의 일시적인 20초 지연을 작업 종료로 오인하지 않도록 질의 제한을 40초로 늘렸다. 한 차례 시간초과 직후 같은 node16 PID2492093/bridge PID1552286을 다시 확인했고, 재질의에서는 정상 응답과 증가한 실제 읽기 바이트를 받았다.

## 08:49 KST 최종 공통 모집단·평가 인계 감사

- node16의 공식 성공 교집합 파일에는 중복 없는 **501개 ID**가 있고, 외부 7방법 원본 reference **502개 ID**의 정확한 부분집합이다. 제외된 한 ID는 공식 Pixal3D RGB1의 기록된 native empty-sparse 실패 사례와 일치한다. 9방법 합치기는 502개 외부 reference에서 이 501개를 선택해 평가기를 다시 실행한다. 이후 Ours 4조건도 같은 501개 reference를 정확한 JSON 행 단위로 대조해 합치고, 해당 subset에서 학습형 지표와 FID/KID를 새로 계산한다. `502` 원본 FID/KID를 주 표에 복사하는 경로는 없다.
- node16에서 현재 실행 중인 9방법 평가 orchestration 코드와 교집합 병합 코드의 SHA가 로컬 정본과 각각 일치했다. 후속 Ours 바인딩·camera8 점수·13방법 병합 스크립트 세 파일은 미리 node16에 복사해 로컬/원격 SHA 일치를 확인했다. Ours watcher는 시작 직전에 같은 세 파일을 다시 복사한다. 공식 evaluator PID**195556**, Ours renderer/evaluator watcher PID**586065/586066**, bridge PID**1552286**이 생존 중이며, 외부 7방법 evaluator PID**2492093**이 계속 실행된다.

## 08:52 KST 통합 표의 metric 분류·표기 점검

- 공통 10방법 표 생성기는 user가 지정한 **source 3개, FID/KID 2개, target alignment 5개**와 방향별 최고 bold·2위 underline을 실제 수치 기준으로 적용한다. 기존 머리글에서 CD/F-Score가 별도 `Geometry` 그룹에 놓여 target alignment에서 빠져 보이던 부분을 **target alignment 5열**로 바로잡았다. 입력 가용성(T1, S3, C)과 Pixal3D 1536 estimated-camera/공식1024 given-camera 차이를 표 캡션에 유지한다.
- FID/KID는 최종 공통 **N≤501**건의 camera9/10 렌더 **2N장**을 대상으로 재계산하도록 `image_count=n*2`가 강제된다. 원고 프로토콜의 'global naturalness' 표현을 실제 범위인 **평가 source 카메라 분포의 자연스러움**으로 명확히 했다. camera8 target 지표는 별도 GT 렌더와 동일 사례의 raw/+source-M0 보조 표에서 재계산한다.

## 08:54 KST baseline 입력 권한 표기 수정

- 통합 표 `Input` 열의 ShapeUp=`T1`, SpaceControl=`T1+A1` 표기는 실제 생성 입력을 축소했다. node16의 `/home/seunghyun/editing_reproduction/setup/run-shapeup.py`는 `textured_source_mesh`를 `source.glb`로, M1 target RGB를 `target.png`로 복사해 모델에 준다. `/home/seunghyun/editing_reproduction/setup/run-spacecontrol.py`는 같은 사례·seed의 TRELLIS.2 **raw predicted target proxy mesh**와 M0@8 anchor RGB를 사용한다. 따라서 표를 ShapeUp=`T1+M0`, SpaceControl=`T1+P+A1`로 수정하고 M0/P/A1을 캡션에서 정의했다. 이는 성능 수치 변경이 아닌 입력 권한의 정확한 공개다.
- 원고 공통 프로토콜에 두 입력과 MVAdapter/CorrAdapter가 target RGB에서 자체 생성한 multi-view를 ReconViaGen에 넘긴다는 사실도 명시했다. 외부법을 동일 T1 입력으로 오인하거나 ShapeUp의 source mesh 접근을 숨기지 않도록 한다.
- node16의 실제 native launcher 전 행을 다시 대조했다. TRELLIS.2·기존 공식 Pixal3D는 M1@8 RGB만, 직접 ReconViaGen은 `[M1@8,M0@9,M0@10,M0@11]`, MVAdapter/CorrAdapter는 M1@8 RGB에서 자체 생성한 여섯 target view만, ShapeUp은 M1@8 RGB+textured M0 mesh, SpaceControl은 TRELLIS.2 raw target proxy+M0@8 anchor RGB를 소비한다. 표의 `T1/S3/M0/P/A1` 표기와 일치한다. SpaceControl의 proxy는 same-sample/seed TRELLIS.2 출력이고 target GT geometry는 입력되지 않는다.

## 08:58 KST 외부 다섯 번째 방법 검증

- node16 외부 learned evaluator PID**2492093**이 ShapeUp 결과 파일을 새로 닫아 **5/7**이 됐다. 기존 네 방법과 새 ShapeUp 모두 **동일 502개 ID**, source view 1,004개, 실패 0건, 사례별 8개 유한 지표, 유한 FID/KID, `complete_population`/`full_metric_computation` 플래그를 직접 검사해 통과했다. 남은 원본 두 방법은 TRELLIS.2와 SpaceControl이다. 최종 논문 주 표에서는 공식 RGB1 성공 교집합의 N≤501로 9방법 전체를 재평가하므로 이 502건 값을 복사하지 않는다.
- 같은 시각 Ours V1 **274/508**, Selected 및 paired **271/508** file4 SHA 보관이다. primary 0–3, tail 6/7, tail 연장 대기 프로세스, node16 bridge가 모두 생존 중이다. 최종 10지표와 공통 주 표는 아직 선행 단계 대기다.

## 09:05 KST 계획·실행 재점검

- 실시간 file4 SHA 보관은 V1 **284/508**, Selected와 동일 사례 paired **279/508**이다. GPU 0–3의 primary worker 네 개와 GPU 6/7의 서로 겹치지 않는 tail worker 두 개가 모두 살아 있고, node16 외부 evaluator PID2492093은 **5/7**개 방법의 원본 502사례 결과를 완료한 뒤 실제 읽기 바이트가 약 **115 GB**까지 증가했다. 공식 RGB1은 native 성공 교집합 **501/501**의 surface 및 렌더용 GLB export를 마쳤다. 최종 Cycles 렌더는 외부 평가 완료 후 GPU 6/7을 인계받도록 대기한다.
- 계획 순서는 그대로다: 동결 V1/Selected 두 arm의 TEST508 생성·file4 검증 → 공식 RGB1을 포함한 최종 공통 성공 모집단의 Cycles 렌더 → source M0만을 사용한 생성 후 Sim(3)과 공통 10지표 평가 → paired CI, 통합 주 표, 실제 렌더 teaser → 최종 논문 PDF/시각 검수. 재학습은 하지 않는다. 현재 최종 공통 N은 공식 쪽 성공 501을 상한으로 하고, Ours 전체 성공 영수증과 공통 바인딩 완료 전까지 확정하지 않는다.
- 9방법 학습형 지표 중 외부 7방법을 중복 추론할 필요가 없도록, 이미 저장된 **per-case 8지표와 per-view Inception-2048 feature**를 검증하고 최종 교집합에서 정확히 재집계하는 evaluator를 배치했다. 먼저 완료된 외부 5방법 각각의 원본 502사례 평균·CI·FID·KID를 평가기 원본 함수로 재현했으며 최대 절대 오차는 **6.3e-11** 미만이다. 이후 공식 RGB1 raw/+M0 두 방법은 최종 교집합에서 새로 평가한다. 캐시의 manifest/protocol/extractor/row ID hash, 7방법 전부의 재현 오차, 공통 N·2N 이미지 수를 최종 receipt에서 다시 강제한다. 따라서 기존 502 모집단의 FID/KID를 501 모집단에 그대로 복사하지 않는다.
- node16의 기존 9방법 평가 watcher PID195556은 의도적으로 `T` 상태로 멈추고, 새 캐시 재집계 watcher PID3010914가 대기 중이다. 기존 watcher를 보존해 새 경로에 오류가 생기면 복구할 수 있다. local live ledger에는 두 프로세스의 상태와 새 watcher 소실 경고를 추가했고, 단발 실측·`py_compile`·whitespace 검사를 통과했다. 현재 캐시 경로는 **아직 최종 9방법 평가 통과를 주장하지 않는다**.

## 09:09 KST 캐시 evaluator 연결 복구

- 실시간 감시가 새 캐시 evaluator PID3010914의 소실을 잡았다. SSH 세션은 코드 오류/평가 결과 없이 **exit 255**로 종료됐고, node16의 작업 디렉터리는 여전히 비어 있으며 기존 9방법 watcher PID195556은 `T`로 보존됐다. 외부 7방법 평가 PID2492093과 Ours worker 여섯 개는 계속 실행 중이다.
- 같은 고정 인자와 빈 작업 디렉터리를 확인한 뒤 새 evaluator를 node16에서 SSH 세션과 분리된 `setsid` 프로세스 **PID3140957**, 로그 `unified-nine-method-cached-v2.log`로 다시 시작했다. PPID=1 및 live ledger의 PID 탐지를 확인했다. 계산은 공식 렌더·외부 평가 영수증을 계속 기다리는 상태이며, 완료 수치나 표에는 영향을 주지 않았다.

## 09:11 KST GPU3 세션 종료와 담당 구간 복구

- live ledger에서 primary rank3 PID669224가 사라진 것을 확인했다. 원래 실행 세션은 **exit 143**으로 종료됐고, 마지막 출력은 index195 V1의 file4 SHA archive PASS였다. index195 Selected 및 이후 rank3 사례는 아직 미완료다. 저장된 PASS receipt를 손실/실패로 취급하지 않았다. 다른 primary rank0–2, tail GPU6/7, node16 evaluator/bridge 및 복구 watcher는 생존했다.
- 동일 frozen campaign, SS 완료 영수증, file4 경로, GPU3/rank3/world-size4와 index11 별도 복구 skip ID를 유지해 rank3을 SSH/exec 세션과 분리한 **PID748587**, `gpu3-recovery-v3.log`로 재시작했다. 새 worker는 index3–191의 기존 두-arm receipt를 재확인하고 index195의 미완료 Selected부터 이어간다. PPID=1, `/proc` rank3 탐지, paired 진행 **286/508**을 확인했다. 결과 집합·checkpoint·학습 step은 바뀌지 않았다.
- 09:12 KST에는 index195의 Selected PBR·Shape까지 file4 checksum 영수증을 통과해 해당 두 arm paired 사례가 닫혔다. 현재 paired **289/508**, rank0–3·tail6/7 및 node16 외부/캐시 평가 작업은 생존한다. 원고 preview 25쪽의 page3–4에서 overview와 matching/retargeting 세부 그림의 PDF 크기 글자·캡션 배치를 눈으로 점검했다. 현재 teaser는 내부 scale 실험의 임시 그림이므로 최종 공정 teaser로 교체하기 전에는 논문 그림 검수를 완료로 표시하지 않는다.

## 09:15 KST 실시간 감시 지연 축소

- node16 원격 상태 질의 한 번이 40초 관측 제한에 걸려 live ledger에 일시적 `temporarily unavailable`가 표시됐다. 이어진 독립 `ps` 조회는 외부 evaluator PID2492093과 캐시 watcher PID3140957의 생존을 확인했으며, Ours local paired 영수증은 **290/508**으로 증가했다. 관측 제한만으로 작업을 재시작하지 않았다.
- 9방법 평가가 아직 대기 중일 때마다 불필요하게 file4 완료 영수증을 열던 원격 상태 query를 변경해, node16 로컬 `summary.json`이 생긴 이후에만 file4 archive를 확인한다. 새 query의 단발 실측은 0.51초에 공식501/외부5/7/캐시 watcher/primary4/tail2 상태를 정상 반환했고, `py_compile`·whitespace 검사를 통과했다. live watcher도 세션과 독립된 **PID751550**, 로그 `live-watcher-v4.log`로 교체했다.

## 09:16 KST 생성 입력·scale 중간 재감사

- [전수 계획과 현재 archive 감사](2026-10-28-test508-image-scale-contract-audit-0916.json)를 다시 실행했다. 508개 TEST 사례의 V1/Selected plan 쌍과 image-only scale prediction 508개를 전수 대조했고, 현재 보관된 V1 **295**, Selected **292**, 두 arm paired **292**, 실제 generation **587건** 및 Selected Top-1 **292건**이 같은 RGB·카메라·추정 scale SHA/값을 사용함을 통과했다. `M0_M1_mesh_or_GT_scale_used_at_inference=false`; 전체 508쌍 완료는 여전히 false로 표시한다. 중간 결과의 데이터 누출 방지 확인이며 최종 품질 수치는 아니다.

## 09:17 KST 외부 여섯 번째 방법 검증

- node16 외부 learned evaluator가 SpaceControl 결과를 닫아 **6/7**이 됐다. 직접 읽은 원본 6개 결과 파일 모두 동일 502개 sample ID, camera9/10 **1,004**개 per-view row, 실패 0건, 8개 유한 지표와 유한 FID/KID, 완전 모집단·full-computation 플래그를 충족한다. 남은 방법은 TRELLIS.2다.
- SpaceControl의 저장된 1,004×2,048 Inception feature 및 사례별 결과를 benchmark 원본 `summarize`/FID/KID 함수로 다시 집계했다. 원본 502 결과 대비 평균·source-cluster CI·FID/KID 최대 절대 오차 **1.21e-11**로 PASS했다. 이 검사는 최종 501 이하 공통 사례 재집계를 위한 캐시 정확성 중간 확인이며, 최종 표 자체는 아직 미완료다. 같은 시각 Ours paired SHA 보관은 **294/508**이다.
- 새 캐시 evaluator의 reference root도 원래 9방법 evaluator와 동일하게 검증하도록 `data -> /file4`, `renders -> frozen shared/gt-a02/renders` 두 resolved path 조건을 코드에 추가했다. node16 실제 두 링크가 해당 경로로 resolve됨을 독립 조회로 확인했고, 코드 compile/whitespace 및 원격 SHA 일치를 통과했다. 대기 중인 PID3140957은 계속 살아 있으며, 이미 시작한 프로세스의 런타임 상태는 변경하지 않았다.

## 09:25 KST 대형 렌더 mesh 사례 완료

- GPU6 tail index433 Selected는 25만/50만/100만/200만 face 렌더 bake가 형상 한도에 미달해 500만 face fallback을 실행했다. 하위 PID747717이 실제 CPU를 사용하며 약 14분간 계산한 뒤 **원본 19,911,998 faces 대비 렌더용 5,000,000 faces, q99 world 0.0019125**로 사전 한도 0.01을 통과했다. V1·Selected Top-1 PBR·Selected Shape 모두 file4 SHA archive receipt와 `PASS_TEST_CASE_TWO_ARMS_ARCHIVED`가 닫혔고 GPU6은 index431로 넘어갔다. 원본 geometry metric용 표면은 변경되지 않았다.
- live ledger는 Ours paired **301/508**, 외부 learned **6/7**로 증가했다. 마지막 TRELLIS.2 evaluator PID2492093의 실제 읽기 바이트도 120.3 GB까지 증가해 실행 중이다. 공식 렌더 handoff와 최종 공통 표는 아직 선행 영수증 대기다.

## 09:48 KST 빈 GPU7의 비중복 TEST 사례 활용

- 첫 tail supervisor의 GPU7 index400이 완료된 뒤 GPU6은 index425→421을 처리 중이어서, GPU7은 후속 tail 관리자 시작 전 잠시 비었다. 공식 export bridge는 아직 외부 7방법 완료를 기다리고, 기존 v4는 index398부터 시작하도록 대기함을 확인했다. 겹치지 않는 동일 frozen campaign/world-size508 단일 사례 **index398**을 GPU7에서 먼저 실행해 V1/Selected/Top-1 PBR 세 file4 SHA archive 및 `PASS_TEST_CASE_TWO_ARMS_ARCHIVED`를 확인했다. v4가 나중에 같은 index를 재방문하면 기존 영수증을 검증하고 건너뛰는 worker 계약이다.
- 같은 방식으로 index396을 detached PID772131에서 시작했다. 각 사례 시작 직전에 공식 export handoff 영수증이 없음을 확인한다. 공식 렌더 인계가 먼저 준비되면 추가 GPU7 사례를 시작하지 않는다. node16 외부 TRELLIS.2 PID2492093은 간헐적 NFS RPC 대기 뒤 다시 실행 상태로 돌아왔고 실제 읽기 바이트가 계속 증가한다. 이 선행 tail 작업은 실험 조건이나 공통 성공 집합을 바꾸지 않는다.

## 09:58 KST 계획·실행 중간 재점검

- 범위는 frozen V1·Selected의 GT-scale-free TEST508 생성, 외부 7법·공식 RGB1과 동일 최종 성공 교집합의 source-M0-only 사후 Sim(3) 및 10지표 평가, paired CI·통합표·실제 teaser·최종 PDF이다. 새 학습이나 추가 arm 생성은 없다. 기존 M1-derived-scale 내부 ablation의 수치는 새 주 비교표에 섞지 않는다.
- file4 SHA 보관 기준 V1 **346/508**, Selected/Top-1 PBR 및 양 arm paired **341/508**이다. GPU0–3의 본 worker 네 개와 GPU6 index417, GPU7의 비중복 index396 worker가 살아 있다. index396은 V1 archive를 마치고 Selected PBR을 계산 중이다. 대형 렌더 bake의 fallback 때문에 GPU 점유율 순간값이 낮아도 CPU 작업이 진행될 수 있으므로, 종료 여부는 사례 영수증으로 판정한다.
- node16 공식 RGB1 native 성공 507/508, 최종 외부 공통 집합에 속한 surface/export **501/501**이다. 외부 7방법 learned 결과는 **6/7**이고 마지막 TRELLIS.2 평가 PID2492093은 실행 중이며 누적 읽기 바이트가 약 **130.5 GB**로 증가했다. 공식 bridge PID1552286와 캐시 기반 9방법 watcher PID3140957은 살아 있으나 외부 7방법 완료를 기다린다. 공식 Cycles 렌더 watcher는 GPU6/7 인계 전까지 의도적으로 정지돼 있다.
- 현재 최종 공통 N·10지표 표·V1↔Selected 및 공식 RGB1 paired CI·공정 teaser·배포용 PDF는 없다. 가장 최근 약 41분 동안 paired archive가 294→341로 47건 증가했다. 같은 속도만 외삽하면 남은 167건은 약 2.4시간이지만, GPU6/7의 공식 렌더 인계와 대형 mesh fallback이 있어 생성 종료는 **대략 2.5–4시간**으로만 예상한다. 이어 공통 렌더·평가·원고 반영이 더 필요하며, 전체 완료 시각은 마지막 외부 평가 및 렌더 처리량을 본 뒤 재산정한다.

## 10:00 KST 생성 계약·렌더 근사 재감사

- [이미지 입력·scale 감사](2026-10-28-test508-image-scale-contract-audit-1000.json)를 같은 508쌍 frozen plan과 현재 archive에 다시 실행했다. V1 **347**, Selected/paired **343**, 실제 generation **690건**과 Selected Top-1 PBR archive **343건**의 RGB·카메라·image-only scale SHA/값 일치를 통과했다. M0/M1 mesh 또는 target GT scale은 생성 추정에 사용되지 않았고 `all_cascade_results_complete=false`다.
- [렌더 mesh 근사 감사](2026-10-28-test508-render-fidelity-incremental.json)는 동일한 V1 **347**, Selected **343** archive의 GLB·원본 prediction·report SHA와 사전 정의한 q99 world **≤0.01**을 통과했다. 최대 q99는 **0.0098672**이고, 실제 face budget 25만/50만/100만/200만/500만은 **621/33/24/9/3**건이다. export 완료 후 archive 진행 중인 기록 1건은 성공 수에 포함하지 않았다. 원본 표면은 geometry metric용으로 그대로 보존한다.

## 10:10 KST 최종 표 모집단 검증 보강

- 최종 10지표 표 생성기를 검토해 target-camera PSNR/SSIM/M-IoU 보조 평가 3개가 기존에는 `samples=N`만 검사하고 실제 sample ID 집합은 확인하지 않는 빈틈을 발견했다. 외부 7법·공식 RGB1·Ours target8 평가 결과에 `sample_ids`를 기록하고, 표 생성 시 세 집합 모두 최종 13방법 공통 manifest의 ID와 정확히 같은지 검사하도록 보강했다. 유한한 10지표 값과 평가 영수증 status/schema도 검증한다.
- 수정된 세 target8 평가 스크립트는 node16 대기 작업이 실행되기 전에 배포했고 로컬/원격 SHA 3개가 각각 일치함을 확인했다. 외부 평가 결과는 아직 6/7이며 공식 Cycles 렌더는 0청크라 새 평가 파일은 생성 전이다. 단발 합성 검증에서 같은 1사례 ID의 10방법×10지표 표는 통과했고, **사례 수는 같지만 ID가 다른** target8 입력은 표 생성 전에 거부됐다. `py_compile`과 `git diff --check`도 통과했다.
- 이 시각 Ours paired file4 SHA 보관은 **356/508**까지 증가했고, 선행 GPU7 index396은 두 arm 모두 보관을 통과했다. node16 마지막 TRELLIS.2 평가 PID2492093의 실제 읽기량은 약 **133.5 GB**로 증가해 실행 중이다. 최종 표와 PDF 완료는 여전히 주장하지 않는다.

## 10:28 KST 외부 7방법 종료·공식 렌더 인계

- node16 외부 7방법의 원본 common502 10지표 평가가 `PASS_SOURCE_M0_FULL502_SEVEN_METHODS_TEN_METRICS` 영수증을 작성했다. 마지막 TRELLIS.2 결과를 별도 확인해 중복 없는 **502개 사례**, source view **1,004개**, 실패 0건, 유한한 사례별 지표·FID/KID, 완전 모집단 플래그, Inception feature **1,004×2,048**을 통과했다. benchmark 원본 aggregate/FID/KID 함수로 재집계한 최대 절대 오차는 **3.34e-11**이다. 이는 외부 502건 정본의 검증이며 공식 RGB1 성공 교집합 N≤501의 최종 표가 아니다.
- 공식 RGB1의 501개 렌더 mesh가 `PASS_OFFICIAL_FAST_EXPORT_BRIDGE`로 검증된 경로에 연결됐다. 실제 face budget 25만/50만/100만/200만은 **488/9/2/2**건이다. 기존 1M watcher는 의도적으로 정지된 채 보존됐고, 로컬 공식 Cycles watcher는 현재 GPU6 tail 사례 종료 후 GPU6·7을 넘겨받도록 대기한다. 캐시 기반 node16 9방법 평가 watcher도 외부 완료를 확인했고 공식 32청크 렌더를 기다린다.
- 인계 직후 tail GPU6가 정지 신호를 관측하기 전에 index405 한 사례를 시작한 경합이 있었다. 이 사례의 V1은 이미 file4 SHA 보관됐고 Selected 계산이 진행 중이다. 현재 사례가 닫히면 관리자가 공식 입력 준비를 감지해 더 낮은 index를 시작하지 않고 렌더 watcher를 재개하는지 영수증으로 확인한다. Ours primary GPU0–3은 계속 생성 중이며 paired **369/508**이다.
- 10:31 KST 추가 확인: index405는 두 arm archive까지 닫혔지만 30초 원격 확인 주기 전에 **index403도 시작**됐다. 공식 export 입력은 수동 SSH의 동일 `test -f`로 확인됐고, tail 관리자 PID713405의 main thread는 새 사례를 시작하는 poll sleep이 아니라 실행 중인 GPU6 worker를 기다리는 join 상태다. 따라서 index403을 끝낸 뒤 공식 렌더로 넘기는 경로를 관찰한다. 중간 사례를 강제 종료해 불완전 산출물을 남기지 않는다.

## 10:42 KST GPU 인계·첫 공식 Cycles 청크 통과

- tail GPU6는 설정된 마지막 index401까지 V1·Selected SHA 보관을 끝냈다. tail 관리자 PID713405는 종료됐고 공식 렌더 watcher PID543994가 `T`에서 `S`로 재개됐다. 인계가 30초 원격 확인 창에 겹쳐 index403·401까지 추가로 시작된 점을 확인해, 후속 tail 코드에는 **각 사례 시작 직전에도 공식 입력을 확인**하는 조건을 추가했다. 현재 공식 렌더와 tail 생성은 GPU6/7에서 겹치지 않는다.
- 공식 RGB1 Cycles 캠페인은 501사례의 raw/+source-M0×camera8/9/10, 총 **3,006 작업·32청크** 계획을 검증하고 시작했다. 첫 `chunk000`의 16사례·96작업은 동일 protocol에서 렌더·file4 영수증·로컬 staging 회수를 **121.4초**에 통과했다. `chunk001`이 staging 중이다. 첫 청크 속도를 단순 외삽한 32청크 약 65분은 대형 GLB와 네트워크 편차를 반영하지 않은 참고치이며 완료 예정 시각으로 확정하지 않는다.
- Ours paired file4 SHA 보관은 **383/508**까지 늘었다. 외부 7방법 원본502는 검증 완료됐지만 공식 RGB1·Ours와 합친 최종 공통 N·10지표는 아직 없다. node16의 캐시 기반 9방법 evaluator는 공식 32청크 완료를 기다린다.

## 10:44 KST 공식 렌더 청크 실측·자동 진행표 보강

- 공식 RGB1 `chunk001`도 16사례·96작업 렌더와 file4 영수증을 **109.24초**에 통과했고 `chunk002`가 시작됐다. 첫 두 청크는 2/32이며, node16 캐시 기반 9방법 평가 watcher가 같은 완료 청크 수를 관측했다. 이 속도의 단순 외삽은 약 1시간이지만 대형 mesh·전송 편차가 있으므로 실제 완료 영수증으로 판단한다.
- [실시간 진행표](2026-10-28-scale-ungiven-live.md)에 공식 RGB1·Ours의 `PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4` 청크 수를 추가했다. 갱신 스크립트의 단발 실측은 공식 **1/32**·Ours plan 대기를 표시했고, 기존 watcher를 안전하게 종료한 뒤 분리 실행 PID799903으로 재시작했다. 새 watcher가 `PPID=1`에서 실행하고 정상 ledger를 작성함을 확인했다. `py_compile` 및 whitespace 검사도 통과했다.

## 10:46 KST 계획·진행 중간 점검

- 사용자 범위를 다시 대조했다. 추가 학습 없이 frozen V1/Selected만 새 camera-given·객체별 GT scale 미제공 TEST508에서 생성한다. 공식 RGB1과 외부 7방법은 같은 최종 성공 ID에서 raw와 **생성 후 source M0만 이용한 Sim(3)**를 채점한다. 기존 M1 유래 scale의 내부 ablation은 별도 표로 유지한다.
- file4 SHA 영수증은 V1 **387/508**, Selected/Top-1 PBR과 양 arm paired **385/508**이다. GPU0–3의 네 primary worker 및 해당 계산·export·전송 자식이 살아 있다. rank3 복구 worker는 index279까지 양 arm 영수증을 추가했다. 대형 mesh의 render-bake fallback과 파일 전송으로 GPU 순간 점유율이 낮을 수 있으며, 현재 단계의 완료 판정은 GPU 사용률이 아니라 사례별 영수증이다.
- 공식 RGB1 native 생성은 **507/508**, 최종 후보의 source-M0 surface와 GLB export는 **501/501**이다. GPU6·7의 공통 Cycles 렌더는 raw/+M0×camera 3개, 총 3,006작업 중 **3/32청크**를 file4에 보관했고 네 번째 청크의 Blender 자식 두 개가 실행 중이다. 첫 세 청크 소요는 **121.4/109.24/103.53초**였다.
- 외부 7방법 원본 common502의 10지표 평가는 완료됐고 사례별 502 ID, source 1,004 view 및 Inception feature·FID/KID 재집계 정확성을 검증했다. node16의 캐시 기반 통합 평가 watcher는 공식 렌더 완료를 기다린다. Ours surface→렌더→4조건 평가→10방법 통합 표→paired CI→공정 teaser는 완료 영수증에 순서대로 연결돼 있다. 아직 **최종 공통 N, Ours 10지표, 표, 새 teaser, 최종 PDF는 없다**.
- 최근 약 46분의 paired 증가 **343→385**를 단순 외삽하면 생성만 2시간 이상 남는다. GPU6·7 공식 렌더 인계와 대형 mesh 편차를 고려하면 생성 종료는 잠정 **2.5–4시간**이다. 그 뒤 surface·Ours 렌더·learned metric·논문 조판이 남으므로 전체 완료 시간을 이 수치에 포함시키지 않는다. 현재 로컬 가용 공간은 약 **1.0 TiB**이며 추가 재학습·arm 확장은 계획하지 않는다.

## 10:49 KST 생성 입력·렌더 근사 감사

- [입력·scale 계약 감사](2026-10-28-test508-image-scale-contract-audit-1048.json)는 전체 508쌍 plan과 현재 file4 보관된 **V1 390건, Selected 387건**, 실제 generation **777건**을 대조해 통과했다. 두 arm의 RGB·카메라·image-only scale 입력이 같고, 생성 시 M0/M1 mesh·GT scale 미사용 플래그와 scale prediction SHA가 일치한다. 완료 508쌍 감사로 확대해 해석하지 않는다.
- [렌더 mesh 근사 감사](2026-10-28-test508-render-fidelity-1048.json)는 같은 777개 render bake의 영수증·원본 prediction SHA·GLB SHA를 대조해 통과했다. world-frame q99 최댓값 **0.0098672**는 사전 한도 0.01 이내다. 원본 geometry metric용 표면은 유지됐고, GT를 이용한 export 정합은 없었다. GPU6·7의 공식 공통 Cycles 렌더는 이 시점 **4/32청크**를 완료했다.
- 남은 incomplete index는 주로 268–399이며, 현재 GPU0–3이 각자 처리 중이다. 공식 렌더 종료 후 GPU6·7을 재투입할지 결정할 때 각 rank의 미완료 구간과 현재 실행 사례를 다시 확인해 겹치는 사례를 만들지 않는다. 현재는 공식 렌더가 두 GPU를 사용하므로 생성 worker를 추가하지 않는다.

## 10:56 KST 최종 방법 설명과 frozen 모델 대조

- 실행 campaign이 가리키는 최종 개발 freeze 파일의 SHA256 **`09bcde32d2841bd7c2895132504733bc00fc31e9aff32bcfc05740aa2ed4f5ab`**를 실제 파일과 대조했다. `selected_pipeline_model_contract`가 V1·Selected의 SS480과 PBR1900 binding을 모두 통과했다. Selected는 SS `E_FIXED_RAW`(10개 source tensor, learned attention head 0), PBR `top1-only`(10개 source tensor)이며, V1은 두 stage 모두 source tensor 0이다.
- Selected SS 실제 config의 `topk=64`, `score_temperature=0.05`, frozen D1, semantic weighted 후보 집계, native bilinear projection, `task_attention=None`을 확인했다. PBR은 별도 `candidate_aggregation=top1`, `appearance_mode=top1-only`다. 이는 원고의 **가중 Top-K SS와 별도 Top-1 PBR** 설명 및 새 Q/K/V head를 필수라고 주장하지 않는 문장과 일치한다. 구조·가중치 binding 감사이며 최종 TEST 성능 증거는 아니다.
- 같은 시각 Ours paired file4 보관 **393/508**, 공식 RGB1 공통 Cycles 렌더 **8/32청크**다. node16 GPU7은 다른 작업이 약 13 GiB를 점유하고 있어 learned-metric 단계가 시작되면 속도와 메모리를 확인한다. 해당 작업은 건드리지 않는다.

## 11:00 KST 최종 공통 사례·평가 입력 연결 사전 확인

- node16의 외부 평가 binding은 502개, 원본 frozen baseline reference는 508개, 공식 RGB1 성공 교집합은 501개 ID다. 이 **501개 전부**가 두 reference 파일에 존재하며, ID별 reference JSON 행의 불일치는 **0개**였다. 이후 Ours binding과 9방법 binding이 같은 reference를 요구하는 merge gate의 현재 입력 조건을 충족한다. Ours 렌더가 아직 없으므로 최종 13조건 merge 완료를 뜻하지 않는다.
- Ours binding·target8 평가 코드가 공식 성공 ID, raw/+M0 각각의 camera8/9/10 렌더 영수증과 SHA, 동일 502개 surface score의 transform을 검사하고, 최종 표 생성기가 세 target8 scorer의 `sample_ids`를 공통 manifest와 대조하는 것을 재확인했다. 최종 FID/KID와 10지표는 공식·Ours 렌더 및 learned evaluator 완료 후에만 작성된다.

## 11:06 KST Ours surface 채점 병행 준비·실행

- node16에서 실제 보관된 동일 사례 하나를 기존 scorer로 채점해 V1·Selected **2개 surface 결과**를 통과했다. 새 사례별 캐시 경로에 같은 사례를 다시 계산한 결과와 캐시 재사용 결과를 대조하니 세 JSONL의 SHA256이 모두 **`68cdc1401501afb77fdd28109d510f2b9693b6d9a27dc90f7d1c99fbcfd0dfb6`**로 완전히 같았고, 재사용 실행은 1초 이내였다. 한 사례의 동등성 검사이며 502사례 결과는 아니다.
- `score_minimal_test_ours.py`에 선택된 사례 ID의 선계산과, 동일 manifest 행·source/target mesh stat·V1/Selected 파일서버 archive/export 영수증 SHA가 맞을 때만 캐시를 재사용하는 경로를 추가했다. 원래 최종 surface watcher는 cascade 완료 후 이 스크립트를 다시 배포하고 전체 502사례×2행을 검사하므로, 선계산에 빠진 사례는 기존처럼 새로 채점된다. 캐시가 불일치하거나 깨졌으면 이유를 로그에 남기고 해당 사례를 원본에서 다시 계산한다.
- 현재 file4 검증이 끝난 395개 공통 사례를 고정한 목록으로 node16의 nice 10·CPU 1 thread 작업자 **PID 1840428/1840429/1840431/1840432**를 시작했다. node16은 96코어 중 시작 시 load 약 35였고, 로컬 GPU6·7의 공식 렌더는 계속 진행한다. [실시간 진행표](2026-10-28-scale-ungiven-live.md)에 선계산 사례 수와 작업자 PID를 추가하고 watcher를 **PID 814372**로 교체했다. 현재 Ours paired **403/508**, 공식 렌더 **11/32청크**다. 선계산 결과는 최종 공통 사례 10지표의 대체물이 아니다.

## 11:20 KST 계획·실행 중간 재점검

- 계획 범위는 그대로다. 재학습 없이 고정 V1·Selected 두 arm만 TEST508에 생성하고, 공식 Pixal3D RGB1·외부 7방법과 최종 동일 성공 ID에서 raw/+source-M0 사후 Sim(3), 공통 Cycles 및 10지표를 평가한다. 기존 M1 유래 scale 내부 ablation 수치는 주 비교표와 분리한다.
- file4 checksum archive 영수증 기준 V1 **418/508**, Selected·Top-1 PBR 및 양 arm paired **414/508**이다. GPU0–3의 네 worker와 각 Shape/PBR/export 하위 프로세스가 현재 살아 있고, 실제 자식은 모델 계산 또는 250k-face GLB 내보내기를 수행 중이다. 이전 worker 로그의 예외는 복구 전 기록이며 현재 네 PID의 종료 증거가 아니다.
- 공식 RGB1은 surface/export **501/501**을 마쳤고, GPU6·7의 raw/+M0×3 camera Cycles 작업은 **15/32청크**를 file4에 검증 보관했다. 16번째 청크는 렌더 결과를 node16/file4로 전송 중이어서 순간 GPU 사용률 0%가 작업 정지를 뜻하지 않는다. node16 캐시 기반 9방법 evaluator PID **3140957**은 공식 렌더 완료를 대기한다.
- node16의 Ours surface 선계산은 초기 395개 계획 중 **125개 사례**의 캐시 파일을 작성했다. nice 10·단일 CPU thread 작업자 네 PID가 살아 있으며, 최종 scorer는 cache key를 다시 확인하고 빠진 사례를 직접 계산한다. 외부 7방법의 502개 원본 10지표는 완료됐지만 공식·Ours와 합친 최종 N과 10방법 표는 아직 없다.
- 이 시점에 Ours 생성은 paired **94쌍**이 남고, 공식 렌더는 **17청크**가 남는다. 최근 paired 11쌍/약 13분과 청크별 2–6분의 편차를 고려하면 생성과 공식 렌더가 각각 대략 **1.5–2.5시간**, **0.5–1.5시간** 더 걸릴 수 있다. 이후 Ours 502사례 surface·6,024건 렌더·learned metric·paired CI·실측 원고·PDF 검수가 남으므로 이를 전체 완료 시간으로 오인하지 않는다. 로컬 저장공간은 약 **1,011 GiB** 여유다.
