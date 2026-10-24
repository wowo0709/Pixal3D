# 완료된 SS 모듈 비교의 voxel 오류 분석

2026-10-24. 실제 완료된 10개 모델의 동일 DEV128개·93 source cluster·생성 noise 3개·480 update 결과를 분석했다. 독립 training seed는 101 하나다. 원래 12개 control 중 개별 voxel query와 mean matching control은 아직 완료되지 않았다. 새로운 모델 학습이나 추론 없이 저장된 TP/FP/FN을 사용했다.

## 비교 가능한 실제 학습 입력

10개 모델의 실제 `training-noise-trace.jsonl` 전체를 확인했다. 각 파일에 step 1부터 480까지 정확히 480개 record가 있고, 모든 step의 asset ID, noisy latent SHA256, timestep SHA256, global conditioning SHA256가 V1과 정확히 일치했다. Global batch는 96이다. 각 원본 trace와 config는 기존 file4 closure로 확인했다.

이 결과는 입력 순서·noise·timestep·global conditioning 차이를 현재 endpoint 차이의 설명에서 제외한다. 모델 weight나 module computation이 같다는 의미는 아니다. 독립 training seed 반복이나 모듈의 작동 원인을 증명하는 결과도 아니다.

## IoU 차이의 정확한 count 분해

같은 GT에서 `IoU = TP / (GT + FP)`다. 두 모델 사이의 차이를 대칭적인 TP 변화 항과 FP 변화 항으로 정확히 분해했다. 아래 수치는 case macro IoU의 차이이며 percentage point가 아닌 IoU 단위다. 두 항의 합이 실제 차이와 같다.

| 비교 | IoU 차이 | TP 변화 항 | FP 변화 항 |
|---|---:|---:|---:|
| Retargeting 추가, decoder Top-1 조건 | +0.003929 | +0.000100 | +0.003829 |
| Raw semantic 경로 − spatial 경로 | +0.001117 | +0.000044 | +0.001074 |
| Decoder cosine − raw cosine | +0.000648 | −0.000883 | +0.001531 |
| Learned scalar − decoder cosine | −0.000408 | +0.000894 | −0.001302 |
| Fixed Top-K − Top-1 values | +0.000527 | −0.000227 | +0.000755 |
| Learned attention − fixed Top-K | −0.000667 | −0.000404 | −0.000263 |
| Full − V1 | +0.004723 | −0.001288 | +0.006011 |

Retargeting의 현재 conditional 개선은 대부분 extra voxel 억제와 연관된다. Full도 V1보다 FP를 줄였지만 TP가 줄어 이득 일부를 상쇄했다. 전체 prediction/GT voxel 수 비율은 V1 약 1.1741에서 Full 약 1.1512로 줄었다. 따라서 높은 IoU를 곧바로 더 정확한 세부 geometry나 더 좋은 rendering으로 해석하면 안 된다. 생성 geometry의 CD·F-Score와 실제 cascade rendering으로 확인해야 한다.

Decoder의 cosine correspondence는 raw cosine보다 FP 억제 쪽에서 좋아졌지만 TP 손실이 있었다. Learned scalar는 정답 voxel 쪽의 기여가 양수였으나 FP 쪽 손실이 더 컸다. Learned attention의 point difference는 fixed Top-K보다 TP·FP 양쪽에서 나빴다. 이는 후보 선택·혼합·전달 문제를 조사할 근거이며, attention 내부의 특정 원인을 증명하는 intervention은 아니다.

## 통계와 해석 범위

원래 module family 10개와 pipeline family 2개의 source-cluster CI를 그대로 유지했다. 새로운 검정이나 family 축소를 하지 않았다. 모듈별 작은 point difference, 특히 retargeting을 포함한 family 보정 CI의 한계를 숨기지 않는다. Full−V1 pipeline contrast는 원래 보정 CI에서 양수지만 독립 training seed는 하나다. Retargeting 효과는 decoder Top-1 조건에서 측정한 conditional contrast다. Spatial 경로에서 semantic 경로로의 비교도 현재 경로 전체의 대체이며 순수 matcher 하나의 효과로 분리되지 않는다.

GT occupancy 크기별 descriptive strata는 deformation 강도별 분석이 아니다. TP/FP 분해도 재학습된 두 모델의 count를 설명하는 대수적 결과이며, neural mechanism이나 inference intervention으로 해석하지 않는다.

## 다음 검증과 실제 저장 위치

현재 최고 case macro endpoint인 decoder-cosine Top-1 + retargeting은 0.47300814, V1은 0.46735504다. Fixed Top-K는 0.47274517, 기존 Full은 0.47207818이다. Macro와 micro를 섞어 0.43에서 0.47로 개선됐다고 주장하지 않는다. 집계 검증은 `2026-10-24-SS-IoU-micro-macro-audit-ko.md`에 별도로 기록했다.

다음 단계는 원래 미완료 control과 shared-evidence fixed/learned 비교를 완료하고, 실제 SS prediction으로 Shape512/1024와 학습된 PBR을 통과시키는 DEV joint 평가다. Stage IoU 최고 모델을 최종 모델로 미리 채택하지 않는다. 최종 TEST와 10개 metric 비교는 아직 완료되지 않았다.

실제 결과는 file4의 `rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래에 보관했다.

- `SS-module-error-decomposition-v194/`: 실제 count 분석, CSV, 원래 CI, 입력 SHA와 file4 closure.
- `SS-module-error-decomposition-figures-v194/`: 같은 수치로 만든 최종 PNG/SVG. 범례를 데이터 영역 밖으로 옮겼으며 원래 그림은 보존했다.
- `SS-ten-model-training-trace-parity-v194/`: 실제 10개 모델 전체 480-step 학습 trace 비교와 원본 file4 증거.

검증: CPU affinity 92/93, CUDA 비활성화 상태에서 count 분해·역방향 부호·macro/micro 차이·GT/noise 불일치 거부·실제 study row 연결을 확인하는 20개 테스트를 통과했다. 학습 프로세스와 기존 checkpoint·원본 metric·실행 중인 source는 변경하지 않았다.
