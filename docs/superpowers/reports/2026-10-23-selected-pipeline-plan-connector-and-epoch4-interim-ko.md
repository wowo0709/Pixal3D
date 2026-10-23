# 후보 pipeline 계획 생성·연결 구현과 4에폭 중간 결과

기준 시각: 2026-10-23 23:04 UTC. 구현 커밋: `dc2a764`.

## 새로 완료한 구현

`selected_pipeline_plan_factory.py`는 닫힌 기존 per-asset template에서 선택 모델의 SS archive, 실제 geometry/guide, 출력 경로를 연결한다. RGB 입력·카메라·원래 Shape stage·sampling seed·sampler·렌더링·원래 M1 material reference는 유지한다. 기존 수치 코드의 source hash를 바꾸는 계획도 거부한다.

머신 작업 경로의 `connect_selected_pipeline_CPU_v182.py`는 실제 닫힌 SS archive와 PBR1900 네 rank/NN/source/config를 확인하고 다음 작업을 만든다.

1. 선택한 SS와 동일한 archive·occupancy hash를 쓰는 geometry가 이미 있으면 재사용한다. 같은 arm 이름만으로 재사용하지 않는다. 없으면 자신의 SS→원래 Shape512/1024→공통 trained V1 PBR case를 만든다.
2. 그 geometry와 원래 native guide에서 선택 PBR를 seed-reset replay한다. 이미 동일 조합의 replay가 있으면 재사용한다.
3. Vanilla 및 Top-1+retargeting 기준선과 중복을 제거한 2개 또는 3개 pipeline의 공식 metric과 source/target별 결과를 연결한다.

새 GPU 작업에는 기존 GPU 0~3 permit와 첫 case 성공 의존성을 유지한다. 계획 파일을 실제 file4 SHA256/size로 확인한 뒤에만 기존 queue에 append할 수 있다. 기존 작업 설정이나 queue state를 변경하지 않는다. 학습, 모델 최종 선택, TEST 실행을 수행하는 연결기는 아니다.

## 실제 자료로 검증한 범위

관련 CPU 테스트 **232개 통과**, 새 테스트는 91개다. 원래 sampling/RGB/camera/Shape/GT reference 보존, 14개 SS와 네 PBR mode의 물리 binding 및 잘못된 model/source/seed/receipt 거부를 검사했다. CPU neural forward는 없다.

실제 완료된 archive와 checkpoint를 사용해 아래 세 가지 **미활성화 검증용 계획**을 생성하고 file4에서 대조했다. 후보를 선택하거나 GPU 작업을 실행하지 않았다.

| 검증용 조합 | 재사용 | 생성된 작업 정의 | 실제 큐 추가 |
|---|---|---:|---:|
| 기존 Full SS + vanilla PBR | 자신의 기존 SS geometry | PBR replay 32개 + metric 단계 3개 | 0 |
| Top-1+R SS + Top-1+R PBR | 기존 기준선 case와 공식 metric | CPU 정리 1개 | 0 |
| V1 SS + Top-1+R PBR | 기존 V1 geometry와 해당 PBR replay | metric 단계 3개 | 0 |

별도로 실제 SS occupancy와 원래 template를 사용한 새 geometry 계획 생성도 통과했다. 하위 `ss-artifacts` 파일은 상위 SS archive의 실제 receipt로 확인했다. 이 검사는 계획 생성 검증이며, 새로운 geometry NN forward를 수행하지 않았다.

전역 queue 파일은 v181의 닫힌 queue snapshot과 byte 단위로 동일하다. v181에서 연결한 기준선 CPU 정리 작업 1개는 그대로 대기 중이다. 이번 구현에서 새로 활성화한 CPU/GPU 작업은 없다.

## 동일한 DEV32의 4에폭 결과

모든 행의 sample ID, noise seed, GT hash가 일치한다. 아래는 **asset별 IoU 평균**, 생성 noise 1개·학습 seed 1개의 중간 패널이다.

| 모델 | 288 step | 384 step |
|---|---:|---:|
| V1 | 0.46897292 | 0.47070421 |
| Top-1+retargeting | 0.46799147 | 0.47257893 |
| 기존 Full | 0.46649383 | **0.47481289** |
| Raw semantic Top-1 + R | 0.47073808 | 0.47193911 |
| Decoder feature cosine Top-1 + R | 0.46703344 | 0.47178594 |
| Native spatial Top-1 + R | 0.46731529 | 0.46973110 |
| Decoder correspondence, R 제거 | 0.46924330 | 0.47094461 |

이 패널에서 raw semantic−spatial은 +0.00220800, Top-1+R−NoR는 +0.00163432다. Raw와 decoder cosine은 가깝다. Full은 3→4에폭에서 회복했다. Full−Top-1+R는 Top-K와 readout 등을 포함한 조합 차이이므로 attention 단독 효과로 해석하지 않는다.

기존 최종 DEV128×3에서는 Full 0.47207818, Top-1+R 0.47260025였다. 중간 DEV32의 순위가 다른 만큼, 이번 값으로 최종 우열이나 각 모듈의 필요성을 확정할 수 없다. 기존 다섯 모델의 최종 비교에서 학습형 attention의 추가 이득은 아직 입증되지 않았다. 현재 arm의 480 update 및 같은 최종 평가, shared-evidence 두 후보, 독립 학습 seed 검증을 계속한다.

## 실제 실행 상태와 남은 완료 조건

GPU 0/1/2/3은 각각 399/388/399/409 update, GPU 6/7은 278/89 update다. 여섯 wrapper와 실제 NN child의 PID/birth/command를 확인했다. 기존 308개 source hash가 일치하고, 현재 로그에 OOM·traceback·nan loss 표시는 없다. 기존 정상 NN을 재시작하지 않았다.

최종 후보는 아직 정하지 않았다. 이후 실제 DEV 결과로 후보를 정하면, 검증용 목적이 아닌 명시적 DEV candidate 입력과 새 출력 경로로 연결기를 실행해야 한다. 필요한 실제 joint case/metric, 선택 후보의 seed202 SS/PBR 검증, VAL36 두 번 fresh-load, DEV 기반 설정 동결, 그 이후 TEST508 전체 cascade 및 10개 metric 평가가 남아 있다. 모델 선택과 실제 GPU 실행 검증은 이번 준비 작업으로 대체되지 않는다.

증거는 기존 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `selected-pipeline-connector-preparation-v182/`, `selected-pipeline-unselected-*-preview-v182/`, `selected-pipeline-new-geometry-plan-preflight-attempt2-v182/`, `focused-status-v182/`에 보관했다. 전체 실험 목표는 계속 진행 중이다.
