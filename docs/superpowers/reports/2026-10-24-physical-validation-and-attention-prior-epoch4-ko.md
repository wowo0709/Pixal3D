# 물리 모델별 재로딩 검증 연결과 attention prior 중간 결과

기준 시각: 2026-10-24 00:43 UTC. 구현 커밋 `3e6edf1361c7e89b987e3ebed28c8e941ad612b6`.

## 새로 관측한 결과

GPU 6에서 진행 중인 `FULL_QK_ONLY`가 384 update 검증을 완료했다. 아래는 같은 DEV32·학습 seed 101·생성 noise 1개·384 update의 asset별 평균 IoU다. Sample ID·source cluster·noise·GT hash가 모두 같다.

| 모델 | IoU |
|---|---:|
| Vanilla V1 | 0.47070421 |
| Matching decoder Top-1 + retargeting | 0.47257893 |
| 기존 Full | 0.47481289 |
| Full에서 additive matching-score prior 제거 | **0.47717633** |

Additive prior 제거−Full 차이는 **+0.00236343**이다. 이 arm은 semantic correspondence 탐색과 Top-K ranking을 유지하고 attention logit에 더하는 matching prior를 제거한다. 사용자가 정의한 No-score=Top-1 arm과 구분한다.

작은 중간 패널에서는 matching confidence를 attention bias로 직접 넣는 방식이 task에 맞는 feature 조합을 제약할 가능성을 시사한다. Correspondence를 찾는 점수와 feature를 실제로 사용하는 효용이 다를 수 있다는 가설이다. 모든 matching score가 불필요하다거나, learned attention의 최종 추가 이득을 증명했다는 뜻은 아니다. 이 arm의 480 update·DEV128×3 및 독립 학습 seed 확인이 필요하다. 기존 최종 DEV128×3의 Full 0.47207818, Top-1+R 0.47260025, Fixed Top-K 0.47274517과 이번 DEV32 값을 직접 섞어 비교하지 않는다.

완료된 epoch row를 정상 NN이 실행 중인 상태에서 별도 immutable snapshot으로 복사했다. 이후 실제 validation 파일의 해당 byte prefix와 복사본이 정확히 같은지 재확인했다. 표·원본 결과 artifact를 별도 W&B run으로 발행하고, remote API에서 8개 모델의 수치와 artifact 등록을 확인했다. 실제 학습 run은 수정하지 않았다.

[W&B: additive matching prior 대조군의 4에폭 DEV32 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-unified-QK-interim32-epoch4-s101-v186)

## 최종 검증에서 새로 연결한 부분

`selected_population_validation.py`는 이미 대기 중인 V1·Top-1+R·Full의 실제 validation36 두 fresh-load 결과를 다시 추론하지 않고 검증한다. 실제 평가 계획·출력·wrapper·checkpoint/config/source hash·서로 다른 종료 PID/birth를 확인하고, 72개 NPZ의 latent·packed occupancy hash를 직접 검사하며 TP/FP/FN도 배열에서 재계산한다. 최초32 기록은 자신의 final128 prefix와 비교하고, tail4를 포함한 전체36 기록 및 latent/occupancy 배열은 두 실행 사이에 일치해야 한다.

원래 세 native family는 과거 binding schema를 사용하므로, 실제 같은 config·NN/source bytes·variant·parameter count·측정 배열임을 확인한 경우에만 현재 물리 binding metadata로 연결한다. 나머지 구조를 기존 모델로 대체하지 않는다. Legacy model binding과 새 binding의 lineage를 함께 보존한다. 단계가 다른 checkpoint, 달라진 source, 잘못된 noise·array·source provenance, 한 번의 로딩 또는 같은 PID/birth 재사용은 거부한다.

V1·Top-1+R·Full 각각에 **CPU reduction 1개씩, 총 3개**를 실제 main queue에 연결했다. 선행 GPU 작업은 v185의 정상 복구 successor ID를 사용한다. 기존 descriptor 및 queue state는 유지했고, 새 GPU 작업이나 학습 arm은 추가하지 않았다. 이 세 parity 검증은 아직 대기 중이다.

`run_selected_population.py`는 선택한 14개 물리 구조를 자신의 기존 evaluator로 실행할 수 있게 준비한 worker다. 명시한 GPU·CPU affinity·Python·source checkout·canonical closure를 사용하며, 실제 child와 session이 종료된 뒤에만 출력을 저장 완료 처리한다. Validation36과 동결 후 TEST508 모두 기존 sampler를 유지한다. 실제 닫힌 V1·Top-1+R·Full 계획으로 실행 context의 source·assignment 검사를 통과하고 file4에 보관했다. 이 새 worker의 GPU forward를 수행한 것은 아니며, 아직 선택하지 않은 후보를 실행하거나 TEST를 활성화하지 않았다.

## 검증 및 실행 상태

관련 CPU 검사 **322개 통과**, 새 검사는 103개다. 순수 metadata 검사에 더해 실제로 생성한 테스트용 NPZ 72개를 읽는 경로, receipt/hash·계획 provenance·array 손상·중복 process identity 거부를 검사했다. 테스트 자료는 synthetic이며 실제 학습 결과나 GPU 검증으로 해석하지 않는다.

실행 중인 NN의 원래 source **325개** 및 science config 3개의 hash가 계획의 pin과 일치함을 직접 확인했다. 정상 학습이나 두 dispatcher를 재시작하지 않았다. 최초 metadata source 스캔에서는 PBR와 SS wrapper의 config 경로 필드 차이로 조회가 끝나지 않았고, 각각의 실제 command/plan 형식을 사용해 세 실행을 모두 다시 검사했다. 첫 process/queue snapshot은 보존하고 별도 source 검증 receipt를 저장했다.

현재 관측한 진행은 raw Top-K PBR **99/1900**, additive prior 제거 SS **392/480**, 개별 voxel query SS **127/480**이다. PBR는 복구 이후 동일한 NN/session을 유지하며 GPU 0~3 DDP4, global batch 24·micro batch 6·accumulation 1로 진행한다. GPU 6·7의 SS도 같은 실제 handle을 유지한다. MeanMatch는 proof와 임시 3-update capacity를 완료했고, 실제 480-update science는 PBR 종료 뒤 GPU 0에서 실행하도록 대기한다.

## 남은 완료 조건

남은 대조군과 두 shared-evidence 후보의 실제 학습·DEV128×3를 마친 뒤, 실제 SS→Shape→PBR joint 결과로 모델을 선택해야 한다. 선택 모델 및 강한 기준선의 독립 seed 확인, validation36 두 fresh-load, DEV 기반 설정 동결, TEST508 전체 cascade와 10개 metric도 남아 있다. 현재 물리 모델별 SS 평가 준비를 전체 cascade·두 original zero control·최종 metric 실행에 연결하는 작업은 계속 필요하다. 전체 목표는 진행 중이다.

실제 file4 검증 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `selected-population-validation-and-execution-v186/`, `focused-QK-epoch4-SS32-v186/`, `paired-QK-epoch4-SS32-WandB-v186/`에 보관했다.
