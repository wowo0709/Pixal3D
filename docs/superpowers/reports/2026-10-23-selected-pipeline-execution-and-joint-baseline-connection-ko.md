# 선택된 SS/PBR 조합의 실제 파이프라인 평가 연결

기준 시각: 2026-10-23 22:33 UTC. 구현 커밋: `86af675`.

## 해결한 실행 연결 문제

기존 공동 평가 worker는 Vanilla와 Top-1+retargeting 두 조합으로 고정되어 있었다. 이후 다른 물리 SS arm이 후보가 되면, 그 모델의 실제 SS occupancy에서 생성한 geometry/native guide와 선택한 PBR checkpoint를 연결해야 한다.

`selected_pipeline_execution.py`는 기존 12개 SS arm과 shared-evidence 두 후보, 네 PBR mode를 구분한다. 같은 조합이 기준선과 겹치면 2개, 다른 조합이면 3개 pipeline을 비교한다. SS 모델이 같으면 같은 geometry 집합을 재사용한다. 모델·학습 seed·source parameter 수·SS archive·32개 sample/cluster 순서·원래 Shape stage를 검사한다.

Replay 전에는 선택 모델 자신의 SS archive와 실제 base case를 대조한다. 다른 모델의 occupancy, 다른 noise/GT, 다른 SS source binding, 재구성한 guide, GT geometry 사용, 정렬 후 평가, 종료되지 않은 child 기록은 허용하지 않는다. 원래 Shape512 step4210과 Shape1024 step16850 및 native guide를 유지한다.

머신 작업 경로에는 다음 두 worker를 추가했다.

- `run_selected_pipeline_replay_GPU03_v181.py`: 선택한 SS의 실제 geometry/guide에서 선택 PBR1900을 replay한다. 실제 네 rank의 완료와 자기 NN/source/config binding을 확인한다. 아직 큐에 넣거나 GPU에서 실행하지 않았다.
- `run_selected_pipeline_metrics_CPU_GPU03_v181.py`: 2개 또는 3개 실제 조합의 공식 metric과 source/target별 결과를 처리한다. 기존 native replay/render/shared/official metric child 명령을 그대로 사용한다.

Original worker의 `subprocess.Popen` AST는 동일하다. Case worker의 child 명령과 child 소유·종료·session drain 함수도 AST가 동일하다. 이는 기존 수치 실행 경로를 보존했다는 소스 검증이며, 새 worker의 GPU 실행 성공을 대신하지 않는다.

## 실제 큐에 연결한 작업

**추가한 작업은 CPU 결과 정리 1개다. 새 GPU 작업이나 학습 arm은 없다.**

`joint-baseline-selected-contract32-summary-CPU-v181`은 기존 `joint-V1-retarget-pipeline32-official-v176` 완료에 연결되어 있다. 기존 V1/retargeting 공동 평가 64개 case와 공식 metric 192개 view를 새 물리 모델 검사로 다시 정리한다. 현재 상태는 pending이다. 기존 학습·case·공식 metric 작업의 명령이나 설정은 변경하지 않았다.

이 결과는 선택 후보의 후속 전체 파이프라인 평가와 같은 검사를 사용하는 기준선이 된다. 실제 새 후보의 공동 평가 계획은 후보와 checkpoint가 정해진 뒤 생성해야 한다. 이번 연결이 모델 선택이나 최종 평가 완료를 뜻하지 않는다.

## 확인한 증거

- 관련 CPU 테스트 **177개 통과**, 그중 새 검사 84개. 14×4 조합의 물리 binding, 기준선 중복 제거, geometry/noise/GT/source mismatch 거부 등을 synthetic metadata로 검사했다. CPU neural forward는 없다.
- 실제 file4에 보관된 V1·Top-1+R·Full의 기존 SS→Shape case 세 개와 기존 공동 평가 index를 읽어 새 검사에 통과했다. E 또는 seed202 모델 결과를 읽지 않았다.
- Worker/검사 기록 9개 파일, 실제 case 검사 기록 1개, 진행 snapshot 1개를 각각 새 폴더에 보관하고 실제 file4 SHA256/size를 대조했다.
- 기존 여섯 NN의 wrapper와 child PID/birth를 확인했다. 현재 계획에 고정된 308개 소스 파일은 그대로다.

## 현재 학습과 남은 작업

GPU 0~3의 C_RAW/C_SPATIAL/C_D1_COS/NoR는 각각 350/341/350/359 update다. GPU 6 QK-only는 240, GPU 7 voxel-query는 77/480이다. 모두 계속 학습 중이다. 새 최종 품질 결과는 아직 없다.

원래 12개 arm과 연결된 shared-evidence 두 후보의 최종 DEV128×3 평가, 실제 SS/PBR 공동 평가, 독립 seed202 검증, VAL36 두 번 fresh-load, DEV 기반 모델·설정 동결, 그 이후 TEST508 전체 cascade와 10개 metric 평가가 남아 있다. 후보가 정해지면 자신의 SS archive→원래 Shape→선택 PBR→공식 metric으로 이어지는 실제 case 계획을 생성하고 연결해야 한다. 전체 실험 목표는 진행 중이다.

새 증거는 기존 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `selected-pipeline-execution-workers-v181/`, `selected-pipeline-execution-actual-preflight-v181/`, `joint-baseline-selected-contract32-v181/`, `focused-status-v181/`에 있다.
