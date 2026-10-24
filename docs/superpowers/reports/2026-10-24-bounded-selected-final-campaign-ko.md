# 선택된 pipeline의 bounded final cascade 실행기

2026-10-24. 실제 최종 모델 선택, DEV-only freeze와 TEST508 실행은 아직 완료되지 않았다. 이 작업은 최종 실행에 필요한 운영 코드를 추가하며, 새로운 학습 arm이나 architecture 선택을 추가하지 않는다.

## 실행 구조

`run_selected_final_campaign.py`는 기존 GPU0–3 dispatcher와 GPU6/7 dispatcher를 재시작하지 않는다. 각 큐에 자기 GPU 전체를 예약하는 keeper 하나를 넣고 CPU coordinator가 두 실제 예약의 PID/birth, dispatcher PID/birth, 부모 관계, command, GPU 환경과 running descriptor를 확인한다. 준비 파일이나 오래된 state만으로 소유권을 인정하지 않는다.

`prepare_selected_final_campaign.py`는 실제 closed DEV freeze와 control freeze, 원래508 SS archive 및 정확히 deduplicate된4/5 모델을 검사한 뒤 descriptor를 만든다. 두 큐에 모두 append하고 실제 descriptor 보존을 확인한 후 공통 activation receipt를 발행한다. 한쪽 append 실패나 concurrent write가 발생하면 그 receipt를 발행하지 않는다. 기존 job, FAILED/HELD 상태와 dispatcher state는 변경하지 않는다. 실제 큐 등록은 이번 작업에서 실행하지 않았다.

각 asset은 `zero-source4`를 먼저 완료해 M1 original-material reference를 만든다. 나머지 모델은 그 reference를 사용한다. 같은 SS를 쓰는 추가 PBR 조합은 실제 canonical geometry owner의 case 종료 후에만 시작한다. 원래 frozen Shape512/Shape1024, native guide archive와 case worker의 수치 경로를 그대로 사용한다.

## 소비자까지 포함한 디스크 예약

개별 geometry producer만 예약하면 window가 producer로 찼을 때 후속 PBR consumer가 들어오지 못할 수 있다. 새 `selected_campaign_resources.py`는 asset의 모든 실제 모델을 함께 admit한다. Group 수는 명시적으로 제한하며, worker 하나가 GPU 하나를 사용한다.

각 group은 case budget 합계의2배를 기존 atomic DiskBudget ledger에 예약한다. Case가 시작할 때 `run_selected_campaign_case.py`가2C future credit을 원래 case reservation C로 atomic transfer한다. 나머지 C는 실제 파일이 쓰여 free disk가 줄어드는 비용을 감당한다. 기존 DiskBudget가 raw case의 전체 C를 계속 계산하므로 필요한 이중 계산 보상이다. 이 adapter가 바꾸는 부분은 디스크 reserve뿐이다. Native 모델, input, sampler, optimizer, phase numerical entry는 바꾸지 않는다.

모든 모델이 완료되면 CPU raw retirement worker를 한 번에 하나씩 실행한다. 실제 file4 fresh SHA256·size, 실제 종료 worker/phase, 현재 reader와 inode를 재확인하는 기존 executor를 사용한다. Case-owned local prediction/geometry NPZ duplicate만 정리하고 canonical remote 원본, guide, checkpoint, scored RGB, GT-Pred 이미지와 report는 보존한다. Group의 모든 case reservation이 실제 retirement 후 해제되어야 다음 group credit을 해제한다. Case 전체 파일 크기가 지정 budget을 넘으면 새 launch를 멈추고 실패 데이터를 보존한다.

## 중단과 별도 세션

각 case와 그 phase에는 campaign/worker ownership token이 상속된다. Phase가 별도 session을 만들거나 parent가 먼저 종료된 경우도 실제 process 환경을 조사해 추적한다. Worker 실패를 관측한 것만으로 keeper를 해제하지 않는다. 정상 worker의 SIGINT handler가 자신의 phase를 drain하게 하고, worker가 먼저 사라졌으면 정확히 자기 token을 가진 orphan만 종료·회수한다. 관련 없는 프로세스는 건드리지 않는다.

Keeper의 SIGINT/SIGTERM 또는 source/queue 검증 오류는 stop request를 만들며, actual descendant drain과 canonical release receipt를 확인할 때까지 예약을 유지한다. 종료되지 않은 작업을 관측 timeout으로 재시작하거나 reservation을 만료시키지 않는다. 실패한 campaign을 자동 재사용하거나 재시도하지 않는다.

## 검증 범위

새 운영 검사28개와 기존 관련124개, 총152개가 CPU92/93에서79.68초에 통과했다. 앞선 관련 검사들과 합하면464개다. 두 CLI help 검사도 통과했다.

검사는 실제 CPU child PID/birth/session, actual queue descriptor와 환경, 실제 파일/atomic credit transfer를 사용한다. 2개 synthetic asset×5 모델을1-asset window·2 worker slot으로 끝내고, reference/owner 이전 실행 금지, canonical local-mirror 보존, 모든 credit의 최종 해제를 확인했다. Worker 실패 후 별도 session orphan, keeper interrupt/source guard 오류, live descendant가 있는데 release를 주장하는 잘못된 기록, partial cross-queue append도 검사했다.

이들은 작은 임시 CPU fixture의 운영 검사다. 실제 neural model, real selected final case forward, real file4 raw 정리 또는 held-out508 quality 검증으로 해석하지 않는다. 실험 원본 raw/checkpoint는 이번 작업에서 삭제하지 않았다.

## 활성화 전 남은 연결

실제 core12 및 shared-evidence 후속 결과, joint SS/PBR 평가, 독립 학습 seed와 validation36 repeatability gate를 먼저 완료해야 한다. `evaluation_protocol.source_SHA256`에는 campaign·credit adapter·preparer·retirement와 향후 최종 metric source 전부를 실제 byte hash로 추가하고, 원래 numerical generation/render/shared/PBR template도 동결해야 한다. 이 코드를 새로 작성했다는 이유로 기존 closed freeze나 source pin을 수정하지 않는다.

그 뒤 실제508 SS archive가 준비되면 preparation request에 source inputs, 모델별 PBR template 및508 RGB/cache hash/receipt, GT surface reference, measured case/GPU budget, caller-assigned GPU/CPU와 현재 dispatcher identity를 명시한다. `prepare_selected_final_campaign --plan <closed-request>`가 준비와 두 큐 등록을 수행한다. 아직 이를 실행할 실제 선택 모델/동결 입력이 없다.

최종10 metric과 dynamic4/5-model population 수집 연결은 여전히 남아 있다. Campaign 성공 자체는 native cases와 raw retirement 완료만 증명하며 `all_ten_metrics_collected=False`, `full_goal_complete=False`를 유지한다. Metric 수집을 생략하거나 old fixed6-model TEST 계획을 활성화해 목표를 축소하지 않는다.
