# TEST508 proc-access 실패와 저장 처리 복구

현재 실제 campaign은 **v586**이다. 이전 v570은 최종 완료되지 않았으며,
완료 결과·partial output·실패 로그를 보존했다. 모델 재선정이나 재학습은 없다.
최종 TEST10metric과 논문 결과 작성은 계속 남아 있다.

## 실제 실패 원인

v570의 중간 raw duplicate 정리 worker가 `/proc/2863585/fd/0`을 조사하다
`psutil.AccessDenied`로 종료됐다. 코드가 미상 reader를 무시하지 않고 중단한 것이다.
실패 지점은 fresh NAS hash 확인과 intent 저장 이후의 두 번째 reader scan이며,
첫 unlink 이전이다. 해당 프로세스는 이후 실제로 종료됐다.

이 실패가 coordinator의 controlled drain을 유발했다. 뒤이어 guard가 종료 중인
coordinator command를 확인하다 assertion으로 끝난 것은 후속 현상이다.
Guard의 finally는 자신의 CPU stop을 해제했으며 GPU/NN signal은0이었다.
Root cause는 이 retirement의 process-inspection 권한 오류이며 모델 OOM 로그가 아니다.

진행 파일의 마지막 count는141이었지만, 실제 zero-exit case terminal은142개다.
27 asset은 다섯 모델과 전체 raw retirement까지 완료한 closed group이다.
이 결과를 최종2540 case 완료로 쓰지 않는다. 원래 failed result/release와 모든
완료·partial 결과를 보존했다.

## 보완한 운영 경로

원래505개 source의 byte는 그대로 유지하고, 별도 운영 adapter 세 개를 추가했다.

1. `operational_process_access.py`: protected process의 `open_files`/`environ`을
   실제로 읽을 수 있거나 해당 PID/birth의 프로세스가 종료될 때까지 기다린다.
   AccessDenied를 empty read나 reader 부재로 치환하지 않으며 timeout·권한 변경·
   다른 프로세스에 대한 signal도 없다.
2. `operational_retirement_wait.py`: 별도로 pin한 adapter source를 확인하고,
   위 wait 범위 안에서 원래 frozen raw-retirement 함수를 호출한다. Canonical NAS
   SHA/size·실제 consumer terminal·원본 inode·두 번의 reader 검사·local duplicate만
   삭제하는 계약을 유지한다.
3. `operational_campaign_entry.py`: CPU retirement를 동기적으로 기다린 뒤 ledger를
   갱신한다. 기존 GPU case는 계속 실행한다. 이로써 original reader-plan snapshot이
   검사 도중 바뀌지 않는다. 외부 SIGSTOP watcher는 사용하지 않는다. Native case
   command·모델 phase·sampler·입력/seed·metric 정의는 그대로다.

실제 non-dumpable owned reader를 만드는 negative control과 원래 campaign/retirement
회귀 검사를 실행했고 **24 passed in47.57s**였다. 보호된 live reader를 종료 전까지
부재로 처리하지 않는지, 실제 retirement subprocess가 끝나기 전에 다음 ledger가
쓰이지 않는지, native case command/인자가 유지되는지를 포함한다. CPU fixture는
운영 순서 검사이며 실제508 품질이나 새 모델의 검증으로 쓰지 않는다.

`actual-proc-access-failure-retirement-and-dead-credit-recovery-v584`에서 실패했던
실제 두 raw file도 새 경로로 재검사했다. Fresh actual NAS hash/size와 원래 모든
guard를 통과한 뒤 local duplicate만 삭제했고 원격 원본은 보존했다. 종료된
coordinator/keeper와 모든 campaign token의 부재, canonical failure/release를
확인한 후16개의 나머지 dead future credit만 cancellation receipt로 해제했다.
모든 남은 partial/완료 output은 그대로 보존·file4 검증했다.

## 실제 후속 실행과 재평가 범위

`actual-selected-five-model-TEST508-cascade-v586`과 준비 context
`actual-selected-TEST508-cascade-preparation-v586/prelaunch-v1/context.json`을
기존 두 dispatcher에 연결했다. Coordinator와 두 keeper는 실제로 시작했고
GPU0·1·2·3·6·7에 여섯 native case worker를 배치했다.

원래 final collector는 case audit/worker terminal의 실제 output이 하나의
`campaign_root/cases/<sample hash>/<alias>`와 일치하도록 검증한다. 다른 namespace의
worker terminal이나 archive 경로를 새 실행인 것처럼 고쳐 쓰지 않는다. 이 동결된
계약을 유지하기 위해 새 namespace에서2540개 cascade case를 평가한다. 기존142개
case도 재실행되므로 이전 약1.5시간의 평가 계산이 다시 필요하다. SS508×5의 기존
2540 archive와 모든 학습 checkpoint는 그대로 재사용하며 재학습하지 않는다.

새 모델이나 metric 변경이 아니라 저장 운영 복구다. Parent freeze558, controls556,
다섯 physical model, 실제 RGB/cache, camera, common Shape, PBR checkpoint, generation
noise와 full508/134-source population을 유지한다. 테스트 결과로 후보를 바꾸지 않았다.

후속 경로는 v586 cascade→v586 ten-metric collector→v587 Main/qualitative→v588
W&B readback이다. 이전 failed v570 metric activation과 미실행 v571/v572 callback은
그 실패 이력에 남기고 다시 활성화하지 않는다. 새 collector는 원래11개 GPU metric
job, TEST8C=16 family, signed raw FID/KID,1524 image/model을 유지한다. 정성 그림은
원래 v355 정책과 Main/Supplement 배치를 따른다.

14쪽 개발 원고(v581)는 이미 compile·검수·file4 보존됐다. 실제 full TEST 수치,
최종 표·정성 그림·결과에 따른 Abstract/Results/Discussion/Conclusion, 최종 PDF 검수는
아직 남아 있다. 현재 실행 PID와 최신 count는 후속 agent-handoff observation을 따른다.
