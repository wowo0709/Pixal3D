# TEST508 디스크 입장을 위한 종료된 PBR 예약 해제

2026-10-26 15:41 UTC. 실제 TEST508 `v636` 실행 중 공용 `DiskBudget`에
2026-10-23 실패한 `top1-plus-feature` PBR 학습의 미래 checkpoint 쓰기
예약 16 GiB가 남아 있었다. 원래 예약 소유 PID 536397과 wrapper 자식 PID
536409는 모두 종료됐고, 큐 상태는 `FAILED`/returncode 1이다. 원래
`wrapper/failure.json`도 `FAILED_NATIVE_PBR_SCIENCE_PRESERVED`이며
`native/ckpts`에는 `.pt`가 없다. 해당 학습의 계획·native/wrapper 경로를
사용하는 활성 프로세스가 없음을 검사했다.

별도 `actual-dead-PBR-feature-credit-cancellation-v648`에서 원본 실패·terminal·
progress 4개 파일의 실제 file4 SHA256/크기를 확인하고, 원래 wrapper에 예약
해제 증명과 file4 receipt를 남겼다. 그 증명이 닫힌 뒤 `DiskBudget.release`로
**미래 쓰기 예약만** 해제했다. 공용 장부의 해당 entry는 `released=true`이며
별도 audit/작업 코드도 file4에 닫혔다. 원래 505개 수치 source, 모델·데이터·
checkpoint·scored image 파일은 삭제하거나 변경하지 않았다. 물리 디스크
16 GiB를 회수했다는 의미가 아니다.

해제 뒤 실제 free는 약159.7 GiB, 두 완전 자산 그룹의 새 입장 필요량은
112 GiB, 여유는 약47.7 GiB다. `v636` coordinator/worker는 계속 살아 있고
새 평가 사례는 37/1,155개 완료됐다. 예약 정리는 운영 안정성 조치이며
평가 metric이나 논문의 성능 claim은 아직 없다.
