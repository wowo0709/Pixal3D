# TEST508 세 producer 재사용과 완료 그룹 guide 수명 주기

2026-10-26 15:31 UTC. 최종 모델 및 수치 protocol을 동결한 `v636`은 이전 세
실행의 완료 그룹 129+98+50=277개(1,385 case)를 원래 물리적 경로에서
재사용하고, 나머지 231개 그룹(1,155 case)을 실행 중이다. 실제 manifest를
읽기 전용으로 재검증해 세 producer의 완료 그룹 수와 다섯 모델 구성을 확인했다.

장시간 실행 중 로컬 디스크 여유를 유지하기 위해, 기존
`operational_closed_current_guide_retirement.py`의 완료 그룹·case/phase
terminal·reader·파일서버 정본 SHA256·inode·원래 scored archive 재검증을
그대로 사용하는 별도 세 producer 어댑터를 추가했다. 원래 guide 정리 절차와
실행 중인 campaign의 505개 수치 source는 수정하지 않았다. 어댑터가 재사용
manifest reader만 교체하고 성공·실패 뒤 원래 binding을 복원하는 3개 검사와
재사용 계약 검사 11개가 통과했다. Git `c4cfd1d`로 푸시했다.

실제 pilot `actual-closed-current-guide-offload-v647-pilot`에서는 **완전히 닫힌
v636 그룹 1개/5 case**의 로컬 `native-shape-guides.pt` 중복본 5개,
133,911,961 bytes(0.1247 GiB)만 정리했다. 정리 전 새 파일서버 SHA256/크기,
완료 terminal, reader queue와 열린 파일을 검사했고, 정리 후 같은 5개 case의
scored RGB·GT·alpha archive 15개를 원래 observation과 정확히 대조했다.
정본 파일서버 guide와 case/group audit, scored archive, 데이터와 checkpoint는
보존했다. `retirement/audit.json`은
`PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`, 실제 file4 receipt는
`PASS`/`local_equals_actual_file4_SHA256=true`다. 평가 coordinator와 GPU
작업자는 15:31 UTC에 계속 살아 있었다(새 case 26/1,155).

이 pilot은 캐시 수명 주기 검증이며 품질 지표·모델 효과·최종 논문 결과가 아니다.
완료 그룹의 로컬 중복 guide가 충분히 쌓일 때 같은 별도 절차로만 추가 정리한다.
전체 508개 자산의 cascade 종료, 10metric, Main/정성 결과, 결과에 따른 원고
서술과 최종 PDF 검수는 계속 남아 있다.

15:45 UTC 저장 공간 관측: v636의 닫힌 그룹 8개를 실제 파일 크기로 집계하면
그룹당 총 보관량 평균은 약0.220 GiB, 그중 guide는 약0.163 GiB다. Pilot로
guide를 정리한 그룹도 포함한 작은 이질적 표본이므로 전체231개 그룹의 확정
크기로 외삽하지 않는다. 다만 완료 그룹에서 guide가 보관량의 큰 부분이므로,
로컬 여유와 `DiskBudget`의 두 그룹 동시 입장 margin을 보면서 완료 그룹만
추가 정리한다. Scored archive는 평가 입력으로 계속 보존한다.

15:51 UTC 후속: pilot과 분리한 `actual-closed-current-guide-offload-v650-bulk1`에서
추가로 완전히 닫힌 **9개 그룹/45개 guide**, 1,456,518,625 bytes(1.3565 GiB)의
로컬 중복본만 정리했다. 원래 세 producer manifest, 그룹 audit, 모든 다섯 case의
terminal과 8개 raw retirement, 활성 reader와 열린 파일, 매 batch의 실제 파일서버
SHA256/크기를 확인했다. 정리 후 원래 scored RGB·GT·alpha archive **135개**가
case observation과 정확히 일치했다. `retirement/audit.json`과 file4 receipt는
각각 `PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED` 및
`PASS`/`local_equals_actual_file4_SHA256=true`다. 이전 pilot의 그룹을 다시
선택하지 않았고, 실행 중인 그룹·정본 guide·scored archive·모델은 보존했다.
`v636` coordinator/worker는 계속 실행 중이며 새 case는 51/1,155개다.

16:49 UTC 후속: `actual-closed-current-guide-offload-v652-bulk3`에서 추가로
완전히 닫힌 **13개 그룹/65개 guide**의 로컬 중복본
**2,821,241,413 bytes(2.6275 GiB)**를 정리했다. 실행 전 세 producer의
완료 그룹, 모든 case·raw retirement, reader·열린 파일과 실제 파일서버 정본
SHA256을 재검증했다. 정리 후 원래 scored archive **195개**가 guide 없이도
기존 observation과 정확히 일치했다. 최종 `retirement/audit.json`은
`PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`, 실제 file4 receipt는
`PASS`/`local_equals_actual_file4_SHA256=true`다. 기존 pilot/bulk1 그룹은
guide가 이미 없어 다시 선택되지 않았다. 원래 505개 수치 source·정본 guide·
scored image·case audit·실행 중인 그룹은 유지했다. 이 시각 `v636` coordinator와
작업자는 살아 있고 신규 118/1,155 case가 완료됐으며 로컬 여유는 약160.8 GiB다.

같은 입력을 가리키는 `v651-bulk2`는 준비 파일만 file4에 검증·보존하고 실행하지
않았다. 준비 audit의 대문자 상태명이 이전 bulk1을 가리키는 표기 오류가 있어,
새 `v652-bulk3` 준비에서 이름을 바로잡고 실제 정리에는 그 policy만 사용했다.
`v651-bulk2/retirement` 산출물은 없다. 이 운영 정리는 최종 품질 수치나 모듈
효과의 근거가 아니다.

17:57 UTC 후속: `actual-closed-current-guide-offload-v653-bulk4`에서 다시 완전히
닫힌 **16개 그룹/80개 guide**의 로컬 중복본 **3,370,518,224 bytes
(3.1390 GiB)**를 정리했다. 매 batch의 파일서버 정본 SHA256, 완료 group/case,
8개 raw retirement, reader·열린 파일을 새로 확인했고, 정리 후 원래 scored
archive **240개**가 guide 없이도 기존 observation과 정확히 일치했다. 최종
`retirement/audit.json`은 `PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`,
실제 file4 receipt는 `PASS`/`local_equals_actual_file4_SHA256=true`다. 이전에
정리한 그룹·정본 guide·scored image·원래 505개 수치 source·실행 중인 그룹은
변경하지 않았다. 이 시각 `v636` coordinator/worker는 살아 있고 신규
196/1,155 case가 완료됐으며 로컬 여유는 약159.9 GiB다. 이는 운영 공간 확보로,
최종 10metric 또는 논문 품질 결론의 근거가 아니다.

18:58 UTC 후속: `actual-closed-current-guide-offload-v654-bulk5`에서 추가로
완전히 닫힌 **15개 그룹/75개 guide**의 로컬 중복본 **2,291,930,103 bytes
(2.1345 GiB)**를 정리했다. 매 batch의 실제 파일서버 정본 SHA256/크기,
완료 group/case와 8개 raw retirement, reader·열린 파일을 재검증했고,
guide 없이 원래 scored RGB·GT·alpha archive **225개**가 기존 observation과
정확히 일치했다. 최종 `retirement/audit.json`은
`PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`, 실제 file4 receipt는
`PASS`/`local_equals_actual_file4_SHA256=true`다. 이전에 정리한 그룹,
정본 guide, scored image, 원래 505개 수치 source와 실행 중인 그룹은
변경하지 않았다. 이 시각 `v636` coordinator와 작업자는 살아 있고 신규
273/1,155 case가 완료됐으며 로컬 여유는 약158.9 GiB다. 이는 운영 공간 확보이며
최종 품질 수치나 모듈 효과의 근거가 아니다.

19:55 UTC 후속: `actual-closed-current-guide-offload-v655-bulk6`에서 추가로
완전히 닫힌 **15개 그룹/75개 guide**의 로컬 중복본 **2,218,048,887 bytes
(2.0657 GiB)**를 정리했다. 새 파일서버 정본 SHA256/크기, 완료 group/case와
raw retirement, 활성 reader·열린 파일을 매 batch 다시 검사했고,
guide가 없어도 원래 scored RGB·GT·alpha archive **225개**가 기존 observation과
정확히 일치했다. 최종 `retirement/audit.json`은
`PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`, 실제 file4 receipt는
`PASS`/`local_equals_actual_file4_SHA256=true`다. 정본 guide·scored image·
원래 505개 수치 source·실행 중인 그룹은 변경하지 않았다. 이 시각 `v636`
coordinator와 5개 작업자는 살아 있고 신규 346/1,155 case가 완료됐으며
로컬 여유는 약158.6 GiB다. 운영 공간 확보이며 최종 품질 근거는 아니다.

21:01 UTC 후속: `actual-closed-current-guide-offload-v656-bulk7`에서 실제
완료 audit·raw retirement·로컬 guide가 모두 갖춰진 **14개 그룹/70개 guide**의
중복본 **2,616,690,206 bytes(2.4370 GiB)**만 정리했다. Case 완료 수만으로
15번째 그룹을 포함하지 않고 준비 시점의 종료·reader·열린 파일 및 실제
파일서버 정본 SHA256/크기 검사를 따랐다. Guide 없이 원래 scored RGB·GT·alpha
archive **210개**가 기존 observation과 정확히 일치했다. 최종
`retirement/audit.json`은 `PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`,
실제 file4 receipt는 `PASS`/`local_equals_actual_file4_SHA256=true`다.
정본 guide·scored image·원래 505개 수치 source·실행 중인 그룹은 유지했다.
이 시각 `v636` coordinator와 5개 작업자는 살아 있고 신규 421/1,155 case가
완료됐으며 로컬 여유는 약156.0 GiB다. 이 작업은 저장 공간 운영이며 최종
품질 수치나 모듈 효과의 근거가 아니다.

22:00 UTC 후속: `actual-closed-current-guide-offload-v657-bulk8`에서 실제 완료
audit·raw retirement가 닫힌 **15개 그룹/75개 guide**의 로컬 중복본
**2,960,447,351 bytes(2.7571 GiB)**를 정리했다. 실행 전 완료 그룹·reader·
열린 파일 및 파일서버 정본 SHA256/크기를 검증했다. Guide 없이 원래 scored
RGB·GT·alpha archive **225개**가 기존 observation과 정확히 일치했다. 최종
`retirement/audit.json`은 `PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`,
file4 receipt는 `PASS`/`local_equals_actual_file4_SHA256=true`다. 정본 guide,
scored image, case audit, 원래 505개 수치 source와 실행 중인 그룹은 보존했다.
이 시각 `v636` coordinator와 5개 작업자가 살아 있고 신규 491/1,155 case가
완료됐으며 로컬 여유는 약155.6 GiB다. 이는 운영 공간 확보이며 최종 품질
수치나 모듈 효과의 근거가 아니다.

23:01 UTC 후속: `actual-closed-current-guide-offload-v658-bulk9`에서 다시
완전히 닫힌 **15개 그룹/75개 guide**의 로컬 중복본 **2,366,645,431 bytes
(2.2041 GiB)**를 정리했다. 실행 전 그룹 audit·8개 raw retirement·활성
reader/열린 파일과 실제 파일서버 정본 SHA256/크기를 확인했다. 정리 후 원래
scored RGB·GT·alpha archive **225개**가 guide 없이도 observation과 정확히
일치했다. `retirement/audit.json`은
`PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`, 실제 file4 receipt는
`PASS`/`local_equals_actual_file4_SHA256=true`다. 정본 guide·scored image·
case audit·원래505개 수치 source와 실행 중인 그룹은 보존했다. 이 시각 `v636`
coordinator와 작업자들이 살아 있고 신규 567/1,155 case가 완료됐으며 로컬
여유는 약154.8 GiB다. 저장 공간 운영 기록이며 최종 품질 효과의 근거가 아니다.

23:33 UTC 후속: `actual-closed-current-guide-offload-v659-bulk10`은 새로 닫힌
**7개 그룹/35개 guide**의 로컬 중복본 **1,249,637,871 bytes (1.1638 GiB)**만
정리했다. 준비 기록은 원래505개 수치 source, 살아 있는 coordinator,
완료 그룹 audit와 실제 file4 SHA256을 확인했다. 실행은 active reader와
열린 파일을 검사한 뒤 16+16+3개로 진행했고, guide가 없는 상태에서도
원래 scored RGB·GT·alpha archive **105개**가 observation과 정확히 일치했다.
`retirement/audit.json`은
`PASS_ACTUAL_CLOSED_CURRENT_GUIDE_DUPLICATES_OFFLOADED`, 실제 file4 receipt는
`PASS`/`local_equals_actual_file4_SHA256=true`다. 원격 정본 guide·scored image·
case audit·원래505개 수치 source와 실행 중인 그룹은 보존했다. 실행 뒤 동일
`v636` coordinator와 작업자가 살아 있고 신규 602/1,155 case가 완료됐으며
로컬 여유는 약147.1 GiB다. 이는 저장 공간 운영 조치이며 최종 품질 효과의
근거가 아니다.
