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
