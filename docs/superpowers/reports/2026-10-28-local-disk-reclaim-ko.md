# 로컬 디스크 회수 기록

2026-10-28, 사용자 요청에 따라 `/root/dev/.cache/deformnext-file4`의 과거 file4 복제본을 정리했다. 현재 진행 중인 TEST 생성·렌더의 입력과 체크포인트는 유지했다.

- file4 원본과 이름/크기 parity 및 표본 SHA-256을 확인한 **24개 로컬 캐시 트리, 1096.2 GB**를 제거했다. 원본은 file4에 남아 있다. 전체 파일의 SHA-256을 새로 계산한 것은 아니며, [감사 JSON](2026-10-28-local-disk-reclaim-audit.json)에 각 트리의 수량·바이트·표본 검증을 기록했다.
- 사용하지 않는 10-step smoke 체크포인트 두 파일 **5.6 GB**는 file4로 옮겨 전체 rsync checksum 비교 후 로컬 복사본을 제거했다.
- 종료 시 `/root/dev` 여유 공간은 **1.00 TiB**다. 시작 시 약 0.4 GB까지 떨어졌던 공간을 확보했다.
- 현재 V1/Selected SS 체크포인트, Shape/PBR checkpoint, TEST508 RGB·transport와 native guide 관련 폴더, 활성 렌더 청크는 보존했다. Selected SS 체크포인트가 `consistent-raw-evidence-followup-v177` 안에 있어 이 폴더도 보존했다.
- `rgb_v1/runs_r9_v5`는 로컬 전용 `LOCAL_CACHE_ARCHIVE.json`이 있어 자동 정리에서 제외했다. 과거 mirror 동기화 worker는 cycle 3810의 `synced` 상태와 remote-delete 비활성 확인 후 종료했다.

이 정리는 학습·평가 결과 자체를 삭제하지 않는다. 기존 수치는 리포트에 보존되고 대용량 원본은 file4에서 재사용할 수 있다.
