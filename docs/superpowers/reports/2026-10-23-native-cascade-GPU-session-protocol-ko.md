# Native DEV cascade의 GPU 예약 종료 처리

대기 중인 native8그룹 DEV32 factory의 case/official worker는 자식 프로세스를 기다리지만,
부모가 비정상 종료되는 경우 session의 남은 GPU 자식까지 기다리는 finally 처리가 없다.
원래 worker·학습·평가 계획은 보존하고 외부 실행 래퍼에서 GPU 예약의 수명을 관리한다.
현재 실행 중인4GPU PBR 학습은 수정하거나 재시작하지 않는다.

새 factory revision은 원래128개 case와 official768 작업1개, 총129개 GPU producer를
각각 별도 OS session에서 실행한다. Inner worker가 종료돼도 같은 session의 자식이
살아 있으면 GPU 예약을 반환하지 않는다. Archive 진행 기록 저장이 실패해도 session
drain을 생략하지 않는다. 이후 실제 child returncode0과 원래 producer의 canonical
audit를 검사하고 guardian receipt까지 생성해야 작업 완료로 취급한다.

Inner command vector·plan SHA256·worker SHA256을 검증하며, consumer 검사에는 원래
inner plan 전체를 그대로 노출한다. CPU summary는 GPU producer로 감싸지 않는다.
모델·데이터·camera·sampling seed·sampler·metric·학습 예산을 바꾸지 않는다.
Guardian은 neural forward나 optimizer update를 추가하지 않는다. 물리 GPU0–3만 허용한다.

새 factory에서 import·guardian helper·case/official 감싸기·운영 audit 필드를 제거하면
원래 factory와 AST가 정확히 같음을 실제 두 소스 파일에서 검증했다. 원래 factory SHA는
51cfd803f55d4d55e0d282ef4baa1416d7c96e1607d7e7d9d1620a5a14ad6c78이다.
이 검사는 과학적 실행 본문 보존을 확인하며 실제 GPU 품질 평가를 대체하지 않는다.

Contract 및 기존 session drain의 CPU15개 검사를 통과했다. 실제 OS 부모 프로세스가
exit7한 뒤 살아 있는 자식이 같은 session에 남는 경우와 progress 저장에서 예외가
발생해도 drain이 끝까지 기다리는 경우를 포함한다. 테스트에는 CUDA나 neural 작업이 없다.

적용은 pending native8 factory descriptor 하나의 command와 준비 receipt 요구만
새 revision으로 바꾼다. 기존 job 이름·dependency·priority·outputs·과학적 작업 수258개는
유지하며 queue-state를 편집하지 않는다. 나머지 학습·historical 재현·legacy6 DEV 비교·
독립 seed·최종508개10 metric 범위도 유지한다. 준비·적용·미래 GPU 실행은 별도로 기록한다.
