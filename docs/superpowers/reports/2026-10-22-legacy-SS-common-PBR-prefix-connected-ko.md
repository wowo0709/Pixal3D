# 공통 PBR 비교의 engineering prefix 실제 연결

기존 retargeting(C1)·feature-spatial·feature-DQ의 같은 첫 asset에 original PBR16850과
공통 native vanilla PBR1900을 적용하는3개 GPU 작업과 CPU 검증1개를 추가했다.
전체96개 case의 비교를 완료한 것이 아니며, 새 GPU 결과는 아직 없다.

원래96개 geometry의 SS prediction·producer plan·Shape config/checkpoint 기록·RGB
binding 등549개 입력이 기존 canonical receipt와 일치하는지 확인했다. 계약 관련
CPU synthetic14개 검사와 worker 구문 검사를 통과했다. 실제 GPU forward는 대기 중이다.
구현과 프로토콜은 `ac44b00`으로 요청한 원격 브랜치에 push했다.

현재 Top-1+R PBR1900 학습과 기존 실험 설정은 바꾸지 않았다. Prefix는 예정된 native
8그룹 DEV32 cascade의 실제 summary closure를 기다린다. Retargeting·spatial은 GPU2에서
순서대로, DQ는 GPU3에서 실행하도록 연결했다. 각 작업의 모든 GPU 자식이 끝난 뒤
예약을 반납하며, 원래 geometry를 덮어쓰거나 새 Shape inference를 수행하지 않는다.

Prelaunch13파일1,201,060bytes, queue2파일7,813,836bytes, 적용 기록1파일678bytes를
실제 file4 SHA256/size로 검증했다. Queue3553개가3557개가 됐고 기존3553개 descriptor는
모두 그대로다. 전체 DAG를 확인했으며 queue-state를 편집하지 않았다.

Canonical root는 `native-flow-cold3-v1/legacy-SS-common-native-PBR-bridge32-v1/`이다.
Engineering prefix가 통과하면 남은93개·own raw 정리·official576개 view·native8개와
합친14그룹 비교를 구현·연결해야 한다. Prefix의 존재를 전체 비교 완료나 모듈 채택으로
취급하지 않는다. Historical 재현, 독립 학습 seed, 최종 모델 선택과508개10 metric도
계속 남아 있다.
