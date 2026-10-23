# Native cascade GPU session 처리의 실제 연결

Pending 상태인 `activate-native-SS480-PBR1900-matched-fullcascade-v1`의 command와
준비 receipt 요구만 새 guarded factory revision으로 변경했다. 기존 이름·dependency·
priority·outputs와 과학적 작업258개는 그대로다. 나머지3747개 descriptor도 정확히
유지했고 전체3748개 DAG를 검증했다. Queue-state는 편집하지 않았다.

새 factory는 같은128개 case와 official768 GPU 작업을 생성하되, 각 원래 worker를
별도 OS session에서 감싼다.129개 GPU producer의 자식이 모두 종료되고 실제 원래
audit가 canonical receipt와 일치해야 guardian 완료를 기록한다. 기존 scientific worker와
activation plan은 원래 SHA 그대로 보존했다.

계약·OS session drain CPU15개 검사,2개 CLI help 및 구문 검사를 통과했다. 실제 두
factory 소스에서 운영상 추가 부분을 제거한 AST가 원래 과학적 본문과 정확히 같았다.
새 source/protocol은a2725aa로 요청한 원격 브랜치에 push했다. 새 GPU forward를
실행하거나 품질 재현을 확인한 것은 아니다.

| 실제 file4 SHA256/size 검증 | Files | Bytes |
|---|---:|---:|
| Revision prelaunch | 12 | 519,154 |
| Revision queue | 2 | 8,233,918 |
| Revision applied | 1 | 1,259 |

Canonical 경로는 `native-flow-cold3-v1/matched-fullcascade-v1/`
`GPU-reservation-revision-v1/`이다. 적용 당시 실제 Top-1 PBR native/outer PID와 birth를
다시 확인했고 학습은1161에서1196 update로 계속 전진했다. 진행 중인 학습·config·
checkpoint·epoch observer는 수정하거나 재시작하지 않았다.

이 변경은 실행 수명 관리다.3에폭 fixed4의 관측을 넘어서는 새 성능 근거는 없다.
나머지 PBR 학습·historical 재현·native8/legacy6 DEV·독립 TRAIN seed·실제 모델 선택과
최종508개10 metric 및 scientific report까지 계속 남아 있다.
