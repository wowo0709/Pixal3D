# TEST508 완료 그룹 재사용과 디스크 복구

## 후속: 완료된 재사용 guide의 NAS 보존·로컬 offload

v601에서 재사용이 확정된 기존129개 group의 **guide645개,24,524,331,737 bytes
(22.840GiB)**를 실제 NAS hash·크기와 다시 대조한 뒤 로컬 중복본만 정리했다.
현재 실행 중인 v595와 partial group,scored image,case/group/terminal metadata,
GT/data·모델·checkpoint는 선택하지 않았다. Canonical guide는 원래 file4 경로에
그대로 있다. v592 당시 guide 보존과 구분되는 후속 lifecycle이다.

먼저 완료된129개 group의 다섯 case와8개 raw consumer,실제 phase/worker 종료를
확인했다. 두 dispatcher와 old/current private reader ledger,실제 open file을 검사하고,
NAS intent를 먼저 닫은 뒤 reader·inode를 다시 확인했다. **41개 batch가 실제 완료**했고
큐 descriptor가 바뀐9개 attempt는 삭제하지 않고 defer·보존한 뒤 새 검사로 진행했다.
Pending/HELD/unknown reader나 unreadable plan,protected process를 생략하지 않는다.

Guide가 모두 로컬에 없는 상태에서 원래 scored-pixel reader로 **1,935개 archive를
다시 읽어 producer RGB/GT/alpha·uint8 observation과 모두 일치**함을 확인했다.
후속 official metric·표·그림은 이 scored archive와 보존된 metadata를 사용한다.
CPU NN forward/학습은0이다. 실제 경계 조건79개 검사도 통과했다.

원래505개 과학·수치 source와 v595 context·coordinator·GPU worker를 수정하거나
재시작하지 않았다. 실제 여유 공간은 약174GiB로 늘었다. 이는 새 group의 기존
disk admission을 돕는 운영 조치이며 max2·case4GiB·workspace16GiB 조건이나
과학적 arm·평가 protocol을 바꾼 것이 아니다. 아래04:26 관측은 그 시점의 이력이다.

실제 근거: `actual-immutable-reused129-guide-offload-v601/prelaunch-v1`,
`actual-immutable-reused129-guide-offload-v601/retirement/audit.json` 및 실제 NAS receipt.

2026-10-26 04:26 UTC 실제 관측이다. 최종 선택은 TEST 전 v558에서 동결한
**Fixed SS + Top-1 R PBR**다. SS는 geometry CD/F-Score, PBR는 동일 geometry의
source appearance를 기준으로 해석한다. 이번 작업은 운영 복구이며 새로운 과학적
arm이나 TEST 이후 모델 선정이 아니다. 최종10metric과 결과 의존 writing은 미완료다.

## 실제 중단과 보존

v586은651개 case 완료 뒤 case-level disk admission에서 종료됐다. 외부 파일 증가로
실제 free space가 이미 허용한 future credits와 workspace를 충족하지 못한 것이
직접 원인이다. 실패 worker log의 오류는 `actual disk no longer covers all admitted
consumers and workspace`다. CUDA OOM 또는 모델의 품질 실패로 분류하지 않는다.
실제 coordinator·worker·두 keeper의 종료와 descendant drain을 확인했다.

전체651개 가운데 다섯 모델이 모두 완료되고 raw retirement까지 끝난 것은
**129개 asset group,645개 case**다. 나머지6개 완료 case와 실제 partial/실패 worker
기록은 그대로 보존한다. 불완전한 세 group은 새 namespace에서 다섯 모델 모두
실행한다. 원래 failed result·case·terminal을 덮어쓰거나 새 terminal로 대체하지 않는다.

## 디스크 정리와 예약 해제

완료된 DEV640의 늦은 geometry CD/F 평가도 끝났으므로, 실제 canonical NAS SHA와
크기·producer receipt·모델·reader·inode를 재검증한 뒤 **로컬 geometry/prediction.npz
640개,79,804,772,166 bytes(74.324GiB)**만 정리했다. 원본 NAS raw, guide,
scored image, 통계·모델·데이터는 보존했다. v591은128개를 안전하게 정리한 뒤
이전 큐의 stale HELD history 문제로 거부됐으며, 그 실패와 네 성공 batch를 닫았다.
새 v592 정책에서 증명된128개만 명시적으로 이어받아 나머지512개를 완료했다.

원인은 실제 descriptor가 없는 이전 큐의 HELD history가 현재 소유 큐의 completed
상태를 덮어쓴 것이었다. 상태는 실제 descriptor의 소유 큐에 귀속하도록 수정했다.
실제 HELD/pending/running descriptor와 unreadable plan, protected/open reader는
계속 보호한다. 중복 descriptor의 두 소유 큐는 허용하지 않는다.

실제로 종료가 확인된 v586의15개 future credit,160GiB도 각 cancellation proof를
먼저 실제 NAS에 보존한 뒤 해제했다(v593). 이는 장부상의 예약 해제이며 물리적
디스크를160GiB 정리했다는 뜻이 아니다. 정리 후 실제 free space는 약153.8GiB다.

## 재사용의 실제 증거

최종 코드 pin을 사용한 v596은129개 original group과645개 native case의 실제
worker/phase terminal, 원래 SS prediction hash, frozen SS/Shape/PBR binding,
parent/control freeze와 **1,935개 scored archive의 RGB/GT/alpha·uint8 identity**를
다시 확인했다. 모델 사이 original-material M1 GT가 같다는 것도 확인했다.
이 재사용 결정은 완료·lineage만 사용하며 TEST 품질로 사례를 선택하지 않는다.

원래505개 과학·수치 source는 모두 동일하다. 새 운영 adapter는 남은379개만 실행해
기존129개와 합친 실제508×5 case index를 만든다. Metric loader는 원래 case-root
표현식 한 곳만 두 실제 producer namespace에 연결한다. 다른 AST 문장과 원래의
모델·SS·terminal·pixel 검사는 유지하며 원래 source 파일을 수정하지 않는다.
11개 GPU metric worker의 원래 command와 numerical plan도 유지한다.

검사는47개 DEV retirement/process checks,85개 reuse·serialized actual CPU worker·
strict retirement checks,추가28개 최종 pin 호환성 검사까지 실제 통과했다.
초기 검사 실패도 보존했다. Fixture의 shared-dict/중복 descriptor 문제는 수정했고,
원래 비동기 CPU fixture에서 드러난 unlink/stat race는 동기적 실제 CPU worker
실험으로 운영 경로가 회피함을 확인했다. Frozen 원래 source를 고쳐 통과시키지 않았다.

## 현재 실제 실행과 후속 작업

v595의 실제 coordinator는 PID3225511/birth1792988661.44다. 두 원래 dispatcher가
GPU0–3 및6–7의 keeper를 실제로 실행 중이다. 첫 두 zero-source4 case worker는
GPU0/1에서 시작했고04:26 관측 시 살아 있다. 각 asset의 공통 material reference가
완료되면 나머지 모델을 병렬 배치한다. 새 물리적 목표는379 assets/1,895 cases이고,
재사용645개를 포함한 최종 논리적 목표는 원래508 assets/2,540 cases다.

최대 open group은2,case당4GiB,workspace16GiB를 유지한다. 실제 free space가
허용하면 두 group을 사용하고 부족하면 새 whole-group admission을 기다린다.
Case의 원래 atomic disk guard도 유지한다. Max2는 모델 batch나 수치 protocol을
바꾸는 설정이 아니다. 추가 외부 disk 증가가 없다는 보장은 하지 않는다.
완료129개에서 관측한 retained footprint를 남은379개에 외삽하면 약89.9GiB이고
예상 종료 free space는 약63.9GiB다. 개별 asset 변동을 포함한 확정 상한은 아니다.

1. 새379개 실제 case·raw retirement·drain과 기존129개 검증을 합쳐 전체508을 닫는다.
2. v595 후속 gate가 original perceptual5개와 Inception6개 GPU 작업 및 원래 collector를
   실행한다. Source cluster134,원래16 adjusted endpoints,1,524 images/model을 유지한다.
3. v597은 실제10metric 완료 후 원래 Main·사전 qualitative exporter를 실행한다.
4. v598은 실제 Main50개 numeric cell·16구간·모든 실제 qualitative PNG를 W&B에
   게시하고 원격 table을 ULP까지,PNG를 RGBA pixel까지 대조한다.
5. 그 실제 결과로 Abstract/Results/Discussion/Conclusion을 완성하고 최종 PDF를
   compile·시각 검수·NAS 보존한 뒤 두 원래 계획 전체를 completion audit한다.

현재 callback은 실제 등록됐고 선행 완료를 기다린다. Metric·Main·qualitative의 최종
artifact는 아직 없으며 개발 PDF를 최종 논문이라고 표시하지 않는다.

실제 보존 근거: `actual-TEST508-disk-admission-failure-preservation-v590b`,
`actual-completed-DEV-geometry-raw-retirement-v591/actual-partial-terminal-v1`,
`actual-remaining-DEV-geometry-raw-retirement-v592`,
`actual-dead-TEST508-credit-cancellation-v593`,
`actual-complete-TEST508-group-reuse-verification-v596`,
`actual-selected-five-model-TEST508-remaining-cascade-v595`,
`actual-remaining-TEST508-and-metrics-activation-v595`,
`actual-TEST508-main-and-qualitative-export-activation-v597`,
`actual-final-TEST508-WandB-activation-v598`.
