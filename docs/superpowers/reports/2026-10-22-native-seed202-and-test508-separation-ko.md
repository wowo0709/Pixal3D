# Native seed202 재현 설정과 최종 test508 분리 확인

## 현재 학습과 재현 범위

PBR vanilla는 실제 native1900 학습을 계속 진행 중이다. 이 작업의 모델·data·
batch·optimizer·프로세스는 변경하지 않았다. 나머지 세 PBR arm과 full-cascade
평가도 기존 순차 실행 계획을 유지한다.

SS에서는 기존 seed101의 최종128개·세 noise에서 Top-1+R의 V1 대비 개선이
관측됐다. 이를 독립 학습 seed202에서 확인하기 위해 실제 완료된 native
V1/Top-1+R 설정을 바탕으로 두 모델의 proof/capacity/science 설정 여섯 개를
만들고 file4 SHA/size를 검증했다.

Seed202에서 바꾼 필드는 training seed, output, W&B 이름, 단계별 logging과
engineering 표시, Top-1의 recipe reference뿐이다. TRAIN9149, 원본 SS4210
base+LoRA 초기화, fresh optimizer, global96/micro96/accumulation1, 내부 FFN
chunk24, native flow objective, learning rate, dropout, 480 update/5에폭,
매96 update 고정32 validation은 유지한다. Matcher와 Omega는 기존 frozen
가중치를 공유하며 이들을 독립적으로 재학습하는 실험은 아니다.

Top-1 validator가 요구하는 R19 reference도 같은 seed202로 만든다. 이는
수치 계약을 비교하기 위한 **설정 파일**이며 세 번째 모델을 학습한 결과가 아니다.
바뀐 필드 외에는 원래 설정과 완전히 같음을 역변환해 확인하고, seed101 혼입·
batch·teacher loss·learning rate·data 변경을 거부하는 CPU 검사9개가 통과했다.

두 native identity/backward proof와 두 micro96 capacity 작업을 기존 queue에
추가했다. 네 PBR1900의 실제 완료와 cache 정리 뒤에 실행한다. 현재는 네
engineering 작업이 pending이며, 실제 seed202 GPU 결과가 나온 것은 아니다.
두 source0/10의 실제 cold 동등성·gradient·3 update·공통 noise/data trace 검증,
본 학습의 disk reservation, 최종128×3와 두 학습 seed의 paired 비교 연결은 남아 있다.

## Test508은 개발용 validation508과 별개다

현재 학습이 읽는 train9149와 validation508, 준비된 native test508 및 원래
legacy test508 manifest를 직접 읽고 실제 file4 SHA/byte와 다시 대조했다.

| Identity | Train ∩ validation | Train ∩ test | Validation ∩ test |
|---|---:|---:|---:|
| sample_id | 0 | 0 | 0 |
| source_uid | 0 | 0 | 0 |
| source_cluster_id | 0 | 0 | 0 |

현재 DEV32와 DEV128은 validation에만 속하며 test508과 겹치지 않는다.
Native test manifest의508개 순서는 원래 test manifest와 같다. 기존 실제
all508 native cache/loader QA의 published-manifest SHA도 현재 파일과 일치한다.
이번 점검은 metadata만 읽었고 candidate test forward, 품질 지표나 geometry
값은 읽지 않았다. 모델 선택을 동결한 뒤의 native test adapter와10개 metric
평가는 여전히 필요하다.

실제 새 근거는 `native-flow-cold3-v1/final508-current-split-audit-CPU-v1/`
`{audit.json,file4-closure.json}`이다. Seed202 설정은 같은 root의
`independent-seed202-v1/preparation-v1/`, engineering 계획과 queue snapshot은
`engineering-prelaunch-v1/` 및 `engineering-queue-v1/`에 있다.

## 시작 전 실행 인자와 저장 공간 점검

DDP4 복구에서 복제했던 역사적 구조 재현 작업13개의 CLI에는 새 plan 경로와
이전 plan의 checksum이 함께 들어 있었다. 모두 시작 전인 것을 확인하고,
각 새 plan이 이미 검증·저장된 file4 byte임을 대조한 뒤 CLI checksum만
바로잡았다. 원래 bound plan·source·설정·과학 결과는 수정하지 않았다.
현재 해당13개 command의 checksum과 실제 plan SHA가 모두 일치한다.
변경 전후 전체 queue snapshot과 원인은 `DDP4-recovery-v1/command-pins-v1/`
에 canonical 기록했다.

완료된 native SS의 step96 replay·observer·최종 평가 소비자도 점검했다.
Pending/running reader가 없는 옛 step96 local checkpoint11개만 실제 원격
SHA/byte와 다시 대조해 정리했고 **10.6807 GiB**를 회수했다. File4 canonical
checkpoint, 최종480, 원본4210/16850, 역사적 재현에 쓰는 별도의 warm96 및
dataset은 유지했다. 초기 소비자 수 검사와 준비 폴더 생성에서의 CPU 실패
로그도 보존했으며 실패한 모델 학습을 재시작하거나 성공으로 바꾼 일은 없다.
