# Native SS480의 matching score와 Top-K 소비 경로 진단

후보 탐색 개선이 최종 품질로 연결되지 않는 병목을 좁히기 위해, 실제 native
SS flow objective로480 update 학습한 R_raw_topk 체크포인트를 고정하고 아래
inference 개입을 연결했다. 이전 feature-teacher112 모델의 결과와 구분한다.

| 조건 | 고정하거나 변경하는 경로 | 확인할 질문 |
|---|---|---|
| Normal shadow | 같은 입력으로 normal을 추가 실행하고 실제 기존32개 기록을 정확히 재현 | 관측 코드가 동작과 RNG를 보존하는가? |
| Wrong retrieved values | semantic 후보·score·Top-1 carrier·source use를 유지하고 후보 안에서 V 대응만 순열 | 후보 내 값의 대응이 실제 출력에 영향을 주는가? |
| Semantic prior only | learned attention weight residual을 제거하고 frozen matching posterior로 집계 | 학습한 Q/K·utility의 weight 조정이 필요한가? |
| Top-1 feature only | feature 집계 확률을 matching Top-1 하나에만 부여 | 저장된 모델에서 Top-K 집계가 Top-1보다 유용한가? |
| No transformed value | 실제 posterior와 raw Top-K 집계를 유지하고 learned V correction만 제거 | learned feature 변환이 추가로 기여하는가? |

이는 **재학습한 ablation이 아니다.** Top-1 feature 개입은 새 physical Top-1
학습 arm을 대체하지 않고, semantic-prior-only는 uniform No-score가 아니다.
Wrong-V도 spatial No-corr 학습을 구현하지 않는다. Decoder의 독립 효과를
이 검사만으로 입증하지 않는다.

다섯 조건 모두 동일 native480 NN/source·config·RGB·GT voxel·sample 순서·
source cluster·생성 noise를 사용한다. 고정 개발32개, generation batch8,
12-step/CFG3, 기존 deterministic reduction을 유지한다. Native 모델 초기화와
loader, pretrained weights, sampler 및 수치 코드는 변경하지 않았다.
추가 학습 update와 teacher supervision은 없다.

각 module forward에서 normal shadow와 candidate index·semantic prior·
Top-1 carrier·source use·semantic fusion scale·source row·CPU/CUDA RNG·
parameter version이 같은지 검사한다. 실제 source correction이 소비되는 위치의
delta RMS와 변경 channel 비율을 기록한다. Matching prior/learned posterior의
effective candidate 수와 Top-1 밖 확률, prior로부터의 KL, R 입력/출력 크기도
실제 native forward에서 기록한다.

첫 normal 검사는 기존 archive32의 전체 per-asset 기록·quick metric·loaded
trainable SHA를 정확히 재현해야 한다. 그 실제 canonical proof와 terminal0이
완료돼야 나머지 네 조건을 시작한다. 실제 output이 달라지면 음성 결과와 원래
파일을 보존하며, 정상 검사 통과나 개입 효과를 미리 가정하지 않는다.

현재 실행 중인 PBR vanilla는 유지한다. Vanilla1900 완료 후 실제 checkpoint
cache 보관/정리가 끝나면 normal을 GPU 하나에서 실행하고, 이후 네 개 개입을
GPU0~3에서 하나씩 병렬 실행한다. 진행 중인 학습을 끊지 않고 PBR 모델 간
구간을 사용한다. 이후 기존 PBR source 학습이 이어진다.

GPU worker는 실제 자식 PID/birth/terminal을 기록하고, 소유 OS session의
하위 프로세스가 모두 종료될 때까지 GPU 예약을 유지한다. CPU 집계는 실제
voxel counts로 metric을 재계산하고 source-cluster bootstrap4000회/seed20261023의
paired interval을 보고한다. 한 학습 seed·한 생성 noise의 개발 진단이며,
보정 전 interval을 attention 성능 개선의 확정 근거로 사용하지 않는다.
결과 표와 차이/CI를 W&B youngwoojeon-unist에 기록한다.

새6개 job으로 queue는3529개가 됐다. 기존 descriptor·queue-state·학습 config는
바꾸지 않았다. Prelaunch15개452,797 bytes와 queue2개7,736,355 bytes를
실제 file4 SHA/size와 대조했다. Native480 입력 경계 및 기존 개입·signal
관련 CPU25개 test가6.49초에 통과했다. 실제 GPU 진단은 아직 대기 중이다.

Canonical root는 native-flow-cold3-v1 아래
`native480-readout-diagnostics-v1/`이며 `summary-v1/audit.json`은 실제 다섯
평가 종료 후 생성된다. 최종 모델 선택 전에 이 결과와 PBR isolation·full cascade·
독립 학습 seed 결과를 함께 검토한다. 전체 실험과 test508/10 metric은 아직 남아 있다.
