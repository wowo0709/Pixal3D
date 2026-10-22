# Original zero-shot508과 여섯 모델 cascade 입력 검증

2026-10-22 UTC. Native PBR vanilla는 기존 GPU0–3 학습을 유지하며
586/1900 update 이후로 진행 중이다. 이번 작업은 최종 평가에 원본 zero-shot 두 조건을
정확히 포함하기 위한 연결이며, 새로운 성능 향상이나 test508 평가 완료 결과가 아니다.

## 원본 zero-shot 두 조건을 test 전에 고정

기존 multi-view SS4210 checkpoint와 zero-shot config를 그대로 사용한다.
Deformation fine-tuning update는0이며 기존 evaluator의 `step=0` metadata를
`step=4210`으로 바꾸지 않는다. Multi-view pretraining step4210은 별도로 기록한다.

- `zero-source4`: 변형 전 target 방향 camera8을 포함하는 M0 RGB 네 장.
- `zero-edited4`: 변형된 M1 target RGB camera8과 원본 M0 source9/10/11.

두 조건의 checkpoint/config SHA는 같고 RGB 입력 모드만 다르다. Original Shape5124210,
Shape102416850, PBR102416850의 config와 checkpoint도 함께 고정해 보존했다.
기존 `zero-edited-source3` 기록은 target1+source3의 총4장 조건이며 새 이름은
이를 더 분명하게 나타낸다. Source-only를 세 장으로 축소하거나 depth를 넣지 않는다.

`original_zero_population.py`는 기존 deterministic evaluator의 dataset 선택 지점만
바꾼다. 원래 sampler·CFG·native projection·RGBA 처리·모델 로딩은 유지한다.
실제로 로드된 전체 base와 LoRA tensor를 원본 checkpoint와 비교하고, 평가 전후
LoRA 값과 optimizer state/step도 확인한다. CPU 작업은 저장 tensor의 비교이며
CPU neural forward나 학습이 아니다.

## 실제 설치된 작업과 아직 실행 전인 부분

두 조건×validation36 독립 load2회, 조건별 CPU 반복 판정2개, 최종508 activator1개를
추가해 **7개 작업을 실제 대기열에 연결했다**. 총3522개 dependency DAG에서
누락 참조나 cycle이 없다. 기존3522개 미만의 descriptor나 queue-state는 수정하지 않았다.

Validation36은 기존 완료된128 평가의 prefix이며 앞32개 기록이 정확히 같아야 한다.
마지막 batch4개까지 두 독립 실행의 실제 prediction/GT latent와 occupancy가 같아야
최종 평가를 허용한다. 해당 GPU 검사들은 아직 pending이며 PBR 학습 뒤 실행한다.

Native 네 모델의 개발 평가·독립 학습 seed·재현성 검사와 모델 고정이 완료되고,
원본 두 조건의 반복 검사도 통과하면 activator가 원본 SS508 작업 두 개를 추가한다.
Native 네 모델과 동일한508 sample 순서·134 source cluster·noise map을 사용한다.
현재는 실제 model test forward/score를 열지 않았다.

사전 준비12개 파일243,143 bytes와 append-only queue snapshot2개 파일7,719,445 bytes의
실제 file4 SHA256/size를 검증했다. Canonical 준비 기록은 native-flow 루트 아래
`original-zero-population-validation36-v1/{prelaunch-v1,queue-v1}/`에 있다.
최초 CPU 준비에서 과거 reference의 receipt가 한 단계 상위 model 폴더에 있음을
확인했다. 출력이나 queue를 만들기 전에 멈췄으며, 기존 plan에 기록된 정확한 receipt
경로를 사용해 준비를 완료했다. 실패 전 코드도 보존했다. 학습 프로세스는 건드리지 않았다.

## 전체 cascade 연결 코드

`original_zero_cascade_population.py`는 기존 원본 cascade 수치 코드를 실행하면서
metadata validator만 새508 SS archive 형식으로 교체한다. 새 held-out 데이터에
과거 prefix replay가 있었다고 꾸미지 않는다. 실제 SS508, 원본 base+LoRA 검증,
camera/input mode, 세 downstream stage와 고정된 checkpoint SHA가 맞아야 한다.
GT geometry는 기존 수치 코드와 동일하게 생성 이후에만 읽는다.

`native_final_case_plans.py`는 원본 두 조건과 native 네 모델의 case 계획을 만든다.
Native 모델은 실제 SS480→원본 Shape512/1024를 거친 뒤, 생성된 sparse support에
자기 PBR1900 NN/source0·10·19를 적용하는 image-only replay를 사용한다.
Replay row에는 RGB descriptor와 camera의12개 필드만 전달하며 GT geometry·latent·
projection/matching-label 경로를 condition에 넣지 않는다. 자신의 모델/config SHA와
실제 sample 입력 hash를 명시해야 하며 기존 개발 sample의 input binding을 재사용하지 않는다.

원본 source4가 각 sample의 original-material M1 GT를 같은 renderer로 한 번 생성하고,
나머지 모델은 그 동일 RGB/alpha·camera를 참조하도록 계획한다. Target은 camera8,
source는9/10이다. Cascade seed와 texture seed는 model 결과를 보지 않고 명시한다.
Original zero model과 native trained model의 실행 단계가 다르다는 점도 기록한다.

이 코드는 **계획 생성과 metadata 연결 구현**이다. 최종508 cascade GPU worker·실제 queue,
생성 이후 공통 surface-reference join, bounded raw cache 정리와 official perceptual/
Inception·전체10 metric 집계 연결은 아직 남아 있다. 구현된 factory를 실행 완료로
표현하지 않는다.

## 실제508 입력으로 수행한 CPU 검사

완료된 test cache QA와 canonical asset packet의 ordered508을 대조했다.
전체508의 두 RGB 모드·원본/edited camera8 일치·source9/10/11·세 scoring camera,
immutable asset fields와12-field PBR row를 검사했다. Asset 파일 **4718개·1,246,864,961
bytes**를 실제로 다시 해시해 packet의 SHA와 같음을 확인했다. Descriptor 파일은
모든 sample에서 실제 존재/size를 확인했고 array/schema 검사는 이미 완료된
native cache QA를 참조한다. 이번 검사에서 descriptor array나 GT geometry/latent
array를 로드하거나 모델을 실행하지 않았다.

결과2개 파일8,877,456 bytes를 실제 file4 SHA/size로 검증했다.
Canonical `final508-six-model-case-data-CPU-v1/{audit.json,resolved-case-inputs.json,file4-closure.json}`은
입력 준비의 증거이며 생성 품질의 증거가 아니다. 관련 새 original/cascade/case-plan
및 기존 zero-shot CPU 검사67개가 통과했다. Native population/freeze 검사를 포함한
최종 통합 검증은 **146 passed in1.98s**였다. CPU 검사를 GPU 실행·재현성 통과로
해석하지 않는다.

전체 실험 목표는 진행 중이며 retargeting 이외의 추가 attention 효과나 최종 full 모델
채택 여부는 실제 비교 결과로 판단한다.
