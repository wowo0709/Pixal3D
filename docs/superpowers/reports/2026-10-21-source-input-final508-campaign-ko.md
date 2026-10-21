# 새 source-input 모델의 최종508 평가 연결

2026-10-21 UTC. 새 Original19/SourceContrast19의 실제 joint480 학습과 최종508
GPU 평가는 아직 실행하지 않았다. 후보 선택이나 test 결과를 만들지 않고,
기존 campaign에 새 evaluator를 명시적으로 연결하는 경로를 추가했다.

## 기존 평가와의 연결

`source_input_frozen_campaign.py`는 별도 실행 진입점이다. 기존 sealed campaign과
population evaluator 파일은 수정하지 않는다. 새 경로를 선택한 process 안에서만
`evaluate_source_input_joint`를 허용하고, 종료나 예외 시 원래 함수와 허용 목록을
복원한다. 이후 예약되는 controller와 GPU stage도 같은 새 진입점을 사용한다.

새 SS 모델은 실제 동일 completed480 run의 config, denoiser, source conditioner를
사용해야 한다. Freeze는 progress/startup/source-input runtime 완료 기록과 새
constructor/loader/evaluator의 source hash도 고정해야 한다. Teacher112 checkpoint를
joint480 결과로 대신하거나, 별도 teacher 초기화 인자를 평가에 넘길 수 없다.

원래 model freeze, final508 cache QA, TRAIN/validation/test의 sample/source/cluster
분리, RGB·GT·noise identity, checkpoint/config/source binding 검사를 그대로 수행한다.
새 경로는 QA나 freeze receipt를 만들거나 후보를 선택하지 않는다.

최종508은 기존 bound population의 native sampling과 SS artifact capture를 사용한다.
개발128용 archive exporter를 중첩 실행하지 않는다. 새 SS 출력 이후의 Shape512,
Shape1024, PBR cascade와 지표 계산은 기존 경로를 사용한다.

## 검사 범위

CPU 검사 **67개가 9.72초에 통과**했다. 새 adapter, 기존 population plan/binding,
기존 campaign의 검사들을 함께 실행했다. 추가한 전체 controller 검사는 실제
plan/freeze 검증과 dispatch 함수를 실행하고, 수치 계산 및 원격 전송만 대체한다.
학습하거나 GPU 추론한 결과가 아니다.

- 여섯 단계 SS → cases → perceptual → eight → distribution → table의 후속 controller가
  새 adapter로 계속 연결된다.
- 5개 모델의 SS와 2,540개 cascade가 유지된다. 기존 source-only/edited-RGB zero-shot
  두 대조군을 유지하고, 동일 asset의 reference 의존성을 유지한다.
- 기존 모델과 perceptual/Inception 수치 계산 진입점은 그대로 사용한다.
- 완료 기록 또는 source pin 누락·변조, 미완료 cache QA, 후보 고정 전 test quality
  열람 표시, teacher112 step 대체를 거부한다. 예외 뒤 원래 binding도 복원한다.
- 임시 queue만 사용하며 실제 queue descriptor/state, 실행 중인 NN, GPU 배치를 바꾸지 않는다.

## 실험을 진행할 조건

소규모 source-contrast 개선 신호는 frozen SS 기반 32개 결과이며, joint SS나 semantic
attention의 효과를 입증하지 않는다. 이미 예약된 별도 72개 자산·3 sampling noise의
Original19/Reference20/Contrast19 비교 및 실제 case identity 검사를 먼저 진행한다.
사전 지정된 Contrast−Original의 paired CI 하한이 양수인 경우에 native warm
field/velocity/backward 정합 및 각 모델의 micro96 capacity 검사를 진행한다.

그 검사를 통과하고 실제 저장 공간을 재확인한 뒤에만, 두 모델 모두 teacher112
초기화를 사용한 동일 SS4210 기반 joint480 학습으로 비교한다. 일부 모델만 teacher를
사용하는 비교로 input 설계 효과를 주장하지 않는다. 독립 학습 seed 확인과 DEV128/32
평가를 거쳐 후보를 고정한 후, 실제 test508 QA와 freeze가 준비돼야 이 adapter를
실행할 수 있다. 현재 이 조건들이 완료됐다는 주장은 하지 않는다.

현재 PBR Direct 4GPU 학습과 기존 예약 작업은 유지한다. 최종 평가는 source
LPIPS/DreamSim/CLIP, target PSNR/SSIM/M-IoU/CD/F-Score 및 공식 Inception의 FID/KID를
포함한다. 이 코드 연결만으로 해당 metric의 실제 계산이 끝난 것은 아니다.
