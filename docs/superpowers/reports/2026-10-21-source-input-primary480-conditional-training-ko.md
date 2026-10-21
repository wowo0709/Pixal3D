# Teacher-warmed 두 SS 모델의 조건부 본 학습 연결

2026-10-21 UTC. 앞선 source-disjoint72 비교와 두 모델 각각의 native warm 정합·
micro96 검사를 통과한 뒤 진행할 Original19/SourceContrast19의 본 학습을 연결한다.
이 문서는 새 학습이 완료됐거나 개선이 입증됐다는 보고가 아니다.

## 동일하게 유지할 비교 조건

두 모델 모두 원래 multi-view LoRA SS4210에서 fresh scientific optimizer로
시작하고, 자신의 TRAIN1536 teacher112 전체 source state를 utility 로딩 **뒤**, native
optimizer 생성 **전**에 로드한다. 둘 다 같은 추가 teacher 학습 예산을 사용한다.
SourceContrast만 teacher로 초기화하는 비교를 하지 않는다.

- 실제 DeformNeXt TRAIN 9,149개, validation 508개를 사용한다. 이전 multi-view
  Pixal3D 학습의 clean80k를 deformation 학습 데이터로 재사용하지 않는다.
- 각 모델 world1/global96/micro96/accumulation1, training seed101, 480 updates다.
  서로 다른 GPU 한 장씩 사용할 수 있다. FFN chunk24는 gradient accumulation이 아니다.
- 원래 AdamW, learning rates, LoRA360개 tensor, source19개 tensor, AMP/BF16,
  sampler, camera/RGB, latent, frozen matching head와 native projection을 유지한다.
- 시작 시점과 매96 update마다 같은 validation32로 voxel 지표와 시각화를 기록한다.
  최종 checkpoint는 고정 step480이며 best epoch를 선택하지 않는다.

`source_input_joint_pair.py`는 원래 full recipe validator로 두 config를 검사하고,
실험 이름·결과 경로·명시한 source-input 설계를 제외한 전체 config가 일치해야
허용한다. Model/data/cache/optimizer/validation이 달라지면 거부한다.
새 factory의 실제 데이터 경로를 사용한 CPU 확인에서 train9149/validation508과
sample/source UID/source cluster overlap 0을 확인했다.

## 본 학습에 앞서 실제로 필요한 결과

기존 source72 admission은 **engineering 전용**이므로 그 기록만으로 본 학습을
시작하지 않는다. 별도 scientific admission에서 다음을 확인한다.

1. 사전 지정한 별도72 자산·61 source cluster·3 sampling noise의
   Contrast−Original gain과 paired CI 하한이 모두 양수다.
2. 두 모델의 실제 native warm proof가 같은 TRAIN RGB·latent를 사용했고,
   standalone teacher와 native global/projection/negative 및 same-noise velocity가
   일치한다. 원래 LoRA와 source19에 실제 SS loss gradient가 도달하고 optimizer는
   업데이트하지 않았으며 matcher와 원래 parameter가 유지됐다.
3. 두 모델 각각 실제 micro96/global96/world1/accumulation1 capacity가 통과했다.
   세 disposable optimizer update, frozen-weight integrity, 실제 gradient 및
   원래 optimizer 상태를 확인한다. 이 disposable 모델로 과학적 학습을 이어가지 않는다.
4. 정합 proof·capacity·본 학습이 같은 각 teacher112 checkpoint/audit hash를 사용한다.
5. 학습 쌍 전체의 checkpoint 예산과 추가 여유 공간을 현재 파일시스템에서 확인한다.
   실행 직전에도 다시 검사하며 데이터·checkpoint를 자동 삭제하지 않는다.

두 본 학습이 진행되면 실제 완료480·checkpoint·epoch validation 기록을 검사하고
파일서버4의 실제 SHA256/크기 검증을 마친 뒤에만 완료 처리한다. 실패 시 로그와
checkpoint를 보존하며 자동 재시작하거나 다른 배치로 바꾸지 않는다.

## 검사 범위와 아직 남은 평가

새 config/engineering prerequisite 검사의 **CPU 24개가 통과**했다. 사용한 proof
fixture는 synthetic metadata이며 실제 GPU 통과 증거가 아니다. 실제 source72와
native warm/capacity 결과는 기존 예약 작업이 생성해야 한다. 현재 PBR 4GPU 학습과
기존 실행 파일·optimizer·대기열 상태를 변경하지 않는다.

학습 후 두 모델의 native epoch32 곡선과 paired voxel 차이를 비교하고 WandB에
기록한다. 이 곡선으로 후보를 고정하거나 최종 개선을 주장하지 않는다. 최종
native128×3, DEV32 cascade, 필요한 개별 matching/score/decoder/correspondence
대조군, 두 번째 joint seed 검증 및 최종 test508 절차가 여전히 남아 있다.

이 비교의 직접 목적은 **같은 matching+Top-K+feature attention+retargeting 아래서
source-relative 입력 설계의 효과**를 확인하는 것이다. 이것만으로 attention,
score 또는 decoder의 개별 효과를 입증하지 않는다. 기존 cold V1과의 비교에는
추가 teacher 학습 여부의 차이도 있으므로 그 효과를 source-input 설계 하나로
돌리지 않는다. 실제 개선이 없으면 부정 결과를 보존하고 원인 분석을 계속한다.
