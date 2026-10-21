# Source teacher 초기화의 native SS 검증 순서

2026-10-21 UTC. [Source-input pilot](2026-10-21-source-input-pilot-results-and-joint-integration-ko.md)의
소규모 개선을 곧바로 full 모델 효과로 취급하지 않는다. 실제 source-disjoint72×3
확인과 native GPU 검증을 통과한 뒤 동일 teacher supervision의 Original19와
SourceContrast19를 비교한다.

## 확대 확인에 따른 실행 조건

이미 예약한 72 assets/61 source clusters × sampling noise 3개 결과와 실제 asset/source
binding 검사가 완료된 뒤 판단한다. 주 비교는 같은 19개 parameter를 사용하는
**Contrast−Original의 source-cluster bootstrap 95% CI 하한 >0**이다. 결과를 보기 전에
이 기준을 기록한다. Reference20과의 비교도 보고하되 이 engineering 실행의 주 기준은
parameter 수가 같은 두 모델의 비교다. 세 generation noise는 독립 training seed가 아니다.

조건을 만족하지 않으면 새 native GPU proof/capacity 실행을 열지 않는다. 이 판단은
모델 채택 또는 full SS 학습 승인과 다르다. 현재 정상 학습 중인 PBR Direct와 기존
keeper/dispatcher를 유지하고, 기존 post-PBR lease 안에서만 실행한다.

## Warm native proof: 모델별 GPU 한 장, optimizer update 0

`probe_source_teacher_native_initialization.py`가 다음을 검증한다.

1. 원래 multi-view SS4210 base+LoRA를 로드하고, historical utility 뒤에 해당 모델의
   teacher112 전체 state를 strict load한다. Native optimizer를 생성할 때 source19와
   LoRA360이 모두 연결되고 optimizer state는 비어 있어야 한다.
2. 입력만으로 보정 support가 존재함을 확인한 동일 TRAIN sample 하나를 두 모델에
   사용한다. RGB는 M1 target8와 M0 source9/10/11이다. Conditioning hash와 실제
   RGB/GT SS latent tensor hash를 기록한다. 추가 teacher label은 읽지 않는다.
3. Integrated conditioner와 standalone teacher checkpoint를 같은 native encoder와
   projection 경로에 넣어 global/proj/negative의 실제 tensor가 정확히 같은지 확인한다.
   Global/negative는 원래 native 조건과도 같아야 한다. Warm source의 projection
   보정은 실제로 0이 아니어야 한다. CPU/CUDA RNG를 보존한다.
4. 같은 noise/timestep의 실제 denoiser velocity와 SS flow loss도 정확히 같은지
   확인한다. 이 read-only 비교에서 deterministic algorithms를 사용한다.
5. 실제 SS flow loss를 backward해 source19의 finite nonzero gradient와 LoRA gradient를
   확인한다. 여기서는 production bilinear projection backward의 원래 CUDA 알고리즘을
   사용한다. Optimizer update는 없으며 source state와 LoRA weight는 그대로여야 한다.

Reference 교체는 world1의 두 model mapping이 alias인 경우까지 복원하며 optimizer의
parameter 참조를 교체하지 않는다. Warm 초기 함수를 cold native zero-output과 같다고
주장하지 않는다. 이 검사는 batch1의 mechanics이며 생성 품질이나 micro96 증거가 아니다.

## 별도 micro96 capacity

각 warm native proof 이후 기존 native capacity 경로를 새 source-input binding으로
실행한다. Global96/world1/micro96/accumulation1, 원래 FFN chunk24와 BF16/AdamW/
LoRA/sampler를 유지하고 **실제 update 3회**를 수행한다. Frozen 상태 일치, source19
gradient, 원래 LoRA360과 source optimizer state379, 메모리·시간을 검사한다.
이 임시 모델과 optimizer는 버리며 scientific checkpoint는 내보내지 않는다.

두 모델의 proof와 capacity를 모두 통과한 뒤 다음 joint 학습을 준비한다. 이 단계에서
추가 full SS 학습을 예약하지 않는다. 다음 학습은 동일 teacher supervision을 받은
Original19/Contrast19의 TRAIN9149/native480 비교이고, 독립 training seed와
source/target/full-cascade 평가 요구는 그대로 유지한다.

## 현재 검증 범위

새 warm-forward/reference 복원 CPU 검사 5개와 이전 검사 28개가 통과했다.
실제 native GPU proof와 warm micro96 capacity는 아직 실행하지 않았다.
진행 중인 PBR의 GPU를 나누거나 기존 queue 상태를 바꾸어 검증을 시작하지 않는다.
