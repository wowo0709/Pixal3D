# 새 warm joint SS480의 최종128 평가 및 feature 사용 진단

2026-10-21 UTC. 새 두 SS 모델의 본 학습은 source72/native proof 이후에만 시작하는
조건부 상태다. 아래 평가는 **실제 동일 run의 완료480 결과**가 생긴 뒤에 진행한다.
현재 새 모델의 GPU 평가나 성능 개선을 완료했다는 보고가 아니다.

## 고정한 평가와 비교 대상

두 모델 각각 같은 개발128개에 sampling noise 3개를 사용한다. 12 sampling steps,
CFG3, generation batch8이며 checkpoint는 고정 step480이다. 정상 평가 6개와 아래의
저장 weight 진단 2개를 실행한다. 기존 128개 panel과 seed map 3개를 그대로 사용한다.
자산별 세 seed가 모두 다르고, 같은 noise끼리는 두 모델에 같은 seed를 사용한다.

비교는 다음과 같이 구분한다.

- Warm Contrast19−Warm Original19: 같은 teacher 학습 예산 아래 source-relative
  입력 설계의 차이. 두 모델 모두 같은 원래 SS4210 기반으로 native SS480을 학습한다.
- Warm Original19−기존 cold Original19: 추가 teacher112 초기화가 있는지의 차이를
  포함한다. 입력 설계가 개선됐다는 근거로 사용하지 않는다.
- 새 두 모델−기존 cold Original19/V1: 기준 성능과의 비교다. 기존 V1은 새 warm
  두 모델과 학습 lineage가 일치하는 비교가 아니며, 개별 모듈 효과로 해석하지 않는다.
  V1을 새로 학습하지 않는다.

TP/FP/FN을 세 noise에 걸쳐 집계하고 source cluster 단위 paired bootstrap으로
차이를 계산한다. 같은 자산의 반복 noise를 독립 학습 seed처럼 세지 않는다.
Per-asset seed·source cluster·GT voxel hash와 geometry를 확인한다. 개발 과정의
반복 비교 CI이며 다중 비교 보정 또는 독립 joint seed 재현을 대신하지 않는다.

## Feature correspondence를 실제로 사용하는지

각 새 모델의 저장된 weight에서 noise0의 동일128개를 한 번 더 평가한다. 실제
forward에서 Q/K/score 및 네 head의 attention 확률, 원래 Top-1 feature와 source-use를
유지하고 **선형 projection 뒤 V 값의 candidate 대응만 순열로 바꾼다**. 이 불변성이
실제 forward에서 확인돼야 진단 결과를 통과시킨다. Parameter나 RGB 입력을 바꾸지 않는다.

정상 결과와 이 진단 결과의 차이는 주어진 matching/attention 아래 전달한 feature
대응의 유용성을 살펴본다. Retraining ablation이 아니며 No-corr의 spatial matching,
No-score의 Top-1, decoder 또는 score의 개별 효과를 대신하지 않는다. 해당 대조군이
필요한지는 실제 결과를 본 뒤 원래 실험 범위를 유지하며 검증해야 한다.

SourceContrast의 실제 작은 CPU 모듈과 기존 구조 검사 **7개가 통과**했다. Normal과
V 순열의 출력이 달라지고, 원래 확률·source-use·Top-1 및 전체 state가 유지되며,
실패 시 hook/source context가 복원됨을 확인했다. 이는 native SS GPU 추론이나
성능 향상 증거가 아니다. 기존 numerical source 파일은 수정하지 않았다.

## Cascade용32는 같은 생성 결과에서 추출

정상 noise0 평가 중 native128 sampler와 decoder가 실제 생성한 prediction/GT latent와
occupancy를 한 번 저장한다. 나머지 정상 noise1/2와 V 진단에는 archive를 중첩하지 않는다.
최종 평가 때 teacher checkpoint나 M1 source teacher label을 모델 입력으로 로드하지 않는다.

기존에 지정한32개가 이128개에 포함되고 noise0 seed가 일치함을 실제 파일에서
확인했다. CPU selector는 parent file4 receipt와 artifact hash를 확인한 뒤 정확한
32개 prediction 파일을 byte 그대로 복사하고 지표를 다시 계산한다. 빈 결과도 유지하며
GT Shape/SS로 대체하거나 성능을 보고 자산을 제외하지 않는다. 이 결과로 native
64→32 Shape support를 기록해 이후 cascade에 넘긴다. 두 번째 SS sampling은 없다.

## 남은 판단

이 연결은 정상 평가6·V 진단2·CPU subset2·비교 요약1의 워크플로다. 실제 완료480
checkpoint/config/progress/startup/source runtime와 canonical receipt가 있어야 GPU
평가가 실행된다. 실패는 보존하고 자동 재시도하지 않는다. 결과와 비교 표는 파일서버4와
WandB에 기록하며, 이 요약 자체로 후보를 채택하지 않는다.

이후 새32개의 full cascade 생성과 source LPIPS/DreamSim/CLIP, target PSNR/SSIM/
M-IoU/CD/F-Score, 적절한 matching 대조군과 두 번째 joint seed 확인, 고정 후보의
최종 test508 및 공식 FID/KID가 여전히 필요하다. SS128 voxel 결과를 이 전체 평가의
완료로 바꾸어 표기하지 않는다.
