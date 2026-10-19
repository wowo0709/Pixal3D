# Frozen matcher와 utility attention 후속 실험

## 첫8개 구조의 실제 고정 결과

동일 DeformNeXt128개·93source·3개 생성 noise,각480update/5epoch의 결과다. V1은 IoU43.2650%,matched Top1+C1은44.2757%,기존 C1 T0.005는44.2193%,static-C1 T0.05는44.0938%,learned Full은43.9236%,native-spatial은44.1701%였다. 실제 `candidate-selection-v1.json`은 `NO_PROMISING_COMBINATION_YET`이며6개 Top-K 후보 모두 같은 retargeter의 Top1 대조 기준을 통과하지 못했다. 모델을 채택하지 않는다.

Static-C1−Top1+C1은 −0.182pp,95% source-cluster CI[−0.308,−0.079]pp,Full−Top1+C1은 −0.352pp,[−0.624,−0.127]pp였다. 기존 C1−Top1+C1은 −0.056pp,[−0.131,+0.007]pp로 추가 이득이 확인되지 않았다. Retargeter를 더한 Top1+C1−Top1은 +0.823pp,[+0.427,+1.222]pp였다. 이 결과는 retargeter가 없는 Top1을 대조군으로 쓰면 Top-K의 효과를 잘못 읽을 수 있음을 보여준다. 모든 CI는 한 training seed에 조건부인 탐색 결과다.

실제 고정128의 feature 진단에서는 decoder/score 공동 학습으로 prior 실효K가4.67에서10.66으로 늘고,argmax가45.32% query에서 바뀌었다. 이는 정확도 하락의 증거나 SS 성능 차이의 확정 원인이 아니다. Frozen R9의 정보와 초기 가중합을 유지한 상태에서 생성용 utility만 학습하는 분리가 아직 검증되지 않았으므로,다음 단일 모델로 이를 검사한다. 첫 round의 보고서·선정 결과·모델·평가를 덮어쓰거나 기준을 완화하지 않는다.

## 후속 모델 F1

Frozen Omega epoch7과R9 D1 decoder/scalar head를 유지한다. 같은 semantic64 후보에서 `frozen_score/0.005 + utility_logit`으로 가중치를 구한다. Utility query는 native target DINO와 frozen contextual Omega, key/value는 원래 source DINO다. Query/key의128차원 projection으로 만든 elementwise product·절대차256차원에서 작은 scalar logit을 예측하며,마지막 linear weight를0으로 초기화한다. 초기에는 frozen matching prior만 읽고,SS 생성 loss로 utility output과 이어지는 query/key projection을 학습한다. Candidate value 자체를 바꾸는 projection은 추가하지 않는다.

C1 retargeter·fusion gate·denoiser LoRA는 같은 조건으로 학습한다. Deformation-query channel은 물리적으로 없고 retargeter 입력은2048차원이다. 기존 task 설정의`decoder`/`score_prior`는 forward 경로 존재 여부로 유지하고,새`train_decoder=false`/`train_score=false`가 학습 여부를 명시한다. 이전 Full·No-decoder·No-prior의 기본 동작과 checkpoint key는 그대로 유지한다.

사용자 SS4210에서 fresh seed101,global96/DDP2/micro48/accumulation1,5epoch/480update로 시작한다. 기존 native LoRA continuation·데이터·optimizer/LR·warmup25·validation32/12steps/CFG3·final128×3/생성batch8 조건을 유지한다. 기존 DeformNeXt world-v4 train9149/validation508과 RGB/Omega/DINO/proposal/projection cache를 재사용하며 새 raw/depth/matching 데이터 구축은 없다. GPU2/3의 이미 실행 중인 No-retargeter Top1 seed202는 유지하고 F1은 GPU0/1을 사용한다.

초기 native condition의 exact equality,실제 SS loss의 LoRA·fusion·retargeter gradient,temporary restored probe의 utility gradient와 matching decoder/score gradient0을 검사한다. DDP 두 update memory probe에서 micro48/accumulation1·각rank batch/유한 loss를 확인하고 benchmark optimizer를 재사용하지 않는다. 예상 optimizer state는371개(LoRA360+C1의6+utility5)이며 실제 값과 frozen matcher24개 tensor를 따로 검사한다. 매epoch validation/W&B 수치·그림,실제 raw checkpoint·optimizer·file4 대조를 유지한다.

## 판정과 후속 범위

원래 raw480/fixed128×3에서 V1과 같은 retargeter의 Top1 대비 paired source-cluster CI가 양수이고3noise 모두 같은 개선 방향이어야 개선 후보로 인정한다. 기존 C1 및 No-retargeter Top1 대비 결과도 함께 보고한다. Best epoch를 고르거나 initial/native equality 및 gradient 통과를 품질 개선으로 간주하지 않는다.

F1이 통과하면 fresh seed202 F1과 matched Top1+C1 seed202를 병렬 학습해 Top-K utility의 추가 효과도 두 optimization seed에서 검증한다. 이미 진행 중인 No-retargeter Top1 seed202는 전체 조합의 추가 기준선으로 재사용한다. 그 뒤 원래 예약한source-disjoint additional121/39source에서 고정 모델과 기준선을 비교하며,이 validation panel을 독립 미사용 test라고 부르거나 결과로 후보를 다시 고르지 않는다. 현재는 CPU 경로 검사61개를 통과한 구현 단계이며,실제 GPU 정합·memory proof·F1 본 학습·성능 향상은 아직 완료되지 않았다.
