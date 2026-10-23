# 핵심 SS 다섯 모델의 공통 PBR cascade 평가 연결

시각은 UTC 기준 2026-10-23이다. 브랜치는 `wowo0709/feature-retargeting`이다. 본 기록은 준비·연결 상태이며 실제 cascade 생성 결과를 보고하는 문서가 아니다.

## 연결한 전체 평가 경로

기존 V1, Top-1+R, Full 후보와 새 Top-1+V, 고정 Top-K의 다섯 SS 모델을 비교한다. 각자의 실제 SS 예측에서 Shape512 → Shape1024 → PBR1024를 생성한다. Shape512 step 4,210과 Shape1024 step 16,850은 사용자가 지정한 원래 multi-view LoRA checkpoint로 고정한다. PBR은 실제 1,900-step 학습이 완료된 vanilla V1 하나로 고정한다. GT geometry는 생성이 끝난 뒤 평가에만 사용한다.

공통 PBR checkpoint의 SHA256은 `8ad3900dc1918f858d2593e7d87bdae26f58ae82ba3c7f079f4039f94adf9c69`이다. 실제 config, source multiview step 16,850, 5 epoch, 학습 asset 9,139개, global batch 24, GPU당 microbatch 6, 4개 rank의 완료 기록을 확인했다. 추가 source conditioner가 없는 vanilla 모델이다.

기존 세 SS 모델의 original32 archive와 새 두 SS 모델의 original128 capture 중 첫 32개를 사용한다. asset·source cluster·GT·noise·occupancy count가 같은지를 검사한다. 새 SS forward를 추가해서 기존 예측을 다시 생성하지 않는다. Shape의 실제 native guide를 원래 cascade 호출에서 저장하고, renderer의 수치 코드는 변경하지 않는다.

전체 연결은 다음과 같다.

1. 모델별 같은 개발 asset 32개에서 생성·native guide 보관·렌더링·공통 참조 점수 계산: 5×32=160건.
2. target camera 8, source camera 9·10의 총 480개 이미지에서 공식 DreamSim·CLIP 평가.
3. source LPIPS/DreamSim/CLIP과 target PSNR/SSIM/M-IoU, 원래 생성 geometry의 CD/F-Score를 paired source-cluster bootstrap으로 비교하고 W&B에 수치와 GT/Pred/Diff를 기록.

같은 PBR 가중치를 사용해 SS 변화가 downstream 품질로 이어지는지 확인하는 구성이다. SS가 달라 생성 geometry가 다른 경우 texture noise 배열까지 같다고 주장하지 않는다. 원래 cascade RNG 절차와 asset별 seed를 동일하게 유지한다.

공통 참조는 변형 M1 geometry에 원래 material을 유지한 렌더다. source camera 점수는 변형에 적응하면서 material을 보존하는 품질을 측정한다. 이를 undeformed M0 RGB와의 직접 비교로 부르지 않는다. 다른 모델에 다른 GT 픽셀·camera·crop을 제공한 결과는 거부한다.

## 실행 자원과 초기 gate

이 경로는 GPU 0–3의 기존 dispatcher에 연결했다. 현재 PBR 학습을 중단하지 않는다. GPU 6·7의 SS 학습도 유지한다. GPU 4·5는 사용하지 않는다. 각 그룹의 첫 생성 case가 성공해야 같은 그룹의 나머지 31건이 실행된다. child process의 실제 종료와 session drain, 생성 artifact의 file4 SHA256 일치를 확인한 뒤 성공을 기록한다.

새 계약·통계 검사는 22개가 통과했다. 실제 GPU 생성에 대한 검사 완료를 뜻하지 않는다. 준비된 163개 case/metric plan에서 923개의 고유 정적 입력 검사를 수행했다. 누락된 정적 의존성은 없으며, 새 SS 두 모델의 최종 evaluation과 해당 child terminal 네 파일만 정상적으로 대기 중이다.

## 중복 예약 정리

대기 상태였던 예전 cascade factory 및 이전 구조를 전제로 한 TEST 예약 4개, 중복 common-PBR case/분석 예약 99개를 보류했다. 기존에 보류한 지엽 작업 36개도 유지했다. 실행 중인 정상 학습이나 생성 결과를 삭제·재시작하지 않았고 queue state를 직접 수정하지 않았다.

기존 legacy14 또는 native8 전체 완료를 새로운 핵심 비교의 선행 조건으로 요구하지 않는다. 최종 TEST를 취소한 것이 아니다. 새 핵심 결과로 구조를 선택하고 독립 학습 seed를 확인한 뒤, 그 결과를 바탕으로 구조·checkpoint를 동결해 TEST 508개를 다시 연결해야 한다.

## Attention 학습 여부에 대한 제한적인 확인

기존 Full 후보의 실제 480-step source checkpoint를 초기 함수 anchor buffer와 비교했다. Q weight의 상대 L2 변화는 약 7.32%, K는 5.64%, utility output은 47.04%였다. 파라미터가 업데이트되지 않은 채 방치됐다는 설명은 이 기록과 맞지 않는다.

이 검사는 CPU에서 tensor 값만 비교했으며 neural forward나 학습은 수행하지 않았다. 파라미터 변화는 attention 기능의 개선이나 최종 성능 기여를 입증하지 않는다. 고정 Top-K 대 학습 attention의 matched 생성 결과와 실제 cascade metric으로 판단해야 한다.

## 남은 필수 작업

12개 arm의 학습 및 최종 SS 평가, 이 다섯 모델의 실제 cascade 생성·8개 metric 계산이 남아 있다. 이후 선택한 조합과 강한 기준 모델을 독립 학습 seed로 확인하고, 올바른 freeze 뒤 TEST 508개와 FID/KID를 포함한 10개 metric을 완료해야 한다. 아직 최종 Full 모델을 채택하지 않았다.
