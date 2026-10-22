# PBR의 같은 epoch 간 자동 중간 비교

완료된 vanilla와 세 source PBR 모델을 학습 중에도 같은 epoch로 비교한다.
새 학습이나 추론을 실행하지 않고, native training이 이미 게시한 validation
기록과 GT/Pred/Diff panel을 CPU에서 읽는다. 기존1900 완료 수집·모델 간 비교는
그대로 유지하며, 이 중간 비교로 대체하지 않는다.

| 학습 중인 모델 | 완료된 비교 기준 | 수집 step |
|---|---|---|
| Top-1+R | Vanilla | 380, 760, 1140, 1520 |
| Feature+R | Vanilla, Top-1+R | 380, 760, 1140, 1520 |
| Raw Top-K+feature+R | Vanilla, Top-1+R, Feature+R | 380, 760, 1140, 1520 |

새 CPU observer는 모델당 하나이며, 아직 게시되지 않은 epoch는 기다린다.
NN의 실제 PID/birth와 명령·mode·output을 확인하고, checkpoint의 실제 file4
검증 및 validation 행 게시 이후에만 해당 epoch의 비교를 기록한다. NN이 먼저
종료된 경우에는 같은 owner의 실제 종료0을 요구한다. Pending 상태나 기다린
시간을 학습·평가 완료로 바꾸지 않는다. 작업 실패 이력도 보존한다.

비교마다 네 자산의 RGB/noise/GT PBR/GT Shape/camera identity, 초기 예측의
정확한 일치, 네 rank의 해당 step까지 실제 asset/noise/timestep/native-global
training trace, GT panel의 실제 pixels를 대조한다. 같은 자산 이름이나 같은
epoch 번호만으로 비교하지 않는다. Reference는 원래1900 완료 snapshot을 쓴다.

PSNR은 pooled-MSE와 mean-asset dB를 각각 기록하고, LPIPS·crop LPIPS·SSIM 등
기존11개 빠른 지표의 모델 간 차이와 자산별 차이를 저장한다. 각 모델의 기존
W&B 학습 로그와 별도로 같은-epoch 차이와 panel을 비교용 run에 기록한다.
모델·데이터·optimizer·GPU 배정·학습 recipe는 바꾸지 않는다.

이 검사는 고정4개·GT Shape 조건·target camera8의 **경향 확인**이다. 추론적
신뢰구간, source preservation, predicted-shape full cascade, 전체 stage 완료,
모델 채택 또는 최종508 성능을 입증하지 않는다. Positive/negative 차이를 모두
보존하고, n=4만으로 중간에 epoch나 full 구조를 선택하지 않는다.

새 `native_pbr_epoch_comparison.py`는 완료된 reference와 실제 source epoch
prefix만 비교한다. 잘못된 input identity, 초기 예측, 미게시 epoch, PSNR 집계,
평가 범위 및 evaluation 중 optimizer update를 거부하는 CPU 검사16개가 통과했다.
실제 step0 preflight와 실행 상태는 별도의 canonical prelaunch/queue 기록으로
구분한다. 실제380 이후 결과가 나오기 전에는 중간 비교가 완료됐다고 주장하지
않는다.
