# PBR 실제 capacity에 따른 후속 학습 연결

SS의 fixed final480/128×3 비교, direct 경로의 다섯 진단 및 outside-proposal32 비교를 먼저 완료한다. PBR은 기존 V1 step762를 공통 시작점으로 쓰며 기존 SS 프로세스를 중단하지 않는다.

본 학습 전에 두 PBR arm의 capacity를 각각 실제로 검사한다. 실제 TRAIN maximum31,900-token asset을 rank당 microbatch 수만큼 반복하고, train3 update → 실제4 사례 rendering validation → 추가 train1 update를 실행한다. DDP 동기화, common NN360 Adam moment 복원 및 새16 tensor update도 확인한다. 반복된 동일 샘플이므로 이 시간은 실제 학습 throughput으로 보고하지 않는다.

1. GPU0/1의 V1 control과 GPU2/3의 direct-DQ를 각각 world2/micro12로 검사한다.
2. 둘 중 실제 CUDA OOM이 발생하면, 기록을 보존하고 두 arm 모두 world4/micro6으로 별도 검사한다. OOM 이외 오류는 그대로 실패시키며 자동 재시도하지 않는다.
3. 두 arm이 같은 world의 global24/accumulation1 검사 및 validation 이후 train step을 통과하고 실제 file4 SHA receipt가 생겨야 학습을 연결한다. World4도 실패하면 scientific 학습을 시작하지 않는다.
4. World2이면 두 arm을 GPU0/1과2/3에서 동시에 실행한다. World4이면 두 arm을 GPU0~3에서 순서대로 실행한다. 두 모델의 total budget1905, source762, 추가1,143 update, 데이터9,139개 및 sampler global offset48은 동일하다.

본 학습은 원본처럼 `balanced_sampler=False`, classifier-free dropout0.1을 사용한다. 기존 NN LoRA360의 LR5e-6 및 Adam 상태를 유지하고, 새 R6/attention9/DQ1은 LR1e-4와5 update warmup으로 시작한다. 학습 seed는 두 arm 모두101이다. Epoch마다4 사례 rendering metric과 이미지, 학습 loss 및 attention 통계를 `youngwoojeon-unist/pixal3d-deformnext-retargeting` W&B에 기록한다.

Final1905 이후 동일32 asset/동일 noise/anchor camera08/GT target Shape 조건에서 V1 control과 direct-DQ를 비교한다. 추가로 final direct weight의 outside-proposal value 진단을 수행한다. Foreground PSNR, target-crop LPIPS/SSIM 및 source-cluster paired CI를 보고한다. Original V1 step762와의 비교는 학습 예산이 다른 맥락 비교로 표시한다.

이 phase는 texture-stage 개발 진단이다. Target-crop LPIPS를 source preservation으로 해석하지 않으며, full cascade·novel-view·독립 training seed·정식10개 metric의 최종 성공을 주장하지 않는다. Composite 개선이 나오면 같은 delivery의 물리적 module 제거 및 독립 seed로 효과를 분리해야 한다.

실제 initial32 rendering 동등성 receipt도 본 학습의 필수 조건이다. Native micro1 검사가 통과했다는 이유로 이 검사나 fullbatch capacity를 생략하지 않는다.

Canonical root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/direct-retargeting-v1/PBR-joint-continuation-v1`이다. `prelaunch-v1`의117개 source binding,8개 machine config, world별6개 후속 job 및 실제 file4 SHA가 검증됐다. `queue-connection-v1`에 capacity/선택분기2개 job의 append-only 기록이 있다. 이 문서 시점에는 capacity와 PBR scientific 학습이 아직 실행되지 않았다.
