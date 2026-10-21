# PBR 동일 예산 3-arm 결과의 자동 게시

2026-10-21 UTC. Direct PBR은 아직 학습 중이다. 기존 V1/R-only의 실제 최종32개
평가는 완료됐고, Direct 최종1905 평가와 3-arm 요약은 이미 대기열에 있다.
추가한 작업은 이 기존 결과를 검증하고 WandB에 게시하는 CPU 작업 하나다.
새 학습·sampling·GPU metric pass를 추가하지 않는다.

## 결과 게시 전에 확인하는 조건

각 모델은 공통 V1 step762에서 시작해 같은1143 updates를 더 학습한 최종1905다.
각 실제 run의 canonical config와 initialization 기록에서 seed101, world4,
global24/micro6/accumulation1 및 공통 NN/Adam 상태를 확인한다. 총1905 updates를
처음부터 새로 학습한 실험으로 표기하지 않는다.

최종 평가32개는 source cluster32개, generation batch1, sampling12 steps,
CFG3, edited RGB4, target camera08 및 고정 GT Shape 조건이다. 모델 간 같은
sample/source/seed, conditioning·noise·RGB·GT PBR·GT Shape hash뿐 아니라
foreground fraction도 확인한다. 같은 GT로 계산한 PSNR weighting인지 확인하기
위함이다. 기존 native panel의 첫 GT 열도32개 모두 pixel 단위로 같아야 한다.
이미 완료된 V1/R-only의 GT 열은 실제32개 모두 일치함을 확인했다. Direct의
일치 여부는 실제 평가가 완료된 뒤 검사한다.

기존 foreground PSNR의 MSE pooling과 crop LPIPS/SSIM 집계를 재사용한다.
같은 bootstrap seed20261020과 source cluster4000 resamples로 계산한 세 비교
R−V1, Direct−V1, Direct−R의 차이·CI가 기존 요약과 정확히 같아야 게시한다.
기존 metric을 바꾸거나 이미지에서 수치를 재추정하지 않는다.

## 게시할 결과와 해석

WandB에 세 비교의 metric/CI 표, 원래 GT/Pred/AbsDiff 이미지96개(3×32),
전체32개를 순서대로 담은4개 overview page, 비교 plot을 기록한다. CSV와
standalone PNG/PDF도 file4에 보존하고 SHA256·size를 확인한다. 성능에 따라
일부 자산만 골라 게시하지 않는다.

이 결과는 **target08의 고정 GT Shape 아래 PBR appearance stage 진단**이다.
여기의 crop LPIPS는 paper의 source preservation 지표가 아니며, alpha IoU가
좋다는 이유로 geometry가 개선됐다고 해석하지 않는다. Direct−R은 attention과
DQ가 함께 추가된 차이로, matching decoder·score의 개별 효과가 아니다. 이 PBR
모델은 기존 target-patch query 계열이며 새 source-camera SS19와 구분한다.

1,524-step의 기존 native4개 validation은 별도 CPU 모니터가 실제 NN PID와 birth를
확인하며 기다린다. 완료된 validation을 같은 조건의 V1/R/Direct와 비교해
WandB에 기록한다. 네 자산의 중간 경향으로 checkpoint를 선택하지 않는다.
최종 모델 조합의 generated cascade, source/target 역할별8개 metric, 독립 학습
seed와 final test508/FID/KID는 별도로 완료해야 한다.
