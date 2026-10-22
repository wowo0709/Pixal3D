# Native PBR Top-1+R의 실제2에폭 비교

Top-1+R PBR이760 update·두 에폭의 저장 및 고정4개 검증을 완료했다.
첫 에폭의 V1 대비 하락은 줄었고, pooled PSNR과 target LPIPS는 같은760 update의
V1보다 좋아졌다. 자산별 평균 PSNR은 아직 낮다. 작은 중간 패널의 결과이며
예정된1900 update·5에폭을 유지한다. 이 Top-1 arm에는 추가 QKV feature head가 없다.

## 같은 시점·같은 입력의 수치

원래 PBR16850에서 fresh LoRA/source로 시작했다. GT Shape 조건·target camera8,
동일4자산·RGB/noise/GT/camera·native12-step/CFG3/render256이다.
LPIPS는 이 target 카메라의 빠른 진단이며 논문의 source-preservation LPIPS가 아니다.

| 모델·시점 | Pooled PSNR ↑ | Mean-asset PSNR ↑ | Target LPIPS ↓ | Target crop LPIPS ↓ |
|---|---:|---:|---:|---:|
| 공통 초기0 | 26.05731 | 29.46333 | .01010496 | .09512621 |
| V1 step380 | 26.34439 | 29.67193 | .00968757 | .09242722 |
| Top-1+R step380 | 26.07872 | 29.44108 | .00991524 | .09369044 |
| V1 step760 | 26.60228 | 29.87889 | .00927335 | .09007769 |
| Top-1+R step760 | **26.62532** | 29.82373 | **.00871067** | **.08610232** |

Step760의 Top-1+R−V1은 pooled PSNR+.02305dB, mean-asset PSNR−.05517dB,
LPIPS−.00056268, crop LPIPS−.00397537이다. Foreground PSNR도+.02305dB이며,
base-color pooled PSNR은+.28027dB다. 첫 에폭의 pooled PSNR 차이−.26567dB에서
회복했지만, 서로 다른 집계의 PSNR 방향이 같지는 않다.

| 자산 | Top-1+R−V1 PSNR (dB) | LPIPS 차이 |
|---|---:|---:|
| Axe | +.27125 | −.00222790 |
| Bed | −.51962 | −.00003428 |
| Bookcase | +.01902 | +.00001285 |
| Bottle | +.00869 | −.00000137 |

LPIPS 개선은 대부분 Axe에서 나왔다. Bed의 PSNR 하락은 첫 에폭−1.01149dB보다
줄었으나 여전히 남는다. 실제 두 모델의 Axe·Bed GT/Pred/Diff를 확인했으며
GT 대비 밝은 외형 오차가 계속 보인다. 큰 시각적 개선이나 모든 자산의 회복을
주장하지 않는다. GT Shape를 공유한 PBR stage이므로 silhouette 일치로 생성 geometry의
정확도를 주장하지 않는다.

## 실제 검증과 자원

네 rank 각각 첫760 update의 asset·x_t·timestep·native global trace가 V1과 같다.
초기 prediction 및 실제 GT PNG pixel도 같으며, 동일 epoch 비교가 자동 W&B에
기록됐다. Canonical `matched-epoch-observers-v1/top1-only/epoch2-step0000760/`의
16개 파일2,766,377bytes를 실제 file4 SHA256/size로 검증했다.

Global24/DDP4/micro6/accumulation1, 물리 GPU0–3이다. 네 rank의 최근20 update
median compute는14.133–14.140초,760 update 동안 peak allocated 최대44.297GiB,
reserved 최대49.424GiB였다. 이 prefix에서 OOM은 없었으며 학습이 재개됐다.
CPU observer는 다음1140 update를 기다린다.

현재 개선은 Top-1+R 조합의 작은 target-stage 추세다. Semantic decoder·member
matching·R 각각의 독립적 기여나 matching-aware attention의 성공을 의미하지 않는다.
실제1900 완료 뒤의 전달 경로 진단, 남은 PBR arm과 isolation128, source/target
DEV cascade·독립 seed·모델 선택 및 최종508개10 metric은 계속 필요하다.
[W&B 동일 에폭 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-top1-only-matched-epoch-fixed4-v1).
