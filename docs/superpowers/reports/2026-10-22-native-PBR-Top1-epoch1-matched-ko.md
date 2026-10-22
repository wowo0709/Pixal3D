# Native PBR Top-1+R 첫 epoch의 같은-step 비교

Top-1+R가 실제380 update와 첫 validation을 완료했고, 자동 CPU observer의
같은-step 비교도 통과했다. 학습은 다음 epoch로 재개됐다. 원본 PBR16850,
fresh Adam, seed101, TRAIN9139, global24/DDP4/micro6/accumulation1이다.
이 arm에는 추가 QKV feature attention이 없다.

## 고정4 target-camera 평가

같은 네 자산·noise·RGB·GT PBR·GT Shape·camera를 사용한다. Source 보존이나
predicted-shape cascade 평가는 아니다. PSNR의 두 집계를 구분한다.

| 모델 / step | Pooled-MSE PSNR ↑ | Mean-asset PSNR ↑ | LPIPS ↓ | Crop LPIPS ↓ |
|---|---:|---:|---:|---:|
| 공통 학습 전 / 0 | 26.0573 | 29.4633 | .0101050 | .0951262 |
| Vanilla / 380 | 26.3444 | 29.6719 | .0096876 | .0924272 |
| Top-1+R / 380 | 26.0787 | 29.4411 | .0099152 | .0936904 |
| Top-1+R − Vanilla / 380 | −.26567 | −.23085 | +.00022767 | +.00126322 |

현재 네 자산 평균에서는 Top-1+R가 vanilla보다 낮다. 학습 전과 비교하면
Top-1+R의 pooled PSNR과 LPIPS는 소폭 개선됐지만 mean-asset PSNR은 소폭
감소했다. 서로 다른 PSNR 집계를 같은 값으로 취급하지 않는다.

| 자산 | Top-1+R − Vanilla PSNR (dB) | LPIPS 차이 |
|---|---:|---:|
| Axe | +.08813 | −.00086904 |
| Bed | −1.01149 | +.00177246 |
| Bookcase | −.00290 | +.00000429 |
| Bottle | +.00286 | +.00000299 |

평균 하락은 주로 Bed에서 발생했다. 이 자산은 학습 전25.0433dB에서
Top-1+R 24.7394dB로 낮아졌으며, vanilla는25.7509dB였다. Base-color PSNR도
같은 epoch의 vanilla보다 .9924dB 낮다. 두 모델의 실제 GT/Pred/Diff 이미지를
확인했고, Bed에서 남아 있는 밝기·재질 차이는 Top-1+R 쪽이 더 컸다.
반면 Axe에서는 PSNR·LPIPS·base-color PSNR 모두 vanilla보다 개선됐다.

## 비교 정합과 해석 범위

학습 전 네 자산의 예측 latent와 metric은 두 모델이 정확히 같았다. 실제
네 rank의 첫380 update asset/noise/timestep/native-global trace도 모두
일치했고, 초기 및380-step GT panel pixels와 입력 fingerprint를 대조했다.
관측된 차이는 초기화부터 예측이 깨진 현상과 구분되는 학습 중 변화다.
이 검사들이 모듈의 전체 설계·학습 목표가 적절하다는 증명은 아니다.

380번째 train pack에서는 fusion gate .0055422와 R residual RMS .0090388을
기록했다. 이는 해당 학습 pack에서 경로가 작동한다는 근거이며, validation
Bed의 원인을 분리한 R-off 결과가 아니다. 평균 하락의 원인을 QKV attention으로
돌릴 수도 없다. 이 arm은 물리적인 Top-1+R 모델이며 feature head가 없다.

n=4·첫 epoch만으로 전체 성능 저하나 이후 회복 여부를 판정하지 않는다.
예정된1900 budget을 유지하고760/1140/1520 및 최종1900에서 같은-step 경향을
비교한다. SS에서 관측한 R 이득을 PBR로 자동 일반화하지 않는다. 현재 PBR
완료 후 먼저 실행하도록 연결한 target-patch/R96 SS 재현과, 남은 PBR feature 및
raw Top-K arm도 모두 유지한다. Source-preservation·generated-geometry isolation·
DEV full cascade·최종508/10 metric 평가는 별도로 남아 있다.

실제 비교는 `PBR-native-science-runtime-v1/matched-epoch-observers-v1/`
`top1-only/epoch1-step0000380/{audit.json,file4-closure.json}`이다.
16파일·1,567,154 bytes의 실제 file4 SHA256/size 검증이 통과했다.
[W&B 같은-epoch 비교 및 panel](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-top1-only-matched-epoch-fixed4-v1).
