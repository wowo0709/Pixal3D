# PBR Direct step1524의 실제 3-arm 경향

2026-10-21 약22:01 UTC에 step1524의 빠른 validation과 자동 비교가 완료됐다.
학습은 같은 run에서 재개됐다. 완료된 CPU 모니터를 학습 실패로 해석하거나
재시작하지 않는다. 아래 결과는 개발4개·source cluster4개·sampling noise1개·
training seed101이며, 고정한 최종1905 checkpoint를 바꾸는 선택 기준이 아니다.

## 동일 조건의 비교

세 arm은 공통 NN/Adam step762에서 시작해762 updates를 추가한 동일 step1524다.
World4/global24/micro6/accumulation1, generation batch1, sampling12 steps,
256px 렌더링을 유지했다. 실제 PBR train 자산은 기존 token limit을 통과한9139개다.
세 모델의 같은 자산·source cluster·seed·conditioning/RGB/GT PBR/GT Shape/noise
hash와 foreground fraction을 확인했다. Native per-asset 값으로 재집계한 수치가
기록된 metric과 일치한다. Foreground PSNR은 foreground 비율로 MSE를 모아 계산한다.

| 지표 | V1 | R-only | Direct | Direct−V1 | Direct−R |
|---|---:|---:|---:|---:|---:|
| Foreground PSNR ↑ | 15.691054 | 15.693310 | 16.496055 | +0.805001 dB | +0.802745 dB |
| Target crop LPIPS ↓ | 0.087304 | 0.085981 | 0.088039 | +0.000735 | +0.002058 |
| Target crop SSIM ↑ | 0.953248 | 0.954168 | 0.956806 | +0.003558 | +0.002638 |

PSNR/SSIM은 Direct가 높지만 LPIPS는 두 reference보다 나쁘다. 이 표의 LPIPS는
**GT Shape를 고정한 target camera08 appearance 진단**이다. 논문의 source9/10
preservation LPIPS나 generated geometry의 CD/F-Score를 측정한 결과가 아니다.

## 어떤 자산에서 변화했는가

| 같은 panel 순서 | Direct−V1 PSNR | Direct−V1 LPIPS | Direct−V1 SSIM |
|---|---:|---:|---:|
| Asset1 | +0.309174 dB | −0.014676 | +0.006244 |
| Asset2 | +3.153687 dB | +0.017102 | +0.008228 |
| Asset3 | +0.043568 dB | +0.000082 | +0.000037 |
| Asset4 | −0.021965 dB | +0.000434 | −0.000276 |

Direct−V1의 pooled MSE 개선분 가운데 asset2가 약74.8%, asset1이 약25.5%를
차지한다. Asset3의 기여는 작고 asset4는 조금 악화됐다. Asset2의 PSNR은
step1143 때 V1보다1.904 dB 낮았는데 step1524에는3.154 dB 높다. LPIPS는 이
자산에서 여전히 나쁘다. 따라서 평균 PSNR의 반등을 전체 자산·지각적 유사성의
일관된 개선으로 해석하지 않는다. 이것만으로 overfitting 원인을 확정하지도 않는다.

모델별 두 epoch 사이에도 위9개 입력 정합 항목이 모두 같다. 저장된 Direct의
native GT/Pred/AbsDiff panel 두 장을 읽어 GT 영역이4개 모두 byte 단위로 같은지
확인했다. 이전/current panel과 자산별 기여도는 별도 CPU 기록에 보존한다.
V1/R의 step1524 native GT 픽셀 비교는 이 검사에서 수행한 것이 아니며,
최종32개 게시 작업이 세 arm 전체의 GT 픽셀 검사를 별도로 수행한다.

## 해석과 다음 단계

Step1143의 Direct−V1 PSNR −0.232272 dB에서 step1524 +0.805001 dB로 방향이
바뀌었다. 아직 네 자산의 중간 관측이며, 개발 모집단의 유의한 효과 또는 독립 학습
seed 재현을 보인 결과가 아니다. 학습이 무의미하다고 중단할 근거도 부족하므로
원래 예산대로1905까지 진행하고, 이미 연결된 동일32개 최종 평가에서 확인한다.
이 관측에 맞춰 새 LR/loss/arm이나 조기 checkpoint 선택을 추가하지 않는다.

현재 PBR Direct는 기존 target-patch query 계열이다. Direct−R은 feature attention과
deformation query가 함께 추가된 차이이며 matching decoder·score·correspondence의
개별 효과를 분리하지 못한다. 새 source-camera SS19의 warm Original/Contrast
실험과도 구분한다. 새로운 SS 학습은 source-disjoint72와 native 초기화·capacity
검증을 통과해야 실행한다. 최종 generated cascade의 source/target 역할별 지표,
독립 joint seed, test508 및 공식 FID/KID도 완료 범위에 남아 있다.

Canonical 결과는 `PBR-joint-continuation-v1/epoch-validation-trends-v1/`
`Direct-step001524-three-arm-v1`과 `Direct-step001524-same-case-transition-CPU-v1`
아래에 보존한다. 각각 실제 file4 SHA256·size를 검증한다.
[동일 step1524 비교 표와 plot](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-step1524-three-arm-paired4-v1),
[같은 자산의 epoch 변화와 원래 GT/Pred/차이 이미지](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-step1524-same-case-transition-v1)에 기록한다.
