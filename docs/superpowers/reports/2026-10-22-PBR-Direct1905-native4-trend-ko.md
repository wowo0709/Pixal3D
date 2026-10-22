# PBR Direct1905 학습 종료와 native4 최종 경향

PBR Direct+DQ의 학습·최종 validation이 완료됐고 wrapper가 returncode0으로 종료됐다.
최종 checkpoint3개와 config·validation·초기화 기록 등12개 artifact,
2,995,237,703byte의 실제 file4 SHA256/size 검증이 통과했다. 최종 stage32 평가와
outside-proposal 대조 평가가 GPU0·1에서 시작됐으며, GPU2·3에서는 기존 SS
bounded-prior 모델의 cascade32 평가가 이어지고 있다. 전체 실험 완료를 뜻하지 않는다.

## 맞춘 비교 조건

세 모델은 같은 native V1의 step762 NN·Adam 상태를 공유하고, 이후1143회 업데이트해
총1905step을 비교한다. Seed101, world4/global24/micro6/accumulation1이며 PBR의
실제 eligible TRAIN 자산은9139개다. Native 검증은 target08·고정 GT Shape·4개 자산,
4개 source cluster·고정 sampling noise1개·12 sampling step·render256 조건이다.

1143·1524·1905의 세 모델에서 sample/source cluster, seed, conditioning input,
실제 noise/RGB/GT PBR latent/GT Shape 및 foreground fraction의9개 필드를 확인했다.
Direct의 세 epoch native GT RGB pixel도4개 자산 모두 같았다. Native V1/R의 GT
pixel 비교는 이 관측에 포함하지 않으며 stage32 publication에서 별도로 검사한다.

이 PBR Direct는 **기존 target-patch query 기반 feature attention+retargeting+DQ**다.
새 `bounded-source-contrast-NoDQ`의 source-camera ray SS 후보와 구분한다.
기존 bounded-prior SS의 cascade 평가도 새 결합 후보의 결과가 아니다.

## 고정 final1905의 빠른4개 검증

| 모델 | Foreground PSNR↑ | Target crop LPIPS↓ | Target crop SSIM↑ |
|---|---:|---:|---:|
| V1 | 15.614101 | 0.0877213 | 0.9529912 |
| R-only | 15.431804 | 0.0874335 | 0.9527243 |
| Direct+DQ | 15.704213 | 0.0851822 | 0.9534409 |
| Direct−V1 | +0.090112 | −0.0025391 | +0.0004497 |
| Direct−R | +0.272409 | −0.0022512 | +0.0007166 |

Foreground PSNR는 foreground pixel 수로 합친 MSE를 dB로 변환한 값이며 자산별
PSNR의 단순 평균이 아니다. Crop LPIPS/SSIM은 자산 평균이다. 모든 aggregate를
실제 per-asset 기록으로 다시 계산해 일치함을 확인했다.

Direct−V1의 pooled foreground MSE 감소는2.05%이며, 그 감소의73.35%가 asset2,
25.86%가 asset1에서 발생했다. PSNR와 LPIPS는 각각3/4개 자산에서 개선됐지만 나머지
한 개의 변화는 매우 작다. 일부 자산에 기여가 집중된 작은 개발 패널이므로 전체
분포의 개선이나 모듈 효과가 확정됐다고 해석하지 않는다.

## Epoch 경향과 현재 해석

| Step | Direct−V1 PSNR | Direct−V1 target LPIPS | Direct−V1 target SSIM |
|---|---:|---:|---:|
| 1143 | −0.232272 | −0.0015205 | +0.0007730 |
| 1524 | +0.805001 | +0.0007354 | +0.0035583 |
| 1905 | +0.090112 | −0.0025391 | +0.0004497 |

1524의 큰 PSNR 차이는1905에서 유지되지 않았다. 대신 고정 final1905의 LPIPS는
V1/R보다 낮아졌다. 따라서 모든 metric이 함께 계속 좋아졌다고 말할 수 없다.
미리 정한 final1905를 비교하며, 작은 패널의 최고 epoch를 사후 선택하지 않는다.
GT−Pred−Diff 이미지에는 여전히 눈에 띄는 밝기·재질 차이가 남아 있다. 작은 수치
이득을 이 차이가 해결됐다는 뜻으로 해석하지 않는다.

현재 LPIPS는 **target08 crop 내부의 빠른 학습 monitor**다. 논문의 source preservation
LPIPS가 아니며, 그 평가는 생성된 full cascade의 M1 source9/10 reference로 수행한다.
GT Shape를 고정했으므로 alpha IoU1을 geometry 개선으로 세지 않는다. CD/F-Score,
source DreamSim/CLIP, global FID/KID의 결과도 이 표에서 주장하지 않는다.

Direct−R은 feature attention·직접 전달·DQ의 결합 차이로, decoder·matching score·
attention 각각의 causal effect를 입증하지 않는다. 다음 판단에는 실제32개 stage
평가·source-cluster interval·outside-proposal 대조와 생성 cascade 결과가 필요하다.
새 source-relative bounded 후보의 GPU 정합·capacity·teacher112 및 후속 비교도
별도로 대기 중이다. Full joint SS·source-disjoint 확인·독립 seed·test508 계획을
이4개 monitor로 대체하지 않는다. 모델 채택과 전체 목표 완료는 아직 하지 않는다.

[최종4개 비교 표·자산별 차이](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-step1905-three-arm-paired4-v1),
[동일 GT/noise의 epoch 경향·실제 GT/Pred/Diff](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-step1905-same-case-transition-v1),
[학습 run](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-joint-v1-direct-retargeting-dq-s101-w4)에 기록했다.
