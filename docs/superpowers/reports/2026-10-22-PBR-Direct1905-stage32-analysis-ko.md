# PBR1905 stage32: 기존 Direct 조합은 채택하지 않는다

PBR Direct의1905step 학습, 최종 validation·checkpoint의 file4 검증,32개 stage 평가,
outside-proposal 대조 평가와3-arm publication이 모두 정상 완료됐다. **기존 Direct는
V1/R보다 약하다.** 작은4개 학습 monitor의 개선을32개 전체의 개선으로 확대하면 안 된다.
새 bounded-source-contrast SS 후보는 아직 학습 전이며 이 결과와 구분한다.

## 같은 조건에서의32개 결과

공통 Deform V1의762step NN·Adam prefix에서1143회 더 업데이트한1905step을 비교한다.
Step762는 deformation 학습 도중의 공통 prefix이며 zero-shot 모델이 아니다.
World4/global24/micro6/accumulation1, seed101이며 actual PBR TRAIN은9139개다.
검증은32개 자산/source cluster32개, sampling noise1개, target08·고정 GT Shape다.

| 모델 | Foreground PSNR↑ | Target crop LPIPS↓ | Target crop SSIM↑ |
|---|---:|---:|---:|
| V1 | 15.605368 | 0.1204092 | 0.9208989 |
| R-only | 15.891211 | 0.1164771 | 0.9227996 |
| Direct+DQ | 14.624105 | 0.1261741 | 0.9100598 |

PSNR는 foreground pixel 수로 합친 MSE의 dB 값이다. Crop LPIPS/SSIM은 자산 평균이다.
Source-cluster paired bootstrap4000회, seed20261020의 보정되지 않은 개발 interval이다.
하나의 training seed이며 독립 학습 반복을 수행한 결과가 아니다.

| 비교 | PSNR 차이 [CI95] | LPIPS 차이 [CI95] | SSIM 차이 [CI95] |
|---|---|---|---|
| R−V1 | +0.285843 [−0.203128,+0.703399] | −0.003932 [−0.009224,+0.000682] | +0.001901 [−0.001450,+0.005490] |
| Direct−V1 | −0.981263 [−1.758833,−0.198223] | +0.005765 [−0.002947,+0.016099] | −0.010839 [−0.023436,−0.000890] |
| Direct−R | −1.267106 [−2.274708,−0.199785] | +0.009697 [+0.000508,+0.021595] | −0.012740 [−0.025419,−0.002479] |

R-only의 평균은 좋지만 세 CI가 모두0을 포함한다. Direct는 V1 대비 PSNR/SSIM,
R 대비 세 metric의 interval이 악화 방향이다. 현재 target-patch 기반 Direct를
full 모델로 채택하거나 같은 구조의 추가 epoch만으로 해결됐다고 주장하지 않는다.

##4개 monitor가 왜 다르게 보였나

세 모델 모두 native validation의 처음4개와 standalone32의 처음4개에서 실제
RGB/noise/GT latent 등9개 필드 및 모든 자산별 metric이 **정확히 같았다**. 집계도
다시 계산해 일치했다. 이4개에 대해서 runtime/입력/sampler 차이가 반전의 이유는 아니다.
3-arm32개의 GT RGB pixel도 모두 같았고, 같은 GT Shape와9개 입력 identity를 확인했다.

| Direct−V1 | PSNR | Target LPIPS | Target SSIM |
|---|---:|---:|---:|
| 처음4개 | +0.090112 | −0.002539 | +0.000450 |
| 나머지28개 | −1.115902 | +0.006951 | −0.012452 |
| 전체32개 | −0.981263 | +0.005765 | −0.010839 |

전체에서 PSNR는10/32개, LPIPS는15/32개, SSIM은14/32개에서만 개선됐다. Pooled
foreground MSE는 V1보다25.35% 증가했다. 첫4개는 빠른 monitor로 고정했지만,
나머지28개의 모델 선택 기준을 대신하지 못했다. 이 분할 분석은 사후 진단이며
별도 held-out confirmation으로 취급하지 않는다. Panel을 바꾸거나1905 대신 좋은
epoch를 사후 선택하지 않았다. Source-disjoint 확인과 test508은 계속 필요하다.

## Correspondence를 안 쓰는 실패인가

같은 Direct weight에서 추가 경로의 값을 semantic64 후보 밖의 RGB-valid source
patch 값으로 교체했다. Candidate key·cached score·attention probability와 기존
Top-1 carrier는 유지하고 appearance/RAW value가 같은 donor 주소를 사용한다.
GT를 추가하지 않는 저장 weight 진단이며 native-spatial NoCorr 또는 재학습 arm이 아니다.

정상 Direct−outside의 PSNR는+0.146825, CI[−0.163743,+0.612695]로 불확실하다.
LPIPS는−0.007432, CI[−0.013942,−0.001795], SSIM은+0.008898,
CI[+0.000951,+0.018262]다. 즉 **이 실패한 Direct 안에서도 correspondence value는
일부 perceptual 결과에 유용하게 작동한다. 그러나 정상 value를 써도 V1보다 약하다.**
이 둘을 구분해야 한다. Matching decoder/score/attention 각각의 학습 효과가 입증된
것도 아니고, 모든 matching을 제거해야 한다는 근거도 아니다.

## 다음 설계에 반영할 것

현재 Direct는 이전 target-patch query의 feature attention·direct 전달·DQ 조합이다.
R-only와의 차이에 여러 경로가 포함되므로 attention 또는 DQ 하나만 원인으로 확정하지
않는다. Worst-case 실제 panel에서도 GT가 같은 상태에서 appearance 오차가 커지는
것을 확인했다. Forward 내 DINO/NAF branch별 상대 correction 크기·방향, source-use,
matching posterior와 실제 전달을 측정하는 것이 다음 PBR 원인 분리의 후보다. 아직
그 scale mismatch를 실제로 측정했거나 원인으로 확정한 것은 아니다.

이미 연결한 새 SS 후보는 source-camera ray member query, source-relative R와
matching prior를 보존하는 bounded attention을 같은19개 tensor 예산에서 비교한다.
CPU12개 runtime 검사와 기존 module 검사를 통과했지만 GPU native32 정합·micro96
backward·teacher112·native 품질은 아직 확인 전이다. 완료된 과거 GT 진단에서 후속
utility/QK가 GT 질량을 줄였다는 근거를 시험하는 구조이며, 이번 PBR의 성능 향상을
달성했다고 대신 주장하지 않는다. 결과가 나오면 native quality와 GT matching을 함께
판정하고 source-disjoint 확인과 joint SS로 연결한다. 기존 full cascade 평가도 유지한다.

이 표의 LPIPS는 target crop monitor다. 논문 source preservation LPIPS/DreamSim/CLIP,
생성 geometry의 M-IoU/CD/F-Score, test508의 global FID/KID를 대신하지 않는다.
모델 채택·전체 실험 완료는 아직 하지 않았다.

[32개3-arm CI·실제96개 GT/Pred/Diff](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-final1905-three-arm-paired32-publication-v1),
[4개 monitor와 epoch 경향](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-step1905-same-case-transition-v1)에 기록했다.
