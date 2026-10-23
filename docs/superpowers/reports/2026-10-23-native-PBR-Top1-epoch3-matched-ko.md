# Native PBR Top-1+R의3에폭 실제 중간 결과

Top-1+R은1140/1900 update까지 학습한 뒤 고정4개 target 평가를 완료했고 학습을
재개했다. 같은1140 update의 vanilla보다 pooled PSNR+.4338dB, 자산별 평균 PSNR
+.1662dB, target LPIPS−.0012555였다.2에폭보다 차이가 커졌으므로 예정된5에폭을
유지한다. 이 arm에는 추가 QKV feature attention이 없다.

## 같은 epoch와 같은4개 GT-shape 조건

| 3에폭·1140 update | Vanilla | Top-1+R |
|---|---:|---:|
| Pooled-MSE PSNR ↑ | 26.722470 | **27.156272** |
| 자산별 평균 PSNR ↑ | 29.965552 | **30.131767** |
| Foreground pooled PSNR ↑ | 15.611604 | **16.045404** |
| Target LPIPS ↓ | .00897729 | **.00772177** |
| GT-only crop LPIPS ↓ | .08823027 | **.07874926** |
| Target SSIM ↑ | .99246727 | **.99296382** |

평가 camera는 target8 하나이며4개 모두 같은 GT shape를 사용한다. 따라서 위 LPIPS는
source preservation이 아니다. Alpha IoU1은 주어진 GT shape에서 나온 값이므로 생성
geometry의 정합 개선으로 해석하지 않는다. 배경을 포함한 foreground fraction은
7.743%여서 전체 PSNR만으로 외형 차이를 판단하지 않고 foreground/crop도 함께 본다.

## 시간에 따른 차이와 샘플별 편차

| Top-1+R − 같은 epoch Vanilla | 1에폭 | 2에폭 | 3에폭 |
|---|---:|---:|---:|
| Pooled PSNR ↑ | −.2657dB | +.0230dB | **+.4338dB** |
| Mean-asset PSNR ↑ | −.2308dB | −.0552dB | **+.1662dB** |
| Target LPIPS ↓ | +.0002277 | −.0005627 | **−.0012555** |

최초 step0의 실제 예측 latent·metric은 vanilla와 같았다. 이후의1에폭 하락과
3에폭 회복은 학습 중 변화이며, 모듈 삽입 직후의 초기 불일치라고 설명하지 않는다.
3에폭의 Axe PSNR+.7872dB/LPIPS−.0046792가 평균 개선 대부분을 차지한다.
Bed PSNR은1에폭−1.0115dB,2에폭−.5196dB에서3에폭−.1469dB로 회복했지만
아직 vanilla보다 낮다. Bed LPIPS는−.0003759로 개선됐다. 나머지2개는 PSNR
+.0345/−.0100dB, LPIPS+.0000148/+.0000182로 차이가 작다.

실제 Axe·Bed의 Top-1/V1 GT-Pred-Diff4개를 열어 확인했다. 두 모델 모두 회색 GT보다
밝은 예측이 남아 있고 육안에서 극적인 개선이라고 부르기 어렵다. 고정4개 중간 평가로
일반적인 성능 향상이나 module adoption을 주장하지 않는다.

## 실제 증거와 다음 단계

4rank 모두 최초1140 update의 asset IDs·x_t·t·native global fingerprint가 vanilla와
같고, 실제 GT panel pixel도4개 모두 같다. Observer가 저장한 source validation prefix와
완료된 vanilla snapshot으로 public compare_epoch를 다시 실행해 모든 수치가 정확히
같음을 확인했다. 복사된4rank training trace도 원래1140 prefix와 다시 대조했다.
이 재검증은 CPU92/93의 metadata/hash/통계 계산이며 neural forward나 optimizer update가 없다.

Epoch3 기록16개3,966,070bytes의 file4 closure를 확인했다. 추가 검토2파일28,164bytes도
실제 file4 SHA256/size로 검증했다. Canonical 경로는 각각
`PBR-native-science-runtime-v1/matched-epoch-observers-v1/top1-only/epoch3-step0001140/`와
`PBR-native-science-runtime-v1/top1-epoch1140-actual-review-v1/`다.
[W&B 중간 곡선 및 시각화](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-top1-only-matched-epoch-fixed4-v1)에1140 step logging을 확인했다.

다음 중간 평가는1520 update다.1900 update와 완료 proof, source delivery/R-off 진단,
후속 feature/raw arm, PBR generated-shape 평가와 DEV14그룹 비교를 계속 진행한다.
이번 조합의 개선을 semantic Top-K·matching-aware QKV 또는 R 하나의 독립적 공로로
돌리지 않는다. 독립 TRAIN seed, 실제 모델 선택과 frozen508개10 metric까지 남아 있다.
