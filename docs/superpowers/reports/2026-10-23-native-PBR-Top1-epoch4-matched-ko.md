# Native PBR Top-1+R의4에폭: 중간 개선이 유지되지 않음

1520/1900 update의 고정4개 target 평가에서 Top-1+R은 같은 step의 vanilla보다
pooled PSNR−.2801dB, 평균 PSNR−.2630dB, target LPIPS+.0005498로 나빠졌다.
3에폭의 양성 결과만으로 이 조합을 채택할 수 없다. 학습은1900 update의 동일 비교
예산을 유지하며, 이 arm에는 추가 QKV feature attention이 없다.

| 4에폭·1520 update | Vanilla | Top-1+R |
|---|---:|---:|
| Pooled PSNR ↑ | 26.744367 | 26.464233 |
| Mean-asset PSNR ↑ | 29.965153 | 29.702138 |
| Foreground pooled PSNR ↑ | 15.633500 | 15.353366 |
| Target LPIPS ↓ | .00888394 | .00943378 |
| GT-only crop LPIPS ↓ | .08768353 | .09120020 |
| Target SSIM ↑ | .99249581 | .99217521 |

같은4개·GT shape·target camera8의 보조 모니터링이다. Source preservation이나 실제
생성 geometry의 CD/F-Score·mask 정합 평가가 아니다. Alpha IoU1은 주어진 GT shape의
결과다. FID/KID나 전체 데이터의 성능 변화도 여기서 추정하지 않는다.

## 하락의 위치

| Top-1+R − 같은 epoch Vanilla | 1에폭 | 2에폭 | 3에폭 | 4에폭 |
|---|---:|---:|---:|---:|
| Pooled PSNR ↑ | −.2657 | +.0230 | +.4338 | **−.2801** |
| Mean-asset PSNR ↑ | −.2308 | −.0552 | +.1662 | **−.2630** |
| Target LPIPS ↓ | +.0002277 | −.0005627 | −.0012555 | **+.0005498** |

Axe의3에폭 PSNR+.7872dB/LPIPS−.0046792 개선은4에폭+.0434dB/−.0003011로 작아졌다.
Bed는3에폭 PSNR−.1469dB/LPIPS−.0003759에서4에폭−1.0948dB/+.0024828로 악화됐다.
나머지2개 PSNR은+.0121/−.0128dB로 차이가 작다.4개 actual GT-Pred-Diff를 열어
확인했으며 두 모델 모두 GT보다 밝은 예측이 남고, Top-1의 Bed가 V1보다 밝다.
주요 변화는 이4개 중 Axe·Bed에 집중되며 일반화된 하락이나 과적합으로 단정하지 않는다.

## 입력 불일치와 학습 목적을 구분

최초 step0의 실제 예측은 V1과 같았다.4rank 모두 최초1520 update의 asset IDs·x_t·t·
native global fingerprint가 V1과 정확히 같고,4개 GT panel pixel도 같다. 저장된 source
validation prefix와 완료된 V1 snapshot으로 public compare_epoch를 다시 계산해 모든
수치를 정확히 재현했다. 이번 하락을 초기 삽입 불일치나 평가 입력 변경으로 설명할
근거는 없다. 그렇다고 모든 구현·설계 선택이 올바르다는 독립적 증명이 되는 것은 아니다.

실제4rank runtime에서 각380 update의 matched TRAIN MSE를 다시 집계했다.

| 학습 epoch | V1 TRAIN MSE | Top-1 TRAIN MSE | Top-1 − V1 |
|---|---:|---:|---:|
| 1 | .26292154 | .26281542 | −.00010612 |
| 2 | .25958307 | .25947240 | −.00011067 |
| 3 | .25994794 | .25988093 | −.00006702 |
| 4 | .25980121 | .25972309 | −.00007812 |

4에폭 TRAIN MSE는 V1보다 약.030% 낮지만,4개 target 생성 품질은 나빠졌다. TRAIN loss의
큰 발산이 관측된 경우가 아니며, 미세한 학습 MSE 개선이 이 생성 평가의 단조 개선을
보장하지 않는다.4에폭의 per-update paired MSE 중앙값은+.00000106, MSE가 더 낮은
update 비율은49.2%로, 평균 차이 자체도 작다. Step을 독립 학습 반복으로 세지 않는다.

로그의 rank0 endpoint fusion_gate는3에폭.015975에서4에폭.022592로 증가했다.
Epoch 평균 logged retarget residual RMS는.014931에서.019536으로 커졌다. 각 batch의
logged RMS 평균이며 Bed의 고정 입력 보정 크기나 실제 적용 gate의 인과적 효과가 아니다.
보정 강도 변화는 원인 후보지만, R·직접 Top-1 전달·NN 업데이트의 효과는 분리되지 않았다.

## 저장 증거와 다음 판단

Epoch4 실제16파일5,163,522bytes의 canonical closure를 확인했다.4rank TRAIN runtime의
첫1520개를 별도로 보존한 재검증7파일4,355,466bytes도 실제 file4 SHA256/size와 일치했다.
재검증은 CPU92/93의 metadata/hash/통계 계산이며 neural forward나 optimizer update가 없다.

중간 평가 observer는380·760·1140·1520의4회를 모두 완료하고 실제 returncode0으로
종료했다. Dialogue 대기 스크립트는 이 정상 종료를 다시 PID 조회하다 예외로 처리했다.
그 오류 기록을 보존했고, 실제 학습이나 observer를 재시작하지 않았다.1900의 최종
완료 snapshot은 별도 작업이 처리한다.

[W&B의4에폭 곡선·시각화](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-top1-only-matched-epoch-fixed4-v1)에1520 logging을 확인했다.
Canonical review는 `PBR-native-science-runtime-v1/top1-epoch1520-actual-review-v1/`이다.

계획된1900까지 완료하고 모든 epoch를 함께 보고한다.1140의 양성4개만 골라 최종
모델로 채택하지 않는다. 예정된 source 전달/R 제거 진단, feature/raw 후속 arm 및
native8/legacy6의 역할별 DEV 비교에서 원인과 조합의 실제 이득을 확인한다.
독립 TRAIN seed·적절한 선택 조합 확인·frozen508개10 metric과 최종 보고서는 계속 남아 있다.
