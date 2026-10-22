# Native PBR vanilla 5 epoch 실제 완료

원본 multi-view PBR16850에서 시작한 vanilla가 fresh optimizer로1900 update를
완료했다. TRAIN9139, seed101, global24/DDP4/micro6/accumulation1이며 물리 GPU
0–3을 사용했다. 원래 학습 child의 종료 코드는0이다. 네 rank의 실제 Adam1900,
frozen base/matcher/anchor 및 DDP parameter 검사가 통과했다. NN·Adam checkpoint
다섯 개 epoch와 최종 native/wrapper 결과의 실제 file4 SHA256/size 검증도 끝났다.

## 고정4 validation 경향

각 epoch는 같은 네 자산, RGB·noise·GT PBR·GT Shape·camera를 사용한다.
아래는 GT target Shape를 조건으로 넣은 target camera8 렌더 평가다.

| Epoch / step | Pooled-MSE PSNR ↑ | Mean-asset PSNR ↑ | LPIPS ↓ | Crop LPIPS ↓ |
|---|---:|---:|---:|---:|
| 학습 전 / 0 | 26.0573 | 29.4633 | .0101050 | .0951262 |
| 1 / 380 | 26.3444 | 29.6719 | .0096876 | .0924272 |
| 2 / 760 | 26.6023 | 29.8789 | .0092733 | .0900777 |
| 3 / 1140 | 26.7225 | 29.9656 | .0089773 | .0882303 |
| 4 / 1520 | 26.7444 | 29.9652 | .0088839 | .0876835 |
| 5 / 1900 | **26.8765** | **30.0736** | **.0087072** | **.0864544** |

Pooled PSNR은 squared error를 합친 뒤 dB로 변환한다. Mean-asset PSNR은 자산별
dB를 평균하므로 서로 다른 수치다. LPIPS와 crop LPIPS는 자산별 값의 평균이다.

4→5 epoch에서 pooled PSNR +.13215dB, mean-asset PSNR +.10840dB,
LPIPS −.00017669, crop LPIPS −.00122914였다. 네 자산 중 두 자산의 변화가 주된
요인이다. 학습 전 대비 자산별 PSNR 차이는 +.43314/+2.01913/−.00135/−.01001dB다.
나머지 두 자산은 거의 그대로다. 마지막 GT/Pred/Diff panel도 확인했으며, 두 번째
자산의 밝기·재질 차이와 첫 번째 자산의 큰 색/밝기 차이는 여전히 남아 있다.

따라서 이 네 자산에서는 5 epoch까지 개선을 관측했지만, 이 결과만으로 전체
데이터 개선이나 추가 epoch의 필요성을 확정하지 않는다. Alpha IoU1은 GT Shape를
고정한 조건의 결과이며 geometry 생성 성능의 증거가 아니다. Source preservation,
predicted geometry를 쓰는 PBR isolation/full cascade, 독립 학습 seed 및 최종508
평가는 별도로 남아 있다. Source 모델과의 우열이나 attention의 효과도 아직 아니다.

## 실제 시간·메모리와 다음 작업

Wrapper 시작부터 canonical 완료까지7시간28분9초였다. 마지막10 update의
중앙 compute 시간은 rank별12.883–12.887초였다. 최대 allocated는43.6595GiB,
최대 reserved는48.4609GiB였다. 저장·validation·전송 시간을 compute 시간과
혼동하지 않는다. Source 모델의 실제 처리 속도는 해당 모델 실행 후 측정한다.

완료 수집 CPU 작업도 실제 종료0으로 끝났으며, 고정6 validation·4 rank1900
trace·GT PNG identity를 보존했다. Snapshot36파일·5,396,061 bytes의 실제 file4
검증이 통과했다. 완료된 중간 checkpoint의 로컬 중복본만 검증 후 정리했고,
canonical file4와 최종1900 checkpoint는 유지했다.

Native SS480 readout 정상 재현 진단이 GPU0에서 실행되기 시작했다. 정상 재현이
통과하면 네 가지 개입을 GPU0–3에서 병렬 실행하고, 이후 Top-1+R PBR1900,
feature+R, raw Top-K+feature+R 학습을 순서대로 비교한다. 아직 실행 중이거나
대기 중인 결과를 완료로 기록하지 않는다.

실제 snapshot은 `PBR-native-science-runtime-v1/`
`vanilla-stage1900-completed-evidence-v1/{audit.json,file4-closure.json}`이다.
[W&B 학습·epoch별 수치 및 panel](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-original16850-vanilla-s101-v1-DDP4).
