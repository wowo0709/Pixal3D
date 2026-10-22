# Zero-shot·generated PBR·native SS 기여 진단 종합

2026-10-22 UTC. 아래 결과는 실제 terminal 0과 file4 SHA closure가 있는 개발 평가다.
최종 모델 채택, 독립 학습 seed 또는 test508 완료를 의미하지 않는다.

## 현재 판단

**기존 Full-R 조합의 target alignment 개선은 관측됐지만, semantic matching + matching-aware
attention의 추가 효과와 source preservation 개선은 아직 입증되지 않았다.**
새 R-mediated source 모듈은 32개에서 소폭 좋아 보였으나, 별도 source cluster의 72개·세 noise에서
기존 Contrast19를 넘지 못했다. 같은 proxy objective의 epoch/후보 탐색을 확대할 근거가 부족하다.

## 원본 MV zero-shot을 포함한 8 metric

원본 checkpoint는 SS64/Shape512 step4210, Shape1024/PBR1024 step16850의 base+LoRA 및 대응 config다.
Zero-source4는 deform 전 camera8을 포함한 M0 RGB 네 장, Zero-edited는 M1 target8과 M0 source9/10/11이다.
Target-only나 depth 입력은 사용하지 않았다.

64개 zero-shot cascade의 생성·렌더와 두 조건의 공식 DreamSim/CLIP 192 view가 완료됐다.
기존 trained-PBR 다섯 모델의 480 view와 함께 **7모델×32개×3카메라=672 view**를 집계했다.
기존 다섯 모델의 모든 all-view 통계와 CI를 정확히 재현했다(최대 차이 0).

| 모델 | target PSNR ↑ | target SSIM ↑ | target M-IoU ↑ | source LPIPS ↓ | source DreamSim ↓ | source CLIP ↑ | CD ↓ | F-Score ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-source4 | 22.6626 | .94095 | .82017 | .055638 | .157438 | .940495 | .013338 | .820778 |
| Zero-edited | 23.8777 | .94889 | .86993 | .053487 | .154792 | .943472 | .012519 | .845919 |
| Full-V1 | 24.2682 | .95049 | .88096 | .053055 | .155995 | .940614 | .012458 | .849527 |
| Full-R | 24.7413 | .95167 | .88493 | .052596 | .154441 | .943435 | .012115 | .861474 |
| Full-Direct | 24.2567 | .95087 | .88987 | .053431 | .155872 | .942331 | .012264 | .856702 |
| PBR-only R | 24.3004 | .95045 | .88099 | .053184 | .157008 | .940912 | .012458 | .849527 |
| PBR-only Direct | 23.9970 | .94970 | .88099 | .053497 | .161607 | .940248 | .012458 | .849527 |

Target는 camera8, source는 camera9/10 평균이다. Source reference도 **M1 geometry에 원래 material을
적용한 동일 GT RGB/alpha**이며 undeformed M0 이미지가 아니다. CD/F-Score는 정렬 없는 source bbox
좌표계, surface20k/20k, threshold .02다. PBR-only 두 모델은 Full-V1의 geometry와 초기 texture noise를 공유한다.

주요 paired source-cluster 95% CI:

- Zero-edited − Zero-source4: target PSNR +1.2151 dB [+.7425, +1.7517], M-IoU +.04976
  [+.03551, +.06487], CD −.000819 [−.001323, −.000311], F-Score +.02514 [+.00757, +.04362].
  Edited RGB 하나만으로도 원본 multi-view 모델이 edit 정보를 상당 부분 활용한다.
- Full-V1 − Zero-edited: target M-IoU +.01102 [+.00428, +.01809]. PSNR +.3905 dB의 CI는
  [−.0856, +.8353]이다. Source preservation과 CD/F-Score 이득은 확정되지 않았다.
- Full-R − Zero-edited: target PSNR +.8636 dB [+.2621, +1.4646], M-IoU +.01500
  [+.00737, +.02281]. Source LPIPS −.000891 [−.003632, +.002070], DreamSim −.000351
  [−.012132, +.013941], CLIP −.000037 [−.005646, +.005626]로 source 개선은 불확실하다.
- Full-R − Full-V1: target PSNR +.4731 dB [+.2216, +.7641]. 그러나 full-stack 비교는
  SS와 PBR이 함께 바뀐다. PBR-only R에서는 개선이 확인되지 않았고 PBR-only Direct는 일부 지표를 악화시켰다.

원본 full cascade와 새 PBR replay는 texture noise stream이 같지 않다. Zero-shot 대비 수치는
동일 개발 패널에서의 **deployment 비교**이며, fine-tuning이나 단일 모듈만의 noise-matched 인과 실험이 아니다.
Full-R은 기존 stage 조합이다. 특히 그 PBR은 기존 target-patch query 구조이며, 새 source-camera-ray
R-rawTopK112를 PBR까지 학습한 모델이 아니다. 두 구조의 근거를 합쳐서 주장하지 않는다.
모든 CI는 학습 seed 하나·noise 한 번의 반복 개발 비교이며 다중 비교 보정 전이다.
FID/KID는 final508의 계획된 view pool에서 계산한다.

[W&B: 7모델 8 metric 및 CI](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/original-zero2-and-trained-PBR5-generated32-eight-metrics-v1).
Canonical 결과는 feature-readout 루트의
`original-zero-and-trained-PBR32-comparison-v2/summary-v1/{audit.json,statistics.json,file4-closure.json}`이다.

## 새 source-ray SS 모듈의 소비 단계 검사

동일 R-rawTopK112에서 normal-shadow, Top-K 내부 wrong-V, semantic-prior-only,
Top1-feature-only, no-transformed-value를 실제 native32로 평가했다.
기본 경로의 기록된 GT/noise/occupancy count/metric 및 실제 module 출력/RNG 동등성을 먼저 검증했다.
개입은 source-use, semantic prior, candidate와 Top1 carrier를 유지했고 새 학습 update는 없었다.

Normal IoU .431005에 대한 차이는 각각 wrong-V +.000014, prior-only −.000153,
Top1-only −.000281, no-transformed-value −.001037이다. **네 CI 모두 0을 포함했다.**
따라서 attention 보정/soft mixture/value 변환의 개별 이득을 현재 자료로 주장하지 않는다.
Wrong-V는 semantic Top-K 안의 순열이므로, 비슷한 후보들 사이의 재배치에 둔감한 결과를
semantic matching 자체가 불필요하다는 결론으로 확대할 수도 없다.
Top1-only는 uniform이 아니며, wrong-V는 spatial NoCorr 재학습이 아니다.

[구체적인 개입 정의와 수치](2026-10-22-R-mediated-mechanisms-and-next72-ko.md),
[W&B: native 개입 결과](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-R-mediated-rawTopK112-saved-controls32-v1).

## 72개·세 noise에서는 새 후보 이득이 재현되지 않았다

새 후보의 checkpoint/epoch를 바꾸지 않고 step112 하나를 평가했다. 기존 TRAIN1536/112 update,
19 tensor source 모듈의 Original19와 Contrast19 reference를 재사용했다. SS denoiser는 모두 원본4210으로
동결돼 있다. 72개는 pilot32와 source cluster가 겹치지 않는 61개 cluster지만,
이전에도 사용한 개발 패널이므로 untouched confirmation/test로 부르지 않는다.

| source 모듈 | 72개·세 noise pooled IoU |
|---|---:|
| Original19 teacher112 | .412554 |
| Contrast19 teacher112 | .414531 |
| R-rawTopK teacher112 | .413590 |

R-rawTopK − Original19: +.001036, CI [−.001360, +.003585].
R-rawTopK − Contrast19: **−.000941**, CI [−.002702, +.000448].
기존 Contrast19−Original19 통계도 정확히 재현했다. 새 후보를 채택하지 않으며
기존 plain-contrast72 admission HOLD를 바꾸지 않는다.

[W&B: 고정 source72/세 noise](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-Rraw112-fixed-development72x3-v1).
Canonical 결과는 `source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/fixed-source-disjoint72-v1/summary-v1/`이다.

## 다음 학습을 정당화할 검증

현재 결과는 “R 경유 전달 + raw Top-K 추가를 feature-teacher loss로 학습하면 충분하다”는 가설을
지지하지 않는다. Data/camera/checkpoint 정합 검증과 cold native 동등성은 통과했으므로,
이 결과만으로 다시 전처리를 전부 하거나 같은 proxy 학습 에폭을 늘리지 않는다.

다음은 **실제 native SS flow objective와 consumer까지 포함한 검증**이어야 한다.
기존 V1의 실제 config/data/LoRA 초기화/optimizer/학습 예산을 다시 대조하고,
새 구조를 native 학습에 통합한 뒤 실제 flow loss가 source19와 LoRA에 도달하는지,
초기 동작과 RNG가 의도한 reference와 같은지, 요청된 batch에서 메모리가 맞는지부터 확인한다.
Proxy feature fit이나 nonzero gradient만으로 품질 향상을 주장하지 않는다.
새 objective 실험의 admission은 별도 lineage로 설계하며 과거 HOLD를 성공으로 바꿔 적지 않는다.

Feature-teacher objective와 실제 3D task의 차이, ray pooling의 정보 손실, frozen score prior의
보정 여유 및 native global context 등의 가설은 아직 원인이 확정되지 않았다.
모든 원인을 동시에 바꾸는 grid보다, 위 진단과 실제 task gradient/출력 증거로 병목을 좁혀야 한다.
Source-preservation·독립 seed·새 모델의 native joint 학습·full cascade·final508/10 metric은 계속 남아 있다.
