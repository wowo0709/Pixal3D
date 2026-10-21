# PBR Direct 1,143 step의 실제 세 arm 비교

2026-10-21 20:00 UTC에 Direct의 epoch 검증이 완료됐고, checkpoint 저장 후 학습이
재개됐다. 아래는 **개발 자산 4개·sampling noise 1개·training seed 101**의 빠른
경향 검사다. 모델 채택이나 모집단 성능 향상의 근거로 사용하지 않는다.

## 비교 조건과 수치

V1, retarget-only(R), Direct 모두 공통 NN/Adam step762에서 시작해 동일한
381 update를 추가한 step1143 결과다. World4/global24/micro6/accumulation1,
원래 sampler, 12 sampling steps, generation batch1, 256px 렌더링을 사용한다.
Actual eligible PBR train asset은 원래 token limit을 적용한 9,139개다.

세 arm의 sample/source cluster/seed/RGB/input fingerprint/GT PBR latent/
GT Shape/noise/foreground fraction을 확인했다. 저장된 per-asset 값으로 native
PSNR 집계와 나머지 지표를 재계산해 원래 결과와 일치함을 확인했다. Foreground
PSNR은 자산별 dB의 단순 평균이 아니라 foreground 비율로 오차를 모아 계산한다.

| 지표 | V1 | R | Direct | Direct−V1 |
|---|---:|---:|---:|---:|
| Foreground PSNR ↑ | 15.586639 | 15.786715 | 15.354367 | −0.232272 dB |
| Target crop LPIPS ↓ | 0.088885 | 0.089722 | 0.087364 | −0.001521 |
| Target crop SSIM ↑ | 0.952599 | 0.952662 | 0.953373 | +0.000773 |

이 검사는 **GT target Shape를 고정한 camera08의 appearance 평가**다. Crop LPIPS는
이 표에서 target alignment 진단이며 논문의 source preservation LPIPS가 아니다.
GT Shape로 인해 alpha IoU가 1인 것도 생성 geometry 향상이 아니다. 전체 cascade,
source9/10 보존, CD/F-Score와 FID/KID 결과로 대체해 해석할 수 없다.

## 평균 뒤에 있는 차이

같은 panel 순서의 asset1은 V1 대비 PSNR +0.618026 dB, LPIPS −0.027325,
SSIM +0.012059로 개선됐다. Asset2는 PSNR −1.904169 dB, LPIPS +0.021075,
SSIM −0.009012로 세 지표 모두 나빠졌다. Asset3/4는 차이가 작다.
따라서 현재는 안정적인 개선보다 자산별로 상반된 correction 효과를 관측했다.
네 자산으로 유의한 모집단 효과나 checkpoint 선택 결론을 내리지 않는다.

Native GT/Pred/Diff 이미지도 학습 WandB에 기록됐고, per-asset 차이 PNG/PDF,
CSV 및 입력 정합을 확인한 snapshot을 canonical 파일서버4에 저장했다.
추가 비교 표와 figure는 [WandB 비교 기록](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-step1143-three-arm-paired4-v1)에 있다.

## Query 지원 영역 누락인지 확인

GPU·모델 forward 없이 이 네 자산의 실제 input cache와 target alpha에서 native
target-patch query의 후보 유효성 조건을 재구성했다. Matching schema, Q2C hash,
input fingerprint를 확인하고 production 코드의 source validity, target alpha 및
candidate mask를 그대로 적용했다.

| Panel asset | 유효 source token | Active target patch | 후보 64개인 active patch | Active patch에 매핑된 GT Shape token 비율 |
|---|---:|---:|---:|---:|
| 1 | 171 | 41 | 41 | 80.15% |
| 2 | 528 | 109 | 109 | 89.01% |
| 3 | 447 | 113 | 113 | 92.43% |
| 4 | 199 | 61 | 61 | 83.74% |

모두 최소 source64 조건을 통과하고 유효 target patch마다 후보64가 있다.
Asset2의 악화를 **후보가 없어서 source 경로가 꺼진 결과**로 설명할 수는 없다.
다만 이 수치는 지원 영역만 검사한다. 후보의 semantic 정확성, 실제 learned attention
분포, 유효한 feature correction인지 또는 GT Shape 밖의 가시성을 증명하지 않는다.

## 다음 판단

현재 PBR Direct는 이전 target-patch query 계열이며 새 source-camera SS
Original19/SourceContrast19 모델과 다르다. Direct−R에는 feature attention과
RAW deformation query 차이가 함께 있으므로 attention, score, decoder의 개별
효과를 분리했다는 주장은 할 수 없다.

고정한 학습 예산을 유지해 step1524/1905 추세와 최종 matched32 결과를 확인한다.
그 뒤 full cascade의 source preservation/target alignment를 구분해 판단한다.
이 4개 관측만으로 새 arm이나 LR·loss 변경을 추가하지 않는다. 지원은 충분한데
일부 자산이 악화된다는 관측은 이후 후보 정확도·feature 사용 및 correction에 대한
진단에서 확인해야 할 가설로 남긴다. 모델 채택과 전체 실험 완료는 아직 아니다.

Canonical 결과는 `PBR-joint-continuation-v1/epoch-validation-trends-v1/` 아래
`Direct-step001143-three-arm-v1/`(8개 파일)과
`Direct-step001143-query-support-CPU-v1/`(2개 파일)에 있다.
각 `file4-closure.json`으로 실제 원격 SHA256·크기 일치를 확인했다.
