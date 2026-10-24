# 같은 PBR을 사용한 SS stage의 DEV32 생성 geometry 비교

2026-10-24. V1과 C_DECODER 각각의 실제 SS prediction을 원본 Shape5124210·Shape102416850과 **같은 학습된 vanilla V1 PBR1900**으로 연결한32개 결과가 모두 완료됐다.32개 asset·source cluster,64개 생성 case, target8/source9/10의192개 view에 대한 비교다. SS 모델 선택의 downstream 효과를 보는 실험이며, 새 E_FIXED_RAW나 PBR Raw Top-K를 결합한 최종 모델의 결과는 아니다.

## 실제 관측된 여섯 endpoint

Source LPIPS는 source camera9/10의 fullframe 값을 asset 안에서 평균한 뒤32개 asset 평균을 냈다. Target image metric은 camera8이다. CD는 normalized coordinate의 symmetric L1 Chamfer이고 F-Score는 threshold0.02, 각 surface20,000 samples, per-output alignment 없이 측정했다.

| Metric | V1 SS | C_DECODER SS | C_DECODER − V1 |
|---|---:|---:|---:|
| Source LPIPS ↓ | 0.053644 | 0.053502 | −0.000142 |
| Target PSNR ↑ | 24.26623 | 24.36422 | +0.09799 |
| Target SSIM ↑ | 0.950695 | 0.950655 | −0.000040 |
| Target mask IoU ↑ | 0.880957 | 0.885747 | +0.004790 |
| CD L1 ↓ | 0.01245167 | 0.01241337 | −0.00003830 |
| Surface F-Score ↑ | 0.849149 | 0.851420 | +0.002271 |

보조 foreground PSNR도12.15107→12.22861로+0.07755dB였다. 위 PSNR은 fullframe이며 보조값과 섞지 않았다. Voxel F1과 위 surface F-Score도 다른 metric이다.

SS 모델 변경의 효과가 일부2D mask와3D surface 지표까지 이어지는 작은 신호는 관측됐다. Source LPIPS 변화는 작고 SSIM은 약간 낮아 모든 metric이 좋아졌다고 하지 않는다. 한 training seed의 DEV32이며 이번 읽기는 새 통계 검정·CI·최종 모델 선정을 수행하지 않았다. C_DECODER는 semantic Top-1과 retargeting을 함께 사용하므로 이 비교만으로 retargeter 단독의 인과 효과를 분리했다고 하지 않는다.

서로 다른 SS support에서는 downstream sparse coordinate와 tensor shape가 달라질 수 있다. 원래 native texture sampling protocol은 같지만 initial texture noise tensor가 bitwise 같다고 주장하지 않는다. 이 결과를 같은 입력 support에서 PBR 모듈만 바꾼 효과로 해석하지 않는다.

## 데이터와 픽셀 대응 확인

64개 case의 실제 성공 queue state와 closed generation/shared 결과, 각 case의 세 owned child 종료를 확인했다. 자신의 SS prediction이 geometry 생성에 사용됐고 원본 두 Shape checkpoint 및 같은 PBR checkpoint가 사용됐는지 검증했다. GT geometry를 생성에 사용하지 않고 생성 후 metric에서만 읽었으며 per-output alignment가 없다는 원래 실행 계약을 확인했다.

실제192개 shared NPZ와 자신의 renderer NPZ를 별도로 읽었다. 저장된 prediction·alpha가 자신의 renderer 출력과 같고, GT float RGB·alpha SHA가 기록 및 두 모델 사이에서 같았다. 원본 material을 edited M1 geometry에 적용한 reference다. 정당한 deformation을 오류로 벌점 주지 않도록 undeformed M0 RGB와 직접 비교하지 않는다.

실제 alpha에서 계산한 mask IoU가 모든192개 view에서 기록과 일치했다. NumPy float64로 PSNR식을 교차 계산한 최대 차이는 fullframe3.74e−6dB, foreground2.44e−6dB였다. 원래 GPU float32 값을 바꾸지는 않았다. SSIM·LPIPS는 원래 GPU 측정값을 읽었고 CPU에서 neural forward나 training을 수행하지 않았다. 새 GPU inference나 학습 arm도 추가하지 않았다.

## 다음 비교

원래5개 SS 모델의 공통 PBR DEV32 실행과480개 view의 공식 DreamSim·CLIP 집계는 그대로 진행한다. 이 보고서에는 여섯 endpoint만 있으며 DreamSim·CLIP·FID·KID를0이나 대체값으로 채우지 않았다. 공통 생성 geometry에서 PBR 네 모델의 효과, 기존 joint baseline과 D1-cos/E_FIXED_RAW candidate joint도 필요하다. 새 shared-evidence learned attention 및 mean/voxel query 학습, 독립 training seed 확인과 최종 TEST508·10개 metric 역시 미완료다.

현재 데이터만으로 frozen Shape interface를 실패 원인으로 단정하거나 새 architecture 실험을 추가하지 않는다. 먼저 예정된 실제 source/target·surface 비교로 모듈과 조합의 효과를 확인한다. 최종 full 모델은 아직 채택하지 않았다.

실제 file4 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/common-V1-retarget-SS-pipeline-six-metrics-v201/` 아래의 `audit.json`, `actual-case-records.json`, `actual-pixel-binding-v1/`, `WandB-v1/`에 있다. [W&B의21개 scalar·두 표·실제 GT/Pred 패널 두 개와 artifact](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-V1-retarget-common-PBR-six-metrics-DEV32-v201)의 원격 완료를 확인했다.
