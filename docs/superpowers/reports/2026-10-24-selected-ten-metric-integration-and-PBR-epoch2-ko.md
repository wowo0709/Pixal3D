# 선택된 pipeline의 10 metric 연결과 PBR 2에폭 중간 결과

2026-10-24. 학습과 실제 최종 모델 선택은 진행 중이다. 현재 결과로 semantic matching·learned attention 각각의 필요성이나 전체 cascade 개선을 확정하지 않는다.

## 실제 PBR 760-update 비교

같은 seed101·global24·DDP4·micro6·accumulation1, 5에폭1900 예산 중2에폭 결과다. 고정4개 asset/4 source, target camera8, GT Shape 조건의 PBR stage 진단이며 source preservation과 generated geometry 평가가 아니다.

| 구성 | Foreground PSNR, 기존 pooled 기준 | Crop LPIPS↓ | Crop SSIM↑ |
|---|---:|---:|---:|
| Vanilla V1 | 15.49141 | 0.090078 | 0.952008 |
| Top-1 + retargeting | **15.51446** | **0.086102** | **0.952942** |
| Top-1 + feature | 15.42118 | 0.087515 | 0.952306 |
| Raw Top-K + feature | 15.50221 | 0.086654 | 0.952740 |

Raw Top-K+feature는 초기값 대비 pooled foreground PSNR14.94645→15.50221, crop LPIPS0.095142→0.086654, crop SSIM0.949160→0.952740이다. 동일760 V1 대비 crop LPIPS는 약3.8% 낮다. 하지만 retargeting Top-1보다 crop LPIPS가0.000552 높고 crop SSIM이0.000201 낮아, Top-K/feature 조합의 추가 이득은 아직 없다. 4개뿐인 작은 진단이며 샘플별 방향도 섞여 있다.

Foreground PSNR 로그는 foreground 크기로 가중한 MSE를 합친 뒤 log를 취한다. Per-asset dB의 산술 평균과 다르다. Source 코드의 기존 `native_pbr_stage_summary.aggregate`로 기록을 재현했으며 별도 mean-asset 수치도 artifact에 보관했다. PSNR 방향이 aggregation에 따라 달라질 수 있으므로 하나의 작은 pooled 차이로 정합 개선을 주장하지 않는다. 최종508 표의 target PSNR은 각 asset의 camera8 PSNR 평균과 source-cluster interval을 사용한다.

실제 네 모델의 rank0/1/2/3 각각760 update의 asset ID, x_t/t/global trace가 모두 정확히 같았다. 검증 RGB·GT PBR latent·GT Shape 조건·camera·seed/noise hash도 모든 모델/epoch0/380/760에서 일치했고, 초기 predicted latent hash도 정확히 같았다. 기존 세 baseline의 canonical file4 로그와 실제 종료 child를 확인했다. 학습 중인 raw arm의 선택된 실제 행과 trace prefix를 별도 immutable snapshot으로 보관했으며 mutable live log를 이미 종료·봉인된 파일이라고 표시하지 않았다.

증거: file4 experiment root 아래 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/PBR-epoch760-four-arm-fixed4-matched-v193/`. 실제 선택된 validation 행, 네 rank의 동일한 training prefix760와 audit·closure가 있다. Alpha IoU1.0은 GT Shape 조건의 결과이므로 생성 geometry 정확도의 증거가 아니다.

## 실제 선택 모델 수에 맞춘 평가 코드

기존 고정6-model 수집기를 보존하고 다음 파일을 추가했다.

- `selected_final_metric_summary.py`: 실제 deduplicate된4/5 모델의8개 pairwise metric, source-cluster 통계와 동일 이미지 FID/KID를 결합한다. 선택 모델이 V1 또는 RETARGET이면 별도 SELECTED 행을 만들어 중복하지 않는다.
- `selected_final_metric_inputs.py`: 실제 DEV-only parent/control freeze,508 SS archive, 완성된 campaign, case/phase/worker terminal, scored RGB/reference/alpha 및 crop/conversion을 확인한다.
- `selected_inception_entry.py`: 기존 Inception 수치 body를 보존하고 GPU membership literal 하나만 명시적 할당으로 바꾼다. AST 복원과 source hash proof를 남긴다.
- `run_selected_final_metric.py`: 한 GPU에서 기존 official DreamSim·별도 OpenAI CLIP 또는 Inception2048 entry를 실행한다. 입력 픽셀과 weight/source registry를 실제 GPU forward 전후 확인한다.
- `prepare_selected_final_metrics.py`: 모든 실제 case가 완성된 뒤 같은 이미지의 모델별 GPU metric 작업과 CPU reducer를 기존 두 큐에 등록한다. 두 큐 등록 후 공통 activation receipt가 있어야 시작한다.
- `collect_selected_final_metrics.py`: 실제 종료한 GPU worker/phase와 feature archive를 확인하고 공식 FID/KID를 계산해 CSV와 canonical10-metric report를 만든다. CPU에서는 neural forward/training을 수행하지 않는다.

10개 metric의 역할은 source9/10의 LPIPS·DreamSim·CLIP, target8의 PSNR·SSIM·M-IoU와 공통 M1 표면에 대한 CD·F-Score, 전체8/9/10 이미지 pool의 FID·KID다. 모든 모델의1524개 이미지와 같은 원본 M1 material reference/alpha를 사용한다. Fullframe이 primary이고 GT-only crop은 supplementary다. FID/KID는 공식 Inception-v3-compat2048 FP32 feature와 float64 통계, KID100 subsets×1000 images/seed42017을 유지하며 음수 unbiased KID를 clamp하지 않는다. KID subset std를 confidence interval로 해석하지 않는다.

실제 선택 모델 대 V1/RETARGET의8개 pairwise endpoint에 source134 cluster 단위 bootstrap과 명시적 family interval을 제공한다. 일반 CI·source-weighted 평균도 남긴다. FID/KID에는 별도 CI/significance를 주장하지 않는다. 이 비교·metric 역할·방향·설정은 `evaluation_protocol.selected_final_metric_contract`에 동결한 뒤 TEST를 실행해야 한다. Final evaluation은 한 pipeline training seed이며 개별 SS/PBR 독립 seed gate가 full-cascade seed 반복을 대신하지 않는다.

## 검증과 다음 실행

새 검사62개를 포함해 관련205개 CPU 검사가33.23초에 통과했다. 실제 FP32 native512 NPZ의 CHW/HWC 변환, GT-only crop, feature bank1524×2048 row order와 hash, 잘못된 source/device proof, live worker를 종료로 표시한 state, partial queue append,4/5-model 및 shared-geometry phase 차이를 검사했다. Synthetic metric/feature fixture는 실제 neural quality 결과가 아니다.

별도로 기존에 실제 점수를 계산했던 DEV render4개를 그대로 사용하는 GPU6 entry compatibility 작업을 append했다. 기존 E_FIXED_RAW의 final128 noise0/1/2가 모두 끝난 뒤에 실행한다. DreamSim/CLIP은 같은 실제 이미지의 기존 original score와 비교하고 Inception은 실제 GPU6에서 두 명시적 membership 할당의 feature bank가 bitwise 같은지 검사한다. 이 작업은 아직 대기 중이다. 새로운 학습 arm이나 TEST quality 평가가 아니다. 기존 정상 학습은 중단하지 않았다.

남은 작업은 원래 core12 및 consistent-evidence fixed/learned SS 후속 실험, raw PBR1900, 실제 joint DEV pipeline 비교, 선택 모델/강한 baseline의 독립 seed와 repeatability, 최종 numerical template·source·metric contract 동결이다. 그 다음 실제508 SS/cascade와 새 metric 작업을 활성화한다. 이번 작업에서 actual selected final TEST508이나 최종10 metric 계산을 수행한 것으로 표시하지 않는다.
