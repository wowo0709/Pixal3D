# 공통 R96에서 시작한 SS 두 arm의 실제192 비교

Retargeting control과 direct-NoDQ가 모두192-step 평가를 완료했다.
같은32개 개발 자산·GT voxel·source cluster·생성 seed로 비교한 결과,
attention을 추가한 direct-NoDQ의 주 지표 차이는−0.0304%p로 아직 추가 이득이 없다.
두 학습은480-step 예산을 유지한다.

| 조건 | 자산별 평균 voxel IoU (%) | TP/FP/FN 합산 IoU (%) |
|---|---:|---:|
| 공통 초기96 | 46.4290 | 43.7187 |
| Retargeting192 | 46.8965 | 44.2303 |
| Direct-NoDQ192 | 46.8661 | 44.1801 |

Direct-NoDQ − Retargeting의 mean-asset IoU 차이는−0.030397%p,
source-cluster bootstrap4000회/seed20261023의 탐색적95% CI는
[−0.424206, +0.376417]%p다. 0을 포함하므로 우월성이나 동등성을 입증하지 않는다.
같은 개발 panel의 반복 중간 평가이며 전체 탐색·여러 epoch에 대한 보정은 아니다.
합산 IoU와 자산별 평균을 혼용하지 않는다.

## 초기 하락과 비교 조건

실제 초기96 평가의32개 per-asset metric·TP/FP/FN·GT hash·cluster·seed가 모두
같다. 공통 NN360 LoRA와 R/fusion6개의 Adam 상태도 step96에서 복원됐다.
Direct-NoDQ에는 deformation-query channel과 projection module이 물리적으로
없으며0값 channel로 대체하지 않았다. 초기 성능 하락은 관측되지 않았다.
해당 초기 기록에 predicted latent/full occupancy hash가 없어 bit-exact 예측까지
증명했다고 하지는 않는다.

DataLoader worker는 retargeting8, direct-NoDQ0으로 다르다. 두 실행의 dataset
설정과 data_dir는 같고 데이터 경로에 무작위 augmentation이 없다.
실제 저장된96–192 구간의 두 rank194개 RNG snapshot에서 CPU/CUDA·NumPy·Python
RNG와 consumed sampler 상태가 전부 같았다. Loader 전용 generator는191/192의
두 rank, 총4개에서 달랐지만 학습 RNG는 같았다. Persistent worker와 iterator
수명에 따른 loader generator 차이를 학습 noise 차이로 해석하지 않는다.
원래 training RNG를 그대로 재개한 것은 아니며 두 arm에 같은 rank별101/102
stage seed를 적용한 프로토콜이다. 실제 x_t/tensor hash는 기록하지 않아 그
동일성까지 주장하지 않는다.

## 새 attention branch가 정지했는가

192 checkpoint의 실제 Adam 상태를 CPU에서 대조했다. Retargeting은360+6개
상태가 각각192 update이고, direct-NoDQ는 동일한366개가192 update,
추가 ray-readout9개가 각각96 update다. 모든 momentum tensor가 유한했다.
새 head의 당시 LR은1e−4다.

학습 로그의 step192에는 feature gradient L1 10.70157,
feature residual RMS 0.0147254, direct-retargeted feature RMS 0.0140020,
head별 평균 effective K 4.2303이 기록됐다. 최초 step97의 residual은0이었다.
즉 새 branch가0 출력에 머무르거나 optimizer에서 빠진 상태는 아니다.
이 값은 TRAIN batch 로그이며32개 validation 자산의 정확한 영향량이나
matching 정확도, 성능 정체의 인과적 원인을 증명하지 않는다.

현재는 초기 붕괴 없이 학습된 branch가 아직 추가 생성 품질 이득을 주지 못하는
상태다. 이번192 결과로 설계 전체의 무효나 구현 오류 부재를 확정하지 않는다.
480 최종과128개×3생성 noise, 세 번째 DQ arm 및 공통 PBR cascade를 계속 비교한다.
Spatial3 saved-weight 제어에서 확인한 학습형 재가중의 조건부 기여도 이 semantic
target-patch 재학습 arm의 추가 이득으로 옮겨 주장하지 않는다.

## 실행·저장 범위

GPU0·1과2·3에서 각 DDP2/global96/micro48/accumulation1로 실행한다.
두 모델은 R96에서384회 추가 update를 수행하는480-step 프로토콜이다.
최근 GPU compute는 약17초/update이며 direct-NoDQ는 serial data loading에
약9초가 추가된다. 이를17초 wall-time으로 보고하지 않는다.

두192 NN/source/Adam과 평가 prefix는 실제 file4 SHA256·크기를 검증했다.
추가 CPU 검토는 저장된 상태·metric의 대조이며 새 neural forward/backward나
학습 update가 없다. 최종 frozen tensor·DDP identity를 이 중간 audit의 증거로
주장하지 않는다.

Canonical 경로는 `PBR-native-science-runtime-v1/` 아래
`historical-SS-actual-initial96-v1`, `historical-SS-matched-RNG-through192-v2`,
`historical-SS-matched-epochs-v1/{retargeting,direct-nodq}/step0000192`,
`historical-SS-matched-epochs-v1/two-arm-step192-comparison-v1`이다.
전체 실험·독립 학습 확인·선택 조합과 최종508개10 metric은 아직 완료되지 않았다.
