# Native source control 288-step 및 bounded192 검증

2026-10-21 13:12 UTC. 실행 중인 네 SS 학습을 유지하고 실제 validation
prefix를 복사해 비교했다. 새로운 학습 arm, loss, 입력 데이터 변경은 없다.

## 같은 step의 결과

기준은 동일 step의 unbounded member-feature NoDQ다. 모두 고정된
validation32, 같은 sampling noise와 GT voxel hash를 사용한다.

| Arm | Step | Pooled IoU | 같은 step 기준 | 차이 | Paired 95% 구간 |
|---|---:|---:|---:|---:|---:|
| Native spatial | 288 | 0.444345 | 0.434628 | +0.009716 | [−0.004015, +0.024737] |
| Deformation query | 288 | 0.435263 | 0.434628 | +0.000635 | [−0.001074, +0.001992] |
| Bounded prior | 192 | 0.433470 | 0.434217 | −0.000747 | [−0.002946, +0.001383] |
| Physical Top-1 | 384 | 0.435093 | 0.434842 | +0.000251 | [−0.005311, +0.005407] |

Primary IoU는 TP/FP/FN을 합산해 계산하며 per-asset 평균은 별도 기록했다.
Bootstrap은 source cluster 단위 4,000회, training seed 1개와 sampling
noise 1개다. 여러 중간 관측에 대한 보정은 하지 않았다. 모든 구간이
0을 포함하므로 spatial의 높은 point estimate를 semantic matching의
열세 확정이나 최종 모델 선정으로 해석하지 않는다.

현재 bounded 설계는 matching posterior가 학습 중 과하게 재분배된다는
가설을 시험한다. 192 step까지는 기존 attention보다 개선되지 않았다.
Deformation query의 추가 이득도 작다. 계획된 최종 480-step 결과와
128개 × 3-noise 평가로 이 추세가 유지되는지 확인한다. 유리한 중간
checkpoint만 골라 비교하거나 자동으로 epoch를 연장하지 않는다.

## 캐시 feature와 native encoder의 실제 대조

첫 고정 validation asset의 실제 4개 RGB를 CPU에서 native DINO로
다시 인코딩했다. FP32 native patch와 기존 FP16 cache를 대조했다.

- 네 view의 상대 RMSE는 0.0001961~0.0001968, 평균 cosine은 약 1.0.
- Source foreground 311개 token의 상대 RMSE는 0.0002021.
- 같은 native projection에 통과시킨 grid의 상대 RMSE는
  0.0001350~0.0001405.
- Patch 위치를 31칸 옮긴 음성 대조의 상대 RMSE는 0.31370,
  source view를 바꾼 대조는 0.36819였다.
- Dataset과 cache producer의 RGBA 로딩 결과는 정확히 일치했다.
  기존 input/cache의 SHA256도 검사 전후 같았다.

이 샘플에서는 cache 값의 normalization이나 token/view 배열 순서가
native feature와 크게 다르다는 설명을 지지하지 않는다. 전체 cache,
PBR NAF feature, 실제 CUDA BF16 경로의 일치를 증명한 결과는 아니다.
초기 module 출력 일치 검사와도 별개의 검사다.

CPU 검사 중 dataset 전체 import가 flex_gemm의 CUDA 초기화를 요구해
실패한 기록을 보존했다. 최종 검사는 dataset 파일의 이미지 로더 함수
AST를 수정 없이 읽어 실행했다. CUDA를 초기화하지 않았고 두 CPU core만
사용했다. 학습 코드는 변경하지 않았다.

## GT source RGB의 가용성

파일서버4의 실제 world-v4 train/validation manifest hash와 로컬 운영
manifest hash를 대조한 뒤 raw M1 RGB 존재 여부를 읽기 전용으로 조사했다.

| Split | Asset 수 | RGB 참조 선언 | M1 camera9·10 모두 존재 | M1 camera11 존재 |
|---|---:|---:|---:|---:|
| Train | 9,149 | 9,143 | 9,143 | 0 |
| Validation | 508 | 508 | 508 | 0 |

Train 6개는 선택적 RGB 참조 목록이 비어 있어 미확인으로 남겼다.
해당 파일이 없다고 임의로 판정하거나 경로를 추정하지 않았다. 각 split의
첫 32개 asset에서 target M1/camera8 및 source M0/camera9·10을 raw PNG와
비교한 총 192쌍은 byte 단위로 일치했다. 전수 pixel 비교는 아니다.
초기 조사 코드가 빈 참조 목록에서 종료된 기록과 수정된 조사 코드를
함께 보존했다. Test split은 조회하지 않았다.

향후 source feature adaptation의 병목을 분리할 필요가 생기면 camera9·10만
M1 GT RGB로 바꾼 부분 oracle을 구성할 수 있다. 이는 추가 정답 입력을 쓰는
진단이므로 경쟁 모델 성능으로 보고할 수 없다. Camera11이 없어 확인된
경로만으로 full four-M1 RGB oracle은 구성할 수 없다. Train-only feature
teacher도 가능성을 확인한 단계이며 아직 구현하거나 queue에 넣지 않았다.

## 다음 판정과 보존

Matching prior만 사용한 고정 readout의 pilot32 결과가 좋아 보였던 현상이
source가 겹치지 않는 72개에서 재현되는지, 연결된 3-noise 진단을 먼저
확인한다. 이 결과와 현재 bounded/spatial/Top-1의 최종 비교를 통해
후보 자체의 한계와 학습된 attention의 재분배 문제를 구분한다. Learned
attention이 실제 이득을 냈다고 주장하려면 동일 V/R을 유지한 prior-only
counterfactual 및 실제 attention 분포 검사가 추가로 필요하다.

현재 SS 중간 IoU로 source preservation이나 PBR 성능 향상을 주장하지
않는다. 이미 연결한 cascade32의 source LPIPS/DreamSim/CLIP과 target
PSNR/SSIM/M-IoU/CD/F-Score, PBR 학습, 독립 training seed 확인, 최종
508개와 FID/KID를 포함한 평가 범위를 유지한다.

Bounded arm의 5개 epoch × 3개 checkpoint 파일을 보존하는 CPU archive를
기존 queue의 끝에 추가했다. 아직 실제 archive 완료를 뜻하지 않는다.
기존 job/state를 변경하지 않았으며 로컬 checkpoint도 삭제하지 않았다.

Curve, paired CI, count CSV, 실제 validation prefix를 W&B와 file4에 저장했다.

- [최신 W&B 곡선과 비교표](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-member-native-controls-spatial-dq288-bounded192-s101-v1)
- Curve: `member-feature-controls-v1/paired-spatial-DQ288-bounded192-observation-v1`
- CPU cache 대조: `failure-driven-diagnostics-v1/native-SS-DINO-cache-parity-CPU-v3`
- RGB 가용성: `failure-driven-diagnostics-v1/M1-source-RGB-availability-v1`

상대 경로의 canonical 부모는
`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`이다.
최종 모델은 채택하지 않았고 전체 실험은 진행 중이다.
