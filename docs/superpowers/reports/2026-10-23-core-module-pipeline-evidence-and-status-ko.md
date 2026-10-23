# 핵심 모듈 검증과 최종 파이프라인의 중간 결과

UTC 2026-10-23 16:37 기준. `wowo0709/feature-retargeting`.

실험 목적은 각 모듈의 조건부 효과를 분리하고, 실제 생성 성능이 좋은 하나의 SS→Shape→PBR 파이프라인을 검증하는 것이다. Full은 현재 비교 대상의 이름이며 최종 채택 구조를 뜻하지 않는다. 효과가 없는 모듈까지 넣어야 한다는 조건은 없다. 이미 보류한 지엽·중복 실험을 재개하지 않았고, 이번 변경으로 GPU 학습 작업을 추가하지 않았다.

## 1. 진행 상태

| GPU | 실제 실행 | 최근 진행 |
|---|---|---|
| 0–3 | PBR `top1-plus-feature`, DDP 4 | 1,444/1,900 step |
| 6 | SS `C_TOP1_V` | 299/480 step |
| 7 | SS `C_FIXED_TOPK` | 293/480 step |

모두 실행 프로세스와 `progress.json`을 확인했다. GPU 0–3의 PBR이 끝나면 기존에 연결한 SS `C_RAW`, `C_SPATIAL`, `C_D1_COS`, `C_DECODER_NO_R`을 각각 한 GPU에서 진행한다. GPU 6·7의 기존 두 작업 뒤에는 attention prior, query 단위, member proposal 대조군이 연결되어 있다. SS는 global/microbatch 96, accumulation 1, 5 epoch/480 step이다.

## 2. 새로 확인된 SS 3-epoch 결과

동일한 DEV32 asset·GT·generation seed, raw weights의 **asset 평균 voxel IoU**다. pooled occupancy IoU나 최종 DEV128×3 결과와 섞지 않는다. 모든 다섯 모델의 학습 전 asset별 결과는 동일하다.

| 모델 | 학습 전 | 1 epoch | 2 epoch | 3 epoch |
|---|---:|---:|---:|---:|
| V1 vanilla | 0.45836 | 0.46380 | 0.46745 | **0.46897** |
| Decoder Top‑1 + R | 0.45836 | 0.46162 | 0.46088 | 0.46799 |
| Top‑1 + V/output + R | 0.45836 | 0.46101 | 0.46472 | 0.46598 |
| 고정 matching-score Top‑K + V/output + R | 0.45836 | 0.46082 | 0.46530 | 0.46761 |
| Full: 학습 attention + matching prior + R | 0.45836 | 0.46097 | 0.46459 | 0.46649 |

3 epoch에서 고정 Top‑K는 Top‑1+V보다 +0.001625 IoU, 즉 +0.1625 percentage points다. Full은 고정 Top‑K보다 −0.001115 IoU다. Top‑K 결합의 개선 신호는 있으나, 학습 Q/K·utility attention의 추가 이득은 이 중간 패널에서 확인되지 않는다. V1도 아직 넘어서는 결과가 아니다. 이는 작은 개발 패널의 중간 결과이며 최종 효과 판정은 480-step DEV128×3의 paired source-cluster 분석으로 한다.

기존에 끝난 DEV128×3에서는 V1 0.467355, Top‑1+R 0.472600, Full 0.472078이었다. Full이 단순 Top‑1+R보다 좋다고 주장할 수 없다. 위 DEV32 표와는 panel/noise 수가 다르므로 숫자를 직접 이어 붙여 성장 곡선으로 해석하지 않는다.

## 3. PBR 중간 결과

동일 1,140-step, fixed4 asset, GT target Shape 조건의 빠른 target-view rendering 검증이다.

| PBR 모델 | Render PSNR ↑ | Crop SSIM ↑ | Crop LPIPS ↓ |
|---|---:|---:|---:|
| V1 | 26.72247 | 0.952737 | 0.088230 |
| Top‑1 + R | **27.15627** | 0.956305 | **0.078749** |
| Top‑1 + learned feature + R | 27.14318 | **0.956407** | 0.078876 |

단순 Top‑1+R과 feature 모델 모두 V1보다 좋다. feature 추가가 Top‑1+R을 일관되게 넘지는 않는다. 이 Crop LPIPS는 target-view 지표이며 논문의 source-preservation LPIPS와 구별한다. GT shape에 조건화했으므로 이 표로 geometry 개선이나 실제 full-cascade 성능 향상을 주장하지 않는다. 최종 판단에는 공통 **생성된** geometry/native guides의 PBR 비교가 필요하다.

## 4. 필요한 모듈 비교만 유지

12개 모델을 재사용해 다음 10개 단일 요소 비교를 수행한다. 모든 조합을 돌리는 grid는 아니다.

| 검증 요소 | 비교 |
|---|---|
| Spatial→semantic correspondence | `C_SPATIAL` → `C_RAW` |
| 학습된 matching decoder 사용 | `C_RAW` → `C_D1_COS` |
| Scalar matching score 사용 | `C_D1_COS` → `C_DECODER` |
| V/output feature transform | `C_DECODER` → `C_TOP1_V` |
| 한 correspondence→Top‑K 결합 | `C_TOP1_V` → `C_FIXED_TOPK` |
| 고정 결합→학습 Q/K·utility attention | `C_FIXED_TOPK` → `FULL` |
| Attention 내부 matching prior | `FULL_QK_ONLY` → `FULL` |
| 개별 voxel→source-camera ray query | `FULL_VOXEL_Q` → `FULL` |
| Retargeting의 물리적 유무 | `C_DECODER_NO_R` → `C_DECODER` |
| 평균 descriptor→member proposal | `FULL_MEAN_MATCH` → `FULL` |

Scalar score로 correspondence 순위를 정하는 효과와 score를 attention prior로 쓰는 효과를 구분한다. No-score의 Top‑1은 실제 하나의 feature를 사용한다. No-corr은 원래 spatial projection correspondence를 사용한다. Ray query는 같은 source-camera ray의 voxel 집합으로 query를 만들고, R-only 전달·target/native branch·no-evidence 시 native source 유지 규칙을 공유한다. R의 효과를 Top‑1에서 검증한 결과만으로 모든 attention 조합에서 필수라고 일반화하지 않는다.

## 5. 최종 파이프라인으로 연결하는 순서

1. 480-step SS DEV128×3 결과로 각 요소의 효과와 불확실성을 정리한다. 반복 개발 탐색이며 독립 TEST 증거가 아니다.
2. 기존에 연결한 대표 SS 5개는 같은 학습된 V1 PBR을 사용해 자기 SS→Shape 생성 geometry에서 비교한다. 별도로 PBR 대조군은 공통 생성 geometry/native guides로 비교한다. source LPIPS/DreamSim/CLIP, target PSNR/SSIM/M-IoU/CD/F-Score를 사용한다. 대표 5개 밖에서 유력한 후보가 나오면 해당 후보의 cascade를 추가 측정한다.
3. 실제 측정된 후보에서 SS와 PBR을 선택하고 **그 두 모델을 결합한 실제 cascade**를 평가한다. stage별 이득을 더해서 조합 성능을 추정하지 않는다. 측정하지 않은 새 조합이 필요하면 그 조합의 직접 검증을 먼저 한다.
4. 선택 모델과 강한 기준선을 독립 TRAIN seed202로 확인한다. 모든 12개 arm의 재학습이 목적은 아니다. V1·Top‑1+R의 기존 seed202 작업은 재사용한다. 선택 PBR의 독립 seed 준비는 별도로 필요하다.
5. DEV로 구조·checkpoint·평가를 고정한 뒤 TEST508과 10개 metric을 평가한다. FID/KID는 최종 population에서 측정한다. 원래 multi-view checkpoint의 source4 zero-shot/edited4 zero-shot도 최종 benchmark에 구분해 포함한다.

Source-view 참조는 현재 edited M1 geometry에 original material을 적용한 영상이다. undeformed M0 RGB와 동일한 참조라고 표시하지 않는다. 최종 보고서에서도 참조 정의를 명시한다.

이번에는 독립 seed 확인을 위한 `module_confirmation_recipe.py`를 추가했다. 12개 물리적 모델 family의 proof/capacity/science 설정 변환 36개를 실제 원본 config로 검사했고, numerical/data/optimizer/initial checkpoint의 값과 JSON type을 보존했다. 직접 테스트 49개가 통과했다. 생성한 것은 **미실행 recipe template**이며 새 학습 arm·NN forward·TEST 평가는 없다. V1·Top‑1+R 기존 seed202 설정도 같은 보존 검사를 통과했다. 준비 산출물 39개/463,373 bytes는 file4 실제 SHA256/size 일치로 닫았다.

이 준비는 최종 모델 선택이나 독립 seed의 성능 검증 완료를 뜻하지 않는다. 현재의 학습 attention 추가 이득은 미확인 상태로 유지한다.
