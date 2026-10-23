# PBR 4-epoch 중간 결과와 독립 seed 확인 준비

UTC 2026-10-23. `wowo0709/feature-retargeting`.

## 새로 확인한 결과

현재 PBR feature 학습이 1,520 step/4 epoch validation을 마쳤다. 아래는 같은 fixed4 asset, GT target Shape, target view에서 측정한 값이다. Crop LPIPS는 논문의 source-preservation LPIPS와 구분한다.

| PBR 모델 | 1,140-step PSNR ↑ | 1,520-step PSNR ↑ | 1,140-step Crop LPIPS ↓ | 1,520-step Crop LPIPS ↓ |
|---|---:|---:|---:|---:|
| V1 | 26.72247 | **26.74437** | 0.088230 | **0.087684** |
| Top‑1+R | **27.15627** | 26.46423 | **0.078749** | 0.091200 |
| Top‑1+feature+R | 27.14318 | 26.54535 | 0.078876 | 0.090897 |

3 epoch의 두 source 모델 개선이 4 epoch까지 유지되지는 않았다. Feature 모델이 4 epoch에서 Top‑1+R보다 소폭 좋지만 두 모델 모두 V1보다 낮다. 학습을 더 하면 단조롭게 개선된다는 주장을 할 수 없다.

V1과 Top‑1+R은 이미 1,900-step 학습이 끝났다. 같은 fixed4의 최종 값은 V1 PSNR 26.87652/Crop LPIPS 0.086454, Top‑1+R PSNR 26.86011/Crop LPIPS 0.085186이다. Top‑1+R의 최종 LPIPS는 소폭 좋고 PSNR은 소폭 낮다. 이 두 모델의 최종 결과만으로 PBR의 일관된 우위를 주장하지 않는다. Feature 모델의 최종 1,900-step 결과는 아직 없다.

실제 validation 기록을 검사해 모든 모델·모든 비교 epoch의 asset, source cluster, conditioning fingerprint, seed, **실제 noise**, RGB, GT PBR latent, GT Shape condition, camera SHA가 같음을 확인했다. 따라서 이번 곡선 변화는 입력이나 generation noise가 바뀌어서 생긴 차이로 설명할 수 없다.

Feature의 3→4 epoch Crop LPIPS 변화는 Axe +0.031392, Bed +0.016950가 주도한다. Bookcase·Bottle 변화는 매우 작다. Top‑1+R도 같은 두 asset에서 주로 악화됐다. 이는 두 모델이 공유하는 적응 경로와 작은 패널의 asset별 변동을 확인해야 한다는 신호다. 이번 기록만으로 전체 데이터의 overfitting이나 attention 구현 오류를 확정할 수는 없다.

진행 중인 모델의 공통 5 epoch 예산을 유지한다. 모듈 비교 중 특정 모델만 조기에 멈추거나 유리한 epoch로 바꾸지 않았다. 채택 여부는 예정된 최종 공통 **생성 geometry/native guides 32개**의 PBR 비교와 선택 SS/PBR 조합의 직접 cascade 평가로 결정한다. GT shape에서 alpha IoU가 1인 것을 생성 geometry의 개선으로 해석하지 않는다.

## 독립 seed 확인 준비

`pbr_confirmation_recipe.py`를 추가했다. 실제 PBR science 설정에서 training seed를 101→202로 바꾸면서 모델·full base+LoRA PBR16850 초기화·TRAIN9139·global24/DDP4/micro6/accum1·LR·optimizer·1,900 update·generation noise·validation을 보존한다. 달라지는 나머지 항목은 새 output/logging/config 경로와 DDP port다.

직접 검사 31개가 통과했다. 실제 V1·Top‑1+R의 완료된 native config와 feature/raw-Top-K의 현재 science config로 네 개의 미실행 template을 검사했다. 완료된 두 모델에 대해서는 예전 DDP2 준비본을 쓰지 않고 실제 DDP4/micro6 설정을 사용했다.

기존 engineering proof와 capacity 기록은 seed101의 같은 architecture에 대한 근거로 재사용한다. 이를 실제 seed202 proof가 끝났다고 표시하지 않는다. Seed202 본 학습은 기존 runner에서 지원하며, 자신의 step0 fixed4 입력·noise·PBR latent가 원래 native 모델과 정확히 같은지 검사한 뒤에만 첫 optimizer update를 허용한다. 기존의 명시적 renderer tolerance도 유지한다.

네 template은 후보 선택에 대비한 설정이며 **네 모델의 추가 학습을 예약한 것이 아니다**. DEV에서 선택한 모델과 필요한 강한 기준선만 활성화한다. 실제 활성화 시 자신의 archiver·output·disk reservation을 갖는 닫힌 run plan이 추가로 필요하다. 이번 작업의 CPU/GPU neural forward, 새 학습 arm, 새 GPU job은 모두 0회다.

준비 산출물은 `unified-module-pipeline-v1/selected-PBR-seed202-v171/prelaunch-NOT-QUEUED`, 실제 epoch 곡선과 sample별 비교는 `unified-module-pipeline-v1/focused-status-v171`에 보존한다. 기존 학습은 중단·재시작하지 않았다. 최종 모델 선택·독립 seed 학습·DEV freeze·TEST508과 10개 metric 평가는 아직 완료되지 않았다.
