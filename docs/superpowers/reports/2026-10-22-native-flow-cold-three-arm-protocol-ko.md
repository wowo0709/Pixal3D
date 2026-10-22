# 실제 SS flow objective의 cold 3-arm 비교

## 목적과 기존 결과의 한계

Source-feature teacher112에서 R 경유 전달과 raw Top-K를 추가한 이득은 source72에서 재현되지 않았다.
같은 proxy loss의 epoch·후보를 더 탐색하지 않고, 실제 native SS denoiser와 소비 경로까지 포함해
학습하면 source 정보가 유용해지는지 검사한다. 새로운 task-objective 실험이며 이전 proxy admission의
HOLD를 성공으로 바꾸거나 실패한 teacher 가중치를 채택하는 실험이 아니다.

원래 V1 결과를 확인했다. `rgb_v1/runs_world_v4/ss64/v1/config.json`은 training_seed 항목이 없고,
native 기본 seed0을 사용한다. 별도 `rgb_matching_v2/ss-seed-confirmation-v1/runs_ss64/v1/`은
seed101이지만 world2×micro48, global96으로 480 updates를 완료했다.
새 source 모델은 world1×micro96이므로 이 기존 결과를 같은 rank별 noise의 paired 학습이라고
주장할 수 없다. 기존 V1 결과는 그대로 보존하고 **새 world1 control**을 함께 학습한다.

## 고정 세 조건

| 조건 | 실제 source 파라미터 | R appearance 입력 | 추가 teacher 초기화 |
|---|---:|---|---|
| Vanilla | 0 | source conditioner 자체 없음 | 없음 |
| R_feature | 19 | semantic Top1 + learned four-head feature | 없음 |
| R_raw_topk | 19 | probability-weighted raw Top-K + learned four-head feature | 없음 |

두 source 조건은 retargeter가 own native source를 기준으로 contrast 입력을 만들고,
모든 새 feature를 R을 거쳐 전달한다. Head-to-consumer 직접 bypass는 없다.
Raw와 transformed value는 같은 네 head 확률을 사용한다. Matching prior는 bound1로 보정하며,
Omega/D1 matcher는 동결한다. Source-camera ray-wise query와 native MV4 global/negative 경로를 유지한다.
NoDQ는 물리적으로 deformation 입력/투영이 없다. Target-only나 GT visibility gate를 사용하지 않는다.

모두 원본 multi-view SS4210의 frozen base와 LoRA를 완전히 이어받고, fresh optimizer로 시작한다.
두 source 모듈은 같은 19 tensor 초기값을 사용한다. Cold R과 fusion 출력이 0이므로 기존 native
MV4 함수와 같아야 한다. 실패한 teacher112 source state나 그 optimizer는 읽지 않는다.

## 비교 조건과 난수 보존

TRAIN9149/validation의 기존 aligned DeformNeXt split을 재사용한다. Clean80k는 학습 데이터가 아니다.
Image512, original SS64 latent schema, original camera/anchor 계약을 유지한다.
각 조건은 world1, global/micro96, accumulation1, 5에폭/480 updates, seed101이다.
원래 360 LoRA tensor, AdamW lr1e-5, source lr1e-4, 25-step warmup, BF16 AMP,
CFG dropout .1, 기존 logitNormal timestep과 native clipping을 유지한다.

추가 모듈의 생성이 training RNG를 바꾸지 않도록 source constructor를 RNG fork 안에서 실행한다.
Proof에서는 초기 LoRA와 source19 tensor hash, post-constructor Torch CPU/CUDA·NumPy·Python RNG를
서로 대조한다. Capacity에서는 실제 첫 세 update의 asset ID, noisy latent, timestep, global condition
hash까지 같아야 한다. 실제 학습에서도 같은 trace를 저장해 대응을 검사할 수 있다.

FFN 내부 checkpoint chunk는 24로 시작한다. 이는 batch96을 gradient accumulation으로 나누는 것이 아니다.
자신의 실제 capacity가 실패한다면 별도 namespace에서 chunk12를 검증할 수 있지만,
세 조건의 공통 수치 계약과 micro96/accum1을 유지해야 한다. 자동 재시도나 숨은 batch 축소는 없다.

## 실제 증명 순서

1. **세 조건 각각 actual native proof:** 기존에 입력 support만으로 선택한 동일 TRAIN sample을 사용한다.
   Native global/projection/negative 및 RNG, 같은-noise velocity/flow loss가 cold reference와 정확히 같아야 한다.
   원래 LoRA와 R/fusion에 실제 SS loss gradient가 도달해야 한다. R 출력이 0이므로 첫 head gradient가
   0인 것은 올바른 결과다. 임의의 weight를 바꿔 gradient가 생긴 것처럼 만들지 않는다. Optimizer update는 0이다.
2. **별도 disposable native capacity:** fresh 모델/optimizer에서 global/micro96으로 실제 native update 세 번을 실행한다.
   세 번째 backward까지 source19 gradient가 모두 finite/nonzero인지, 원래 360 LoRA와 source optimizer가
   실제로 업데이트되는지, frozen parameter/buffer가 같은지, peak/속도가 어떤지 검사한다.
   이 모델은 버리며 scientific checkpoint를 내보내지 않는다.
3. **독립 CPU admission:** 세 proof와 세 capacity가 실제 완료되고 공통 계약·초기값·난수·세 update trace가
   대응되는 경우에만 새 task-objective 학습을 허용한다. 이 PASS는 모델 품질을 입증하지 않는다.
4. **각 조건 fresh 480-step 학습:** proof/capacity 상태를 이어받지 않고 처음부터 시작한다.
   첫 optimizer update **전에** initial native32의 기록된 GT/noise/count/metric이 원본 MV4 reference와
   같은지 확인한다. 다르면 학습을 진행하지 않는다. Raw voxel mask는 이 경로에서 보관하지 않아
   mask byte identity라고 표현하지 않는다.

모든 epoch(96 updates)마다 같은 32개·12-step sampler·동일 seed map으로 voxel IoU/precision/recall/F1과
GT/Pred/Diff 이미지를 W&B `youngwoojeon-unist/pixal3d-deformnext-retargeting`에 기록한다.
SS에는 느린 PBR rendering validation을 붙이지 않는다. Checkpoint는 epoch마다 저장한다.

## 결과 해석과 남은 범위

Vanilla와 두 source 모델의 실제 native task 최적화·일반화 경향을 우선 비교한다.
R_raw_topk−R_feature는 R appearance에 raw mixture를 추가하는 효과이며,
semantic matching/decoder/score 각각의 독립 효과를 자동으로 입증하지 않는다.
NoScore는 Top1 feature, NoCorr는 native spatial correspondence라는 사용자 정의를 유지한다.
유효한 task 모델이 확인되면 그 구조에서 필요한 retrained ablation과 독립 seed/128×3/cascade/PBR를 검증한다.
Final508/10-metric 범위는 유지하며, proxy fit이나 engineering PASS를 최종 성공으로 쓰지 않는다.

GPU는 지정된 0~3만 사용한다. 각 작업은 한 GPU를 독점하고 정상 진행 중인 기존 cascade 작업은
종료될 때까지 유지한다. Scientific 3개가 실행되면 남는 GPU에서는 기존 평가를 계속한다.
모든 data/checkpoint/result의 canonical 위치는 file4이며 로컬은 cache다.
