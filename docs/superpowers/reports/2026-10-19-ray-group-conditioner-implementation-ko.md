# 공통 ray-group conditioner 구현

사용자의 최신 지시대로 retargeting adapter는 공통 블록이며 semantic matching/attention/decoder 대조의 query 단위는 source-camera별 native32² ray slot이다. Individual-voxel query와 기존 anchor-patch 공유를 이 모델로 재표시하지 않는다.

`RayGroupConditioner`와 `RayGroupImageConditionedProjFlowMatchingCFGTrainer`를 별도 checkout에 구현했다. 같은 ray의 voxel에서 네 RGB native projected DINO를 mean-pool해 utility query 하나를 만든다. Edited target의 FRAME와 DINO도 동일한 ray membership으로 요약해 각각 semantic 검색 query와 retargeter query로 쓴다. Membership은 실제 camera/native16³ grid와 input edited-RGBA support만 사용한다. GT depth/mesh/correspondence·hidden target RGB·GT latent를 query 입력으로 쓰지 않는다. 기존2D match ID/logit 및 GT spatial-index side channel은 거부한다.

Semantic arm은 ray별 frozen Omega RAW Top64를 새로 찾고 같은 frozen D1/scalar head로 score를 계산한다. Source native DINO가 value다. Retargeter input2048·bounded residual·zero output·별도 semantic fusion gate를 유지한다. 각 ray에서 계산한 update 하나를 source camera별2D map에 쓰고, 원래 bilinear ProjGrid와 MV4의1/4 평균으로 전달한다. 개별 voxel마다 다른 feature를 만드는 직접 voxel splice는 없다.

현재 ray-query feature는 RGB projected DINO이며 과거 target-only predicted SS latent21-channel query 전체를 복제한 것은 아니다. Query 단위·grouping 산술·native carrier를 재사용하는 새 inference-safe 모델이다. Nearest32² query bin과 bilinear feature backprojection은 구분한다. 빈 ray의 map delta는0이며 native feature fallback을 유지한다.

공통 recipe는 원래SS4210 base+LoRA, DeformNeXt world-v4 train9149/val508, seed101, raw480/5 nominal epoch, global96/DDP2/micro48/accumulation1이다. Matching T는0.05로 모든 ray arm에 고정하고 utility가 있는 arm의 scale16·zero-output를 공유한다. T0.005의 prior 집중에서 벗어나기 위한 새 공통 설정이며 ray 결과를 보고 사후 선택한 값이 아니다. Omega/DINO/RGB input cache는 재사용하지만 old anchor-patch match cache는 쓰지 않는다.

| Arm | 실제 대조 |
|---|---|
| ray-top1 | 동일 ray query/selected matcher/R/native projection. Score argmax의 feature 하나,utility 모듈 물리적으로 없음. 사용자 No-score 정의. |
| ray-weighted | 동일 semantic64/score/R/projection,utility 모듈 없이 frozen score만으로 가중합. Top1과의 차이는 multi-candidate score-weighted fusion,Full과의 차이는 learned utility다. |
| ray-full | 동일 query/R/projection에 semantic64+score prior+learned utility attention. |
| ray-native-spatial | 동일 ray query/R/projection/attention이며 source evidence는 각 camera의 original native spatial projection을 ray-group으로 모은3후보다. 사용자 No-corr 정의이며64대3 자연 후보 수 차이를 명시한다. |
| ray-no-decoder | 동일 semantic/full 구성에서 D1 block을 실제 제거한다. 기존 선택된 D1-trained scalar initialization은 유지하므로 optimalD0를 따로 학습한 비교라고 주장하지 않는다. |

CPU 검사32개에서 same-ray voxel 순서 불변성·view/batch 분리·empty-ray mask·Top1 one-feature·physical decoder removal·frozen head·AMP/gradient·native projection 선형성·원래4-view projection/global zero-start를 통과했다. Native projector를 통해 backward가 실제 fusion gate와 retargeter output에 도달하는지 검사했다. Config guard는 budget·R·GT/old-cache·score-prior drift를 거부한다. Static Top1/weighted와 learned attention 모두 value-only intervention에서 실제 evidence가 바뀌고 score/weight/candidate ID는 그대로인지 검사했다.

Actual GPU native/SS-loss proof는 아래와 같이 완료했다. DDP48 memory/time benchmark,본 학습·최종128×3·training-seed replication은 남아 있다. 구현 검증 통과는 새 모델의 SS 성능 개선이나 전체 실험 완료를 뜻하지 않는다. 기존 학습 checkout은 detached/frozen으로 보존하고 새 source는 `wowo0709/feature-retargeting`의 ray-group stage에서만 바꾼다.

## 실제 GPU 첫 검증 및 weighted 기준 추가

첫4개 arm은 실제 SS 모델·원래4RGB/정합GT·LoRA 상태로 GPU native/backward 검증을 통과했다. `global/proj`의 원래 모델 대비 최대 차이는0이고,실제 SS-loss gradient는 LoRA·fusion gate·retargeter output으로 연결됐다. Utility arm은 임시 output probe를 원복한 상태에서 attention gradient와 frozen matcher의0 gradient를 확인했다. Physical No-decoder는 block 수0이었다. GPU0의 할당량20% 제한 안에서 검증했고 single-case peak allocated는4.714~4.744GiB였다. 이는 DDP48 또는5에폭 학습의 메모리·품질 결과가 아니다.

첫 시도는 standalone 환경에서 `ATTN_BACKEND`가 빠져 기본 FlashAttention import에 실패했다. 실행 중인 reference process의 실제 환경을 확인해 SDPA 및 `expandable_segments:True`로 맞췄고,CUDA 할당 전 이를 확인하는 guard를 추가했다. 실패 source/로그는 별도 보존하며 패키지나 기존 학습은 변경하지 않았다.

Score-only weighted arm도 추가했다. Zero-utility Full과 score-only arm의 candidate weight·retargeter·ray delta가 exact한 CPU 대조를 추가해 총25개 검사를 통과했다. Weighted branch의 FP32 softmax는 Full의 divide-first prior 순서를 사용한다. 이 확장 후의 source/config snapshot과 hardware gate는 원래 첫 검증 버전과 구분한다. 아직 새 ray 모델의 실제 본 학습과 개선 판정은 미완료다.

## 최종 실행 경로

Value-only diagnostic을 실제 static ray readout 경계에 연결한 source `5f26e42`로5개 arm 모두 `cohost-final-native-v4` 검증을 다시 완료했다. 각 arm의13개 numerical component SHA256이 실행 시작부터 종료까지 동일하다. Global/proj native 차이는 전부0,LoRA·fusion gate·retargeter output gradient는 양수이며 frozen matcher는 학습하지 않는다. Single-case peak allocated는4.714~4.744GiB다. 이 값으로 micro48 메모리가 보장된다고 해석하지 않는다.

`ray_group_queue.py`는 frozen reference scheduler에서 새로운 checkout의 cwd/PYTHONPATH를 명시하여5개 학습을 실행한다. GPU0~3만2개씩 배정하고 Full/weighted를 먼저,그 다음 Top1/native-spatial,마지막 No-decoder 순으로 시작할 수 있게 구성했다. 각 학습은 원래SS4210에서 새로 시작하며 자체 native gate와2-update 실제 DDP48 benchmark를 먼저 통과해야 한다. 기존 진행 중인 reference 학습/평가는 보존한다.

각 arm에5회 epoch audit·actual W&B scalar/PNG readback·final weight/optimizer audit·trained two-case RGB/BF16/CFG 검증·final128×3 generation·common32 value-address intervention을 연결했다. Ray saved-model 검증은 precomputed match cache가 없는 상태의 fresh ray retrieval 재현성을 확인하고 GT spatial/old2D match 입력은 거부한다. Static Top1/weighted에는 utility module이 없고 No-decoder는 decoder block이 없다. 최종 비교는 Full−Top1,weighted−Top1,Full−weighted,Full−native-spatial,Full−No-decoder 및 V1/두 zero-shot을 포함한다. 기존 query 단위가 다른 Top1-R은 맥락 비교로만 취급한다. 새 ray 구조의 성능 개선은 실제 학습·평가 전까지 미확인이다.
