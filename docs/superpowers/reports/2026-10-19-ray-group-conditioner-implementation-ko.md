# 공통 ray-group conditioner 구현

사용자의 최신 지시대로 retargeting adapter는 공통 블록이며 semantic matching/attention/decoder 대조의 query 단위는 source-camera별 native32² ray slot이다. Individual-voxel query와 기존 anchor-patch 공유를 이 모델로 재표시하지 않는다.

`RayGroupConditioner`와 `RayGroupImageConditionedProjFlowMatchingCFGTrainer`를 별도 checkout에 구현했다. 같은 ray의 voxel에서 네 RGB native projected DINO를 mean-pool해 utility query 하나를 만든다. Edited target의 FRAME와 DINO도 동일한 ray membership으로 요약해 각각 semantic 검색 query와 retargeter query로 쓴다. Membership은 실제 camera/native16³ grid와 input edited-RGBA support만 사용한다. GT depth/mesh/correspondence·hidden target RGB·GT latent를 query 입력으로 쓰지 않는다. 기존2D match ID/logit 및 GT spatial-index side channel은 거부한다.

Semantic arm은 ray별 frozen Omega RAW Top64를 새로 찾고 같은 frozen D1/scalar head로 score를 계산한다. Source native DINO가 value다. Retargeter input2048·bounded residual·zero output·별도 semantic fusion gate를 유지한다. 각 ray에서 계산한 update 하나를 source camera별2D map에 쓰고, 원래 bilinear ProjGrid와 MV4의1/4 평균으로 전달한다. 개별 voxel마다 다른 feature를 만드는 직접 voxel splice는 없다.

현재 ray-query feature는 RGB projected DINO이며 과거 target-only predicted SS latent21-channel query 전체를 복제한 것은 아니다. Query 단위·grouping 산술·native carrier를 재사용하는 새 inference-safe 모델이다. Nearest32² query bin과 bilinear feature backprojection은 구분한다. 빈 ray의 map delta는0이며 native feature fallback을 유지한다.

공통 recipe는 원래SS4210 base+LoRA, DeformNeXt world-v4 train9149/val508, seed101, raw480/5 nominal epoch, global96/DDP2/micro48/accumulation1이다. Matching T는0.05로 모든 ray arm에 고정하고 utility가 있는 arm의 scale16·zero-output를 공유한다. T0.005의 prior 집중에서 벗어나기 위한 새 공통 설정이며 ray 결과를 보고 사후 선택한 값이 아니다. Omega/DINO/RGB input cache는 재사용하지만 old anchor-patch match cache는 쓰지 않는다.

| Arm | 실제 대조 |
|---|---|
| ray-top1 | 동일 ray query/selected matcher/R/native projection. Score argmax의 feature 하나,utility 모듈 물리적으로 없음. 사용자 No-score 정의. |
| ray-full | 동일 query/R/projection에 semantic64+score prior+learned utility attention. |
| ray-native-spatial | 동일 ray query/R/projection/attention이며 source evidence는 각 camera의 original native spatial projection을 ray-group으로 모은3후보다. 사용자 No-corr 정의이며64대3 자연 후보 수 차이를 명시한다. |
| ray-no-decoder | 동일 semantic/full 구성에서 D1 block을 실제 제거한다. 기존 선택된 D1-trained scalar initialization은 유지하므로 optimalD0를 따로 학습한 비교라고 주장하지 않는다. |

CPU 검사23개에서 same-ray voxel 순서 불변성·view/batch 분리·empty-ray mask·Top1 one-feature·physical decoder removal·frozen head·AMP/gradient·native projection 선형성·원래4-view projection/global zero-start를 통과했다. Native projector를 통해 backward가 실제 fusion gate와 retargeter output에 도달하는지 검사했다. Config guard는 budget·R·GT/old-cache·score-prior drift를 거부한다.

아직 actual GPU native/SS-loss proof,DDP48 memory/time benchmark,본 학습·최종128×3·training-seed replication은 남아 있다. CPU 통과는 새 모델의 SS 성능 개선이나 전체 실험 완료를 뜻하지 않는다. 기존 학습 checkout은 detached/frozen으로 보존하고 새 source는 `wowo0709/feature-retargeting`의 ray-group stage에서만 바꾼다.
