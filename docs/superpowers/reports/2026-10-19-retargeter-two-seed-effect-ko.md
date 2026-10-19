# Retargeting adapter의 두 학습 seed 효과

2026-10-19 06:33 UTC. 현재 Top-K attention 후보의 최종 결과와 별개로, 완료된 네 모델에서 retargeting adapter의 추가 효과를 확인했다. 새로운 학습이나 생성은 하지 않았다.

두 모델 모두 frozen R9 D1/scalar head의 raw matching logit으로 Top-1 correspondence feature를 선택한다. `top1-reference`는 retargeter가 없고, `static-top1-c1`은 같은 feature에 C1 retargeter를 추가한다. C1 입력은 evidence1024 + target DINO1024인2048 channel이며 deformation query channel은 물리적으로 없다. Task attention도 두 모델 모두 없다.

| 학습 seed | Top-1, retargeter 없음 | Top-1 + retargeter | IoU 차이 (pp) | source-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|
| 101 | 0.434529 | 0.442757 | +0.822795 | [+0.427361, +1.222321] |
| 202 | 0.434534 | 0.443340 | +0.880580 | [+0.490929, +1.314417] |

각 행은 고정 raw480, 같은128개·93 source·세 생성 noise의 voxel TP/FP/FN 집계다. 세 noise의 차이도 각각 모두 양수였다. seed101은 +0.924864/+0.807963/+0.732365 pp, seed202는 +0.957850/+0.904996/+0.777663 pp였다. 모델 학습 seed는 두 개이며 생성 noise를 추가 학습 seed로 세지 않는다.

두 모델의 model/data/optimizer/validation config를 대조하고, retargeter 추가에 필요한 conditioner arm 및 dataset transport arm 외 설정을 동일하게 유지했는지 확인했다. Config에 남아 있는 score temperature0.005/0.05는 이 대조에서 비활성이다. 두 Top-1 경로가 모두 temperature 적용 전에 raw logit argmax를 선택하므로, 이 차이가 가중치 분포를 바꾸는 weighted attention 비교가 아니다. Positive temperature와 실제 Top-1 소비 경로를 요구하며 weighted arm을 이 대조로 허용하지 않는다.

네 모델 모두 원래 SS4210에서 fresh LoRA continuation으로 시작해 global96/DDP2/micro48/accumulation1, raw480을 완료했다. 저장된 frozen base701개와 원래 matcher24개를 유지하고 LoRA360개를 갱신했다. Retargeter 없는 모델의 optimizer 상태는361개, 있는 모델은366개다. 실제 최종 state/optimizer, 다섯 epoch의 W&B 네 metric 및 PNG, file4 대조와 saved GPU conditioning QA를 확인했다.

양 seed와 양 arm, 세 생성 noise의 총1,536개 case 기록에서 원래 case 순서·source cluster·GT hash/shape/count·생성 seed가 일치했다. Train9149/validation508의 실제 manifest hash도 별도로 보존했다. 원본 dataset과 checkpoint를 수정하지 않았다.

이 결과는 retargeting adapter가 기존 개발 panel에서 두 학습 seed 모두 향상됐다는 근거다. 현재 utility scale16 모델이 vanilla V1보다 좋아 보인다는 관측을 Top-K/attention의 효과로 전부 돌릴 수 없는 이유이기도 하다. 해당 후보는 이 강한 Top-1 + retargeter 기준을 넘어야 추가 이득을 주장할 수 있다. Decoder·score·semantic correspondence의 효과와 새 full 모델의 채택을 이 결과로 대신하지 않는다.

각 CI는 해당 checkpoint의 source cluster 재표집에 대한 조건부 CI다. 두 seed의 학습 모집단 CI 또는 독립 test 성능이라고 주장하지 않는다. 추가121개는 아직 SS 생성 평가를 열지 않았으며, 이전 auxiliary-head validation에 포함됐다는 한계도 유지한다.

실제 결과는 `rgb_matching_v2/semantic-topk-search-v1/orthogonal-utility-followups-v1/retargeter-effect-v1/audit.json`, cross-seed identity 확인은 같은 폴더의 `identity-and-manifests-v1.json`이다. `summarize_retargeter_seed_effect.py`는 이 대조의 원래 config/기록/QA/epoch·final 감사 입력 hash를 함께 저장한다. 관련 CPU 검사는7개가 통과했다. 전체 목표는 여전히 진행 중이다.
