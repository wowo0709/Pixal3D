# E_FIXED_RAW의 PBR 짝비교와 baseline validation 재사용

2026-10-24 15:54 UTC, 추가 vanilla PBR32개, official96개 view와 재사용한 V1/R192개 view, joint 집계 및 같은 SS geometry의 paired 집계가 모두 실제 완료됐다. 모든 case의 canonical closure와 pilot 이후 gate를 확인했다. 신규 SS·Shape 생성과 재학습은0이다.

## 같은 SS에서 PBR 선택의 효과

| 지표 | Vanilla PBR | Top-1 R PBR | R − vanilla와 탐색적95% CI |
|---|---:|---:|---|
| Source LPIPS↓ | 0.053513 | 0.053290 | −0.000223 [−0.000518,+0.000038] |
| Source DreamSim↓ | 0.158346 | 0.157828 | −0.000517 [−0.002224,+0.001118] |
| Source CLIP↑ | 0.942179 | 0.941692 | −0.000487 [−0.002048,+0.000788] |
| Target PSNR↑ | 24.47775 | 24.54433 | +0.06658 [−0.02568,+0.16653] |
| Target SSIM↑ | 0.951534 | 0.951563 | +0.0000295 [−0.000167,+0.000285] |
| Target M-IoU↑ | 0.886571 | 0.886574 | +0.00000261 [0,+0.00000784] |
| CD↓ | 0.0123276 | 0.0123276 | 0 |
| Surface F-Score↑ | 0.856566 | 0.856566 | 0 |

두 그룹의 raw geometry, native guide, 초기 texture noise, 실제 render triangles와 GT reference는 모두 일치한다.

Geometry는 설계상 동일하며 source appearance의 작은 차이는 PBR만 바꿔 비교한 결과다. R의 LPIPS/PSNR 점추정은 좋지만 CI가0을 포함하므로 PBR 단독의 우위는 미확정이다. Vanilla로 바꿔도 E_FIXED_RAW 조합의 source LPIPS/DreamSim은 개선되지 않는다. 따라서 현재의 geometry/appearance tradeoff를 PBR을 vanilla로 변경하는 것만으로 해소할 수 있다는 근거는 없다.

Canonical 결과는 `fixed-evidence-SS-paired-PBR-native32-v209/same-SS-pair-summary-v1/audit.json`이며 실제 W&B는 [same-SS paired PBR32](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/Efixed-SS-same-mesh-PBR-pair32-v209)다. V1/R에 대한 조합 비교는 별도의 `summary-v1/audit.json`과 [joint DEV32](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/Efixed-SS-paired-vanilla-PBR-native32-v209)에 기록했다. 신규 GPU forward96 view와 재사용192 view를 구분한다.

## Baseline VAL36은 재실행 없이 재사용

V1과 C_DECODER의 기존 VAL36 parity audit를 읽어 현재 joint baseline과 model binding 전체가 일치함을 확인했다. 각 모델에서 독립 로드2회,36 asset, 일반 batch8과 tail4, optimizer update0, 저장한 occupancy/target latent array의 일치를 확인한 기존 결과를 재사용한다.

근거는 `baseline-VAL36-reuse-verification-v217/audit.json`이다. 이는 SS baseline2개의 검증이며 새로운 E 후보의 VAL36, PBR의 모든 확인, TEST508 전체의 cache QA를 완료했다는 뜻은 아니다.

E_LEARNED_RAW의 실제 DEV32 cascade를 별도로 시작했다. 현재 R 중심 PBR 후보를 유지하고, stage별 효과를 더해 최종 결과를 추정하지 않고 후보 자체의 mesh/render 출력으로 평가한다.

16:11 UTC 후속 관측에서 E_LEARNED_RAW도 VAL36 fresh-load2회와 실제 저장 배열의 parity 집계를 완료했다. `learned-evidence-candidate-VAL36-v220/parity-v1/audit.json`의 상태는 `PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY`다. 현재480-step seed101 model binding, 일반 batch8/tail4,36개 occupancy 및 prediction/target latent의 정확한 일치를 확인했다. 이는 후보 SS의 재현성 검사이며 최종 모델 채택이나 TEST 품질 평가가 아니다.
