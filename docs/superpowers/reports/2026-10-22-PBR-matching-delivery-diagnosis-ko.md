# PBR matching posterior와 feature 전달 실패 진단

2026-10-22. 기존 Direct1905는 고정 GT Shape·target08 개발 자산32개에서 V1보다
foreground PSNR가0.9813dB 낮았다. 상세 비교와 CI는
[stage32 분석](2026-10-22-PBR-Direct1905-stage32-analysis-ko.md)을 따른다.
이번 작업은 동일 weight에서 원인을 분리한다. 추가 학습 또는 새 full 모델 선정이 아니다.

## 실제 normal forward에서 확인한 두 경로

동일32개 RGB와 일반 dataset의 transport tensor를 정확히 대조한 CPU FP32
normal source forward를 수행했다. Pooled branch 진단은 R-only/Direct64회,
GT matching 진단은 Direct32회다. Source weight와 frozen matcher는 바뀌지 않았다.
Encoder·diffusion denoiser·VAE를 실행하거나 optimizer를 업데이트하지 않았다.
이 수치는 GPU BF16 generation 정합 검사를 대신하지 않는다.

| 관측 | Matching prior | Learned mean-head |
|---|---:|---:|
| Exact Top-1 hit | 21.8853% | 18.0620% |
| Exact GT probability mass | 18.0456% | 14.7705% |
| Near-1 GT probability mass | 45.2482% | 41.0177% |

분모는 GT-observable target patch1517개다. Source cluster32개를 재추출한
bootstrap10000회(seed20261022)의 head−prior CI95는 exact Top-1이
[−5.7242,−1.7988]pp, exact mass가[−4.2409,−2.2931]pp다. 하나의
학습 seed/개발 panel에서 얻은 사후 진단이며 독립 확인 결과가 아니다.
여기 query는 기존 **target-image patch**다. 새 source-camera ray member query와
통계 단위가 달라 SS의 matching 수치와 직접 비교하면 안 된다.

| Pooled branch RMS | R-only | Direct |
|---|---:|---:|
| DINO combined correction / Top-1 reference | 0.00782 | 0.06546 |
| NAF combined correction / Top-1 reference | 0.00769 | 0.06174 |
| NAF sparse 전달 correction | 0.00560 | 0.04512 |

NAF sparse 전달은 실제 consumer의0.75배 R 경로를 사용한다. Semantic NAF delta는
consumer에서0이므로 해당 값은 전체 NAF conditioning correction이다. DINO에는
별도 semantic fusion 항이 있어 표의 sparse R 항만으로 전체 DINO 변경을 표현하지 않는다.
Reference는 **pooled Top-1 feature**이며 native NAF128 고해상도 carrier norm이 아니다.
Pooled DINO/NAF reference RMS는 둘 다 약1이다. NAF reference만 작다는 설명은
이 관측으로 지지되지 않는다. Direct의 추가 head 경로가 큰 correction을 전달하지만,
크기만으로 잘못된 feature 방향이나 최종 품질의 인과를 확정하지 않는다.

GT source-visible1517개와 surface가 있으나 source-invisible993개의 correction
크기는 비슷했다. GT visibility는 forward 후 분석에만 사용했다. 모델 입력이나
전달 mask에 GT 가시성을 추가하지 않는다. DQ projection도 실제로 작동하므로
그 영향이 없다고 결론 내리지 않는다.

## 같은 weight에서 수행할 원인 분리

새 evaluator는 기존 native PBR evaluator와 frozen projection streaming을 그대로
호출하고, eval/no_grad 안에서 다음 process-local hook만 적용한다.

1. `normal`: 전달을 바꾸지 않는다. 기존 Direct의 처음4개에서 모든 입력 identity,
   자산별 metric와 실제 생성 panel이 정확히 같은지 먼저 검사한다.
2. `matching-prior`: candidate64·key·score·Top-1 carrier를 유지하고 attention
   probability를 원래 matching prior로 복원한다. 같은 projected V로 appearance를
   다시 계산하며 RAW2048 aggregation도 같은 probability를 사용한다.
3. `no-direct-delivery`: 정상 head/DQ를 포함한 R 입력과 실제 R 출력을 유지하고,
   consumer에 전달되는 추가 bounded head correction만 제거한다.
4. `matching-prior-no-direct-delivery`: 두 진단을 함께 적용한다.

세 변경 조건은 같은 NN/source1905, RGB·noise·camera·GT Shape의32개 자산에서
비교한다. GPU no-op 정합 검사를 통과하기 전 후속 비교를 해석하지 않는다.
CPU 검사8개는 feature/probability 재계산, 실제 R 전달, weight 불변,
hook 제거 후 normal 출력 복원, 예외 시 hook 정리를 확인했다.

이것은 **저장 weight 개입 진단**이다. NoScore의 Top-1 feature arm, NoCorr의
native spatial correspondence arm, 물리적으로 channel을 제거한 no-DQ arm,
새로 학습한 R-only 또는 R×DQ ablation으로 이름 붙이지 않는다. DQ2048과
R5120 channel 및 원래 학습된 NN은 유지된다. 개입으로 좋아져도 학습 때의
모듈 효과나 독립 test 성능이 입증되는 것은 아니다.

## 설계 판단에 연결하는 기준

Prior 복원만으로 회복하면 learned reweighting의 drift를 우선 개선한다.
Direct 제거로 회복하면 R을 우회하는 전달 경로를 우선 개선한다. 둘 다 필요하면
후보 선택과 전달을 함께 제한해야 한다. 회복하지 않으면 head/R/NN 공동 적응과
target/source feature 방향을 더 살펴야 하며 추가 epoch만 늘리지 않는다.

새 SS bounded-source-contrast 후보는 source-camera ray member query와
source-relative R를 사용한다. GPU micro96 backward와 native 초기 출력 보존
검사를 통과한 뒤 학습 중이다. 본 PBR 진단과 별도로 진행하며 native32 품질,
source-disjoint 확인, joint 학습과 전체 cascade를 통과하기 전 full로 채택하지 않는다.

원본 증거는 file4의 `rgb_matching_v2/semantic-topk-search-v1/` 아래
`ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/`
`direct-retargeting-v1/PBR-joint-continuation-v1/`의
`pooled-feature-branches-CPU-v1/` 및 `point-matching-GT-CPU-v1/`이다.
각 폴더는 실제 file4 SHA256/size 대조 receipt를 포함한다. Source와 실행 로그를
보존하며 원본 실패 결과를 덮어쓰지 않는다.
