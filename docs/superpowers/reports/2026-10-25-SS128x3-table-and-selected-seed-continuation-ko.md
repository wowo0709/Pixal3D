# 최종 SS128×3 집계·cascade 표 통합·선택 후보의 독립 확인 연결

2026-10-25 UTC. 이전 턴은 다섯 control의 own SS 이후640개 cascade/8metric과 논문 집계를 연결했다. 이번에는 빠져 있던 SS128×3 집계와 정확한 checkpoint 기반 표 통합, 선택 Fixed의 두 training seed collector를 연결했다. 기존 NN 학습은 유지하며 신규 GPU arm은 추가하지 않았다. 실제 신규 control의 최종 품질과 TEST는 미완료다.

## 1. SS와 cascade를 같은 실험으로 묶는 코드

새 [SS summary source](../../../experiments/deformnext_retargeting/final_fixed_ablation_ss_summary.py)의 계약:

- 다섯 control과 Top-1 R/Fixed baseline의 실제7모델, 모델당3개 generation noise, 모델당 실제480 checkpoint를 구분한다.
- Own matcher/physical-R schema, raw checkpoint/config/source hash, native RGB camera8/9/10/11·sampler,128개/93 source cluster, GT hash·count·generation seed를 모두 확인한다.
- 저장된 voxel IoU/precision/recall/F1이 per-asset tp/fp/fn과 일치해야 한다. Global occupancy IoU를 macro IoU로 바꿔 넣지 못한다.
- 각 asset 안에서3 noise를 평균낸 뒤 source cluster를 resample한다. Generation noise3회를 training seed3개로 세지 않는다.
- 기존 cascade와 동일한6개 contrast의 **primary IoU family6**를 사전 지정했다.12,000 cluster-bootstrap draws로 보정 구간을 계산하고 다른 voxel metric은 보조 진단으로 보존한다. Cascade8metric의48-comparison family와 별개다.
- Cascade 통합에서는 실제 SS checkpoint/config/source bytes,128개 source pairing, 공통 PBR, cascade의 실제 noise0 SS 기록이 일치해야 한다. SS128×3와 cascade128×1의 범위는 caption에 명시한다.

새 검사21개, 관련 기존 cascade/contract 검사를 포함해 **115 passed in5.71s**다. 실제 GPU NN proof나 신규 품질 측정의 대체가 아니다. Source commit은 `cb463e0`다. 앞서 고정한 NN/worker/cascade source는 수정하지 않았다.

## 2. 실제 완료된 두 SS baseline을 새 검사기로 검증

R과 Fixed의 실제6개 SS 평가를 읽어 per-asset metric/count, 원래128개 source/GT/noise와 실제 saved config seed101, checkpoint bytes를 검증했다. 기존4모델 follow-up 통계와 비교한8개 평균값은 **최대 차이0**이었다.

| 모델 | Macro IoU | Precision | Recall | Voxel F1 |
|---|---:|---:|---:|---:|
| 실제 Top-1 R | 0.4726002471 | 0.5971708594 | 0.6819515498 | 0.6337454496 |
| Fixed | 0.4734709997 | 0.5999328349 | 0.6798032995 | 0.6343779487 |

이는 기존 실제 측정의 재사용이며 신규 control의 결과가 아니다. Canonical `SS-baseline-series-v311`의3개 artifact/1,084,567bytes가 실제 file4 SHA256/size와 일치한다.

## 3. 실제 추가한 CPU 후속 작업

| 새 작업 | 실제 시작 조건 | 결과물 |
|---|---|---|
| 7모델 SS128×3 집계 | 새5개 arm의15개 평가 및 producer 종료·file4 보존, own480 bundle | Macro/global 구분,6개 IoU family 비교, W&B 표 |
| SS+cascade ablation 통합 | 위 SS 집계와 기존에 등록한 실제7모델 cascade 집계 | SS IoU/CD/F-Score/LPIPS/DreamSim의 실제7행 TeX와 bounded contrast evidence |
| 선택 Fixed의2 training seed 확인 | 실제 Fixed202480 및3개 SS 평가 완료·원본 보존 | V1/R/Fixed의 seed101/202 strict collector, paired 통계와 W&B |

총 **3개 CPU 작업**을 실제 main queue에 append-only로 등록했다. 모두 own 실제 결과에 의존해 pending이다. Proof/capacity/science/evaluation를 서로 다른 모델로 세지 않으며, 실제 결과가 없는 weight hash를 미리 꾸며 넣지 않는다. 새 control의 전체 학습·품질이 완료됐다는 뜻은 아니다.

통합 결과의 `contrast-evidence.json`은 보정 구간이 지지하는 개선/악화/불확실성을 metric별로 구분한다. 구간이0을 포함하면 동등성/비열등성으로 해석하지 않는다. 이 파일이나 표 생성이 architecture 선택·manuscript 반영·TEST를 자동으로 완료하지 않는다.

## 4. 선택 후보의 독립 seed collector를 실제 모델로 사전 실행

이미 완료된 seed101의 V1/R/Fixed와 seed202의 V1/R에서 public collector를 실행했다. 실제5개 최종 checkpoint 각각의 **frozen NN tensors701개가 original SS4210과 정확히 같고 LoRA tensors360개만 학습 대상으로 분리됨**을 확인했다. 같은 seed 안의 실제480 input/noise/timestep/global trace와 constructor RNG pairing도 확인했고, V1/R의 두 seed reducer까지 실제로 통과했다.

Global NN forward hook으로 CPU forward를 금지한 상태에서 실행했으며 호출0회, CUDA 초기화False였다. CPU는 저장 tensor 비교와 metadata/통계만 수행했다. Fixed202는 아직 완료되지 않았으며 해당 checkpoint를 읽거나 두 seed 재현이 끝났다고 보고하지 않았다.

Own Fixed202 결과가 나오면 기존 V1/R 네 실행과 Fixed101을 재사용해 **정확히 같은 public collector**로 두 seed 세 모델을 집계하도록 연결했다. Fixed matcher/anchors의 final frozen report도 확인한다. 모든 ablation의 seed를 늘리거나 V1/R을 다시 학습하지 않는다.

Actual preflight의5개 artifact/126,720bytes와 후속 prelaunch의5개 artifact/34,089bytes는 실제 file4와 일치한다.

## 5. 새로 완료된 두 번째 epoch의 범위

GPU7 Uniform-K가 실제192 updates의 quick32 검증을 완료했다. 학습 중인 wrapper/child의 PID/birth를 실제 확인하면서0/96/192의 완료 entry만 snapshot으로 보존했다. 닫힌 V1/Fixed와 같은 sample/source/noise/GT이며 초기0의 실제 기록은 일치한다. **첫192개 실제 training input/noise/timestep/global trace도 세 모델에서 정확히 같다.**

| 모델 | Step192 macro mean-asset IoU | Step192 global occupancy IoU |
|---|---:|---:|
| V1 | 0.4674473490 | 0.4357467600 |
| Fixed | 0.4636343510 | 0.4302203513 |
| Uniform-K | 0.4658034115 | 0.4334766728 |

Uniform−Fixed의 macro 차이는 step96의−0.000653에서 step192의+0.002169로 바뀌었다. 현재 중간 panel은 score weighting의 우위를 지지하지 않는다. 이 순위 변화를 최종 모듈 효과나 코드 실패로 단정하지 않는다.5epochs·DEV128×3와 같은 geometry/PBR의 실제8metric 평가를 유지한다. Quick32·단일 noise와 최종128×3를 혼합하지 않으며, 아직 같은 step192가 완료되지 않은 No-R와 이 표에서 비교하지 않는다.

`shared-Fixed-second-epoch-readout-v316`의8개 artifact/5,570,626bytes는 실제 file4에 보존했다. Intermediate entry snapshot을 학습 중 전체 log의 closure로 표현하지 않는다.

## 6. 기존 PBR 독립 확인과 논문

기존 `required-PBR-controls-two-seed-closed-collection-CPU-v198`와 seed202 vanilla/R의1900 완료 → own snapshot → collector 의존 관계가 정상임을 확인했다. 중복 작업은 추가하지 않았다. 이 collector는 **fixed4 GT-shape, target-camera stage trend**를 검증하며 generated-geometry/source-preservation이나 full-pipeline 두 seed를 입증하지 않는다. 해당 범위를 final freeze와 논문에 유지한다.

논문 protocol에는 별도 SS family6, three-noise macro와 global ratio의 구분, 실제 noise0/cascade checkpoint 정합 조건을 추가했다. 새 compiled PDF는12pages/24references, compile warning0이며5·9·10페이지를 직접 검수했다.28개 artifact/732,710bytes가 실제 file4와 일치한다. 신규 control과 TEST의 cell은 여전히 pending이다.

[현재 개발 PDF](/root/dev/.cache/deformnext-retargeting/paper-preview-20261024-v315.pdf).

## 7. 남은 전체 작업

1. 진행 중 두 control와 등록된 matching3개의 own proof/capacity/480/SS128×3 및 actual cascade8metric 완료.
2. 연결한 실제7모델 표와 통계를 판독해 모델과 모듈 claim을 선택. 다른 control이 유리하면 Fixed를 고집하거나 해당 arm을 E로 재명명하지 않는다. 선택된 실제 family에 맞는 독립 확인·fresh load·최종 실행 계약이 필요하다.
3. 선택 Fixed202/PBR vanilla·R202의 실제 완료와 등록한 collector, 선택 모델의 VAL36 정합 확인.
4. 저장된 native32 ray/candidate/weight 관측을 이용한 실제 기전 그림과 같은 모델의 성공·실패 사례. 추가 head/grid나 새로운 NN 관측부터 만들지 않는다.
5. 최종 선택/protocol/checkpoint를 동결한 뒤 TEST508/134 source cluster의10metric·paired 통계. TEST 전 local 중간물의 실제 file4 보존/rolling 관리를 마련한다.
6. 실제 Main/Ablation/Mechanism/Qualitative와 Results/Discussion/Abstract/Conclusion, 최종 PDF.

전체 목표는 계속 active다. Phase 등록, compatibility 검사, 중간 quick32 값으로 목표 완료나 모든 모듈의 필수성을 선언하지 않는다.

## Canonical 위치

`PBR_NATIVE_SCIENCE_RUNTIME/unified-module-pipeline-v1` 아래:

- `final-Fixed-controls-DEV128-cascade-v306/SS-baseline-series-v311`.
- `final-Fixed-controls-DEV128-cascade-v306/SS-and-table-prelaunch-v312`.
- `selected-Fixed-two-seed-collection-v313/{actual-preflight-v1,prelaunch-v314}`.
- `shared-Fixed-second-epoch-readout-v316`.
- `SS-and-selected-confirmation-admission-v317`.
- `paper-draft-final-Fixed-SS-integration-scope-v315`.

[Cascade 실행 연결](2026-10-25-final-fixed-cascade-metrics-continuation-ko.md), [전체 핵심 행렬](2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md), [claim 원장](../paper/claim-evidence-ledger.md).
