# Semantic correspondence 조합의 병목과 Q/K 보정

## 현재 결론

Retargeting의 이전 positive 결과와 달리, 새 semantic Top-K attention 및 deformation query의 추가 재구성 효과는 아직 입증되지 않았다. 같은 update192의 고정32개 SS IoU는 matched R 0.44216347, input-side attention 0.44190640, input-side DQ 0.44050169, Direct DQ 0.44251782다. Direct−matched R은 +0.0354pp이고 source-cluster95% CI가 [-0.3607,+0.4342]pp라 개선으로 판단하지 않는다. 기존 학습은 유지한다.

이번에 확인한 구체적인 문제는 **pretrained matching 분포 위에 무작위 Q/K logits를 더하는 초기화**다. Feature output이0이라 초기 재구성 자체는 유지되지만, feature branch 내부의 후보 확률까지 유지되는 것은 아니다. 이 구분 없이 zero-output 초기화만 확인하면 잘못된 후보 분포에서 feature 학습을 시작하는 문제를 놓친다.

## 실제 확률 검사

현재 native RGB/Omega transport 및 Top64 cache와 입력·prior bytes를 대조하고, 실제 step192 readout9개만 GPU에서 실행했다. GT는 CPU 집계에만 썼으며 optimizer update는0회였다. 표본은 utility fit에서 제외한32개 source cluster, 관측 가능한 target/source 관계1,275개다. **모두 SS TRAIN에 속한다. Reconstruction validation 또는 독립 test가 아니다.** 여러 표면 관계가 같은 target-patch ray를 공유할 수 있어 확률 mass를 관계별로 집계한다.

| 분포 | Exact 정답 mass | Near1 정답 mass | Near1 Top1 coverage |
|---|---:|---:|---:|
| 원래 scalar matching prior |15.8413%|43.7147%|47.1373%|
| Pretrained utility7 |15.6125%|45.3326%|47.6078%|
| Cold random Q/K 추가 |15.1977%|44.6761%|45.9608%|
| 학습된 Direct step192 |15.8393%|44.9627%|47.5294%|
| Step192 Q projection weight만0 |16.2093%|45.6919%|48.4706%|
| Step192 score prior 제거 |0.3397%|2.5784%|1.3333%|
| Step192 decoder query context bypass |15.7807%|43.8580%|45.7255%|

같은 후보 집합의 exact/near1 Top64 coverage는53.0196%/79.9216%다. Near1의 약20%는 후보 집합에 없고, 후보가 있는 경우에도 현재 attention이 충분한 mass를 주지 못한다. 이 coverage를 SS IoU의 이론적 상한으로 해석하지 않는다. 전체 모델에는 다른 RGB conditioning 경로도 존재한다. 과거 pooled-source-ray 진단과 query domain이 달라 수치를 섞지 않는다.

Source cluster32개를10,000회 paired bootstrap한 near1 mass 차이는 다음과 같다. 단위는 percentage point다.

- Random Q/K−utility7: **-0.6565pp**, CI [-1.0748,-0.2968]pp.
- Step192−utility7: -0.3699pp, CI [-0.9918,+0.1860]pp.
- Step192 Q-weight0−step192: **+0.7293pp**, CI [+0.3870,+1.1203]pp.
- Score-prior-off−step192: -42.3842pp, CI [-50.4906,-33.6164]pp.
- Decoder-context-bypass−step192: -1.1047pp, CI [-2.1905,+0.0517]pp.

따라서 score prior에 의존하는 실제 후보 확률 경로는 확인됐다. 반면 decoder context의 추가 효과는 이 검사에서 확정되지 않았다. 이들은 **saved-weight matching 진단**이며 생성 품질의 인과적 효과나 retrained ablation 결과가 아니다. Score-prior-off는 사용자가 정의한 NoScore(Top1 feature만 사용)와 다르다. Decoder bypass에서도 D1 candidate bank는 유지된다.

## 구현하고 시작한 보정

새 `PriorAnchoredRayFeatureReadout`은 Q projection **weight만0으로 초기화**한다. RGB/Omega 관측, semantic Top64, matching score, context, 원래 Top1 carrier는 보존한다. Deformation feature를0으로 채우는 방식이 아니다. Inherited output projection도0이라 새 branch의 초기 출력은0이다. 기존 실행 코드와 source binding은 수정하지 않았다.

먼저 기존 auxiliary cache의512 selected TRAIN 중 covered-positive가 있는503개로 Q/K 두 tensor만7에폭 학습한다. Pretrained utility5와 value/output 및 모든 SS/R/LoRA/matching decoder는 고정한다. 다중 positive-set GT는 loss에만 들어간다. 추가 raw 다운로드나 DeformNeXt 재전처리는 없다. Calibration32는 기존 utility fit에서도 제외했던 동일 source-disjoint 집합이며 reconstruction SS가 학습한 TRAIN이라는 사실은 변하지 않는다.

실제 native attention과 같은 확률식을 사용하는 Q/K 전용 경로를 구현했다. CPU3개 검사는 초기 pretrained utility 확률 및 output 보존, Q→K 순서의 gradient와 두 tensor만 변경, masked/empty ray 확률 일치를 확인했다. 실제 GPU8192-query profile에서도 native probability의 **byte-exact 일치**, 임시3 update, 두 tensor만 변경을 확인했다. 손실은 같은 TRAIN batch에서1.82640→1.80604→1.78518이었다. Max allocated는**17.3326GiB**, native process exit0이다. 이 값은 학습 가능성과 용량 증거이며 일반화 증거가 아니다.

7에폭 warmup을 physical GPU0에서 시작했다. SS 두 DDP 학습의 peak 약50.4GiB와 별도20% allocator 상한 약19.12GiB를 근거로 함께 실행하며 GPU0–3 정책을 지킨다. Micro8192, accumulation1이다. W&B entity는 `youngwoojeon-unist`, run id는 `ss-prior-anchored-QK-v1-s101-512-e7`이다. Epoch0–7의 matching metric과 checkpoint를 모두 기록하고 **미리 고정한 epoch7**을 비교한다. 중간 epoch의 최고 수치로 선정하지 않는다.

Final7−initial0 near1 mass의 paired source-cluster CI lower bound가0보다 커야 native SS candidate 검사의 근거로 삼는다. 이것만으로 SS 학습을 자동 추가하거나 모델을 채택하지 않는다. 통과해도 실제 SS zero-output identity, native gradient/continuation, 같은480 budget의 재구성 학습 및 fixed128×3의 강한 original R/V1 비교가 필요하다. 추가 auxiliary 학습 예산도 함께 명시해야 한다. SS loss만으로 매칭 보정이 유지되는지, source feature와 deformation difference를 실제 사용하는지도 확인해야 한다.

## 유지한 비교와 해석 범위

- Input-side DQ와 Direct DQ는 현재 고정480까지 계속 학습한다. Physical Direct NoDQ는 별도 native proof를 마쳤고, fullbatch capacity와 같은 R96 continuation이 연결돼 있다. Query 입력을 실제로 제거한 대조군이다.
- Direct 입력을 원래 semantic fusion gate 밖의 bounded retarget residual로 전달한 효과와, 추가 RAW deformation-query의 효과를 분리한다. 기존 scalar composer 학습과 최종128×3도 유지한다.
- Full Direct의 score/decoder/feature/DQ 및 correspondence-value saved-weight 검사, proposal 밖의 value 대조를 유지한다. Wrong-value/outside-proposal을 원래 native-spatial NoCorr의 효과로 보고하지 않는다. 올바른 모듈 ablation과 독립 training seed는 개선 후보 확인 뒤 필요하다.
- PBR는 warm continuation 및 실제 initial32 rendering parity를 이미 검증했다. 실제 global24 capacity와 학습은 SS 결과 이후 연결된 상태다. Appearance 쪽 효과가 더 클 가능성은 가설이며 아직 측정 결과가 아니다.

새 확률 진단은 초기 분포 문제를 특정했지만, 그 문제가 낮은 SS 성능 전체를 설명한다고 확정하지 않는다. 추가 feature의 정보가 기존 target RGB와 중복되거나, 잘못된 correspondence가 RAW difference를 오염시키거나, SS reconstruction objective가 그 경로를 충분히 학습하지 못할 가능성도 남아 있다. 구조와 손실의 변경은 이 실제 검사 결과에 따라 결정한다.

## 증거 위치

Canonical root: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`.

- `direct-retargeting-v1/probability-audit-step192-v1`: 실제8종 확률, 입력/source/checkpoint SHA 및 실제 file4 receipt.
- 그 아래 `paired-source-analysis-v1`: 관계 수를 보존한 source bootstrap.
- `prior-anchored-QK-v1/plan-v1`, `followup-plan-v1`: 고정7에폭 계획과 source snapshot.
- `prior-anchored-QK-v1/profile-v1`: 실제8192 CUDA3 update, native parity 및 terminal0.
- `prior-anchored-QK-v1/head-training-v1`: 시작한7에폭 보조 학습. SS 재구성 학습 완료를 뜻하지 않는다.
- `prior-anchored-QK-v1/warmup-queue-connection-v1`: 기존 job을 바꾸지 않은 append 기록과 fixed7 비교 callback.
- `physical-query-epoch-trends-v1/step0000192`: 실제 같은 update의6개 SS 모델 및 zero-shot 맥락 비교.

성능 개선 또는 full 모델 채택은 아직 선언하지 않는다.
