# Matching confidence, visibility, retargeting 실패 원인과 다음 검증

2026-10-20 UTC. `wowo0709/feature-retargeting`. 실제 생성 품질 향상과 full 모델 채택은 아직 입증되지 않았다. 아래 matching 수치는 별도 표기가 없으면 실제 SS VAL32의 native target-patch ray와 기존 topology teacher를 사용한다. 이 패널은 반복 분석한 개발용 패널이며 독립 test가 아니다.

## 현재 결론

Semantic matching 자체가 무효인 것은 아니다. 별도 fixed7 Q/K 학습은 near1 Top1을 개선했다. 다만 Top-K의 정답 확률 질량, 생성 모델 공동 학습 중 matching 유지, 대응점이 없는 ray의 처리, retargeting 경로의 실제 영향은 서로 다른 문제다. 현재 구조는 이 네 조건을 동시에 만족하지 못한다. 단순히 학습 epoch를 늘리거나 attention 확률을 날카롭게 하는 것만으로 해결됐다고 판단할 수 없다.

이전에 보고한 positive 결과도 구분해야 한다. `reports_final2/2026-09-21-three-module-qualitative-results-ko.md`의 SS oracle 실험에는 GT correspondence/camera 조건이 포함된다. Q2C의 RGB L1 개선은 GT surface를 통제한 APP-CUE 실험의 결과다. 이를 실제 predicted correspondence를 사용하는 native SS/PBR 전체 모듈의 개선으로 바로 해석하면 안 된다. 현재 retargeter의 효과와 matching/attention의 추가 효과도 별도로 비교해야 한다.

## 1. 순위 개선과 Top-K 확률 질량은 달랐다

별도 Q/K 학습의 epoch7을 사전에 고정했다. utility5와 V/O2는 동결하고 Q/K2만 224회 업데이트했다. SS/R 업데이트는 0이다.

Visible-source relation 기준 near1 Top1은 utility 대비 **+3.4077 pp**, source-cluster paired CI95 **[+1.3022, +5.2601]**였다. 그러나 near1 mass 변화는 **−0.2790 pp**, CI95 **[−1.4769, +0.8444]**로 개선이 확인되지 않았고 exact mass는 감소했다. SS-TRAIN의 별도 calibration32에서도 기존 near1-mass gate는 실패했다. 이 gate를 소급해서 바꾸거나 epoch1을 선택하지 않았다. 해당 epoch7 가중치는 현재 SS 학습에 배포하지 않았다.

네 attention head를 pooling한 뒤 confidence temperature만 바꾸는 검사를 추가했다. grid `[.125,.25,.5,.75,1,1.5,2,4]`, objective, tie-break를 사전에 고정하고 auxiliary TRAIN-calibration32에서 선택한 뒤 실제 VAL32를 검사했다. TRAIN의 native query가 없는 두 asset도 유지했다. 첫 prototype의 empty-query 오류는 실패로 보존했고, 이를 처리한 v2가 native/wrapper 모두 종료 코드 0으로 완료했다. 모든 T1 relation 수치는 기존 실제 native 결과와 정확히 같았다.

아래는 **source-view relation을 중복하지 않은 ray-union** 결과다. 위 relation Top1과 분모가 다르다.

| 모델 / temperature | near1 mass | near1 Top1 | covered near1 NLL ↓ | effective K | 최대 확률 평균 |
|---|---:|---:|---:|---:|---:|
| utility / 1 | .466095 | .508899 | 1.726369 | 6.123 | .554 |
| fixed7 QK / 1 | .463127 | .545155 | 1.338979 | 6.539 | .457 |
| fixed7 QK / .125, TRAIN mass로 선택 | .540599 | .545155 | 6.440064 | 1.427 | .882 |
| fixed7 QK / 1.5, TRAIN NLL로 선택 | .418444 | .545155 | 1.224880 | 10.733 | .352 |

Sharpening `.125`는 fixed7 T1 대비 mass를 +7.7472 pp 높였지만, effective K를 약 1.43까지 줄였고 NLL은 크게 악화시켰다. 이는 잘못된 Top1에도 확신하는 효과를 포함한다. **사용자가 기대한 여러 semantic 후보의 유용한 조합을 입증한 결과가 아니다.** NLL objective로 부드럽게 하면 불확실성은 개선되지만 mass는 낮아진다. ranking 변화는 모두 0이었다. Temperature는 live SS에 적용하지 않았다. 이 검사는 사용자의 NoScore arm인 “matched Top1 feature만 사용”과도 다르다.

실제 calibration은 32.27초/peak .147 GiB, VAL은 40.07초/peak .405 GiB였다. NN generation이나 NN/R 학습을 수행한 시간은 아니다.

## 2. 공동 학습 중 matching이 망가지는 현상

현재 geometry-only 모델은 appearance V/O 행렬을 물리적으로 제거했다. native Top1 appearance와 기존 R을 유지하면서, ray-wise attention이 RAW source Omega를 가중합하고 `target_RAW − weighted_source_RAW`를 deformation query로 사용한다. 추가 Q projection은 0으로 초기화해 기존 utility prior와 초기 함수를 보존했다. 관측 feature를 0으로 대체하지 않았다.

실제 step192의 저장된 conditioner로 전체 native forward를 실행했다. query 순서와 여섯 입력 tensor가 닫힌 VAL cache와 정확히 같고, 현재 head의 native probability도 정확히 재현됐다.

| ray-union 지표 | 초기 utility | geometry 공동 학습 step192 |
|---|---:|---:|
| near1 mass | .466095 | .455457 |
| near1 Top1 | .508899 | .483850 |
| covered near1 NLL ↓ | 1.726369 | 1.812332 |

Mass 변화 **−1.0639 pp**, CI95 **[−1.8810, −0.2020]**. Top1 변화 **−2.5049 pp**, CI95 **[−4.6396, −0.5882]**. 현재 Q만 0으로 되돌린 저장 가중치 진단에서도 utility selector의 변화가 남는다. **초기 함수가 일치한다는 것과 matching 품질이 학습 중 유지된다는 것은 다르다.** 다음 생성 실험에서는 matching 동결 또는 별도 correspondence objective로 이 drift를 통제할 필요가 있다. 아직 이 처방의 생성 품질 개선을 주장하지 않는다.

## 3. Deformation 경로는 살아 있으나 영향이 작다

실제 step192의 2,511개 active ray를 hook으로 측정했다.

| 항목 | pooled RMS |
|---|---:|
| appearance 입력 | 1.000002 |
| RAW deformation query | .161084 |
| appearance hidden | 1.310274 |
| deformation projection hidden | .104442 |
| 실제 BF16 hidden 덧셈 변화 | .104466 |
| 전체 R 출력 | .065627 |
| deformation에 의한 R 출력 변화 | .002799 |
| RAW source 주소를 바꾼 R 변화 | .002795 |

BF16에 묻혀서 deformation 경로가 사라지는 것은 아니다. source 주소 변화에도 민감하다. 다만 DQ hidden은 appearance hidden의 약 8%, DQ 출력 변화는 전체 R의 약 4.27%다. Feature scale과 R이 appearance 경로에 의존하는 문제가 후보다. 이 VAL 측정값을 이용해 scale을 fitting하지 않는다. 정규화 실험을 한다면 TRAIN 통계 또는 inference에서 계산되는 per-ray normalization을 사용하고 원래 magnitude를 보존할지 별도 검증해야 한다. Projection-off와 wrong-address는 저장 가중치의 sensitivity 검사이며 재학습한 NoDQ ablation이 아니다.

## 4. Native ray 상당수에는 선택할 정답이 없다

기존 teacher를 읽기만 해서 patch-center foreground, source visibility, native Top64 coverage를 분리했다. 새 렌더/latent/Omega는 만들지 않았다.

| VAL32의 native target ray 분류 | 개수 |
|---|---:|
| 전체 | 2,511 |
| GT patch center가 background여서 판단하지 않는 ray | 1 |
| GT foreground center지만 세 source 어디에도 visible relation이 없는 ray | 993 |
| visible source가 있는 ray | 1,517 |
| 그중 native Top64에 near1 정답 후보가 있는 ray | 1,345 |
| 그중 native Top64가 near1 정답 후보를 놓친 ray | 172 |

가시성은 기존 topology teacher의 patch-center/depth 정책을 따른다. Full patch 전체에 대응점이 없다는 증명은 아니다. Background center를 no-correspondence negative로 잘못 취급하지 않았다.

현재 valid candidate softmax는 적어도 하나의 후보가 있으면 확률 합이 1이다. 이 때문에 source에 정답이 없거나 proposal이 놓친 ray에도 가중합된 feature가 전달된다. 양성 correspondence만 보는 head loss와 coverage 지표만으로는 이 문제를 해결할 수 없다.

## 5. 다음 작업을 실제로 연결한 내용

작은 **correspondence reliability head pilot**을 추가했다. 새 full SS 학습을 PBR 앞에 추가한 것은 아니다.

- 기존 auxiliary TRAIN512 / TRAIN-calibration32 / 실제 VAL32를 사용한다. Source cluster는 role 간 겹치지 않는다. Native query가 없는 asset도 유지한다.
- 입력은 실제 inference의 native query, context query, semantic score, candidate-valid, candidate address뿐이다. GT는 label과 unknown-center 제외에만 사용한다.
- 목표는 “native Top64에 visible near1 후보가 하나라도 있는가”다. No-visible-source와 proposal-miss도 negative 학습에 포함한다.
- TRAIN known ray 41,117개: usable 19,593 / source-invisible 17,937 / proposal-miss 3,587. Calibration 2,314개, VAL 2,510개. Unknown-center는 각각 45/2/1개 제외했다.
- 두 feature를 per-ray 정규화하고 12개 inference summary를 더한 2,060차원 입력, hidden64의 작은 MLP, trainable tensor4개다. Summary mean/std는 TRAIN만 사용한다. 7 epoch, seed101, AdamW LR .001 / WD .01, microquery4096을 사전에 고정했다.
- 최종 epoch7을 사용한다. Threshold는 TRAIN-calibration의 사전 고정 19-grid balanced accuracy로 선택하고, actual VAL label을 열기 전에 selection을 기록한다. VAL AUROC≥.65, AUROC cluster-CI 하한>.5, TRAIN prevalence constant보다 Brier가 낮고 paired-CI 상한<0을 진단 gate로 고정했다.
- GPU2에서 현재 geometry SS가 끝난 뒤 실행하도록 queue에 연결했다. Allocator limit 5%, W&B `youngwoojeon-unist/pixal3d-deformnext-retargeting`. 기존 SS/R/matching 가중치는 수정하지 않는다.
- Cache 약 379 MB와 source/plan을 실제 file4에 SHA256 검증해 보관했다. Head 결과는 아직 pending이다. 진단 gate를 통과해도 생성 품질 개선이나 full 모델 채택을 뜻하지 않는다.

이 pilot이 구별력을 보이면, frozen matching의 soft TopK와 reliability를 이용해 잘못된 추가 보정을 억제하는 retargeting 경로를 검증한다. Baseline의 native appearance/R은 보존하고, 처음 출력이 baseline과 정확히 일치해야 한다. 그 뒤 동일 초기 NN/R·Adam·sampler·seed·update 예산으로 생성 품질을 비교한다. 신뢰도 head, frozen matching, feature normalization을 한꺼번에 성공 원인으로 묶지 않는다. 실제 효과가 확인된 단계에 대해 Top1/semantic-vs-spatial/physical-NoDQ ablation을 분리한다.

## 6. SS 진행과 PBR 순서

새 geometry SS와 physical Direct-NoDQ는 공통 R96의 NN/R·Adam366을 이어 총480 update로 진행 중이다. Global96, DDP2, GPU당48, accumulation1을 유지했다. 두 모델 모두 실제 초기32와 양 rank의 첫97 loss, common optimizer/sampler/RNG가 기준 R과 정확히 일치한다. NoDQ는 channel을 물리적으로 제거한 375-state 모델이며, geometry는 appearance V/O가 없는 374-state 모델이다.

| 같은 epoch, VAL32 IoU | R control | geometry-only | Direct Full | physical NoDQ |
|---|---:|---:|---:|---:|
| step192 | .442163 | .441957 | .442518 | .442128 |
| step288 | .445920 | .445034 | .439131 | 진행 중 |

Geometry step288은 Direct Full보다 +.5903 pp, CI95 [.1098, 1.1275]이나 R control 대비 −.0886 pp, CI95 [−.4412, +.2246]다. **망가진 경로를 제거해 회복한 것과 R보다 좋아진 것은 다르다.** 고정480 및 기존 final128×3 generation noise, strong R/V1, 두 4-RGB zero shot, path/outside-proposal 비교를 유지한다. Noise 반복은 training-seed 재현이 아니다.

01:07 UTC 근처 실제 상태는 geometry 319/480(GPU2/3), NoDQ 210/480(GPU0/1)였다. CUDA allocated peak는 각각 약50.03/49.36 GiB, nvidia-smi 사용량은 GPU당 약55.5 GiB였다. GPU는 각97,887 MiB이다. OOM 징후는 이 체크에서 없었다. Geometry는 약18초/update이며 NoDQ는 data load 약10초가 더해져 약28–30초/update였다. 진행 중인 학습을 재시작하지 않았다.

PBR가 SS 추가 실험 뒤로 계속 밀리지 않도록 이번 round의 순서를 고정했다.

1. 현재 두 SS fixed480 완료.
2. 기존 실제 PBR 초기32/PNG/native correctness 증명을 재사용하고, 실제 paired DDP2 micro12/global24/acc1 capacity를 먼저 검사. 실제 CUDA OOM에만 DDP4 micro6로 전환.
3. 남은 scalar SS arm과 이미 계획한 최종 평가를 완료.
4. PBR common V1 step762에서 vanilla와 FullDirectDQ를 같은 예산으로 step1905까지 학습·비교.

먼저 끝난 SS의 GPU 쌍이 놀지 않도록 최종 평가를 채운다. 두 현재 SS가 모두 step432 이상이면 새 평가 배정을 잠시 막고, actual PBR capacity가 끝나면 평가 배정을 다시 연다. Running native process는 중단하지 않는다. 이 변경은 scheduling뿐이며 scientific config/command/output/GPU assignment는 그대로다. 추가 reliability pilot 한 개를 제외한 491개 queue descriptor도 유지했다.

현재 PBR 두 arm은 전체 stack 대 vanilla 비교다. Matching의 독립적 추가 효과를 PBR에서 주장하려면 동일 예산의 R-only 및 필요한 모듈 ablation이 추가로 필요하다. Quick rendering32/GT-shape 조건부 결과를 full cascade나 논문의 10 metric 최종 검증으로 혼동하지 않는다.

## 산출물과 검증

Canonical root: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`.

- `prior-anchored-QK-v1/pooled-temperature-v1/failed-native-v1`: 실패한 empty-query prototype의 실제 종료 코드1과 원인 보존.
- `prior-anchored-QK-v1/pooled-temperature-v2/{calibration-v1,validation-v1}`: 고정 grid 선택과 실제 native VAL, native/wrapper 종료0.
- `geometry-only-retargeting-v1/usage-step192-v1`: 실제 경로 활성과 matching drift.
- `prior-anchored-QK-v1/observability-audit-v1`: 2,511개 ray의 가시성/coverage 분해.
- `direct-no-query-v1/scientific-startup-v1`: physical NoDQ의 초기 함수/Adam/sampler/RNG 일치.
- `{geometry-only-retargeting-v1,direct-no-query-v1}/epoch-trends-v1`: 같은 epoch 자동 비교 callback.
- `prior-anchored-QK-v1/reliability-head-v1/{cache-v1,prelaunch-v1,queue-v1}`: 다음 작은 head 검사의 데이터/실행 계획/연결. 결과는 `fit-fixed7-v1`에 예정.
- `direct-retargeting-v1/PBR-joint-continuation-v1/{scheduling-round-order-v1,evaluation-window-plan-v1,evaluation-window-queue-v1}`: 유한 SS round와 PBR 전환 순서.

새 CPU checks는 temperature probability/7-gradient/native-width 3개, ray-union metric 1개, empty-query 1개, reliability feature/metric 2개가 통과했다. 실제 native 온도 검사와 conditioner probe도 완료했다. Reliability head와 최종 생성 결과가 끝나기 전까지 성능 향상 달성을 선언하지 않는다.
