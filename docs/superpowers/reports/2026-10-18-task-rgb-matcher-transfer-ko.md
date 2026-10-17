# 재현한 Q2C의 DeformNeXt task RGB 대조

원본63-pair Q2C의24개 run을 수치까지 재현한 다음 단계다. 현재H3의 ranking CE/DINO cosine objective를 원래 RGB task와 같다고 취급하지 않는다. 새 결과는 별도 `task-rgb-transfer-v1` namespace에 기록한다. 진행 중인14개 SS 비교와G2 reference 재생은 유지한다.

## 고정 비교

| Arm | Architecture | 후보 | Loss |
| --- | --- | --- | --- |
| T0 | 원본 R9Scorer D0 pairwise | 기존 frozen Omega epoch7 global RAW64 | controlled RGB L1 |
| T1 | 동일 + 원본 D1 Q2C | 동일64개 | 동일 RGB L1 |

Score-aware soft fusion은 양쪽에 같다. 이것은 Q2C context의 유무 비교다. 사용자가 정의한 No-score(Top1 선택), No-corr(native spatial), physical No-query의 의미를 바꾸지 않는다. Confidence/rejection을 새 head에 임시로 붙이지 않는다.

기존 head-train8570개에서256개, head-calibration573개에서64개를 category round-robin·role/sample hash로 선택했다. 각 pair는 서로 다른 source cluster이며 train/calibration/validation cluster가 겹치지 않는다. 기존508 validation의 이미지나 task label을 이 pilot 평가에 사용하지 않는다. 이64개 calibration 결과를 보고 방법을 선택하면 unseen-test라고 부르지 않는다.

## 원래 task를 patch 공간에 옮기는 계약

이전 `procedural_surface_cue` 함수를 AST 그대로 이식했다. M0 vertex에 label16개·분리된 evaluator16개 RGB cue를 만든다. M1은 같은 topology의 barycentric interpolation으로 같은 source-space identity를 받는다. 대상은 전체 source surface이며, 과거의 vertex-level transport-region carrier와 이 patch-level 확장은 별도 task variant다.

실제512px4-view camera에서 CPU raycast를 수행하고, 기존RGB alpha를 사용해 patch16×16의 foreground-normalized mean cue를 구한다. Sparse pixel→vertex barycentric operator를 사용하므로 nearest-patch hard label로 색을 복사하지 않는다. 원본 topology/raster certification과 이미지·mesh·camera fingerprint를 대조하고, 새 raycast도 input alpha와 검증한다. 원래 descriptor source-valid mask 및 기존Top64 proposal은 변경하지 않는다. GT geometry로 후보를 고르지 않는다.

Cue는 loss supervision이다. Matcher 입력은 기존 실제RGB에서 추출한 Omega epoch7 FRAME1024이며, cue RGB·vertex ID·geometry·camera·sample ID를 feature로 전달하지 않는다. Label/evaluator cue는 별도 파일에 저장한다. GT상 source-visible인 foreground query를 task 대상으로 쓰되, 후보 안의 exact patch 정답 여부로 eligibility를 다시 고르지 않는다. 후보 miss·가시 query 수는 별도 기록한다. 관측 가능한 query가0개인 pair는 임의 대체하지 않고 수와 ID를 보고한다.

## 학습·평가

원본과 같은 pair-uniform sampling·256query/update·1200update를 사용한다. Scalar head LR0.001, added Q2C LR0.0001, AdamW weight decay0.0001, fixed tau0.005, no correspondence CE다. P0-RAW의 preservation term은0이다. Seed42017/42018/42019에서 T0/T1을 각각 fresh로 학습한다. 같은 seed는 동일 pair/query sampler를 쓰고 초기 score가 RAW 및D0와 exact인지 확인한다.

이것은 matching head의 query batch다. 현재 SS의 global asset batch96·LoRA continuation·5epoch 설정은 바꾸지 않는다. 최종 checkpoint와 SHA를 고정한 뒤 evaluator cue를 연다. Calibration64의 pair별 RGB L1·RAW/Top1 참고치·seed별 변화와 source-cluster paired CI를 저장한다. 원본 controlled experiment와 현재 dataset 결과를 합쳐서 집계하지 않는다.

학습된 pilot checkpoint는 자동 production 승격하지 않는다. Task gain·native SS 입력 경로·SS 실제 생성의 순서로 검증한다. 기존 G2의9채널/region-loss 재현, native per-source-camera 경로와 native fallback의 검증은 별도 조건이다. 현재 GT-address32 진단의+1.96IoU pp를 이 learned matcher의 향상으로 사용하지 않는다.

## 자원·실측 준비

GPU는0~3만 사용한다. 현재0/1의C2와2/3의G2 재생을 유지하며 CPU cache는 affinity2core로 제한한다. 첫 실제 asset의512px4-view cache는약3.3초였고 mask IoU는각view0.9973 이상, containment1.0이었다. 이후 실제320개 전체는196.3초(약3.3분)에 완료됐다. 기존 RGB/Omega cache를 재사용했으므로80k 재다운로드나 backbone 재추론은 하지 않았다.

RNG/role seed·barycentric area average·background·split·cluster 독립성의CPU4개 검증과 실제1개 asset의 label/evaluator 파일 분리를 확인했다. 이 PASS는 학습 개선을 뜻하지 않는다. Cache와 source provenance를 file4에 보존하고, 비는GPU2/3에서6개의 짧은 head 학습을 실행한다.

전체320개 중 source-visible task query가 있는 pair는 train251/256개, calibration61/64개다. 나머지8개의ID와 역할은cache summary에 보존하며 다른asset으로 대체하지 않았다. T0/T1의 paired 비교는 같은61개 calibration cluster를 사용한다. 이 제외는 학습 전 geometry/visibility 조건으로 결정됐으며 task error를 본 뒤 선택하지 않았다.

`train_task_rgb_transfer`는원본Scorer/temperature/optimizer budget을 사용하고 실제 patch의 연속RGB 정답을 읽는다. 별도 evaluator cue file은 final checkpoint freeze 뒤에만 연다. Preflight는 train pair 하나에서2-update·첫 score parity·gradient·held-out cue bank의I/O 및auxiliary topology/DINO metric을 확인한다. Calibration pair를 preflight에 쓰지 않는다. D0/D1 seed별head 초기화·pair/query sampler·초기 score/loss 동일성, RGB를 바꿔도 descriptor score가 안 바뀌는 것과 evaluator file이 training에 열리지 않는 것을 포함해 관련CPU7개 검증이 통과했다.

Cal metric에는RGB task error와 함께 exact-patch Top1/proposal recall 및GT-nearest patch DINO teacher cosine error를 기록한다. Learned soft fusion과RAW soft/Top1·learned Top1은같은64 candidate와동일eval cue 위에서 비교한다. 보조 metric도 동일한 관측 가능한 query에서 평가하며, 이전 H3의 full-validation population과 직접 합쳐 비교하지 않는다.

## 실제 6-run 완료 결과

6개 run 모두1,200 updates를 완료했다. 같은 seed의3,600개 paired update에서 sample ID·query hash가 모두 같고, 초기 loss도 세 seed 모두 exact였다. 각 checkpoint의 입력 cache SHA·optimizer budget·final freeze와 evaluator 개방 순서를 확인했다. 모든 run에서 같은61개 source cluster·같은query·같은RAW control 수치를 사용했다. 각 query/channel/cue와 seed를 cluster 안에서 평균한 뒤61개 cluster를10,000회 paired bootstrap했다.

| Metric | RAW soft | D0 pairwise | D1 Q2C |
| --- | ---: | ---: | ---: |
| Held-out cue RGB L1 ↓ | 0.13757989 | 0.09008583 | 0.08467440 |
| DINO teacher cosine error ↓ | 0.06034594 | 0.04545267 | 0.04471921 |
| Exact-patch Top1 accuracy ↑ | 0.02576265 (RAW Top1) | 0.11975704 | 0.12107392 |

Q2C의RGB error 감소는 **0.00541143 (D0 대비6.01%), 95% CI[0.00305711,0.00771686]**다. 세 seed의 감소량은0.00716815/0.00432741/0.00473875로 모두 양수다. Learned Top1으로 값을 읽어도 Q2C 추가 후 error가0.09308689→0.08739888로 감소했다. 이는 사용자가 정의한 No-score를 uniform attention으로 바꿔 평가한 것이 아니다.

DINO error의 추가 감소량0.00073346은CI[−0.00020878,0.00166502], exact-patch accuracy 차이+0.00131688은CI[−0.01073324,0.01329354]로0을 포함한다. **RGB task의 Q2C 효과를 DINO address 개선이나 SS 생성 개선으로 확대하지 않는다.** RAW64 proposal recall은0.68218465이며GT 후보 누락을 제외하지 않았다. 이 pilot의선택/비교에 사용한61개 calibration cluster는 unseen test가 아니다. CI는 학습 seed 모집단의 불확실성을 추정하지 않는다.

집계는 `python -m experiments.deformnext_retargeting.summarize_task_rgb_transfer --inputs <task-root>/inputs-v2 --root <task-root>/heads --output <new-summary.json>`로 재실행할 수 있다. 이미 존재하는 결과는 덮어쓰지 않는다. 결과와 전체 paired audit는 file4 namespace의 `task-rgb-transfer-v1/matched-six-run-summary.json`에 있다. 관련CPU11개 검증이 통과했다. 본학습 loop와 calibration 평가의 실측은 preload 이후약19~24초/run, GPU peak0.69/0.96GiB였으며 전체 데이터 준비시간과 구분한다.
