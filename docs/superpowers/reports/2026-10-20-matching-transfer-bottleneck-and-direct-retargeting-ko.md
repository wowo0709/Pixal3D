# Matching 전달 경로 진단과 direct retargeting 후보

## 아직 개선이 입증되지 않은 이유와 확인 범위

Semantic matching + Top-K + matching-aware attention이라는 이름만으로 성능 개선을 보장할 수 없다. 후보를 여러 개 사용하는지, 후보 간 차이를 담은 정보가 있는지, 그 차이가 생성 모델에 전달되는지를 따로 확인해야 한다.

실제 첫 동일 epoch 비교에서는 추가 이득이 없었다. 같은 old R step96에서 공통 optimizer366, sampler, stage RNG reset을 유지해 96 updates를 추가한 step192 결과다.

| SS 모델 | 시작96 IoU | step192 IoU |
|---|---:|---:|
| Retargeting control | 0.43718719 | 0.44216347 |
| Input-side feature attention | 0.43718719 | 0.44190640 |

32개 asset/32 source cluster, 같은 noise와 target voxel hash로 비교한 attention−control은 −0.00025707, paired source-cluster 95% CI는 [−0.00140694, +0.00092577]이다. 유의한 개선이나 악화가 확인되지 않았다. Final480/128×3 평가나 독립 학습 seed 검증을 대신하지 않는다.

반면 현재 attention은 step190에서 평균 effective K=4.40593, feature residual RMS=0.0312141, head gradient L1=0.516803이었다. 따라서 단순히 전부 Top-1이거나 gradient가 없는 상황은 아니다.

## 구분한 세 문제

1. **후보 가중치 집중:** 기존 PBR 온도0.005에서 Top-1 mass가 평균93.36%였다. 그러나 고정 weights의 32개 온도변경 평가로는 개선되지 않았다. 온도 조절만으로 해결됐다고 주장하지 않는다.
2. **전달 정보:** appearance value가 유사한 후보들은 attention weight가 달라도 비슷한 값을 전달할 수 있다. 그래서 동일 correspondence probability로 조합한 source RAW Omega와 target RAW Omega의 차이를 retargeter의 실제 추가2048개 채널로 전달하는 별도 arm을 구현했다. GT correspondence/deformation을 forward 입력으로 쓰지 않는다.
3. **전달 위치:** native fusion은 semantic replacement와 retargeting residual을 다르게 처리한다. `image_conditioned_proj.py`의 native update는 `gate * semantic_delta + 3/4 * retarget_residual`이다. Input-side attention의 직접 feature 성분은 작은 semantic gate를 거친다. Retargeter를 통한 간접 경로는 gate 밖에 있으므로, attention 전체가 차단됐다는 뜻은 아니다. 이 직접 성분의 감쇠가 품질 부진의 원인인지는 추가 비교로 확인해야 한다.

## 새 후보: bounded transformed correspondence를 retargeting 출력으로 전달

`DirectRayRetargetingConditioner`는 physical-DQ arm의 동일한 semantic Top64, matching score, contextual decoder, edited target patch ray query와 parameter376개를 유지한다. Attention이 만든 보정값을 `0.25 * tanh(delta / 0.25)`로 제한하고 native retargeting residual에 넣는다. Semantic replacement에는 기존 positive Top-1 carrier를 유지한다. 개별 voxel을 attention query로 바꾸지 않는다.

추가 head의 output projection과 DQ projection weight는 0에서 시작한다. 실제 DQ 입력 관측값은 비영 값이다. 기존 NN/R weights와 common Adam366을 보존한다. 이는 기존 reference와 같은 출력에서 출발하면서, 학습된 correspondence feature가 semantic fusion gate 밖의 retargeting 경로로 전달되는지를 시험하는 후보다. 별도의 강제 gate floor나 surrogate gradient는 사용하지 않는다.

실제 동일 2개 TRAIN asset에서 NN/R/matcher를 고정한 검증 결과:

| 검증 | Input-side physical DQ | Direct retargeting |
|---|---:|---:|
| 시작 SS loss | 0.1046660021 | 0.1046660021 |
| 첫 backward의 head output gradient L1 | 1.6319362 | 31.1272583 |
| 세 번째 임시 update에서 측정한 loss | 0.1041570529 | 0.1006616503 |

동일 초기 loss/evidence를 유지하면서 head output gradient가 약19.1배 커졌다. 두 모델 모두 새10개 tensor가 비영 gradient를 받았다. 이 값들은 **학습 가능성과 전달 경로 검증**이다. 단 두 TRAIN asset의 임시 optimizer update이며 모델을 폐기했다. Validation 개선이나 full model 채택의 근거로 쓰지 않는다.

새 CPU 검증에서는 실제 non-reentrant checkpoint backward를 통해 head/DQ gradient가 연결되는 것, 초기 evidence와 R residual의 정확한 동일성, output bound와 hook 제거를 확인했다. Joint micro2 및 초기 generation32 검사는 별도로 진행한다. 독립 DDP2/micro48 capacity를 통과하기 전에는 본 학습을 시작하지 않는다.

## 실행과 효과 판정

- 기존 R control과 input-side attention 본 학습을 유지한다.
- GPU0/1은 control 완료 후 physical-DQ capacity와 본 학습을 진행한다.
- GPU2/3은 input-side attention 완료 후 direct retargeting capacity와 본 학습을 우선 진행한다. 기존 scalar composer 작업은 삭제하지 않고 이후에 유지한다.
- 모두 common old R96에서 출발해 additional384/total480, global96/DDP2/micro48/accumulation1/seed101을 맞춘다.
- Final480/128 asset×3 generation noise, replay32, source-cluster paired CI로 비교한다. 세 noise를 세 학습 seed로 세지 않는다.
- 각 새 경로에는 wrong-value-address, score-prior-off, decoder-context-bypass, added-path-off, DQ-projection-off saved-weight 진단을 연결한다. Wrong-value-address는 appearance와 RAW Omega를 같은 permutation으로 바꾸며 keys/probabilities/score/Top-1 carrier를 유지한다.

위 진단은 재학습 ablation과 다르다. Score-prior-off는 사용자의 Top-1 NoScore arm이 아니고, wrong-value-address는 native spatial NoCorr arm이 아니다. DQ projection 출력 mute는 채널이 실제 없는 NoDQ training arm이 아니다. Direct 모델이 유망해지면 동일 delivery 방식의 physical NoDQ 재학습이 필요하며, input-side NoDQ와 direct-DQ의 차이만으로 DQ 효과를 분리했다고 주장하지 않는다.

현재 full model은 확정하지 않았다. 개선이 없으면 해당 모듈을 성공했다고 포장하지 않으며, direct delivery의 validation 이득도 아직 확인하지 않았다. PBR에서는 새 feature 경로의 DINO/NAF gradient까지만 실제 확인했고, direct-DQ PBR fine-tuning 결과는 없다.

## 기록

아래 경로는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1` 기준이다.

- `epoch192-matched-v1/summary.json`: 실제 첫 동일 epoch 결과와 CI.
- `deformation-query-v1/initial-parity-v1/audit.json`: 기존 모델과 초기32개 voxel counts 및 PNG 정확히 동일.
- `deformation-query-v1/controls-implementation-v1/verification.json`: 실제 SS CLI에서 appearance/RAW correspondence intervention 실행 확인.
- `direct-retargeting-v1/SS-gradient-v1/audit.json`: 실제 초기 SS loss/evidence 동일, genuine10-tensor gradient, frozen NN/R/matcher 유지.
- 각 `evaluation-plan-v1`, `control-plan-v1`: 고정 final evaluation과 진단 queue 기록. Source와 데이터/weights/결과 저장소는 분리한다.
