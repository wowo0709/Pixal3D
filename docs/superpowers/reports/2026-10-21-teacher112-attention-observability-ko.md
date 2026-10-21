# Teacher112의 source 사용·Top-K attention·feature 방향 진단

새 source-camera SS teacher112의 실제 저장 weight와 기존 feature cache를 읽어
source 전달과 attention 분포를 측정했다. 검증32개에 세 모델을 모두 적용한
**96회 CPU FP32 feature forward**이며 약70초가 걸렸다. CPU92/93 두 코어만
사용했고 encoder·denoiser·optimizer·GPU forward는 실행하지 않았다. 동시에
진행 중인 PBR 학습, 원래 코드·가중치·온도·gate·loss는 변경하지 않았다.

## 범위와 정합

Original19, SourceReference20, SourceContrast19는 같은 TRAIN1536에서7 epochs,
112 source-module updates를 완료한 teacher다. Native joint SS480 이전 모델이며
기존 full480 SS 또는 현재 PBR Direct의 attention을 측정한 결과로 해석하지 않는다.
먼저 사전에 고정한 TRAIN 첫4개에서 경로를 확인한 뒤, 원래 검증32개 전체를 측정했다.

기존 `module_inputs`와 normal conditioner forward를 그대로 사용했다. Query는
source-camera ray의 voxel members를 모아 구성하며, point matching posterior를
ray 단위64개 후보로 합친다. 각 ray에서 얻은 실제 네 head 확률과 source-use를
읽는다. 네 head는 value/output projection을 거쳐 별도의 feature correction을
만들고, 원래 Top-1 carrier 및 R과 함께 source map에 전달된다.

검증 teacher label은 normal forward가 끝난 뒤에만 읽고 `not_for_training`과
sample ID·canonical SHA256를 확인했다. TRAIN/validation source cluster는 겹치지
않는다. 모든 모델의 state tensor가 전후 동일하고 gradient가 없으며 CUDA가
초기화되지 않았음을 확인했다. 모델 간 frozen matcher state도 동일하다.
출력 delta가 실제 source-use·semantic carrier·R·direct head 항의 합과 일치한다.

## 전달 경로는 실제로 작동한다

아래 attention 평균은 **유효한 source-camera ray** 또는 ray/head 수로 가중했다.
Source-use와 attention에는 source9/10/11이 포함된다. 무효 ray의 native fallback을
source-use 평균에 섞지 않았다. 유효 ray의64개 prior 확률은 모두 양수였다.
유효 후보 수는 entropy의 지수 `exp(H(p))`이며, 올바른 correspondence 개수라는
의미가 아니다.

| Teacher | Source-use u | Prior 유효 K | 네 head 유효 K | Head argmax=prior Top-1 | Head 간 TV |
|---|---:|---:|---:|---:|---:|
| Original19 | 0.6069 | 10.85 | 6.25 | 28.15% | 0.4222 |
| SourceReference20 | 0.6119 | 10.85 | 6.17 | 28.19% | 0.4178 |
| SourceContrast19 | 0.6260 | 10.85 | 6.32 | 30.10% | 0.4203 |

이32개에서 source 전달이 거의0이 되거나 attention 전체가 하나의 prior Top-1에
붕괴했다는 설명은 관측과 맞지 않는다. Head들이 prior 순위와 다른 후보를 실제로
선택하고 서로 다른 확률을 만든다. 다만 이 사실은 그 후보가 semantic 정답이거나
decoder·score·attention이 최종 성능을 올렸다는 증거가 아니다.

## Teacher가 원하는 변화와의 방향 정합은 약하다

Feature 정답이 있는 source9/10의 M0/M1 foreground union에서, 각 모델이 전달한
delta와 정답 M1−M0 delta를 비교했다. Feature element별 제곱합·내적을 모아
집계했으며 자산별 cosine의 단순 평균이 아니다. Source11 teacher는 없다.

| Teacher | 전달 delta / 정답 norm | 정답 방향 cosine | 보정 후 MSE / 무보정 MSE |
|---|---:|---:|---:|
| Original19 | 0.0852 | 0.0752 | 0.994444 |
| SourceReference20 | 0.1126 | 0.0822 | 0.994160 |
| SourceContrast19 | 0.1110 | 0.0893 | 0.992487 |

Contrast의 source-use는 충분히 열려 있고 여러 후보를 사용하지만, 전달 변화는
정답 방향과 낮은 cosine을 보인다. Feature MSE 개선도 약0.75%다. 이 값은
캐시 feature 공간의 진단이며, 해당 feature 정답에 가까워지는 것이 native SS의
최적 task representation이라는 보장은 없다. 생성 geometry의 성능은 별도
native 평가와 joint 학습 결과로 판단해야 한다.

## 세기를 단순히 높이는 해법은 지지되지 않는다

저장된 같은 delta를 스칼라 α로 배율만 바꿨을 때의 foreground feature error를
추가 forward 없이 계산했다. 정답 `t`, 전달 변화 `d`에 대해 다음 항등식을 사용한다.

`MSE(αd−t)/MSE(t) = 1 + α²·||d||²/||t||² − 2α·<d,t>/||t||²`

실제 α=1의 값이 위 표와 정확히 같음을 확인했다. Contrast는 α=1에서0.992487,
α=1.5에서0.997973, α=2에서1.009622다. 이 진단에서 단순 증폭은 정답 feature와의
오차를 늘린다. GT를 사용해 계산한 전역 scalar 최적값은 약0.805지만, 이를 실제
모델에 적용하거나 validation으로 hyperparameter를 선택하지 않았다.
이 곡선은 SS/PBR를 재생성한 성능 곡선도 아니다.

## 실험 판단

현재 증거로 source-use 강제 증가, 무조건적인 temperature 확대 또는 새로운
가중합 arm을 추가하지 않는다. 예정된 source72 확인과 native warm 검증을 거쳐
같은 예산의 warm Original19/Contrast19를 joint SS480에서 비교한다. 입력 설계가
교정 정보의 방향을 개선하는지와 실제 geometry로 전달되는지를 함께 확인한다.

Semantic correspondence 정확도와 개별 decoder·score의 유용성은 아직 별도 검증
대상이다. 이 관측은 No-score=Top-1 feature, No-corr=원래 spatial correspondence의
학습 ablation을 대신하지 않는다. 기존 모듈이 작동한다는 관측을 성능 향상으로
바꾸어 보고하지 않으며, source/target별 cascade metric 및 독립 joint seed와
최종 test508/FID/KID 범위도 유지한다.

Canonical 진단은 `source-feature-teacher-pilot-v1/` 아래
`TRAIN-first4-matching-attention-observer-CPU-v1`,
`VAL32-matching-attention-observer-CPU-v2`,
`VAL32-matching-attention-observability-publication-CPU-v1`에 저장했고 실제 file4
SHA256·size를 확인했다. [WandB 표와 진단 곡선](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/source-teacher112-observability-VAL32-CPU-v1)에 기록했다.
