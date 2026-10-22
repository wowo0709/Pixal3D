# Native SS480의 matching score·Top-K value 소비 진단

실제 raw Top-K+feature+R native480 checkpoint를 고정하고 기존 DEV32·noise1회에서
정상 재현 및 네 가지 inference intervention을 완료했다. 정상 재현은 기존 own
archive32와 자산별 생성 결과·metric·trainable weight identity가 정확히 같았다.
정상 실행 후 네 개 개입을 GPU0–3에서 각각 실행했고, 모두 실제 종료0 및 file4
검증을 통과했다. Optimizer update는0이다. 이는 재학습 ablation이 아니다.

## 최종 출력 비교

IoU는32개 자산별 IoU의 평균이다. Pooled IoU와 다르며, 최종128×3 결과를
대체하지 않는다. 마지막 열은 개입−정상의 source-cluster bootstrap4000회,
seed20261023에 따른 exploratory 개별95% CI다. 다중 비교 보정은 하지 않았다.

| 저장된 같은 모델에 적용한 개입 | Mean-asset IoU | 정상 대비 차이 (%p) | 차이95% CI (%p) |
|---|---:|---:|---|
| 정상 | .47801183 | — | — |
| 후보 안에서 V 대응만 순열 | .47849356 | +.04817 | [−.23198, +.34841] |
| Matching prior만 사용, learned 재가중 제거 | .47901775 | +.10059 | [−.07876, +.27900] |
| Feature 집계 확률을 Top-1 하나로 고정 | .47633534 | −.16765 | [−.45344, +.10828] |
| Learned value transform만 제거 | .47533650 | −.26753 | [−1.06648, +.54390] |

네 IoU 차이의 CI 모두0을 포함한다. Learned QK/utility 재가중이 기존 matching
prior보다 도움이 된다는 근거는 얻지 못했다. 반대로 prior-only가 우월하다는
결론도 아직 아니다. Top-1-only와 value transform 제거는 평균 IoU가 낮아졌지만
현재 패널에서 안정적인 IoU 이득을 입증하지는 않는다.

보조 지표에서 Top-1-only recall 차이는 −.37309%p,
CI[−.72726, −.04676]%p였다. Value transform 제거의 precision 차이는
−1.28090%p, CI[−2.24485, −.49609]%p였다. 두 결과는 보정되지 않은 탐색적
관측이며, IoU의 불확실성이나 전체 모집단의 성능 검증을 대신하지 않는다.

## 연결·가중치 소비에 대해 실제로 확인한 것

모든 개입의 네 batch에서 후보 indices·matching prior·Top-1 carrier·source use·
fusion scale·source row·CPU/CUDA RNG·parameter version이 정상 shadow와 같았다.
각 개입은 실제 R 입력 또는 R 출력을 바꿨다. 가중치가 갱신됐다는 이전 검사보다
강한 실제 forward 소비 근거다. 정상의 projected correction/native feature RMS
비율은 batch별5.10–5.64%였고, target native/global object는 보존됐다.

정상의 유효 ray/head 수로 가중한 통계는 다음과 같다. 이 단위들을 독립 자산으로
세지 않으며, GT matching 정확도로 해석하지 않는다.

| 정상 실행의 통계 | 값 |
|---|---:|
| Matching prior의 Top-1 확률 질량 | .325435 |
| Matching prior의 유효 후보 수 | 10.8305 |
| Learned head의 유효 후보 수 | 9.3684 |
| Learned head가 matching Top-1 밖에 둔 질량 | .690666 |
| Learned head의 prior 대비 KL | .186596 |

따라서 경로 전체가 꺼졌거나 attention이 완전히 Top-1로 붕괴했다는 설명은
이 실행과 맞지 않는다. 그러나 source 보정이 작동한다는 것과 정확한
correspondence별 value를 유용하게 활용한다는 것은 별개의 주장이다.

## 개입의 정확한 의미와 해석 제한

`semantic-prior-only`는 matching score를 유지하며 추가 QK/utility 재가중을
제거한다. Uniform NoScore가 아니다. `top1-feature-only`도 uniform이 아니라
정확히 하나의 correspondence를 선택한다. 새로 학습한 physical Top-1+R arm과
동일한 실험으로 취급하지 않는다.

`no-transformed-value`는 readout head의 learned `change`를0으로 만들어
R의 입력에서 해당 value transform만 제거한다. Raw Top-K weighted appearance와
retargeting adapter 자체는 유지된다. R을 제거한 ablation으로 해석하지 않는다.

Wrong-V는 semantic 후보 집합 안에서 값의 대응만 순열하고 Top-1 carrier와
use를 유지한다. 후보끼리 비슷할 수 있으므로, 유의한 하락이 없다는 사실만으로
semantic retrieval이 쓸모없다고 결론 내리지 않는다. Spatial NoCorr나 decoder
제거 실험도 아니다. 이 결과를 단순 연결 오류보다 집계·task 활용을 살펴볼
근거로 사용하되, 특정 실패 원인이 확정됐다고 주장하지 않는다.

현재 강한 기준은 native128×3의 Top-1+R이다. 이 기준을 넘지 못한 사실은 그대로
유지한다. 새 QK scale/구조를 임의로 바꾸거나 작은 양수 차이만으로 모델을
채택하지 않고, 진행 중인 PBR 세 source arm 및 DEV full cascade의 실제 결과와
함께 다음 설계와 최종 SS/PBR 조합을 결정한다.

실제 결과는 `native-flow-cold3-v1/native480-readout-diagnostics-v1/`
`summary-v1/{audit.json,file4-closure.json}`이다.
[W&B 진단 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native480-readout-controls32-v1).
