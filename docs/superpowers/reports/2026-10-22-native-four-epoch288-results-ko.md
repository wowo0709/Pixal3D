# Native SS 288 update: 192의 readout 이득이 유지되지 않음

같은 original MV SS4210/fresh optimizer·TRAIN9149·seed101,
world1/global96/micro96/accumulation1의 네 모델에서288 update가 완료됐다.
첫288개 실제 asset/noise/timestep/global condition trace도 동일하다.
같은32개 source cluster·생성 noise1회 panel을 비교했다.

| 모델 | Global occupancy IoU | Mean asset IoU |
|---|---:|---:|
| Vanilla V1 | .433619533 | .468972916 |
| Physical Top1+R | .430609142 | .467991469 |
| Feature+R | .428528739 | .464853213 |
| Raw Top-K+feature+R | .431503597 | .466493834 |

실제 TP/FP/FN으로 asset/global 지표를 재계산하고 GT hash/shape/표현·source·생성 seed를
짝지었다. Source-cluster bootstrap4000회, mean asset IoU의 결과는 아래와 같다.
단위는 percentage point이며 반복 개발 비교·다중 비교 보정 전95% CI다.

| 비교 | IoU 차이 (%p) | Source-cluster CI95 (%p) |
|---|---:|---|
| Top1+R − V1 | −.098145 | [−.736180, +.498195] |
| Feature+R − V1 | −.411970 | [−1.131547, +.221598] |
| Raw Top-K+feature+R − V1 | −.247908 | [−.971629, +.408843] |
| Feature+R − Top1+R | **−.313826** | **[−.553657, −.086926]** |
| Raw Top-K+feature+R − Top1+R | −.149764 | [−.540817, +.207773] |
| Raw Top-K+feature+R − feature+R | +.164062 | [−.130485, +.428604] |

192 update에서는 feature+R − Top1+R이+.478021%p이고 CI도 양수였으나,
288에서는 방향이 반대이며 CI도 음수다. Raw Top-K의 Top1 대비 차이는288에서 불확실하다.
따라서192의 양수 신호를 안정적인 attention 기여로 채택할 수 없다.
세 source 모델 모두 V1 대비 개선을 입증하지 못했다.

현재 결과만으로 원인을 overfitting, matching 오류 또는 구현 오류 중 하나로
확정하지 않는다. 초기 native output identity와 같은 학습 trace는 확인됐지만,
actual attention/consumer signal observer와 최종128×3 noise 및 독립 TRAIN seed 검사가
더 필요하다. 고정480 budget을 유지하고 특정 epoch의 최고값을 골라 full을 정하지 않는다.
V1의 다른 epoch를 source288과 비교해 효과를 주장하지 않는다.

Canonical evidence는 `native-flow-cold3-v1/checkpoint-evaluation-v2/`
`epoch288-CPU-evidence-v2/actual-inputs-v1/`와 `four-native-paired32-v1/audit.json`이다.
실제 saved NN/source/Adam288와 snapshot24개11,471,169,725byte 및 통계152,713byte를
file4의 실제 SHA/size와 대조했다. CPU 집계이며 독립 saved-weight GPU replay를 대신하지 않는다.

최초288 집계 v1은 이전 스크립트의 trace range193이 남아 guard에서 중단됐다.
실패 로그와 빈 output을 보존하고, range289를 검증하는 v2에서 다시 계산했다.
모델 학습·config·checkpoint는 수정하거나 재시작하지 않았다.
