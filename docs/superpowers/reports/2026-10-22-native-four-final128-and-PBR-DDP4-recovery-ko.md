# Native SS 최종128×3 결과와 PBR 네 GPU 복구

원본 multi-view SS4210에서 fresh optimizer로 시작한 네 모델이 모두480 update를
완료했다. TRAIN9149, seed101, global96/micro96/accumulation1이며 첫480 update의
실제 asset/noise/timestep/global-condition trace가 네 모델에서 같다.
최종 저장480을 고정하여128개·93 source cluster·생성 noise3회로 비교했다.
생성 noise3회는 독립 학습 seed3회가 아니며, 이 패널은 개발 패널이다.

## 최종 SS 비교

주 지표는 asset별 voxel IoU를 세 noise에 걸쳐 평균한 뒤 asset 평균한 값이다.
아래 값은 voxel을 전체 합산한 pooled IoU와 다르다.

| 모델 | Mean asset IoU ↑ |
|---|---:|
| Vanilla V1 | .46735504 |
| Physical Top-1+R | **.47260025** |
| Feature+R | .47047280 |
| Raw Top-K+feature+R | .47207818 |

차이와 신뢰구간은 percentage point다. Source cluster 단위 paired bootstrap을
사용했으며, 마지막 열은 주 IoU의 여섯 비교에 대한 family 보정 구간이다.
반복 개발 패널에서 수행한 모든 과거 실험 전체를 보정한 값은 아니다.

| 비교 | 차이 (%p) | 개별95% CI (%p) | 여섯 비교 family95% CI (%p) |
|---|---:|---|---|
| Top-1+R − V1 | **+.524521** | [+.155519, +.920561] | **[+.048024, +1.053614]** |
| Feature+R − V1 | +.311776 | [−.046034, +.666883] | [−.167820, +.804451] |
| Raw Top-K+feature+R − V1 | +.472314 | [+.089903, +.879827] | [−.048874, +1.002791] |
| Feature+R − Top-1+R | −.212745 | [−.438398, +.027599] | [−.511194, +.109903] |
| Raw Top-K+feature+R − Feature+R | +.160538 | [−.076583, +.394079] | [−.172040, +.471408] |
| Raw Top-K+feature+R − Top-1+R | −.052207 | [−.300243, +.213977] | [−.382083, +.289465] |

현재 이 조건에서 가장 강한 대조군은 Top-1+R이다. 이 조합의 V1 대비 이득은
관측됐지만, 이것만으로 semantic correspondence와 retargeting을 각각 분리해
입증한 것은 아니다. 추가 feature attention 및 raw Top-K의 Top-1+R 대비 이득은
확인되지 않았다. Ray-wise QKV의 필수성이나 matching decoder/score 각각의
필요성을 주장하지 않는다. 이전32개 중간 epoch의 양수 차이로 최종 결론을
대체하지 않는다. 독립 학습 seed와 native spatial 대조, full-cascade 및 test508은 남아 있다.

실제 요약은 `native-flow-cold3-v1/physical-Top1-control-v1/checkpoint-evaluation-v1/`
`four-model-final128-summary-v1/{audit.json,file4-closure.json}`에 있다.
세 source 모델의32개 predicted SS archive와 모든128×3 GPU 평가도 완료됐다.
원래 비결정적 epoch96 exact replay 실패 이력은 그대로 보존했고, 결정적 pooling을
사용한 독립 load 반복 검사를 새 프로토콜로 통과한 뒤 원래 최종 평가를 연결했다.

[W&B: 네 모델 최종128×3 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-four-with-Top1-final128x3-s101-v1).

## 실제 PBR 메모리 초과와 복구 범위

DDP2/global24/micro12/accumulation1에서 실제 최대 TRAIN31900-token asset을
반복하는 capacity 검사를 했다. Vanilla는 세 update, 네 asset 렌더 validation,
그 다음 네 번째 update 및 frozen/DDP/Adam 검사를 통과했다. 최대 allocated는
88.604963 GiB다. Source가 있는 Top-1-only, Top-1+feature, raw Top-K+feature는
모두 실제 OOM으로 종료했다. 실패를 통과로 바꾸지 않았다.

새 `PBR-native-DDP4-micro6-capacity-v1`에서 네 모델 모두 world4/micro6,
global24/accumulation1을 검사한다. 원본 NN16850·모델·데이터·objective는
그대로이며, 최대 token4 update와 중간 render4 이후 update까지 통과해야 한다.
계획을 추가했다는 사실은 실제 GPU 검사를 통과했다는 뜻이 아니다.

새 `PBR-native-science-runtime-v1/DDP4-recovery-v1`에는 실제 검사 통과를
필수로 요구하는 네 fresh1900 학습 설정을 준비했다. 각 모델5에폭/1900 update,
매380 update의 고정4 렌더 metric·GT/Pred/Diff·W&B와 canonical checkpoint 저장은
유지했다. 이전에 시작하지 않았던 과학 실험 출력만 사용한다.

새 alias411개로 네 학습, generated-shape PBR isolation128개, 렌더128개,
공식 perceptual metric과 paired 집계, bounded cache retirement, 역사적
target-patch query 재현 세 arm까지 연결했다. 기존 실패·HELD 이력과 bound
설정·명령은 수정하지 않았다. 새 설정에 맞춰 미래 실제 config hash 및
completed-dependency 참조를 갱신했다. 네 GPU 학습은 모델별로 순차 실행하고,
이전 모델의 실제 완료·file4 검증·중간 checkpoint cache 정리 뒤 다음 모델을
진행한다. 원본 데이터와 최종 checkpoint는 유지한다.

해당 새 prelaunch와 queue snapshot은 실제 file4 SHA/size를 검증했다.
관련 capacity/science/native-cascade CPU 검사40개가 통과했다.
새 PBR 성능 개선, full-cascade8 metric, 독립 seed 또는 test508/all10 metric의
완료를 주장하지 않는다.
