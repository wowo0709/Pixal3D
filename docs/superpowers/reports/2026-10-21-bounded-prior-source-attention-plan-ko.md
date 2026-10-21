# Matching prior를 보존하는 bounded source-ray attention

2026-10-21 UTC. `wowo0709/feature-retargeting`.
이번 변경은 성능 향상이 확인된 full 모델이 아니라, 실제 실패 진단으로 좁힌
단일 설계 가설이다. 기존 실험·음성 결과·정상 학습은 보존한다.

## 실제 근거와 해석 한계

최종480-step source-ray QKV NoDQ의128개×3 noise IoU는0.438810이다.
V1 대비+0.006160, source-cluster CI95 [+0.002125,+0.010456]이다.
Member Top-1 대비+0.002547, CI95 [−0.001047,+0.006393],
기존 mean-query spatial 대비+0.001353, CI95 [−0.003071,+0.005820]다.
따라서 V1 대비 개선과 별개로 QKV/semantic matching의 추가 이득은 확정하지 않는다.

같은 최종 weight·32개·noise0에서 다음 inference intervention을 실제 실행했다.
후보·matching prior·Top-1 carrier·R target context·source-use hash가 조건 간에
정확히 같고, 정상32개의 voxel count/GT/noise는 원래 최종128 평가와 일치한다.

| 변경 | IoU | 원래 모델 대비 | CI95 |
|---|---:|---:|---|
| 원래 NoDQ | 0.442638 | — | — |
| Foreground utility query만 사용 | 0.443025 | +0.000386 | [−0.000489,+0.001631] |
| Matching prior만으로 feature readout | 0.445378 | +0.002739 | [−0.000433,+0.006413] |
| Readout의 log prior만 제거 | 0.442242 | −0.000396 | [−0.004201,+0.003683] |
| 추가 feature correction0 | 0.439814 | −0.002824 | [−0.011861,+0.005895] |

모두 CI가0을 포함한다. 독립 training seed·재학습 ablation 또는 rendering 이득이
아니다. 다만 foreground query보다 prior-only 변경의 점 추정치가 크고, 원래
attention은 prior의 평균 effective K10.83을5.00으로 줄였다. 실제 feature output의
RMS는0.1057이며 prior-only로 바꾸면0.0183만큼 달라진다. 이 관측을 근거로
추가 대규모 조합 탐색 대신 content 재가중의 초기화와 범위를 먼저 수정한다.

원본 closed 결과는 result root의
`historical-query-reproduction-v1/causal-diagnostics-report-v1/actual-v1`에 있으며,
GT/Pred/difference panel·효과 그래프·CSV/PDF를 함께 보존했다.
[실제 W&B 진단](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-member-feature-causal-readout-final480-s101-v1)에 동일 자료를 게시했다.

## 설계 한 가지

원래 four-head의 content/utility logit 함수를 `f_theta(x)`라 하고, utility7을
불러온 직후의 동일 함수를 frozen buffer로 저장해 `f_initial(x)`라 한다.
Matching prior `p`와 같은64개 후보는 유지한다.

```
d = f_theta(x) - f_initial(x)
d = d - sum_k p[k] * d[k]
attention = softmax(log(p) + tanh(d))
```

Logit residual bound는1 하나로 고정하며 sweep하지 않는다. 두 후보 사이의
matching-prior odds에 대한 learned log-odds 보정은 최대2다. 처음에는 `d=0`이므로
모든 head의 확률이 matching posterior와 정확히 같다. Q/K를0으로 만들거나
utility7 가중치를 버리는 방식이 아니다. Frozen 초기 함수는 gradient를 받지 않지만,
현재 함수에는 derivative가 남으므로 별도 scalar0 gate 없이 content가 학습된다.
Frozen 입력이 아닌 경우에도 reference 함수의 입력 gradient는 유지한다.

Reference는 Q/K와 utility의7개 tensor를 **buffer**로 직렬화한다. 기존 head9개,
전체 source19개 trainable tensor와 native LoRA360개는 유지한다. V/output 및
R/source-use/semantic Top-1 carrier/direct residual delivery는 원래 경로를 쓴다.
후보 확률이0인 유효 이미지 key도0으로 유지하며 `inf-inf`를 gradient로 전달하지 않는다.
Checkpoint를 읽을 때 선언한 logit bound와 저장된 값이 다르면 거부한다.

Query는 원래의 source-camera ray 단위다. Utility 입력은 edited target의
single-view `target['proj']`이며 MV4 평균이 아니다. R target context, 원래 camera
projection, 모든 ray voxel에 대한 write, native four-view carrier는 그대로다.
Visibility mask·GT depth·GT correspondence를 새 model input으로 사용하지 않는다.
Foreground utility 수정안은 이번 arm에 섞지 않는다.

이는 과거 target-patch query에서 Q를0으로 두고 두 projection만 auxiliary GT loss로
학습했던 prior-anchored QK512/7-epoch 실험의 재실행이 아니다. 그 실험은 최종
calibration gate를 통과하지 못했고 배포하지 않았다. 이번에는 source-ray의 원래
19개 tensor를 native SS loss로 jointly 학습하며 auxiliary label training을 추가하지 않는다.

## 실행과 검증

공유 NN 파일이나 기본 registry를 수정하지 않고 새 모듈과 process-local runtime을
사용한다. 원래 source tensor·RNG 초기화, utility 로딩, checkpoint buffer 복원,
BF16 초기 posterior, 모든 source gradient, projected-V-only 대조와 예외 복구를
CPU에서 검사했다. 관련37개 검사가6.59초에 통과했다. GPU 증거는 별도로 요구한다.

1. 자체 native proof: 초기32개 V1 prediction/count/noise/GT identity, 첫 native
   loss의 gradient, 활성 source19개 gradient, query/source 결손 fallback,
   미관측 target voxel로도 전달되는 원래 projection 및 native FFN parity.
2. 자체 capacity: global96/micro96/accum1로 실제3회 update,379개 Adam state의
   step3, source19개 positive finite gradient와 frozen parameter/buffer 불변.
   Capacity 가중치나 optimizer를 과학 실험으로 이어 쓰지 않는다.
3. 통과 후 fresh seed101, 원래 multi-view LoRA SS4210,480 update를 사용한다.
   Data/camera/latent/sampler/native Adam·AMP·clip/LR/batch budget은 원래 NoDQ와 같다.
   초기 및 매96 step validation32, 최종128×3 noise와 projected-V 대조32를 실행한다.
4. 원래 NoDQ와 paired 비교해 parameterization의 효과를 판단한다. 긍정 후보는
   동일32개 full cascade, 고정 Shape512/1024·V1 PBR의 image/geometry metric,
   독립 training seed와 필요한 matched component control을 추가 확인한다.

No-score는 원래대로 실제 semantic Top-1 feature arm이다. Prior-only frozen 진단은
그 학습 arm의 대체가 아니다. No-corr는 native spatial correspondence다. 새 bound를
사용한 모델에서 semantic-vs-spatial 효과를 주장하려면 같은 bound의 spatial 대조가
필요하다. Decoder의 독립 효과도 별도 matched 대조 없이 주장하지 않는다.

원래11개 SS·14개 모델 dev cascade·두 원래 zero-shot stack·PBR3개·조건부 seed202·
필요시 faithful target-patch/R96 재현과 최종508개/10metric 범위는 유지한다.
새 결과는 별도 alias·출력 경로로 연결하며 기존 결과를 새 모델 결과로 재명명하지 않는다.
V1 대비 개선만으로 full을 정하지 않고 test508을 설계 선택에 사용하지 않는다.
