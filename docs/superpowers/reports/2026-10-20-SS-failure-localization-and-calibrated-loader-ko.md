# SS 하락의 경로별 진단과 calibrated matching loader

## 지금까지 분리한 문제

Actual SS VAL32의 매칭 진단은 추가 Q/K가 pretrained utility 분포보다 정답 근처 확률을 낮춘다는 결과를 보였다. 그러나 Direct192→288의 매칭 mass는 거의 변하지 않았는데 SS IoU는0.44251782→0.43913141로 하락했다. 따라서 매칭 선택 문제만으로 최근 재구성 하락을 전부 설명하지 않는다. 새로운 feature 전달 경로도 따로 검사했다.

이 리포트의 경로 검사는 **동일한 trained NN/R/conditioner step288 가중치**에서 수행한 saved-weight intervention이다. 같은32개, GT hash/seed/source cluster, generation batch8,12 sampling step를 유지했다. 새 학습 arm 또는 whole-module retraining 결과가 아니다. Retargeter는 모든 조건에 있다.

## 실제 매칭 drift

Native256 partition의 VAL32,1,614개 visible-source relation에서 측정했다.

| 분포 | Near1 정답 mass | Near1 Top1 relation coverage |
|---|---:|---:|
| Pretrained utility7 |43.8084%|47.8315%|
| Direct192 |43.0140%|46.2825%|
| Direct192 Q-weight0 |43.5198%|46.9641%|
| Direct288 |43.0787%|47.3358%|
| Direct288 Q-weight0 |43.7646%|48.0793%|

Direct288−utility7 near1 mass는 -0.7296pp, source-cluster95% CI [-1.4043,-0.0151]pp다. Direct288 Q-zero−Direct288은 +0.6858pp, CI [+0.0911,+1.2676]pp였다. 반면 Direct288−Direct192는 +0.0648pp, CI [-0.3174,+0.4010]pp로 거의 유지됐다. 정답을 더 잘 고르는 방향으로 SS loss가 Q/K를 교정했다는 증거는 부족하다. 이 확률 진단은 SS 품질의 인과적 module effect 자체를 증명하지 않는다.

### Denominator 검사

한 target ray에 최대3개 source-view 정답이 있을 수 있으므로 relation-level coverage와 확률 mass의 상한은 다르다. 실제 label을 다시 읽어 source-view positive sets가 서로 겹치지 않는지 검사했다. VAL32는 native target ray2,511개, GT 관측 가능한 unique ray1,517개, visible-source relation1,614개였다.

동일 denominator의 ray-union 기준으로 utility7의 near1 Top64 coverage는88.6618%, Top1 coverage는50.8899%, 정답 union mass는46.6095%다. Direct288 union mass는45.8332%였다. 이 표본에서는 대부분 ray가 한 source-view 정답만 가지므로 denominator 교정 후에도 후보 활용의 차이는 남는다. 단순히88.7% relation coverage와43.8% relation mass를 같은 상한처럼 해석하지 않는다. Near1 relation mass의 실제 oracle ceiling은83.3333%였다. 이 상한도 SS IoU의 상한이 아니다.

## Native generation replay와 경로 검사

먼저 step288 standalone normal을 실행했다. 모든32개의 seed/source/GT hash 및 TP/FP/FN과 metric이 실제 학습 validation과 정확히 일치했다. Predicted3D mask 전체의 byte-level 동일성까지 주장하지 않는다.

| 고정288 가중치의 출력 조건 | SS IoU |
|---|---:|
| Normal: appearance residual + DQ projection |0.43913141|
| DQ projection만 mute, appearance 유지 |0.43601077|
| Appearance residual과 DQ projection 함께 mute |0.44760567|
| Appearance residual만 mute, learned DQ 유지 |0.44905614|
| 같은288 matched R control, 별도 학습 |0.44592041|

추가 두 경로를 함께 끄면 normal보다 **+0.8474pp**, CI [+0.3231,+1.3969]pp 회복됐다. DQ만 끈 차이는 -0.3121pp, CI [-0.7203,+0.1074]pp라 그 단독 효과를 확정하지 않는다. 두 경로를 끈 결과−DQ만 끈 결과는 +1.1595pp, CI [+0.7265,+1.5888]pp다. 적어도 DQ-off 조건에서는 추가 appearance feature가 해로웠다.

두 경로 off−matched R은 +0.1685pp, CI [-0.1160,+0.4729]pp다. NN/R 공통 가중치가 전부 망가져 하락했다고 설명할 근거는 약하다. 이것만으로 새로운 full model을 채택하지 않는다.

추가 appearance output만 끄고 **확률/RAW query/학습된 DQ projection을 유지하는 네 번째 조건**도 실제32개 평가와 file4 확인을 완료했다. CPU 검사는 appearance residual이0이 되는 동안 matching probability, RAW DQ input, nonzero DQ projection output, 모든 가중치가 그대로인지 확인했다. Actual native 평가에서도 nonzero DQ projection이 실행됐다.

이 조건−normal은 **+0.9925pp**, CI [+0.4403,+1.6057]pp다. 두 경로 off 대비 DQ를 유지한 차이는 +0.1450pp, CI [-0.3756,+0.6180]pp라 DQ 자체의 추가 이득은 확정하지 않는다. Matched R 대비는 +0.3136pp, CI [-0.1360,+0.7247]pp, original R288 대비는 +0.7269pp, CI [-0.0178,+1.4137]pp다. Vanilla V1 대비는 +1.7184pp, CI [+0.2152,+3.3428]pp지만 R의 기여와 새 matching/query 기여를 분리해야 한다.

따라서 **이 checkpoint와 panel에서는 추가 transformed appearance 경로가 출력 하락에 관여한다**는 결론을 지지한다. 그 내부 원인이 feature-space 변화인지 학습 과적합인지는 아직 구분하지 못했다. 네 조건 모두 하나의 checkpoint에 대한 sensitivity 검사이며, retargeting×query 유무의2×2 학습을 추가한 것이 아니다.

세 generation 검사의 실제 peak allocated는5.8536GiB였다. Global training은 그대로 유지했고 cohost inference의 allocator를30%로 제한했다. 실제 file4에288 complete NN/base/LoRA 및 conditioner2,774,390,934 bytes와 원래 validation/config의 SHA/size를 확인한 뒤 실행했다.

## 배제한 optimizer 가설

추가 branch가 global gradient clipping을 크게 유발해 기존 NN/R update를 약화시켰는지 로그를 검사했다. R, input feature, input DQ, Direct의 step97–288 각192개 완료 로그에서 clipping threshold는 모두1.0이었다. 최대 gradient norm은 각각0.03017/0.03846/0.04810/0.12892이며 threshold를 넘은 step은0개다. 이 관측 구간의 하락을 clipping으로 설명할 근거는 없다. Optimizer나 clipping 정책을 바꾸지 않았다.

## 구현한 calibrated matching initializer

새 loader는 실제 Q/K fit의 pure9 state를 cold reference와 비교한다. Query를0으로 초기화한 reference에서 **Q/K 두 개만** 바뀌었고 utility5와 value/output2가 동일해야 한다. 입력 파일 SHA, shape, finite 값을 확인한다. Transfer하는 matching tensor는 utility5+Q/K2, 총7개다. Native SS/PBR value/output projection2개는 해당 stage의 width와 초기화를 유지한다.

`CalibratedDirectRayRetargetingConditioner`와 physical NoDQ variant는 기존 forward를 바꾸지 않고 이 matching 초기화만 추가한다. Source는 별도 파일에 두어 frozen training 경로를 수정하지 않았다. CPU3개 검사는 SS1024/PBR2048 width의 feature projection 보존, 잘못된 SHA 및 matching 외 tensor 변경 거부, nonzero R 아래 초기 함수 일치와 utility 재로딩에 의한 덮어쓰기를 검출했다. 처음 scalar hash 처리 오류는 CPU 검사에서 발견해 고친 뒤3개 모두 통과했다.

실제 보조 fit epoch1 checkpoint를 engineering initializer로 export해 native SS micro2에서 검사했다. 초기 loss와 evidence는 기존 R reference와 정확히 같았고, loss는0.10466600209474564였다. 임시3 update까지 새10개 tensor가 모두 실제 gradient를 받았으며 NN/R은 그대로였다. Max allocated5.6151GiB, 측정102.76초, native exit0 및 실제 file4 SHA를 확인했다. Prototype은 폐기한다.

**이것은 fixed7 candidate의 scientific 학습 또는 fullbatch continuation/capacity 증거가 아니다.** Epoch1이 실제 SS에 적용됐다고 보고하지 않는다. 고정7에폭 결과의 자체 native proof, common366 Adam/RNG continuation, 별도 DDP2/global96/micro48/accumulation1 capacity 및 실제 초기 validation은 여전히 필요하다.

## 다음 결정

Q/K 보정은 후보 선택을 다룬다. 현재 결과는 그것만으로 추가 appearance 전달 문제까지 해결된다고 보장하지 않는다. 보조 fit의 epoch0/1/2/3 TRAIN calibration near1 mass는45.3326/45.5663/43.5590/43.2869%였고 training loss는 감소했다. 과적합 징후로 보며 epoch1을 사후 선택하거나 중간 checkpoint를 scientific initializer로 적용하지 않는다. 고정 epoch7과 실제 validation 비교를 끝까지 기록한다.

같은 epoch384 중간 SS IoU는 matched R0.44748878, input feature0.44793061, input DQ0.44877649, Direct0.44038553, original R0.44988546, V10.43429997였다. Direct−matched R은 -0.7103pp, CI [-1.5391,+0.2239]pp라 이 panel에서는 개선을 보이지 않는다.

Epoch288에서 선택한 appearance-off 조건을 **미리 고정한480 checkpoint의128개×3 generation noise와32 replay**에서 검증하도록4개 evaluation 및 strong original R/Vanilla 비교 summary, 총5개 작업을 연결했다. 같은 학습 가중치를 쓰는 inference condition이며 새 학습 arm5개가 아니다. 기존 Full/NoDQ/scalar/PBR job은 그대로 두었다. 중간 epoch나32개에 맞춰 full 모델을 확정하지 않는다.

Appearance 변환의 문제가 유지된다면, SS에서는 native Top1 appearance carrier를 유지하고 semantic Top-K 확률을 RAW deformation query에 사용하는 구조가 다음 후보가 된다. 이때 임의의 추가 value/output projection을 **실제로 제거**하고 기존 R에 geometry correspondence를 전달하는 구현을 검토한다. Source feature 관측값을0으로 바꾸는 arm이 아니다. 이 후보는 아직 구현·학습 완료된 모델이 아니며, 재학습으로 효과를 입증해야 한다. Matching을 계속 SS loss로 갱신할지 auxiliary fit 이후 고정할지도 같은 source/budget 아래 확인할 사안이다.

기존 physical NoDQ 및 scalar composer, Direct/DQ fixed480 평가, strong original R/V1, decoder/score/correspondence controls와 PBR matched continuation은 유지한다. PBR appearance 경로가 SS와 같은 경향이라고 가정하지 않는다. 최종 module ablation은 native-spatial NoCorr, Top1-feature NoScore, physical input removal의 실제 정의와 학습 budget을 지켜야 한다. Decoder-context bypass는 D1 proposal/score bank까지 제거한 ablation이 아니다.

## Artifact

Root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`이다.

- `prior-anchored-QK-v1/SS-matching-drift-v1`: 실제192/288 matching head와 Q-zero 진단6개, source bootstrap.
- `prior-anchored-QK-v1/ray-union-denominator-audit-v1`: 실제 ray/relation denominator 및 oracle ceiling.
- `direct-retargeting-v1/intermediate288-clip-audit-v1`:4개 arm의 동일192-step gradient window.
- `direct-retargeting-v1/intermediate288-weights-v1`: 실제288 가중치와 원래 config/validation snapshot.
- `direct-retargeting-v1/intermediate288-path-probes-v1`: normal 및 두 경로 off, DQ-only off; normal replay 검증과 paired CI.
- `direct-retargeting-v1/intermediate288-appearance-plan-v1`: appearance-only off의 source/CPU 검증 및 실행 recipe.
- `direct-retargeting-v1/appearance-suppressed-final-plan-v1`, `appearance-suppressed-final-queue-v1`: fixed480/128×3, replay32, strong R/V1 비교5개 followup.
- `prior-anchored-QK-v1/calibrated-native-engineering-v1`: 실제 epoch1 initializer, CPU3 및 native micro2 초기 함수/gradient 검사. Scientific candidate가 아니다.

모델 채택이나 성능 목표 달성은 아직 선언하지 않는다.
