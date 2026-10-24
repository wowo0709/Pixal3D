# 필수 PBR 두-seed 결과의 자동 결합 연결

2026-10-24. 앞서 실제 큐에 연결한 두 기준 모델의 seed202 학습과 stage snapshot에 이어, 두 seed를 결합하는 CPU 집계도 연결했다. 새 architecture나 GPU 학습 작업은 추가하지 않았다.

## 실제 frozen base 검증

완료된 seed101 Vanilla·Top-1 PBR checkpoint를 원본 initializer와 직접 비교했다. 두 모델 각각 frozen NN tensor700개가 dtype·shape·값 모두 정확히 같고, LoRA tensor는360개였다. 원본 파일의 SHA도 사용자가 지정한 mixed385 multi-view PBR16850 checkpoint mirror와 같았다. 이 작업은 CPU92/93에서 저장된 tensor를 읽어 비교했으며 neural forward나 training update는 수행하지 않았다.

이는 실제 완료된 seed101 두 모델에 대한 증거다. 아직 시작하지 않은 seed202의 frozen 상태를 미리 통과한 것으로 표시하지 않는다. 새 집계는 seed202가 실제 완료된 뒤 같은 검사를 다시 수행한다.

## 실제 연결한 의존성

`required-PBR-controls-two-seed-closed-collection-CPU-v198`를 실제 dispatcher 큐에 추가했다. 두 모델의 자기 seed202 stage snapshot이 모두 성공한 뒤에만 실행한다. 기존 descriptor와 state는 보존했다.

1. 이미 연결한 Vanilla·Top-1 PBR seed202 native1,900-step 학습.
2. 각 학습의 실제 완료를 기다리는 explicit-seed stage snapshot.
3. 새 CPU collector가 네 개의 실제101/202 catalog를 읽고 closed collection plan을 생성.
4. 기존 public two-seed collector로 실제 checkpoint·frozen tensor·네 rank 전체 trace·초기 proof·GT pixel·epoch curve를 검증하고 결과를 file4에 보관.

현재 seed101 catalog는 실제 SHA로 고정했다. Seed202 catalog의 SHA는 파일이 실제로 생기고 stage snapshot이 닫힌 후 계산한다. 미래 checkpoint의 SHA나 성공 상태를 placeholder로 만들지 않는다. 실제 config의 seed·mode·native output이 계획과 일치해야 한다.

Original initializer와 multi-view config, mode별 기존 engineering proof config도 실제 파일에 묶었다. Seed101 engineering reference를 seed202에서 다시 측정한 proof라고 부르지 않는다. Seed202 native run 자체의 초기4개 exact-latent/bounded-render 검증은 첫 update 전에 별도로 수행된다.

## 실행과 해석 범위

새 작업은 CPU-only이며 affinity92/93와 낮은 priority를 사용한다. NN을 구성하거나 학습하지 않고, public collector가 실제 model byte와 frozen tensor를 확인한다. Child PID/birth와 session drain을 기록한 뒤 결과를 닫는다.

이 집계에서 `selected_mode=top1-only`는 두 필수 기준을 deduplicate하는 reader 인자다. 최종 PBR 구조를 채택했다는 뜻이 아니다. Context와 최종 wrapper는 `REQUIRED_CONTROLS_TRAINING_REPRODUCIBILITY_ONLY`를 명시한다. 다른 PBR mode를 최종 후보로 선택하면 해당 mode의 실제 두 seed를 포함한 추가 결합 결과가 필요하다.

Scope는 fixed4·GT shape·target camera8의 stage 반복이다. 생성 geometry, source preservation, 전체 pipeline의 독립 training-seed 효과로 보고하지 않는다. 두 PBR202 학습, 두 stage readout과 새 combined collector는 아직 pending이다.

실제 등록 후 큐와 source pin17개를 다시 확인했다. 미래202 artifact를101 데이터로 대체하지 않았으며, 세 진행 중인 NN은 원래 PID/birth를 유지했다. Public numerical source는 이번 연결에서 변경하지 않았다.

실제 file4 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/PBR-actual-frozen-base-comparison-v198/`와 `required-PBR-two-seed-collection-v198/` 아래에 있다. 원본 byte/tensor 비교와 prelaunch·queue·applied·실제 큐 검증 closure를 포함한다. 원래 모듈 study·joint DEV·최종 선택과 TEST508·10개 metric은 계속 남아 있다.
