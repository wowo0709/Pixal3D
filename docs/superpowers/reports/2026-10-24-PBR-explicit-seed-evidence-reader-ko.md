# 실제 TRAIN seed를 유지하는 PBR 결과 reader 연결

2026-10-24. 현재 학습을 유지하면서, 실제 완료된 PBR stage와 독립 seed202 결과를 읽는 후처리를 연결했다. 모델·optimizer·sampler·렌더링 numerical source는 변경하지 않았다.

## 초기 검증 계약을 그대로 적용

Native PBR은 이미 `exact-input-noise-latent-bounded-render-v1` 정책을 사용한다. 입력 RGB·noise·GT·camera·생성 PBR latent hash는 정확히 같아야 한다. Renderer metric만 기존 absolute bound인 PSNR0.01dB, SSIM0.0002, LPIPS0.0003을 적용한다. 그 외 metric은 정확한 일치를 요구한다.

기존 독립-seed 집계는 초기 render metric의 비트 단위 일치를 추가로 요구하고 있었다. 실제 Raw PBR의 초기 latent가 같아도 이미 허용된 renderer 차이를 이유로 이후 집계를 막을 수 있었다. 새 reader는 실제 native producer proof를 원래 cold32 reference와 다시 계산해 일치하는지 확인한다. 초기 record의 metric 값을 바꾸지 않고, paired 모델 사이에서도 원래 bound를 확인한다. RGB·noise·camera·GT·latent의 일치 조건은 유지한다.

실제 Raw 초기4개 기록으로 검사했다. Latent와 입력은 정확히 같았으며, V1 초기 기록 대비 최대 render PSNR 차이는0.00247192dB, 최대 crop LPIPS 차이는0.00014541로 원래 bound 안이었다. 실제 producer proof도 다시 계산한 결과와 같았다. 이 검사는 초기4개 데이터만의 검증이며 Raw1,900 update 완료나 seed202 재현을 의미하지 않는다. 진행 중인 native log 전체를 닫힌 artifact라고 취급하지 않고, 초기 producer 파일의 immutable copy와 이미 닫힌 validation snapshot을 사용했다.

## Seed202를 seed101로 바꾸지 않는 stage snapshot

새 public entry `snapshot_selected_pbr_stage.py`는 실제 config의 integer training seed와 네 rank completion의 seed가 계획과 같은지 확인한다. Seed202도 원래 값202를 유지한다. 기존 seed101 전용 snapshot helper를 재해석하지 않는다.

다음 실제 파일을 읽고 확인한다.

- 완료된 실제 science job, 동일 child PID/birth의 성공 terminal 및 현재 child 부재. Seed202는 producer의 session-drain 증거도 요구한다.
- 네 rank의 원래 PBR16850 initializer·1,900 update·global24·micro6·accumulation1·frozen/DDP 상태.
- 실제 native config와 자기 NN/source/Adam final archive. 모델 binding도 실제 seed를 유지한다.
- 네 rank별1,900개 flat micro6 training trace, runtime의 finite loss와 batch, actual raw SHA와 canonical parsed-row digest.
- 실제 여섯 epoch의 고정4개 validation log, GT PNG pixel hash, 원래 초기 검증 proof와 cold32 reference.

두-seed collector도 snapshot row가 실제 closed native validation log와 같은지 다시 읽는다. Metric normalization이나 seed alias를 만들지 않는다. Scope는 여전히 fixed4·target8·GT-shape stage 진단이다. 생성 geometry나 source preservation의 독립 pipeline 재현으로 보고하지 않는다.

## 실제 실행과 검증

관련 CPU 검사 **114개 통과**. Seed101/202·네 mode, 틀린 seed·부분 학습·잘못된 batch/종료 증거의 거부, exact latent 유지와 원래 renderer bound, 실제 curve 값 보존을 확인했다.

실제 완료된 Vanilla와 Top-1 PBR 두 개에서 새 CPU worker를 실행해 각 stage의 네 rank·1,900 update·실제 config/NN/archive/PNG/초기 proof를 읽었다. 두 모델 모두 seed101을 유지했고 결과를 실제 file4에 복사해 remote SHA256/size를 확인했다. 이 실제 실행을 admission 증거로 사용해 다음 두 CPU 작업을 큐에 추가했다.

- `PBR-s202-vanilla-explicit-stage-snapshot-CPU-v197`
- `PBR-s202-top1-only-explicit-stage-snapshot-CPU-v197`

각 작업은 앞서 연결한 자기 PBR seed202 학습의 실제 성공 완료를 기다린다. 추가한 GPU 작업은0개다. 기존 NN과 큐 state는 유지했다. 이 두 작업은 아직 pending이며 독립 seed 결과는 미완료다.

새 worker와 관련 reader source는 이 큐의 SHA로 고정했다. 후속 변경은 해당 pin을 먼저 확인해야 한다. PBR seed202 학습의 native source55개와 기존 학습의 source pin은 변경하지 않았다.

실제 증거는 file4 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `PBR-explicit-seed-stage-readout-v197/`와 `PBR-initial-render-policy-actual-preflight-v197/`에 있다. 실제 seed101 snapshot·model-input catalog·초기 proof·prelaunch·큐 연결 및 검증 closure를 포함한다. 최종 두-seed 결합 집계, 모델 선택·TEST508과10개 metric은 아직 완료되지 않았다.
