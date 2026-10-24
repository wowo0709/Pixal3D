# Decoder-cosine SS와 학습된 Top-1 PBR의 실제 joint DEV 비교

2026-10-24. 실제 완료된 SS480 checkpoint의 DEV128×noise3 case macro IoU에서 가장 높은 `C_D1_COS`를 기존 학습된 `top1-only` PBR1900과 조합하는 실제 DEV32 평가를 큐에 연결했다. 새 training arm은 추가하지 않았다. 최종 모델 채택이나 TEST 실행은 아니다.

## 확인하려는 차이

| DEV pipeline 표기 | 실제 SS | 실제 PBR |
|---|---|---|
| V1 | Vanilla V1, 480 update | Vanilla, 1,900 update |
| RETARGET | Decoder learned-score Top-1 + retargeting, 480 update | 기존 top1-only, 1,900 update |
| SELECTED — 아직 채택되지 않은 비교 후보 | Decoder cosine Top-1 + retargeting, 480 update | RETARGET과 byte가 같은 기존 top1-only, 1,900 update |

`SELECTED`는 기존 실행·집계 코드가 사용하는 후보 alias다. 이번 후보가 논문의 최종 모델로 결정됐다는 뜻은 아니다. PBR은 기존 `top1-only` 구현 그대로이며, SS의 cosine 선택을 PBR 전체에 적용했다고 해석하지 않는다. SS 구조에 따른 최종 출력 차이를 같은 학습된 PBR 아래에서 비교한다.

Stage IoU의 point difference가 실제 surface·target alignment·source appearance에도 이어지는지가 질문이다. 후보는 이미 저장된 자기 SS prediction으로 원래 Shape512/Shape1024를 통과한다. 다른 arm의 voxel이나 geometry를 빌려 쓰지 않는다. 기존 V1/RETARGET baseline과 동일한 edited target RGB8 + original source RGB9/10/11, sample/noise, 원래 Shape weight/config, 생성·렌더링 protocol을 유지한다. GT geometry나 depth를 생성 입력으로 사용하지 않는다.

## 실제 연결한 작업

기존 `connect_selected_pipeline_CPU_v182.py`를 수정 없이 실제 실행했다. 두 모델의 자기 config·denoiser·conditioner checkpoint와 SS archive, 성공한 실제 평가 terminal, PBR 네 rank의 1,900 update 완료 증거를 file4 SHA256/size receipt로 확인했다. SS128 archive의 같은32 prefix가 기존 reference와 sample/source/GT/noise/count 단위로 일치했다.

새 descriptor 67개를 기존 GPU0–3 dispatcher 큐 끝에 추가했다.

- 자기 SS로 geometry를 생성하고 공통 vanilla PBR까지 통과하는 native case 32개.
- 저장된 자기 geometry와 실제 native shape guide 위에서 학습된 top1-only PBR을 실행하는 case 32개.
- 실제 pixel coverage CPU 집계, 공식 DreamSim/CLIP GPU 평가, 최종 paired CPU 집계 각 1개.

공통 vanilla PBR case는 기존 cascade·guide 보존 protocol을 따른다. 최종 joint 후보의 texture 비교는 같은 seed-reset replay protocol을 사용한다. 재생성 시 geometry를 바꾸거나 GT sparse support를 쓰지 않는다. Metric은 edited target camera8과 source cameras9/10을 나누고 source-cluster paired 통계를 유지한다. FID/KID를 포함한 최종 10-metric 전체 TEST 평가는 이 DEV32 확인과 별도의 단계다.

연결 후 실제 큐를 다시 읽어 67개 모두 pending, 새 GPU 작업65개와 CPU 작업2개, 기존 descriptor 전체 불변, source pin414개가 현재 byte와 일치하는 것을 확인했다. GPU 작업은 실제 raw PBR resource-release permit을 기다린다. 현재 네 GPU를 사용하는 PBR 학습 중에는 시작되지 않는다. Main dispatcher PID/birth와 GPU67 dispatcher, 진행 중인 세 NN 및 resource coalescer가 원래 PID/birth로 살아있는 것도 확인했다. 관찰 timeout에 따른 재시작은 없다.

## 실제 상태와 증거

이것은 실행 준비만 만든 preview가 아니다. 실제 dispatcher 큐에 연결했고 각 작업을 dispatcher가 인식했다. 다만 아직 GPU forward가 실행되지 않아 후보의 joint 성능은 미측정이다. 기존의 미완료12-control, shared-evidence fixed/learned 비교, 독립 training seed 확인, 실제 최종 TEST와 10개 metric이 모두 남아 있다. 원래 objective는 완료되지 않았다.

file4 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 증거:

- `decoder-cosine-Top1-joint-native32-inputs-v194/`: 실제 두 모델·SS128 archive·terminal과 입력 plan의 closed binding.
- `decoder-cosine-Top1-joint-native32-v194/prelaunch-v1/`: 원래 numerical template와 자기 SS/PBR로 구성한 실제 plan·index·jobs·audit 총262개 및 closure.
- 같은 root의 `queue-v1/`, `applied-v1/`, `actual-queue-verification-v1/`: 실제 append된 큐, descriptor/pin 검증, 살아있는 원래 PID/birth 및 file4 closure.

SS count 진단은 [별도 분석 기록](2026-10-24-SS-module-occupancy-error-analysis-ko.md)에 정리했다. 실제 count table·PNG/SVG·training input parity artifact는 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-module-occupancy-error-analysis-v194)에 게시한 후 원격 finished 상태와 scalar70개 및 artifact를 다시 확인했다.
