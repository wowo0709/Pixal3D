# 필수 PBR seed202 기준 학습 연결과 실제 trace reader 수정

2026-10-24. 실제 dispatcher 큐를 점검한 결과 SS seed202 반복은 pending successor로 연결돼 있었지만, PBR seed202는 `selected-PBR-seed202-v171/prelaunch-NOT-QUEUED` 설정 준비 상태였다. 앞선 handoff의 “v185 successor가 SS/PBR202를 포함한다”는 메모는 부정확했다. v185 successor는 이 PBR 반복 학습을 실제로 등록하지 않았다.

## 두 기준 모델의 실제 학습을 큐에 연결

`modes_for_selected()`의 모든 가능한 선택에 공통으로 필요한 두 PBR 기준, `vanilla`와 `top1-only`만 등록했다. 최종 후보 architecture를 새로 추가하거나 채택하지 않았다. Top-K/feature 모델의 추가 seed는 실제 DEV 결과에 따른 선택 이후에 판단한다.

| 항목 | 두 기준 run의 설정 |
|---|---|
| 초기 가중치 | 사용자가 지정한 원래 multi-view PBR16850 full base + LoRA |
| 데이터 | 기존 native PBR9139개 train subset |
| 예산 | 5 epoch, 1,900 update |
| 독립 training seed | 202 |
| Batch | global24, DDP4, GPU당 micro6, accumulation1 |
| GPU | 명시적인 physical0/1/2/3 전체 exclusive group |
| Optimizer | 매 run 원래 checkpoint에서 새 optimizer로 시작 |
| 초기 검증 | 자기 run의 fixed4 입력·noise·PBR latent exact identity와 기존 renderer bound를 첫 update 전에 확인 |
| Epoch 검증 | 원래0/380/760/1140/1520/1900 fixed4 target8 GT-shape stage protocol |

실제 완료된 seed101 native config에서 public `make_confirmation`과 `validate_confirmation`으로 설정을 생성했다. Training seed와 output/run/port 같은 실행 식별자 외의 scientific 값과 JSON type이 같다. Seed101의 architecture/capacity proof를 재사용하며 이를 seed202에서 새 proof가 끝난 것으로 표시하지 않는다.

기존 `run_native_PBR_science_render_identity_v157.py`와 checkpoint archiver, native trainer를 그대로 사용한다. 새 output·W&B ID·port·checkpoint archive 경로와 자기 config/plan을 갖는다. 다섯 checkpoint 세트를 고려한 저장 공간은 현재 완료된 기준 모델의 실제 final checkpoint 크기로 계산하고 여유분을 더했다. 준비 시 두 run의 combined budget을 검사했으며, 실제 시작 시 기존 atomic DiskBudget가 다시 공간을 예약한다.

실제 추가한 두 작업:

- `required-PBR-seed202-vanilla-native1900-DDP4-v196`
- `required-PBR-seed202-top1-only-native1900-DDP4-v196`

Vanilla는 현재 Raw PBR과 필요한 SS seed202·MeanMatch 작업의 완료를 기다린다. Top-1은 새 Vanilla PBR의 성공적인 완료를 기다린다. Dispatcher가 네 GPU 모두를 확보할 때만 시작하므로 다른 진행 중인 학습 위에 네 rank를 겹쳐 띄우지 않는다. 현재 두 작업은 실제 pending이며 새 학습은 아직 시작하지 않았다.

등록 전 실제 native `guard_plan`을 두 config/plan에 실행했다. 등록 후 실제 큐를 읽어 기존 descriptor 보존, 두 작업의 pending 상태, source pin55개, dependency closure18개에 FAILED/HELD가 없는 것을 확인했다. 현재 네 GPU가 사용 중일 때 group allocator가 새 작업을 admit하지 않는 것도 확인했다. 세 진행 중인 NN은 원래 PID/birth로 유지됐다. 기존 큐 state를 직접 수정하지 않았다.

## 실제 trace schema와 digest 구분 수정

`collect_selected_pbr_seed_confirmation.py`의 최종 후처리 reader에서 실제 native 기록과 맞지 않는 두 항목을 확인했다.

1. Native PBR은 각 rank의 `asset_ids`를 문자열6개의 flat list로 기록한다. Reader는 중첩된 한 pack을 기대하고 있었다. 실제 flat micro6 schema를 검증하도록 수정했으며 중첩 list·짧은 batch·문자열 하나·잘못된 ID/step/digest를 거부한다.
2. 기존 native stage snapshot은 parsed JSON row를 canonical serialization한 SHA를 기록한다. Reader는 이를 raw `.jsonl` 파일 SHA와 같다고 요구하고 있었다. 원본 파일 SHA와 receipt는 기존 `read_file`로 계속 확인하고, 실제 row의 canonical digest를 별도로 계산해 snapshot과 비교한다. Snapshot이 같은 path·raw SHA·receipt를 가리키는지도 추가로 확인한다.

수정한 파일은 검사 당시 두 dispatcher의 실제 active/pending plan476개와 실행 중인 세 native config에서 참조·pin하지 않는 후처리 코드였다. 진행 중인 모델·trainer·optimizer·sampler source를 변경하지 않았다.

검증 결과:

- 신규 trace regression 및 기존 selected-pipeline freeze 검사 **62개 통과**, CPU92/93·CUDA 비활성화.
- 실제 완료된 Vanilla/Top-1 PBR의4 rank씩, 파일8개·각1,900 update·총15,200 row에 새 reader를 적용했다.
- 모든 실제 trace가 flat micro6 schema를 통과하고, canonical digest가 기존 실제 stage snapshot의 값과 정확히 일치했다. Raw file SHA 검증도 유지했다.

이 결과는 실제 seed101 trace reader 연결을 검증한 것이다. 아직 시작하지 않은 seed202 학습이나 두-seed 성능 재현을 입증하지 않는다.

## 남은 최종 집계 연결

Seed202 완료 후 사용할 stage snapshot은 실제 training seed202를 유지해야 한다. 기존 native stage evidence helper의 seed101 전용 검사를 seed202 결과에 그대로 적용하거나 seed를 바꿔 표시하지 않는다. 또한 후속 PBR 집계는 이미 native 시작 단계에서 사용하는 exact-latent/bounded-render 계약을 유지해야 한다. 초기 render metric의 허용된 작은 부동소수점 차이를 latent 불일치로 취급하지 않으며, 실제 값·원래 bound·producer proof를 그대로 확인해야 한다. 이 후속 연결은 아직 남아 있다.

선택된 추가 SS/PBR의 독립 seed, 원래 모듈 비교와 joint DEV, validation36, 최종 freeze, 실제 TEST508과10개 metric도 계속 필요하다. 이번 작업으로 전체 실험이 완료되지는 않았다.

실제 file4 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/required-PBR-seed202-baselines-v196/`의 prelaunch·native-guard·queue·applied·actual-queue-verification과 `PBR-confirmation-trace-actual-preflight-v196/`에 저장했다. 실제 remote SHA256/size를 확인한 closure를 포함한다.
