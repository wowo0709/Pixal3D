# SS 공유 증거 구조 5에폭 완료와 최종 평가 전환

2026-10-24. E_FIXED_RAW의 실제 480-update 학습과 마지막 DEV32 검증이 완료됐다. Native child가 종료됐고 science wrapper와 checkpoint bundle seal 작업도 성공했다. GPU6에서 128개 최종 평가의 첫 generation noise가 실제 실행 중이다. 아래 32개 결과는 최종 DEV128×3을 대체하지 않는다.

## 같은 480-update DEV32 결과

Training seed101, generation noise1개, 동일32개 asset·source cluster·GT·생성 seed를 사용했다. 각 모델의 초기32개 prediction count와 실제480개 training asset/x_t/t trace가 일치했다. Primary aggregation은 case-macro다.

| Arm | Case-macro IoU ↑ | Global-micro IoU ↑ |
|---|---:|---:|
| V1 | 0.473279 | 0.433639 |
| C_SPATIAL | 0.473084 | 0.432517 |
| C_RAW | 0.473901 | 0.436847 |
| C_D1_COS | 0.474095 | 0.439739 |
| C_DECODER | 0.476967 | 0.440537 |
| C_DECODER_NO_R | 0.473085 | 0.434668 |
| C_TOP1_V | 0.479037 | 0.443045 |
| C_FIXED_TOPK | **0.479107** | 0.443777 |
| FULL | 0.478243 | 0.442061 |
| FULL_QK_ONLY | 0.478177 | 0.443200 |
| E_FIXED_RAW | 0.478940 | **0.444724** |

E_FIXED_RAW는 V1보다 macro +0.005661이지만 고정 Top-K보다 약0.000167 낮다. 384-update panel에서 Q/K-only가 최고였던 순위도 유지되지 않았다. 이 크기의 중간 panel 순위로 full 구조를 결정하지 않는다. 이미 완료된 DEV128×3에서는 C_D1_COS가0.473008로 높았으며, 그 값과 위 DEV32 값을 직접 비교하지 않는다. 0.43대 micro와0.47대 macro의 차이도 모듈 향상이 아니다.

FULL_QK_ONLY는 additive matching-confidence prior를 attention logit에서 제거한 arm이다. Semantic 후보·context와 Top-1 carrier, V/O 경로와 retargeting은 남는다. 사용자 정의의 no-score=Top-1 correspondence arm과 같지 않다. E_FIXED_RAW는 carrier/use/retargeting이 같은 raw Top-K posterior를 사용하고 learned Q/K·V/O가 없는 별도 구조다. 두 arm의 차이를 하나의 요소만 바꾼 비교라고 해석하지 않는다.

## 실제 완료와 자원 검증

E_FIXED_RAW의 frozen base·matcher parameter/buffer가 그대로인 것을 마지막 native proof에서 확인했다. LoRA360개와 source10개가 학습됐고 optimizer state는370개다. GPU 한 장, global96=micro96, accumulation1로480 update를 완료했다. 마지막20개 compute median은36.98초, 전체 training update의 최대 allocated89.84GiB, 최대 reserved94.03GiB다. 실제 science child 부재와 성공 terminal을 확인했고 file4의 native·bundle receipt도 생성됐다.

학습 가능한 상대 Q/K·utility를 추가하는 E_LEARNED_RAW도 실제 GPU6의 cold capacity 검증을 통과했다. 같은 micro96/accumulation1의 임시3 update, 초기 native32 일치, 세 번째 update에서 모든 source17개 gradient의 양수·유한성, frozen parameter/buffer 보존을 확인했다. 최대 allocated92.57GiB, reserved93.29GiB다. 이3개 update는 engineering 검증이며 실제5에폭 학습으로 계산하거나 과학 실험 checkpoint로 재사용하지 않는다. 실제 학습은 기존 평가·metric 검증 순서 뒤 fresh optimizer로 수행한다.

Voxel-query FFN12 재실행은 실제147번째 optimizer update를 OOM 없이 완료했다. 해당 update의 compute는158.27초, peak allocated91.71GiB, reserved93.22GiB였다. 원래 실패한 run의 저장된146개 입력 trace가 새 run의 첫146개와 정확히 같다. 원래 실패한147번째 입력 trace는 없으므로 그 미기록 입력까지 실제 같았다고 주장하지 않는다. 전체480 update의 OOM 해결이나 성능은 아직 미확인이다.

## PBR 경향과 남은 판정

완료된 기준 PBR의 실제1900-step fixed4 로그에서 Top-1 retargeting crop LPIPS는4에폭0.091200에서5에폭0.085186으로 회복됐다. V1의5에폭은0.086454다. 반면 foreground PSNR은 Top-1 15.74925, V1 15.76565여서 metric별 우열이 다르다. 현재 Raw Top-K의1900-step 결과는 아직 없다. 4에폭 악화만으로 과적합이나 matching 실패를 확정하지 않는다.

네 mode의 초기 상태와380/760/1140/1520 update 패널80개를 읽어, mode 내부에서 GT의 uint8 RGB 픽셀이 모든 시점에 동일하고 네 사례 모두 비어 있지 않음을 확인했다. Float render tensor까지 bitwise 같다는 주장은 아니다. 이 평가는 GT Shape·target camera8에 한정되므로 alpha-IoU1을 생성 geometry의 정확도로 해석하거나 source preservation을 입증했다고 보고하지 않는다.

진행 중인 Raw PBR이 실제 완료된 뒤 기존 public explicit-seed stage reader를 실행하는 CPU 작업도 연결했다. 기존 reader·worker는 수정하지 않았다. 실제 자기 seed101·네 rank·1900 trace/runtime·6개 epoch·원본 initial proof·checkpoint byte와 GT panel을 검사해 catalog를 생성한다. 미래 결과의 SHA나 PASS는 만들지 않았다. 새 학습 arm이나 GPU 실험은 추가하지 않았다.

다음 판정에는 E_FIXED_RAW/E_LEARNED_RAW의 실제 DEV128×3, 원래 core12, 공통 생성 geometry의 PBR 비교와 SS→Shape→PBR joint DEV가 필요하다. 이후 실제 독립 training seed 확인·VAL36 재로딩 검증·closed selection을 마친 뒤 TEST508 및10개 metric을 계산한다. 현재 최종 모델을 채택하지 않았다.

실제 file4 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/consistent-fixed-completed480-DEV32-v200/`, `consistent-fixed-completed480-WandB-v200/`, `raw-PBR-explicit-stage-readout-v200/` 아래에 있다. Same-epoch 표28개 scalar와 artifact의 [W&B 원격 완료](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-consistent-fixed-epoch480-all-controls-v200)를 확인했다. 기존 core12·followup summary와 finite statistical family는 바꾸지 않았다.
