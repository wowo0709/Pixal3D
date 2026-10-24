# SS 공유 증거 구조 5에폭 완료와 최종 평가 전환

2026-10-24. E_FIXED_RAW의 실제480-update 학습, DEV32 검증과 DEV128×3 최종 평가가 완료됐다. Science wrapper·checkpoint seal·세 evaluation wrapper·11-arm descriptive collector가 모두 성공했다. GPU6에서는 E_LEARNED_RAW의 실제 fresh 학습이 시작됐다. 아직 원래 core12 중 voxel-query·mean-matching과 이 learned followup은 완료되지 않았으며 최종 모델을 채택하지 않았다.

## 최종 DEV128×3 결과

128개 asset,93개 source cluster, generation noise3개, training seed101, optimizer480 update의 같은 조건이다. Noise3개를 독립 training seed3개로 계산하지 않는다. 아래 값은 각 asset의 metric을 noise에 걸쳐 평균한 뒤 asset 평균을 취한 값이다.

| Arm | Case-macro IoU ↑ | Precision ↑ | Recall ↑ | Voxel F1 ↑ |
|---|---:|---:|---:|---:|
| V1 | 0.467355 | 0.589160 | 0.681455 | 0.628705 |
| C_SPATIAL | 0.471242 | 0.594881 | 0.681915 | 0.632515 |
| C_RAW | 0.472360 | 0.596185 | 0.681512 | 0.633168 |
| C_D1_COS | 0.473008 | 0.599107 | 0.680833 | 0.634261 |
| C_DECODER | 0.472600 | 0.597171 | 0.681952 | 0.633745 |
| C_DECODER_NO_R | 0.468671 | 0.591042 | 0.681865 | 0.629964 |
| C_TOP1_V | 0.472218 | 0.597829 | 0.680199 | 0.633351 |
| C_FIXED_TOPK | 0.472745 | 0.598838 | 0.679785 | 0.633829 |
| FULL | 0.472078 | 0.598227 | 0.679274 | 0.633222 |
| FULL_QK_ONLY | 0.472446 | 0.598270 | 0.679687 | 0.633420 |
| E_FIXED_RAW | **0.473471** | **0.599933** | 0.679803 | **0.634378** |

새 공유 증거 구조의 IoU는 V1 대비+0.006116, C_DECODER 대비+0.000871, 기존 최고 C_D1_COS 대비+0.000463이다.
E_FIXED_RAW가 현재 완료된11개 arm에서는 가장 높지만, 기존 최고 대비 차이는 작다. 새 통계 검정이나 독립 training-seed 확인은 이번 descriptive collector에서 수행하지 않았다. 기존 core12 및 followup의 finite family와 summary는 그대로 남아 있다.

E_FIXED_RAW에는 학습 가능한 Q/K·V/O가 없다. 따라서 이 결과를 learned matching-aware attention의 입증으로 보고하지 않는다. 같은 carrier/use/R·identity-V를 유지하고 상대 Q/K·utility7개만 더하는 E_LEARNED_RAW의 결과가 그 효과를 확인하는 matched 비교다. V1 대비 precision은+0.010773, recall은−0.001652였다. IoU의 상승이 얇은 구조 보존이나 mesh surface 개선까지 뜻하는지는 CD·F-Score 및 생성 geometry 평가로 확인해야 한다.

실제 최종 평가의 asset/source/GT/noise·RGB 역할·sampler·자기 checkpoint·source 구조·성공 terminal을 검증했다. 원래10개 모델의 metric을 다시 계산해 기존 값과 일치함도 확인했다. [W&B에47개 scalar, 표, 실제 V1·C_D1_COS·E_FIXED_RAW의 GT/Pred 패널3개와 artifact](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-consistent-fixed-final128x3-eleven-models-v200)를 게시하고 원격 완료를 확인했다.

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

DEV32에서 E_FIXED_RAW는 V1보다 macro +0.005661이지만 고정 Top-K보다 약0.000167 낮다. 384-update panel에서 Q/K-only가 최고였던 순위도 유지되지 않았다. 위 최종128×3에서 E_FIXED_RAW가 높았던 순위와도 다르다. 이 크기의 중간 panel 순위로 full 구조를 결정하지 않는다. 0.43대 micro와0.47대 macro의 차이도 모듈 향상이 아니다.

FULL_QK_ONLY는 additive matching-confidence prior를 attention logit에서 제거한 arm이다. Semantic 후보·context와 Top-1 carrier, V/O 경로와 retargeting은 남는다. 사용자 정의의 no-score=Top-1 correspondence arm과 같지 않다. E_FIXED_RAW는 carrier/use/retargeting이 같은 raw Top-K posterior를 사용하고 learned Q/K·V/O가 없는 별도 구조다. 두 arm의 차이를 하나의 요소만 바꾼 비교라고 해석하지 않는다.

## 실제 완료와 자원 검증

E_FIXED_RAW의 frozen base·matcher parameter/buffer가 그대로인 것을 마지막 native proof에서 확인했다. LoRA360개와 source10개가 학습됐고 optimizer state는370개다. GPU 한 장, global96=micro96, accumulation1로480 update를 완료했다. 마지막20개 compute median은36.98초, 전체 training update의 최대 allocated89.84GiB, 최대 reserved94.03GiB다. 실제 science child 부재와 성공 terminal을 확인했고 file4의 native·bundle receipt도 생성됐다.

학습 가능한 상대 Q/K·utility를 추가하는 E_LEARNED_RAW도 실제 GPU6의 cold capacity 검증을 통과했다. 같은 micro96/accumulation1의 임시3 update, 초기 native32 일치, 세 번째 update에서 모든 source17개 gradient의 양수·유한성, frozen parameter/buffer 보존을 확인했다. 최대 allocated92.57GiB, reserved93.29GiB다. 이3개 update는 engineering 검증이며 실제5에폭 학습으로 계산하거나 과학 실험 checkpoint로 재사용하지 않는다. 실제 학습은 기존 평가·metric 검증 순서 뒤 fresh optimizer로 수행한다.

Voxel-query FFN12 재실행은 실제147·148번째 optimizer update를 OOM 없이 완료했다. 147번째 update의 compute는158.27초, peak allocated91.71GiB, reserved93.22GiB였다. 원래 실패한 run의 저장된146개 입력 trace가 새 run의 첫146개와 정확히 같다. 원래 실패한147번째 입력 trace는 없으므로 그 미기록 입력까지 실제 같았다고 주장하지 않는다. 전체480 update의 OOM 해결이나 성능은 아직 미확인이다.

## PBR 경향과 남은 판정

Raw Top-K PBR도 실제1900 update·마지막 검증·네 rank의 종료와 저장을 완료했다. 네 모델의 전체1900-update/rank 입력 trace, 모든 시점의 실제 RGB·GT Shape/PBR latent·camera·noise 대응을 확인했다. 새 explicit-seed snapshot도 실제 성공해 Raw 자신의 seed101·checkpoint byte·6개 epoch·runtime·initial proof·GT panel을 검증했다.

| PBR arm | Foreground PSNR ↑ | Crop LPIPS ↓ | Crop SSIM ↑ |
|---|---:|---:|---:|
| V1 | 15.76565 | 0.086454 | 0.953675 |
| Top-1 retargeting | 15.74925 | 0.085186 | 0.954216 |
| Top-1 + feature | 15.71515 | 0.086537 | 0.953868 |
| Raw Top-K + feature | **16.18082** | **0.083294** | **0.955793** |

Raw Top-K 조합은 V1 대비 PSNR+0.41517dB, LPIPS−0.003161이고 Top-1 retargeting 대비 PSNR+0.43158dB, LPIPS−0.001892다. LPIPS가 나아진 사례는 각 비교에서4개 중3개다. Top-1 + feature 대비로는2승2패다. GT Shape 조건·target camera8·사례4개·training seed1개에 한정된 조합 결과이며, learned attention 단독 효과나 source preservation·생성 geometry 성능을 입증하지 않는다. Top-1에 feature correction만 더한 arm은 LPIPS가 더 나빴다. 현재 결과로 어떤 추가 경로가 필수라고 주장하지 않는다.

Top-1 retargeting LPIPS도4에폭0.091200에서5에폭0.085186으로 회복됐다. Raw의4에폭 악화 역시 최종 결과에서는 유지되지 않았다. 4에폭 한 점으로 과적합이나 matching 실패를 확정하지 않는다. [실제 W&B 최종표·6개 시점 곡선·사례별 결과·GT/Pred 패널](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-final1900-fixed4-matched-v200)의 원격 완료를 확인했다.

네 mode의 초기 상태와380/760/1140/1520 update 패널80개를 읽어, mode 내부에서 GT의 uint8 RGB 픽셀이 모든 시점에 동일하고 네 사례 모두 비어 있지 않음을 확인했다. Float render tensor까지 bitwise 같다는 주장은 아니다. 이 평가는 GT Shape·target camera8에 한정되므로 alpha-IoU1을 생성 geometry의 정확도로 해석하거나 source preservation을 입증했다고 보고하지 않는다.

Raw 완료 뒤의 public explicit-seed stage reader는 기존 reader·worker 수정 없이 실제 실행·완료됐다. 미래 결과의 SHA나 PASS를 먼저 만들지 않았다. GPU0–3의 Raw 독점 실행 gate도 실제 해제됐고, GPU0에서 기존 mean-matching arm의 학습이, GPU1–3에서 기존 공통 생성 geometry 평가가 실제 시작됐다. GPU6은 E_LEARNED_RAW, GPU7은 voxel-query 비교군이 사용한다.

실제 DEV128×3을 완료한 E_FIXED_RAW와 기존 RETARGET baseline의 정확히 같은 Top-1 PBR1900을 묶은 DEV32 실행도 실제 큐에 연결했다. 자신의 SS prediction archive와 원본 Shape512/1024 checkpoint를 사용하며, SS에서의 향상이 생성 geometry와 rendering/source preservation에도 이어지는지 확인하는 candidate 평가다.65개 GPU 작업과2개 CPU 작업으로 구성되며, 등록 시 Raw 완료 이후 실행하도록 기존 자원 gate를 유지했다. 등록 당시 source pin414개와 pending job67개를 확인했다. 새 학습 arm이나 TEST 실행, 최종 모델 채택은 아니다. Raw PBR의 최종 fixed4 우위만으로 이 조합의 PBR까지 Raw로 바꾸지 않는다. 공통 생성 geometry DEV32 결과가 먼저 필요하다.

GPU6에서 기존 DEV 이미지4개를 사용한 공식 metric entry 호환성 확인도 실제 완료됐다. 기존 경로와 새 공식 entry의 DreamSim·CLIP 점수 차이는0이었고, 같은 GPU6을 서로 다른 허용 GPU 집합으로 실행했을 때 Inception feature4×2048이 bitwise 같았다. 이는 entry/assignment 확인이며 모델 간 성능 비교나 전체10개 metric 완료가 아니다.

다음 판정에는 E_LEARNED_RAW의 실제 DEV128×3, 원래 core12의 미완료 두 arm, 공통 생성 geometry의 PBR 비교와 SS→Shape→PBR joint DEV가 필요하다. 이후 실제 독립 training seed 확인·VAL36 재로딩 검증·closed selection을 마친 뒤 TEST508 및10개 metric을 계산한다. 현재 최종 모델을 채택하지 않았다.

실제 file4 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `consistent-fixed-final-endpoint-readout-v199/actual-final128x3-v1/`, `consistent-fixed-final128x3-WandB-v200/`, `consistent-fixed-completed480-DEV32-v200/`, `voxel-query-FFN12-recovery-v188/actual-past-original-OOM-v200/`, `consistent-fixed-Top1-joint-native32-v200/`, `raw-PBR-explicit-stage-readout-v200/`, `PBR-final1900-four-arm-fixed4-matched-v200/`, `PBR-final1900-four-arm-fixed4-WandB-v200/`에 있다. 별도로 same-epoch DEV32 표28개 scalar와 artifact의 [W&B 원격 완료](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-consistent-fixed-epoch480-all-controls-v200)도 확인했다. 기존 core12·followup summary와 finite statistical family는 바꾸지 않았다.
