# Matching prior 보존과 source-relative retargeting의 결합 검증

앞선 GT 진단에서는 frozen matching decoder·FusionHead가 correspondence 순위를
개선했으나, 후속 task utility/QK 재가중이 exact GT 질량을 낮췄다. 이를 검증하기 위해
`bounded-source-contrast-NoDQ`를 별도 모델로 실행한다. 과거 bounded-prior 단독
모델의 SS128×3 개선 CI는0을 포함했으므로, prior 보존만으로 성공을 가정하지 않는다.

## 비교가 답하는 질문

| 모델 | Retargeter 입력 | 네 head의 확률 | 학습 tensor |
|---|---|---|---:|
| Original19 teacher112 | matched feature와 target context | 기존 utility+QK+matching prior | 19 |
| Plain SourceContrast19 teacher112 | matched−own-source와 target context | 기존 utility+QK+matching prior | 19 |
| 새 Bounded SourceContrast19 teacher112 | matched−own-source와 target context | matching prior + bounded function residual | 19 |

새 모델−Plain Contrast가 **같은 source-relative 입력에서 attention의 초기 함수와
업데이트 제한을 함께 바꾼 효과**다. 이 두 변경을 각각 분리한 causal ablation은 아니다.
Original과 Plain Contrast는 완료된 같은 예산 실험을 재사용한다. 새 모델은 부모의
teacher112 weight를 가져오지 않는다. 초기 trainable19개 값은 기존 모델과 실제
checkpoint에서 비교하고, frozen 함수 reference만 별도 buffer로 저장한다.

Decoder, matching score, Top-K64, ray member 구성, own-source camera ray 단위 query,
target RGB 입력, 원래 projection, source-use 경로와 source-relative R 입력을 고정한다.
DQ channel은 없다. Bound1은 이번 설계의 고정값이며 validation GT로 sweep하지 않는다.
No-score는 Top-1 correspondence feature, No-corr는 원래 spatial correspondence라는
기존 정의를 유지한다. 이 실험으로 해당 retraining 대조군들을 대신하지 않는다.

## 실행 경로와 초기 검증

새 portable entry는 `train_bounded_source_contrast_teacher_pilot`과
`evaluate_bounded_source_contrast_teacher_pilot`이다. Trainer registry 연결은
평가 프로세스 안에서만 적용하고 종료 시 복원한다. 기존 두 warm joint arm, 현재
PBR, 기존 runtime source·recipe·output은 수정하지 않는다. 새 모델의 strict
checkpoint에는 `bounded_source_contrast_schema`와 frozen initial 함수 buffer가
필요하므로 부모 checkpoint의 우연한 로드를 허용하지 않는다.

실제 GPU에서 다음 순서로 실행한다.

1. TRAIN micro96/accumulation1 backward capacity: weight update0회,
   source correction0, 네 head의 실제 확률이 matching prior와 정확히 같은지 확인한다.
2. Cold native32: 같은 SS4210·RGB·camera·GT·noise로 global/projection/negative tensor와
   RNG, voxel TP/FP/FN까지 기존 native baseline과 일치하는지 확인한다. Source 출력이
   0이어도 attention prior 검사가 실패하면 학습을 시작하지 않는다.
3. 두 실제 결과와 동일한 초기19개 값을 확인한 뒤에만 finite source-only teacher를 학습한다.

GPU는 사용자 지정0–3 안에서 기존 dispatcher가 독점 배정한다. 현재 four-GPU PBR을
중단하거나 병행 적재하지 않는다. 새 GPU 검사는 아직 실행 결과가 아니다.

## 학습·판정

같은 TRAIN1536개의 unique source, 같은 teacher cache·batch 순서, seed101,
global96/micro96/accumulation1, AdamW LR1e−4·warmup6·7epoch/112update를 쓴다.
Teacher는 TRAIN의 M1 source9/10 feature만 사용하며 source11의 가짜 label을 만들지
않는다. VAL label은 loss에 들어가지 않는다. 매 epoch original SS4210을 고정하고
실제 RGB BF16 조건으로 native32를 평가해 IoU와 voxel 시각화를 W&B에 기록한다.
Matcher와 initial function buffer는 학습 전후 byte 값이 같아야 한다.

최종 epoch112를 미리 정한 비교점으로 사용한다. 원래 두 모델과 모든112개 batch의
sample 순서, GT hash·noise seed·source cluster를 대조하고 source-cluster paired
bootstrap과7개 epoch 경향을 기록한다. 검증32의 반복 관측은 exploratory 결과다.
최고 epoch를 사후 선택해 확정 효과로 보고하지 않는다. 학습 후 matching GT 질량과
head 재가중 경향도 재확인해야 한다.

좋은 teacher feature MSE만으로 모델을 채택하지 않는다. Source-disjoint 확인,
실제 native joint SS 학습, cascade의 source preservation·target alignment와
별도 test508 평가는 계속 필요하다. 현재 등록된 Original/Contrast의 source72→native
engineering→조건부 joint480→128×3→cascade32 경로도 그대로 유지한다.

새 runtime CPU 검사는 잘못된 parent alias·warm migration·bound·학습 예산을 거부하고,
동일한19개 실제 초기 값 및 **source delta가0인데 attention만 잘못된 경우**를 검출한다.
CPU 통과를 native GPU 정합·capacity·품질 향상 결과로 바꾸어 보고하지 않는다.
