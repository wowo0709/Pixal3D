# Physical semantic Top1-only native SS control

현재 cold native SS 세 모델은 vanilla, R_feature, R_raw_topk이다. R_feature에도 learned
Top-K feature가 포함되므로 이 모델을 NoScore로 부를 수 없다. Matching-aware attention의
필요성을 확인하려면 semantic Top1 feature만 사용하는 R-only control이 추가로 필요하다.

새 `RMediatedTop1SourceConditioner`는 기존 R19에서 R·fusion·source-use의 열 개 trainable
tensor를 그대로 유지하고, Q/K/V readout의 아홉 tensor와 frozen score utility를 물리적으로
포함하지 않는다. Uniform attention을 사용하거나 head를 달아 놓고 0으로 만드는 구조가 아니다.
Frozen semantic member matcher의 posterior argmax로 선택한 실제 feature 하나만 gather한다.
Top-K candidate **value** tensor도 만들지 않는다. Source-camera ray group, source별 원래
feature reference, target context, missing-match fallback, projection/target carrier는 유지한다.

R 입력은 `[semantic Top1 appearance − own source feature, target context]`다. 원래 source가
native four-view carrier 안에 있으므로 해당 source를 뺀 displacement가 정합된다. Matching
support가 없으면 native four-view condition을 유지한다. 새로운 visibility write gate나
target-only 모델을 추가하지 않는다. NoDQ는 채널이 실제로 없는 구조로 유지한다.

CPU 검증은 다음을 확인한다.

* 열 개 R/use/fusion 초기 tensor가 현재 R19와 모두 동일하고, cold native source delta가 0이다.
* R이 0이 아닌 상태에서도 head output이 0인 R_feature와 결과가 정확히 일치한다.
* 실제 선택된 Top1 feature와 own-ray source reference가 R에 들어가고, 열 개 source weight에
  finite gradient가 도달한다. Frozen matcher는 갱신되지 않는다.
* Matching이 없으면 native delta와 source gradient가 모두 finite 0이고, checkpoint는
  R19/legacy Top1과 schema가 달라 strict loading으로 혼용할 수 없다.

`r_top1_native_recipe.py`는 이미 고정한 R_feature science config를 SHA로 참조하고 native
수치·모델·dataset·validation contract가 동일한지 검사한다. Runtime은 별도 파일이며, 진행
중인 세 모델의 코드나 config를 수정하지 않는다. 실제 constructor에서도 같은 RNG state로
R19 reference를 잠시 CPU에 생성하여 공통 초기 열 tensor가 정확히 일치하는지 확인하고,
CPU/CUDA RNG를 복구한 뒤 실제 source10만 GPU로 이동한다. CPU에서 학습하는 과정은 없다.

GPU proof/capacity가 통과하기 전에는 이 추가 control의 native micro96 학습이 가능하다고
간주하지 않는다. Proof는 actual native field·noise·flow loss·backward를, capacity는 initial32
validation 뒤 실제 micro96 세 update와 optimizer370 state·frozen integrity를 확인해야 한다.
과거 R19 engineering 결과를 source10의 실제 GPU 증거로 대체하지 않는다.

Science를 진행할 경우 기존 모델과 동일하게 original SS4210의 complete base+LoRA,
TRAIN9149·seed101·world1·global96·micro96·accumulation1·480 native flow updates를 사용한다.
새 source와 fresh optimizer로 시작하며, 이전 teacher112 결과를 initializer로 사용하지 않는다.
별도 Top1 checkpoint evaluator는 process-local binding만 사용하며 원래 R19 evaluator 파일을
고치지 않는다. 데이터나 기존 matching/Omega cache를 다시 구축할 필요는 없다.

해석은 R_top1_only−vanilla가 semantic Top1+R/use 전체 효과, R_feature−R_top1_only가
learned Top-K feature/head 추가 효과, R_raw_topk−R_feature가 raw weighted appearance를 R에
추가한 효과다. Decoder 자체의 효과와 native spatial NoCorr는 별도 통제 없이 입증하지 않는다.
최종 DEV128×3 generation noises와 독립 training seed 검증을 성공 판단에 사용한다.
