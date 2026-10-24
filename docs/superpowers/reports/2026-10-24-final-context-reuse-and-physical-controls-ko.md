# 최종 문맥의 재사용 감사와 실제 제거 대조 구현

2026-10-24. 최종 모델 선택 전, 기존 결과를 어떤 비교에 사용할 수 있는지 코드·닫힌 config·실제 생성자 기준으로 확인했다. 이 작업은 학습된 최종 모델이나 새 대조군의 성능 결과가 아니다. 목표는 [필수 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)의 모듈별 claim을 잘못된 대조로 채우지 않고, 필요한 대조만 실행하는 것이다.

## 기존 결과의 재사용 범위

| 기존 비교/arm | 실제 확인 | Shared Fixed/Learned 최종 문맥에서의 처리 |
|---|---|---|
| Physical Top-1 R ↔ Fixed | 공통 core10·carrier/use/R·기본 학습 조건 정합의 기존v225 감사 | Top-1 대비 fixed Top-K 비교에 재사용 |
| Fixed ↔ Learned | 같은 raw identity V·shared evidence, 추가 Q/K·utility7개 | Learned refinement 효과에 재사용 |
| C_SPATIAL/C_RAW/C_D1_COS | 기본 학습 조건은 같지만 appearance가 single-correspondence Top-1 | Weighted Top-K의 matcher 제거 대조로 자동 재사용하지 않음 |
| C_DECODER_NO_R | 실제5개 source tensor, Top-1 appearance | Shared weighted Top-K의 R 제거 대조로 자동 재사용하지 않음 |
| FULL_MEAN_MATCH / 기존 FULL_VOXEL_Q | 원래19개 tensor·value 변환/추가 correction 구조 | Shared10/17개 tensor 구조의 독립 대조로 자동 재사용하지 않음 |

이번 감사의 최종 문맥 비교 대상은 **E_FIXED_RAW/E_LEARNED_RAW**다. 최종 모델이 Top-1 R로 결정되면 원래 Top-1 문맥 대조의 재사용을 다시 검토한다. 아직 선택되지 않은 모델을 전제로 모든 기존 결과를 폐기하거나 새 학습을 등록하지 않았다.

실제 C_SPATIAL/C_RAW/C_D1_COS/C_DECODER_NO_R/FULL_MEAN_MATCH의 닫힌 science config에서 기본 numerical/data contract 차이는 없었다. 그러나 appearance/전달 구조의 차이가 있어 그것만으로 matched final-context 비교가 되지는 않는다. 근거는 runtime root의 `final-context-reuse-and-control-scope-v280/audit.json`과 file4 closure다.

## 서로 다른 score 질문을 분리

1. **Raw → D1 cosine:** 같은 후속 경로에서 decoder descriptor의 효과. Decoder의 독립 기여를 주장하려면 이 중간 대조가 필요하다.
2. **D1 cosine → D1 scalar score:** 같은 decoder에서 score 모듈의 ranking/prior 효과. Member pool의 최종 support가 바뀌면 그 변화도 score 모듈의 효과에 포함됨을 명시한다.
3. **같은 후보의 Uniform-K → Fixed posterior:** 최종 raw evidence를 score로 가중하는 효과. 같은 posterior가 carrier/use/R에 공통으로 쓰이면 공통 evidence weighting의 효과다.
4. **Fixed → Learned:** 추가 Q/K·utility refinement의 효과.
5. **Learned without additive logit prior → Learned:** 같은7개 learned parameter를 유지한 additive attention prior의 효과. 후보·context·utility 입력의 matching 정보는 유지된다.

Learned 전체를 Uniform-K로 바꾸면 frozen prior와 learned refinement를 함께 바꾸므로 순수한 prior 대조가 아니다. **No-score=실제 Top-1**, **No-corr=원래 spatial 대응**이라는 사용자 정의는 유지한다. Uniform-K와 additive-logit-prior 제거는 별도의 이름이다. 위 다섯 질문은 독립적인 추가 model grid가 아니라, 실제 채택한 claim에 맞는 비교를 선택하기 위한 구분이다.

기존 FULL_QK_ONLY의 `NoPriorBoundedReadout`은 원래 readout 전체에 neutral prior를 전달하므로 utility 입력에도 영향을 준다. 새 no-logit-prior는 current/initial score를 원래 prior 입력으로 계산하고 마지막 probability의 additive prior만 neutral로 바꾼다. 따라서 원래19-tensor 대조의 결과를 새17-tensor 직접 logit 항의 대조로 재사용하지 않는다.

최종 표는 matching ladder와 최종 전달 경로의 단일 변경 대조를 구분할 수 있다. Raw와 decoder+scalar를 동시에 바꾼 행의 차이를 decoder descriptor 하나의 효과라고 쓰지 않는다. Spatial의 원래3개 projected source 후보와 semantic64의 비교는 correspondence 생성 방식의 차이를 포함하며, 단순한 score-only 비교가 아니다.

## Query 대조가 실제로 바꾸는 것

Fixed에는 learned Q/K readout이 없다. 따라서 기존 voxel-Q helper의 native attention-query 입력만 바꾸면 Fixed의 attention readout 차이가 생기지 않는다. Fixed에서도 retargeter와 source-use가 ray-pooled target query를 사용한다는 사실은 유지된다.

최종 모델이 Learned라면 같은 identity V·posterior·carrier/use/R에서 learned readout의 query unit만 ray→voxel로 바꾼 matched 대조가 필요하다. 원래 FULL의19-tensor voxel 대조는 그 결과를 대신하지 않는다. Fixed를 선택하고 R/use의 처리 단위까지 바꾸려면 이를 **ray/voxel retargeting·use granularity**로 정의해야 하며 순수 Q/K-query 효과라고 부르지 않는다. 최종 모델에 없는 Q/K의 필요성을 입증하려고 무효인 query arm을 돌리지 않는다. Ray 단위 구현은 구조로 설명할 수 있으나 독립 품질 이득 claim은 유효한 대조 이후에만 쓴다.

## 구현한 실제 대조 경로

| Variant | Parent | 물리적 변경 | Trainable source tensor |
|---|---|---|---:|
| A_FIXED_UNIFORM | Fixed | 동일 valid 후보를 균등 가중; learned head 추가 없음 | 10 |
| A_FIXED_NO_R | Fixed | Retargeter 자체를 None으로 제거 | 5 |
| A_LEARNED_NO_R | Learned | 같은 learned head7개를 유지하고 R 제거 | 12 |
| A_LEARNED_NO_LOGIT_PRIOR | Learned | Q/K·utility7개·identity V 유지, additive logit prior만 제거 | 17 |

제거된 R은 state dict에 parameter가 없고 호출되지 않는다. Residual의0은 존재하지 않는 항을 나타내며 feature/query channel에 가짜0을 추가한 것이 아니다. Uniform-K는 기존 valid 후보 주소를 그대로 사용한다. Learned의 logit-prior 제거는 기존 finite prior support를 유지하고 후보/context/utility의 score 정보를 제거했다고 주장하지 않는다.

[실제 source 모듈](../../../pixal3d/modules/shared_evidence_ablation.py)과 [생성자/native trainer factory](../../../experiments/deformnext_retargeting/shared_evidence_ablation_recipe.py)를 별도로 추가했다. 실행 중인 기존 module·trainer·checkpoint 코드는 수정하지 않았다. 공통 native trainer의 초기화 로직은 source 생성자 호출을 제외하면 AST가 같고, 새 readout은 probability operator를 제외하면 원래 learned head와 AST가 같다.

## 완료한 검증과 아직 안 한 검증

- 기존 관련 검사와 [새 대조 검사](../../../tests/multiview/test_shared_evidence_ablation.py)를 합쳐 **53개 통과,1.26초**.
- Actual file4 D1 checkpoint와 닫힌 Fixed/Learned config로 네 생성자를 구성. 남는 cold parameter가 동일 RNG parent와 같고, global CPU RNG가 유지됨을 확인.
- R 제거의 실제 parameter 부재, frozen matcher, learned head7개·identity V, strict schema 구분을 확인.
- 계산 probe로 같은 evidence가 source-use와 R 입력에 전달되고 원래 native source subtraction이 유지됨을 확인. 모든 NN forward를 금지한 CPU 생성자 검사에서 forward0회·CUDA 초기화 없음.
- 실제 실행 중인6개 assigned worker의 **487개 pinned source SHA256** 일치를 확인. 새 source 파일은 이들의 pin에 포함되지 않아 기존 실험을 바꾸지 않음.

생성자 검증의 최신 실제 artifact는 `final-shared-controls-construction-v279`의5개 파일36,065bytes이고, 앞선v278도 보존했다. CPU arithmetic/probe·실제 checkpoint 생성자 검사는 **GPU native flow identity/backward, micro96 capacity, checkpoint/evaluator binding,480-step 학습**을 대신하지 않는다. Native trainer factory는 구현했지만 실행 entry와 평가 binding은 아직 연결하지 않았다. 최종 후보 선택 뒤 필요한 variant만 이 경로로 입장시킨다. Raw/spatial/D1-cos 및 최종 query 대조의 필요한 구현도 아직 남아 있다.

위 runtime artifact는 `agent-handoff.json`의 `PBR_NATIVE_SCIENCE_RUNTIME` 아래 `unified-module-pipeline-v1`을 기준으로 file4 SHA256/size 확인을 마쳤다. TEST·최종 모델 채택·목표 전체 완료는 모두 아니다.

실행 자원 후속: GPU1의 기존 Fixed24건을 마친 뒤 기존 R 평가20건을 이어 실행하도록 pending task의 위치만 옮겼다. Fixed는 priority−70, 후속 R은−69이며 공통 Fixed GT-reference의 실제 file4 closure를 시작 조건에 추가했다. 원래 job name·dependency·model/seed/output은 유지하고 진행 중인 worker와 queue-state는 수정하지 않았다. 준비64개 파일32,693,725bytes와 실제 등록 관측은 `paired-DEV128-stage-evaluation-v240/GPU1-pending-reassignment-v282` 및 `GPU1-R-reassignment-activation-v282`에 보존했다. R의 새로운 science arm을 만든 것은 아니다.
