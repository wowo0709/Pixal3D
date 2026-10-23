# 실제 선택 파이프라인의 joint 평가·독립 PBR 확인·모델 고정 도구

구현 커밋 `9374ef3`, 브랜치 `wowo0709/feature-retargeting`.
기존 6개 학습과 그 소스/config/worker는 유지했다. 이 작업으로 추가 NN 학습을 큐에 넣지 않았다.

## 완료한 구현

기존 joint 평가의 모델 집합은 원래 12개 arm, 원본 SS→Shape case 검사는 core5개에 한정되어 있었다.
선택 후보가 E_FIXED_RAW/E_LEARNED_RAW이거나 core5 밖의 arm이어도 실제 구조와 이름을 유지하도록
별도 버전을 추가했다. 현재 실행 중이거나 대기 중인 기존 checksum-bound 파일은 변경하지 않았다.

`selected_pipeline_model_contract.py`, `selected_pipeline_cascade_contract.py` 및
`selected_pipeline_joint_summary_v2.py`는 다음 실제 근거를 확인한다.

- 해당 SS480 NN/source/config의 실제 SHA와 source 텐서 수 및 물리적 family가 일치한다.
- 새 E 모델은 기존 Full의 schema/19개 텐서로 바꾸어 읽을 수 없다. raw identity V와 source10/17을 구분한다.
- 실제 저장된 SS 예측이 원래 Shape5124210·Shape102416850으로 전달된다. SS를 재생성하거나 GT shape로 대체하지 않는다.
- 실제 PBR1900 모델을 해당 SS의 예측 geometry와 원래 native guide에 적용한다.
- V1·retargeting baseline·선택 후보는 같은 seed-reset texture protocol과 같은 GT 픽셀·카메라·공식 지표를 사용한다.
- 단계별 이득을 실제 SS+PBR joint 이득으로 바꾸어 보고하지 않는다. 선택 pair가 baseline과 같으면 중복 composition을 만들지 않는다.

새 machine worker `run_selected_pipeline_case_GPU03_v179.py`도 작성했다.
원래 Shape/PBR 생성·render 수식과 sampler는 그대로 사용하고 물리적 모델 검증만 확장했다.
**아직 해당 worker를 실행 큐에 넣지 않았다.** 실제 DEV 후보가 결정되고 자체 SS archive가 준비되면 연결해야 한다.

## 독립 PBR seed 확인

`selected_pbr_seed_confirmation.py`와 `collect_selected_pbr_seed_confirmation.py`를 추가했다.
실제 seed101/202 config, 두 seed 각각의 네 rank1900 완료, 전체 per-rank training trace,
source checkpoint, 실제 종료된 child 및 원본 frozen weight의 tensor data를 확인한다.
seed202를 seed101로 표기하거나 기존101 결과를 복제해서 통과시키지 않는다.

PBR 독립 확인의 렌더링 범위는 기존 **target8·GT-shape 고정4개 패널의 경향**이다.
독립 TRAIN stream과 초기4 정합 및 학습 완료를 확인하지만, 이 패널만으로 source preservation,
생성 geometry의 개선이나 전체 파이프라인의 seed별 유의한 개선을 주장하지 않는다.
그 근거는 별도의 실제 joint DEV32 및 최종 TEST 평가에서 얻어야 한다.

## 선택 모델 고정 도구

`selected_pipeline_freeze_v2.py`와 `freeze_selected_pipeline.py`를 구현했다.
실제 기록을 읽는 CPU entry는 아래 자료가 모두 있을 때에만 `freeze.json`을 생성한다.

1. 전체 원래12개 SS 및 별도 E 두 모델의 실제 DEV128×3 결과.
2. 실제 측정한 SS+PBR joint 선택 pair와 V1/retargeting 두 strong baseline.
3. 선택 SS와 중복을 제거한 control의 실제 독립 TRAIN seed101/202 확인.
4. 선택 PBR과 control의 실제 독립1900 학습 및 명시된 fixed4-stage 확인.
5. 실제 두 번의 validation36 모델 load, prefix32 정합 및 tail4 정합.
6. TRAIN9149/VAL508과 source identity가 분리된 TEST508/134-source, 전체 cache QA 및 고정 noise.
7. 문서화된 DEV 선택 이유와 원래 네 multi-view checkpoint/config.
8. 10개 metric의 역할·방향·카메라, metric/배포 지표 registry 및 생성 template/source의 실제 hash.

NN/source/config의 실제 primary 및 independent-model byte를 다시 읽어 확인한다.
원본 checkpoint의 로컬 복사 위치가 달라도 실제 SHA가 같은지 확인하며, 다른 가중치를 같은 모델로 취급하지 않는다.
모델 고정은 효과의 증명이나 TEST 실행이 아니다. 최종508 생성·full cascade·10개 지표 완료 flag는 여전히 false다.
모든 모듈을 반드시 채택하도록 강제하거나 평균이 높은 작은 DEV 차이만으로 성공을 선언하지 않는다.

## 검증 결과와 실제 상태

관련 검사 **221개 통과**. 새 도구의 CLI도 CPU-only 환경에서 확인했다.
새 selected 계약을 기존 실제 V1·Top1+R·Full SS archive3개와 실제 PBR3개 모드에 적용해
archive/GT/noise/count pairing 및 PBR의 실제 네 rank 완료를 확인했다.
PBR101 binding은 기존 validator가 생성한 값과 정확히 같았다.

실제 검증 경로는 `unified-module-pipeline-v1/selected-pipeline-contracts-actual-preflight-v179`다.
실제 file4 SHA256/size PASS. 준비 자료는 `selected-pipeline-freeze-preparation-v179`에 저장했다.
이 자료는 새 모델이나 seed202가 이미 완료됐다는 근거가 아니다.

관측 당시 SS 진행은 C_RAW261, C_SPATIAL253, C_D1_COS261, C_DECODER_NO_R269,
FULL_QK_ONLY174, FULL_VOXEL_Q54 /480이었다. 새 E 두 모델의17개 작업은 GPU 대기 중이다.

실제 후보 선택, 선택 후보의 case/metric 작업 연결, 독립 seed 학습 활성화, 실제 validation36 두 번,
실제 freeze 및 이후 TEST508 실행은 남아 있다. 현재까지 실제 `freeze.json`을 만들거나 TEST 모델 forward를 실행하지 않았다.
