# 물리 모델별 평가 계획 준비와 PBR 실행 연결 복구

기준 시각: 2026-10-24 00:15 UTC. 평가 계획 생성기 구현 커밋 `d35678f7dc565ee04d7d7d600077b7925dccaa2b`.

## 실제 진행 결과

GPU 0~3에서 진행하던 raw semantic, spatial, decoder cosine, decoder correspondence에서 retargeting을 제거한 네 SS 대조군이 모두 480 update를 완료했고, queue 정상 종료 및 file4 저장을 확인했다. SS는 global/micro batch 96, accumulation 1을 유지했다.

같은 DEV32·학습 seed 101·생성 noise 1개의 asset별 평균은 다음과 같다.

| 모델 | 5에폭 IoU |
|---|---:|
| Vanilla V1 | 0.47327916 |
| Decoder correspondence, retargeting 제거 | 0.47308540 |
| Spatial correspondence + retargeting | 0.47308443 |
| Raw semantic Top-1 + retargeting | 0.47390104 |
| Decoder feature cosine Top-1 + retargeting | 0.47409496 |
| Matching decoder Top-1 + retargeting | 0.47696710 |
| 기존 Full | 0.47824306 |

이 패널에서 raw semantic−spatial은 +0.00081661, matching decoder−raw semantic은 +0.00306607, retargeting 유지−제거는 +0.00388170이다. 같은 입력·seed·GT에서의 비교지만, 작은 한 seed 패널이므로 최종 모듈 효과를 확정하지 않는다. Full−Top-1+R도 Top-K·readout을 함께 바꾼 조합 차이다. 이미 완료한 DEV128×3에서 학습형 attention의 추가 이득이 확인되지 않았다는 판단은 유지한다.

GPU 6·7에서는 score prior 제거와 개별 voxel query 대조군이 실제 NN child를 유지하며 진행 중이다. 별도 W&B 평가 run에 위 7개 모델의 epoch5 표와 원본 평가 artifact를 올리고, remote API에서 7개 수치 및 artifact 등록을 확인했다. Mean asset IoU와 pooled IoU는 별도 열로 기록했다.

[W&B: 동일 DEV32의 5에폭 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-unified-interim32-epoch5-s101-v184)

## 평가 준비에서 완료한 부분

새 `selected_population_plan_factory.py`는 원래 12개 대조군과 두 shared-evidence 후보의 실제 checkpoint/config/source binding을 그대로 유지해 validation36 또는 동결 후 TEST508 SS 실행 계획을 만든다. 모델 선택, neural forward, 학습, queue 설치는 수행하지 않는다.

Validation 계획은 자신의 완료된 final128 평가에서 같은 36개 prefix와 noise를 사용해야 한다. TEST 계획은 실제 DEV 기반 freeze, 동일 모델의 2회 독립 로딩 및 tail4 parity, 전체508 cache QA, train/validation/test 분리와 원래 sampler를 검사한다. 기존 recipe가 외부 manifest 경로를 유지하는 경우에는 명시한 file4 복사본과 원본의 실제 byte hash가 일치해야 한다. 데이터 선택 외에 모델 구조나 계산을 바꾸지 않는다.

관련 CPU 테스트 **217개 통과**, 새 테스트 54개다. 전체 14개 구조의 두 scope와 잘못된 checkpoint, source, reference, noise, coverage, freeze, parity, matcher, manifest 복사본을 검사했다. 실제 V1·Top-1+R·Full의 완료된 checkpoint/config/file4 receipt로 각각 validation36 계획을 생성하고, 새 계획도 실제 file4에서 hash/size를 확인했다. 이는 계획 준비 검증이며, 해당 세 모델의 새로운 validation36 GPU 재로딩이나 TEST 실행을 완료했다는 의미가 아니다.

## PBR 전환 중 실제 오류와 복구

기존 raw Top-K PBR 작업이 시작될 때 `storage-admission-v1/after-top1-plus-feature/file4-closure.json`가 없어 prelaunch guard에서 종료됐다. 원인은 이전 feature PBR 복구 경로의 저장 공간 확인 작업이 오래된 선행 job ID를 가리키며 HELD 상태였기 때문이다. 이 실패는 PBR neural child·wrapper 출력 생성·optimizer update **이전**이었다. GPU OOM이나 학습 발산은 아니었다.

실패한 queue state와 로그는 보존했다. 실제 완료한 feature PBR의 네 rank·terminal·file4 기록을 확인하고, 현재 디스크 공간과 진행 중인 checkpoint 예약량으로 다시 admission을 검사했다. 당시 여유 공간은 약 848 GiB였고, 새 checkpoint budget 및 workspace를 충족했다. Checkpoint를 삭제하지 않고 실제 disk admission과 receipt를 생성한 뒤, 원래 PBR의 모든 source/input/scientific prelaunch 검사를 직접 재실행해 통과했다.

Raw PBR 복구 job은 **원래 worker command·science plan·config·출력 경로를 그대로 사용**한다. 새로운 job ID로 append했으며, 원래 FAILED 항목을 성공으로 바꾸거나 queue state를 편집하지 않았다. 기존 resource controller 종료를 확인한 뒤 같은 backfill permit의 관리를 새 controller로 넘겨 GPU 0~3을 모았다. 그 사이 이미 시작된 MeanMatch의 proof와 임시 3-update capacity 검사는 정상 종료까지 유지했고, capacity도 통과했다. MeanMatch 480-update science는 raw PBR 이후에 실행하도록 계속 대기한다.

실패 때문에 HELD로 바뀐 필수 후속 작업 68개에는 새 dependency ID를 사용하는 successor를 연결했다. 원래 descriptor와 실패/HELD 이력은 유지했고, 실험 arm이나 학습 반복을 추가하지 않았다. 66개 command는 byte 단위로 동일하고, 두 CPU cache 처리 계획은 실제 완료 dependency ID만 바꿨다. 오래된 네 모델 TEST 자동 활성화 3개 및 불필요한 R_feature validation 3개는 되살리지 않았다. 현재 후보를 지원하는 DEV 선택·동결 절차가 여전히 필요하다.

복구 raw PBR wrapper PID **1013382**, 실제 neural child PID **1013437**를 확인했다. 실행은 GPU 0~3 DDP4, global batch 24·GPU당 micro batch 6·accumulation 1·1900 update 설정이다. 이 시각에는 초기 로딩 중이므로 첫 optimizer update나 학습 정상 수렴을 아직 주장하지 않는다. 정상 SS worker나 dispatcher는 재시작하지 않았다.

## 남은 완료 조건

남은 SS 대조군 및 shared-evidence 두 후보의 실제 학습과 DEV128×3, 실제 SS→Shape→PBR joint 평가, 선택 모델과 강한 기준선의 독립 학습 seed 검증이 필요하다. 이어 자신의 validation36 두 fresh-load, DEV 선택 및 설정 동결을 완료한 후 TEST508 전체 cascade와 10개 metric을 계산한다. 기존 네 모델 전용 TEST 활성화 경로를 현재 물리 모델별 실행 경로로 연결하는 작업도 남아 있다. 전체 목표는 진행 중이다.

실제 저장 및 검증 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `selected-population-factory-actual-preflight-v184/`, `focused-epoch5-SS32-v184/`, `paired-epoch5-SS32-WandB-v184/`, `raw-PBR-storage-admission-recovery-v185/`, `raw-PBR-required-successors-v185/`에 보관했다.
