# 핵심 실험 범위와 GPU 네 장의 작업 시작 순서 정리

UTC 2026-10-23 기준. `wowo0709/feature-retargeting`.

기존 학습을 유지하면서 대기열의 자원 전환과 남아 있던 구 실험 분기를 정리했다. 모델·데이터·loss·학습 budget·평가 조건은 바꾸지 않았다. 새 GPU 작업은 추가하지 않았다.

## GPU 작업 시작 순서

현재 dispatcher는 필요한 GPU가 모두 비어 있을 때 DDP 작업을 시작한다. 일부 GPU가 비어 있어도 나머지 GPU에서 평가가 진행 중이면, 낮은 우선순위의 다른 평가가 빈 GPU를 다시 사용할 수 있다. 따라서 네 SS 대조군이 끝난 직후 대기 PBR DDP4가 네 장을 함께 확보하도록 별도 자원 permit을 연결했다. 아직 실제 지연이 발생했다는 뜻은 아니다.

동작은 다음과 같다.

1. 네 SS 작업이 진행 중일 때는 빈 GPU에서 기존 평가를 허용한다.
2. 네 SS science 작업이 terminal이고 모든 phase parent가 끝나면 새 후순위 GPU 작업의 시작을 잠시 막는다. 이미 실행 중인 작업은 완료한다.
3. 대기 중인 raw-Top-K PBR이 GPU 네 장을 함께 확보해 기존 command/config로 학습한다.
4. PBR의 실제 parent와 소유한 process session이 모두 종료되면 후순위 작업을 다시 허용한다. 실패·보류로 끝난 경우도 GPU 자원은 반환하되 과학적 성공으로 바꾸지 않는다.

CPU controller는 현재 PID `743157`, birth `1792775792.83`으로 실행 중이다. CPU affinity 92·93, CUDA visibility는 빈 값이다. 현재 permit은 열려 있다. 네 SS 종료 이후의 실제 coalescence와 PBR 시작·반환은 추후 관측해야 한다.

대기 중인 후순위 GPU descriptor 264개에는 자원 permit 한 항목만 `requires`에 추가했다. 원래 command, scientific gate, dependency, output, priority를 유지했다. 기존 raw PBR과 승격된 SS 12개 phase의 descriptor 전체, 실행 중인 작업, GPU 6·7 대기열은 유지했다. queue state는 직접 수정하지 않았다.

관련 정책과 실제 orphan process/session 종료 검사 **24개가 통과**했다. 자원 permit은 로컬 scheduler 제어 파일이며 scientific PASS나 dataset artifact가 아니다.

## 남아 있던 구 warm-source 분기 보류

추가 점검에서 `native-warm-engineering-v1`, `native-joint-primary480-v1`의 pending 항목 89개를 발견했다. GPU 작업 79개에는 warm-source parity/capacity/training/evaluation과 64개 case cascade 등이 포함된다. CPU 항목은 admission·selection·summary 등 10개다. 구 cascade는 PBR762를 사용해 현재 native PBR16850 continuation과 비교 조건도 다르다.

이 분기는 현재의 finite12 cold native 모듈 비교에 포함되지 않는다. 핵심 SS/PBR/공통 cascade/독립 seed 작업의 모든 job dependency와 조상 dependency의 required output을 조사해 충돌이 없음을 확인한 뒤, 해당 89개의 gate만 명시적 HOLD로 바꿨다. 실제 dispatcher가 **89개 모두 HELD**로 처리한 것을 확인했다. 기존 파일과 결과는 보존했다. 실행 중인 학습은 중단하지 않았다.

현재 실제 대기열은 두 변경을 적용한 예상 JSON과 값·type까지 일치한다. 기존 세 NN 작업의 bound source 286/286/53개 SHA도 모두 유지된다. 검증 산출물은 `unified-module-pipeline-v1/GPU03-resource-coalescence-v172`, `legacy-warm-scope-hold-v172`, `resource-scope-actual-audit-v172`에 보존한다.

이 조치는 모듈의 품질 향상이나 실험 완료를 의미하지 않는다. SS 480-step/DEV128×3, 공통 생성 geometry에서의 PBR 비교, 선택 SS/PBR 조합, 독립 seed, DEV freeze와 TEST508의 10개 metric 검증을 이어간다.
