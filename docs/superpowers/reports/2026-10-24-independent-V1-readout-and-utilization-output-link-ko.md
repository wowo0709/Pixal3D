# 독립 V1 실제 집계와 matching 활용·출력 연결

2026-10-24 19:43 UTC 관측. 이전 goal turn은 실제 활용 진단·논문 수정/검수·commit/push를 완료한 progress였다. 이번에는 완료된 독립 V1 평가를 집계하고, 기존 DEV128 평가를 GPU1에도 연결했으며, 같은32개 활용 관측과 실제 출력의 연관성을 계산했다. 최종 모델 선정이나 TEST 실행 기록은 아니다.

## 독립 V1의 동일 DEV128×3 결과

| 실제 training seed | Macro voxel IoU | Precision | Recall | Voxel F1 |
|---|---:|---:|---:|---:|
| 101 | 0.467355 | 0.589160 | 0.681455 | 0.628705 |
| 202 | 0.467177 | 0.591355 | 0.678820 | 0.628804 |

동일128 asset·93 source cluster의3개 generation noise를 asset 안에서 먼저 평균했다. Seed202−101의 IoU는−0.000178, 탐색용 source-cluster CI95는[−0.001809,+0.001317]이다. Epoch quick/pool IoU0.434774와 비교하지 않았다. 두 실제 V1 checkpoint는 비슷한 점추정을 보이지만, CI가0을 포함하는 것이 동등성 검정은 아니다.

검증한 항목:

- 각 seed의 실제 raw480 checkpoint/config/bundle·평가3개·returncode0 terminal의 file4 SHA256/size와 모델 binding.
- Asset·source cluster·각 generation noise·GT voxel hash의 두 모델 간 일치.
- Native model/data/optimizer numerical recipe의 training seed 외 공통 조건 일치.
- 각 모델의480개 실제 training trace와 startup RNG가 서로 다름. 원래 LoRA 초기 hash는 일치.
- 기존 seed101 보고값 재계산 일치. 기록된 평가 child PID/birth는 현재 종료 상태이며 과거 terminal에 없는 session-drain 필드를 소급해 만들지 않음.

이 결과는 V1 기준선의 독립 실행 확인이다. R/최종 후보의 효과나 PBR/full-cascade의 독립 재현성이 완료됐다는 뜻은 아니다. R seed202는 계속 학습 중이다.

V1/R 두-seed 통합 collector는 `R-native-two-training-seeds-101-202-final128x3-summary-CPU-v1-data-preservation-v156-actual-PBR-recovery-v158-storage-successor-v185`라는 기존 job으로 정상 pending임을 확인했다. Source295개 모두 hash가 일치하며 V1 평가3개와 cache 종료 dependency는 완료, R 평가/cache dependency는 대기다. 이름 필터로 오래된 HELD job만 보고 collector가 없다고 판단하지 않는다. 새 통합 collector를 중복 등록하지 않았다. 실제 검증은 `existing-baseline-two-seed-summary-verification-v277`에 file4 보존했다.

`independent-V1-actual128x3-readout-v274/audit.json`, `per-asset.json`, `file4-closure.json`에3개 산출물95,889bytes를 보존했다. 선행 CPU 집계v273의 native alias lookup 오류도 별도로 보존했다. Native variant `vanilla`를 logical arm `V1`로 잘못 조회한 CPU reducer 문제였으며 새 NN 실행이나 데이터 변경 없이 올바른 참조 키로 집계했다.

## 같은 사례의 probability 변화와 출력 변화

실제 Fixed/Learned 활용 진단32개의 per-asset 차이를 기존 cascade32의 같은 asset 결과와 연결했다. 설명 변수는 prior Top-1 miss/support-hit subset의 **Learned−Fixed exact GT mass**다. 출력 값은 모두 양수가 개선을 뜻하도록 정했다.32개는 서로 다른 source cluster이고4000회 source resampling, seed20261024를 사용했다.

| 출력 개선 지표 | Spearman rho | 탐색용 CI95 |
|---|---:|---|
| CD 감소 | −0.0828 | [−0.4280,+0.2882] |
| F-Score 증가 | +0.0623 | [−0.2831,+0.3943] |
| Source LPIPS 감소 | +0.1683 | [−0.2024,+0.5138] |
| Source DreamSim 감소 | +0.0400 | [−0.3365,+0.4199] |

네 지표에서 모두 뚜렷한 연관성을 확인하지 못했다. 현재32개에서는 학습된 head가 exact GT mass를 바꾼 정도로 asset의 출력 개선을 설명할 근거가 부족하다. 이 진단은 asset-level association이며 relation별 local surface error나 인과 개입이 아니다. Exact patch GT만으로 유용한 semantic 후보 전체를 표현할 수도 없다. Learned head의 무용성 증명이나 새로운 loss/architecture의 근거로 과장하지 않는다.

다음 판단은 진행 중인 동일 DEV128 pipeline 비교와 최종 문맥의 Top-1/Uniform-K/score-weighted Top-K matched 학습으로 한다. 작은 panel의 상관관계를 보고 새 head나 temperature grid를 추가하지 않는다.

`trained-ray-utilization-mechanism-v259/paired-output-association-v275`에 audit·per-asset 값·worker·4-panel PDF/PNG의5개 산출물214,975bytes를 file4 SHA256/size로 확인했다. 원본64개 관측·generation·GT를 수정하거나 재생성하지 않았다.

## 기존 DEV128 평가의 GPU1 활용

GPU1이 비어 있음을 실제 장치 사용량과 queue에서 확인한 뒤, 대기 중인 Fixed 사례24건의 실행 위치만 GPU2에서 GPU1로 옮겼다. 원래 job name·output·dependency를 유지하므로 기존384-case coverage/official metric chain의 입력은 같다. 기존의 닫힌 plan은 수정하지 않고 별도의 successor를 만들었다.

변경은 geometry/replay의 assigned GPU와 wrapper의 GPU/CPU affinity 및 plan pointer뿐이다. Model/config/data·noise·surface seed·native sampler·NN numerical source는 그대로다. 원래 pending tail 사례만 교체했으며 진행 중인 worker나 queue-state는 수정하지 않았다. 첫 GPU1 사례는 실제 Shape/PBR/render/reference/shared scoring과 file4 보존을 완료했고 다음 사례가 실행 중이다.

19:43 UTC의 추가288건은99개 완료·4개 실행·185개 대기였다. GPU2 Fixed·GPU3 R·GPU6 Learned에 GPU1 Fixed가 추가됐다. GPU0은 R seed202, GPU7은 voxel 대조436/480이다. GPU4/5는 사용하지 않았다. 수치는 관측 시점의 기록이며 DEV128 전체 metric 완료를 뜻하지 않는다.

재배정 plan·원래/후속 queue·변경 대조는 `paired-DEV128-stage-evaluation-v240/GPU1-pending-reassignment-v272`의76개 산출물33,893,250bytes로 file4에 보존했다. 동일 case의 중복 실행이나 새로운 science arm을 추가하지 않았다.

## 논문 반영과 남은 필수 작업

V1의 독립 readout과 실제 활용→출력 association을 영문 초안과 claim 원장에 반영한다. 미완료 범위는 [최신 필수 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)을 유지한다: DEV128/voxel 종료, 최종 구조와 재사용 감사 후 matched ablation, R/선택 SS 및 PBR 독립 확인, 최종 freeze, TEST508·10개 metric, 최종 표·그림·논문 검수다.

Runtime root는 `agent-handoff.json`의 `PBR_NATIVE_SCIENCE_RUNTIME` 아래 `unified-module-pipeline-v1`을 기준으로 한다. Training seed2개를 임의의 training seed population으로 확장한 CI를 보고하지 않는다.
