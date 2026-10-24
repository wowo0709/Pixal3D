# DEV128 metric 집계의 실제 출력 통합 검증

2026-10-24 18:16 UTC 관측. 최종 pipeline 평가가 완료된 뒤 schema 오류를 발견하지 않도록, 기존32개와 새1개의 실제 결과를 사용해 production coverage/assembly 경로를 검사했다. 새 모듈 효과·DEV128 최종 성능·TEST 결과를 얻은 작업은 아니다.

## 실제 검사 범위

- 기존 R/Fixed/Learned의32개씩 생성·render·공식 score를 재사용했다.
- 추가 asset 하나에서 세 모델의 실제 render9개 view를 GPU6의 원래 official-perceptual entry로 평가했다. FP32·TF32 off·no AMP를 유지했다.
- 총33 asset·33 source cluster·3모델·297 view를 production metric worker의 coverage/summary AST로 조립했다. 기존 full128 archive 검증은 그대로 수행했다.
- Rehearsal에서만 view count를864→9,1,152→297로 바꾸고, 선택한 사례의 원래 index를 유지했다. 원래128-case 통계 함수와 W&B final 집계 호출 직전에 멈췄다. Production worker와 numerical metric entry는 수정하지 않았다.

## 발견한 오류와 수정

첫 rehearsal `metric-integration-rehearsal-v253`은 기존 R case audit의 `actual_SS_model_binding` 필드가 없어 실패했다. Fixed/Learned의 새로운 기록에는 이 필드가 있지만, 과거 R32의 실제 audit schema에는 없었다. 기존 R 생성/평가의 실패가 아니라 새 집계 코드가 역사적 schema를 지원하지 못한 문제다.

기존 audit에 임의의 binding을 넣거나 원본을 수정하지 않았다. 각 R32 사례에 대해 다음 실제 closed evidence를 대조하고 새 호환 기록을 생성했다.

1. 원래32개 SS 평가의 config·denoiser·conditioner 경로/hash와 실제480 model binding.
2. 원래32개 per-asset 기록과 새 full128 capture의 prefix32가 정확히 일치함.
3. 각 사례의 predicted SS SHA256, 생성 geometry SHA256, native guide SHA256 일치.
4. 실제 Top-1 R PBR의 config·denoiser·conditioner hash와 성공한3개 child terminal.
5. 이미 완료된 prior cascade summary의 실제 SS/PBR composition이 현재 binding과 정확히 같음.

새 index는 R의 첫32개 `case_audit/case_receipt`만 이 호환 기록을 가리킨다. 기존 geometry·PBR prediction·render·score·그 외352개 사례 경로와 모든 model binding은 유지했다. File4에 업로드한 bytes의 SHA256/size를 실제 원격 값과 대조했다.

## 통과한 실제 검증

`metric-integration-rehearsal-v256`에서 다음을 확인했다.

| 검증 | 실제 결과 |
|---|---|
| 조립한 panel | 33 asset·33 source cluster·3모델·297 view |
| 기존 official score 재사용 | 288 view |
| 새 실제 official score | 9 view |
| 기존32개의 per-asset/role metric 비교 | 2,784개 값 정확히 일치 |
| 기존 metric의 최대 절대 변화 | 0.0 |
| Full128 archive/model identity 검사 | 원래 검사 유지·통과 |
| Camera/GT pixel/geometry/native guide/replay 연결 | 실제 파일·기록으로 검사·통과 |
| Rehearsal CPU의 NN forward/학습 | 0 |
| 실행 중인 기존 NN 재시작 | 0 |

9개 official view는 작은 integration check의 실제 GPU forward다. 전체864개 official view 완료와 혼동하지 않는다. Rehearsal의33개를 새로운 model-selection panel이나 논문 성능 표로 사용하지 않는다. 성공한33개 integration은 전체384개 case 완료를 대신하지 않는다.

## 실제 집계 체인 교체

아직 실행되지 않은 v250의 coverage/official/summary3개 job만 queue에서 제거하고, 검증된 index를 사용하는 v257의3개 job으로 교체했다. 원래 대기 상태와 command/plan은 canonical 기록으로 보존했다. Queue state 파일을 성공으로 변경하지 않았다.

- `legacy-R32-binding-compatibility-v254`: 실제 근거로 확인한32개 호환 audit와 successor index.
- `metric-integration-rehearsal-v253`: 첫 schema rejection과 원래 실패 기록.
- `metric-integration-rehearsal-v256`: 실제33개 assembly·2,784개 exact-value 검증.
- `execution-prelaunch-v257`: 새 coverage→official→summary 계획과 교체 전 pending job 기록.
- `metrics-v257/summary-v1/audit.json`: 전체 완료 후 기록될 최종 DEV128 집계 위치. 관측 시 아직 없다.

Production worker는 기존 `run_DEV128_metrics_CPU_GPU_v246.py`를 그대로 사용한다. 최종 chain은384개 실제 case,864개 새 view+288개 재사용 view,128 asset·93 source cluster를 요구한다. 모델·render·metric numerical code와 실행 중인 case job은 변경하지 않았다.

## 같은 시각의 실제 학습/평가 상태

- 추가288개 case: **25개 완료·3개 실행·260개 대기**. 기존32개×3의 재사용은 이 추가 수에 포함하지 않는다.
- SS seed202 V1:426/480, R:252/480. GPU1/0의 원래 PID·command가 계속 살아 있음을 확인했다.
- FULL_VOXEL_Q:398/480. GPU7의 원래 PID·command가 계속 살아 있다.
- DEV128 case wrapper GPU2/3/6도 실제 PID/command가 살아 있음을 확인했다. 단계 간 CPU 작업/공통 reference 대기에서 GPU 사용률이 순간0일 수 있다.
- Local free space는약685GB였다. Final TEST 전체 출력 저장/retirement 계획을 대신하는 수치는 아니다.

학습/평가 수치는 계속 변한다. Full128 pipeline 수치와 최종 구조의 missing matched ablation, 독립 seed 확인, 모델/protocol freeze, TEST508의10개 metric과 최종 writing은 계속 남아 있다.

관련 계획: [논문 완성을 위한 핵심 실험·ablation·writing](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md), [DEV128 실제 실행](2026-10-24-paired-DEV128-actual-launch-and-reuse-ko.md).
