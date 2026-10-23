# 선택 모델의 native TEST508 평가 연결 준비

UTC 2026-10-23. 브랜치 `wowo0709/feature-retargeting`.

기존 `evaluate_native_bound_population.py`는 예전 vanilla/R_top1_only/R_feature/R_raw_topk 네 모델만 지원했다. 현재 12개 모듈 비교에서 선정될 모델을 이 네 모델로 제한하지 않도록 별도 진입점 `evaluate_unified_bound_population.py`와 `unified_population_contract.py`를 추가했다. 기존의 checksum으로 고정된 evaluator·runtime·trainer는 수정하지 않았다.

새 진입점은 각 모델의 기존 checkpoint evaluator와 전용 source loader를 사용한다.

| Arm | 기존 evaluator 계열 | Source trainable tensor 수 |
|---|---|---:|
| V1 / Full | native R | 0 / 19 |
| Decoder Top‑1+R | physical Top‑1 | 10 |
| Top‑1+V / Fixed Top‑K | projected readout | 12 |
| Spatial / RAW semantic / D1 cosine | raw correspondence | 10 |
| No-prior / voxel query / mean proposal | unified design | 19 |
| Top‑1 without R | unified design | 5 |

Bundle schema·variant·480 step·모델별 denoiser/source/config 경로·실제 SHA256·정수 training seed를 검사한다. 다른 계열의 source checkpoint를 같은 이름으로 해석하지 않는다. 실제 평가 결과에도 LoRA 360개와 해당 source tensor 수가 일치해야 한다.

기존 evaluator의 panel 선택 함수에만 dataset binding을 적용한다. 모델·sampler·decoder 계산은 기존 진입점을 사용한다. 원래 online matching의 `None` root를 cached matcher root로 바꾸지 않는다. 원본 config는 실행 후 복원한다.

TEST 실행 조건은 기존 native population 검사를 그대로 유지한다. TRAIN9149/VAL508/TEST508의 asset·source UID·source cluster 분리, 전체 TEST508/134 source coverage, frozen TRAIN matcher와 Omega 23,597 update 캐시, DEV 기반 모델 고정, 실제 validation36의 독립 두 번 재로딩 및 tail4 재현이 모두 필요하다. 새 진입점과 해당 family의 기존 evaluator/runtime/recipe 등 핵심 source pin도 요구한다.

검사 결과는 다음과 같다.

- 새 계약 직접 테스트 59개와 기존 population 검사 33개, 총 **92개 통과**.
- 실제 12개 scientific config 모두 기존 전용 evaluator에 연결되고 TEST dataset 선택만 바꾼 뒤 나머지 값·JSON type을 정확히 복원함을 확인했다.
- 12개 모두 frozen TRAIN q2c와 준비된 TEST cache가 일치함을 확인했다.
- 이미 완료된 V1·Top‑1+R·Full 세 모델은 실제 file4 closure와 일치하는 config/denoiser/source bytes를 확인하고 새 binding을 구성했다.
- 나머지 9개의 최종 checkpoint가 완료됐다는 주장은 하지 않는다. 이들은 원본 scientific recipe의 경로 연결만 검사했다.
- CPU/GPU neural forward, TEST 품질 계산, 새 GPU 작업 추가는 모두 **0회**다.

실제 preflight는 `unified-module-pipeline-v1/population-adapter-v170/preflight-v1`에 저장한다. 이는 TEST508 실행 완료나 최종 모델 고정을 뜻하지 않는다. 새 12-model DEV 결과에 맞는 최종 freeze 생성, 선택 모델·강한 기준선의 독립 seed 확인, 선택된 SS/PBR의 실제 조합 평가와 validation36 GPU 재로딩을 마친 뒤 TEST를 실행해야 한다.

진행 중인 SS 두 작업과 PBR DDP4 작업은 중단·재시작하지 않았다. 실험 arm 수는 12개로 유지한다.
