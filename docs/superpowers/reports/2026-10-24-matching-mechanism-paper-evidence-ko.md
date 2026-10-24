# Matching 기전 근거의 재사용과 해석 범위

기존 완료된32개 development source cluster의 matching audit와 production-parity audit를 canonical file4 closure로 확인했다. 모델 forward나 학습을 새로 실행하지 않았다. 현재 E_FIXED_RAW 설정과 동일한 frozen D1 checkpoint SHA256을 확인했다.

분모는 **관측 가능한 source-camera ray / target patch 관계50,493개**다. 한 ray가 여러 target patch를 포함할 수 있으며, 이 수치를 고유3D point 또는 pixel matching accuracy로 부르지 않는다. GT observability는 평가 분모에만 사용한다. 모델의 feature 전달에 surface visibility gate를 추가한 것이 아니다.

| Matching 방식 | Exact@1 | Exact@64 | 해석 |
|---|---:|---:|---|
| Raw cosine·mean-ray RAW64 support | 1.25% | 50.76% | 원래 후보와 순위 |
| D1 contextual descriptor·같은 RAW64 | 7.73% | 50.76% | 같은 후보 안에서 순위 개선 |
| D1 scalar score·같은 RAW64 | 8.56% | 50.76% | 추가 순위 개선; 후보 포함률은 그대로 |
| Member-posterior mixture64 | 11.19% | 73.36% | 최종 후보 수64를 유지하며 support 개선 |

Member의 exact@64 개선은 +22.60 percentage points이며 탐색적 source-cluster paired95% CI는 [+15.99,+30.12]다. 같은64 support의 순위만 바꾸는 대조군과 support 자체를 바꾸는 member 설계를 구분한다.

Member에서 Top-1은 틀렸지만 정답이 Top-64에 있는 관계는 **31,393개**로, 전체 observable 관계의62.17%, Top-1이 틀린 관계 중70.00%다. 따라서 Top-K 후보의 정보를 활용할 여지는 크다. 그러나 이 사실만으로 learned attention의 유효성이나 최종 생성 품질 개선을 입증한 것은 아니다.

특히 scalar score가 ranking을 개선한 진단과, C_D1_COS→C_DECODER의 SS IoU 점추정이 감소한 결과는 양립한다. 대응 후보의 순위, 원본 feature의 결합, source-use와 retargeting 전달, 최종 geometry/appearance 품질은 서로 다른 단계의 근거다. 이후 최종 ablation은 이 관계를 연결해야 하며, 지표의 scope가 다른 두 실험을 같은 paired causal 비교로 합치지 않는다.

## 검증된 원자료

- `target-first-source-v1/matching-decoder-score-ray-audit-v2/audit.json`: 동일 RAW64·해당 실행 당시 label fingerprint·선택 D1 reference 재현 확인.
- `native-prior-source-ray-v1/member-proposal-audit-v1/audit.json`: 동일 분모의 support/ranking 비교 및 보정 전 cluster CI.
- `native-prior-source-ray-v1/member-production-parity-v1/audit.json`: production candidate address, probability, context, retained mass, member count와 진단 결과의 exact parity.
- 이번 추출과 동일 checkpoint 확인: `unified-module-pipeline-v1/matching-paper-mechanism-readout-v214/audit.json`.

실제 수치를 [논문 matching 진단 표](../paper/tables/dev-matching32.tex), 본문의 기전 해석, claim 원장에 반영했다. Final TEST claim이나 최종 model adoption은 수행하지 않았다.
