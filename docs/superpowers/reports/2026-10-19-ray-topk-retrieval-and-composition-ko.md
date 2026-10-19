# Ray-wise semantic Top-K: 검색과 조합 병목 분리

2026-10-19 UTC. 목표는 정확한 Top-1 선택률 자체가 아니라, 유용한 correspondence를 Top-K에 포함하고 그 정보를 조합해서 SS 성능을 높이는 것이다. 현재 관측은 이 방향을 포기할 근거가 아니다. 다만 검색과 조합을 별도로 측정해야 한다.

## 실제 저장 모델과 진단 범위

현재 Full raw480 conditioner를 고정한 채 기존 개발32개를 원래 순서로 실행했다. 입력은 cached RGB/Omega/DINO와 원래 camera/SS16 projection geometry다. 물리 GPU0에 allocator20% 상한을 두었으며 실제 peak allocated는0.388GiB였다. SS denoiser/decoder를 로드하거나 optimizer update를 하지 않았다. 기존 학습 두 프로세스와 원래 numerical source는 유지했다.

GT topology label32개는 모두 존재하며, source mesh·M1 vertices·실제 RGB·camera의 현재 fingerprint와 일치했다. GT label은 CPU metric 함수에만 전달하며 query/proposal/attention/model condition을 만들지 않는다. 이 probe는 cache 기반 conditioner 진단이고 online DINO equality, SS 생성 성능, 독립 test 또는 training-seed 재현의 증거는 아니다.

Ray 하나가 여러 target patch를 대표하므로 한 정답 주소만 지정하지 않는다. **중복을 제거한 `(source-camera ray, target patch)` 관계**를 구성하고, 해당 target patch의 M0 topology correspondence가 source에서 관측 가능하며 실제 source patch alpha mask로 검색 가능한 관계를 분모로 사용한다. 한 ray에서 하나만 맞히는 느슨한 hit rate와 구별한다.

- Active ray21,845개, distinct ray/target-patch 관계83,783개.
- 관측 가능한 관계50,509개, 해당 관계를 포함하는 ray18,913개.
- 나머지33,274개 관계는 관측 가능한 정답이 없는 조건이므로 검색 실패 분모에서 제외했다.
- 정확한32² patch 일치와 같은 source view의 Chebyshev±1 patch를 별도로 집계한다.

## Top-K에는 Top-1이 놓친 정보가 있는가

| 검색/순위 | 정확한 patch coverage | ±1 patch coverage |
| --- | ---: | ---: |
| Raw pooled FRAME의 첫 후보 | 1.24% | 4.08% |
| Frozen D1+scalar score의 Top-1 | 8.57% | 28.32% |
| 현재 learned attention의 Top-1 | 7.72% | 27.21% |
| 공통64개 후보 전체 | **50.77%** | **80.48%** |

Scalar Top-1은 놓치지만 같은 Top64에는 있는 관계가 전체 분모의42.20%p(exact),52.16%p(near1)였다. 따라서 Top-1이 어렵다는 사실만으로 Top-K 조합의 가능성을 부정할 수 없다. 동시에 exact49.23%·near1 19.52%는 아직64개 밖에 있으므로 검색 병목도 남아 있다.

Frozen D1+scalar reranking은 raw 순서보다 개선됐지만 이 비교는 decoder와 scalar score의 결합 효과다. Decoder 단독 효과의 증명으로 해석하지 않는다. Pure decoder/score-prior ablation의 SS 결과는 진행 중이다. 이 표의 Top-1은 여러 target patch를 포함하는 ray의 coverage이며 기존 단일 patch matching accuracy와 다른 metric이다.

## 현재 attention은 그 정보를 잘 활용하는가

| 동일한64개 후보, 실제 Full480 | Frozen score prior | Learned attention |
| --- | ---: | ---: |
| Ray 내 exact GT 대응 주소 집합에 주는 평균 probability mass | **17.89%** | 17.11% |
| Ray 내 near1 대응 주소 집합에 주는 평균 probability mass | **43.67%** | 42.68% |
| 유효 후보 수 `1 / sum(p²)` | 3.92 | 3.22 |
| 최대 후보 probability | 54.53% | 59.80% |

현재 learned attention은 prior와 다른 가중치를 만들지만, 이 topology 기준에서는 유용한 주소의 비중을 높이지 못했다. Mean probability L1 차이는0.4405로 거의 같은 출력만 내는 경로도 아니다. SS loss에 연결됐다는 검사와 별개로, correspondence 활용의 개선은 확인되지 않았다.

이는 단순히 가중치를 더 평평하게 만들면 된다는 뜻도 아니다. 동일 후보의 uniform 참고값은 exact GT 집합 mass2.08%·near1 10.14%로 prior보다 낮다. Uniform은 공식 No-score arm이 아니다. 공식 No-score는 Top-1 feature만 사용한다. 또한 topology 기반 probability mass는 SS에 최적인 가중치의 정답을 정의하거나 성능 저하의 인과성을 입증하지 않는다. 반복 appearance, patch aliasing과 task-specific 유용성은 이 metric만으로 완전히 설명하지 못한다.

## 검색을 바꾸는 최소 prototype

평균 Omega query로64개를 검색하는 대신, ray가 포함하는 **서로 다른 target patch의 RGB/Omega proposal**을 먼저 구하고 rank별 round-robin으로64개의 고유 source 주소를 합치는 prototype을 구현했다. Ray query 단위·camera membership·최종64개 예산은 유지하고 GT를 후보 생성에 사용하지 않는다. 기존 여섯 arm의 학습 코드에는 연결하지 않았다.

| 동일32개·동일50,509 관계·K64 | 기존 pooled query | Patch proposal union | 차이와 source-cluster95% CI |
| --- | ---: | ---: | --- |
| Exact coverage | 50.77% | 52.17% | +1.39%p [−1.44,+4.47]%p |
| Near1 coverage | 80.48% | 83.84% | +3.36%p [+0.81,+6.18]%p |

Baseline의 ray membership·분모·exact/near1 K64 count는 앞선 actual Full probe와 모든 사례에서 일치했다. Source32개 cluster bootstrap4,000회, seed42017의 개발 진단이다. Exact 개선은 미확정이며 near1의 양성 관측도 SS 생성 개선을 뜻하지 않는다. Source/ray feature readout이나 reranker는 이 prototype에서 학습하지 않았다. GPU peak는0.093GiB였다.

첫 독립 prototype은 membership을 BF16 autocast 밖에서 계산하여 원래 ray 분모와 달랐고 strict 비교에서 거부됐다. 그 시도는 결과를 채택하지 않고 보존했다. 실제 통과한 비교는 원래 model autocast를 유지하며 기존 분모와 count를 모두 재검증한 `bf16-matched-v1` 실행이다. 비교를 통과시키기 위해 분모를 바꾸거나 검사 허용 오차를 늘리지 않았다.

## 후속 검증 방향

1. 현재 여섯 arm의 고정 raw480·동일128×3 SS 평가를 완료한다. Full−가중합, 가중합−Top1, scalar prior 제거, decoder 제거, native spatial을 구분한다.
2. 검색 변경은 고정K에서 ray가 필요로 하는 여러 target patch의 정보를 더 포함하는지 확인한다. Patch union의 개발32개 관측을 독립 test 성능으로 부르지 않는다.
3. 조합 변경은 기존 matching score가 가진 유용성을 유지하면서 여러 correspondence의 보완 정보를 활용해야 한다. 단일 Top-1 정답 강제나 uniform 평균을 목표로 삼지 않는다. Ray 단위 multi-positive supervision이나 여러 constituent token을 보존하는 readout은 후보 설계이며 아직 효과를 주장하지 않는다.
4. 기존 양성 R은 target-anchor query/carrier였다. Ray readout 개선을 다음 학습에 넣을 때 이 검증된 R 경로까지 동시에 바꾸면 검색·조합 효과를 다시 혼동한다. 기존 R을 유지하는 기준선과 추가 ray 경로를 비교하고, 추가 경로는 zero residual 초기화 및 실제 초기 출력 일치를 요구해야 한다.
5. 후보 채택은 SS 생성 성능과 source-cluster 불확실성으로 판단하고 추가 training seed로 확인한다. Matching metric 개선만으로 모델을 채택하지 않는다.

## 산출물

File4 결과 root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/diagnostics/`다.

- `ray-topk-coverage-v1/audit.json`: 실제 Full480의 candidate coverage·prior/learned probability 및32개 current-input GT fingerprint.
- `ray-patch-union-v1/bf16-matched-v1/audit.json`: 고정K64 proposal prototype과 baseline count 재현·source-cluster 비교.
- 각 폴더의 source snapshot은 실제 실행 코드와 metric 코드를 보존한다. CPU regression7개가 통과했다. 기존55개 frozen runtime binding과 진행 중인 학습은 바꾸지 않았다.
