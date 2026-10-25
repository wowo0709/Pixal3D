# 최종 파이프라인·모듈 검증·논문 완성을 위한 남은 작업

작성: 2026-10-25. 후속 관측: v409 완료3대조의 사전 family6 통계, v410 원격 게시, v412 실제 논문 반영·검수.

이번 정리의 남은 본 학습은 **correspondence 대조3개, 선택 Fixed의 SS 독립 seed1개,
PBR R 독립 seed1개 실행의 완료**다. PBR vanilla 독립 seed는 완료됐다. 앞의5개 신규 ablation 중 Uniform/no-R의
480 학습과 DEV128×3는 끝났으며 반복하지 않는다. 나머지 핵심은 같은128개 실제
cascade8metric, 독립 확인, freeze 후 TEST508의10metric과 표·writing 완성이다.
학습 작업 수와 평가/proof/capacity 작업 수를 구분한다.

후속 실제 실행 연결: [다섯 control의 cascade·8metric·논문 집계 등록](2026-10-25-final-fixed-cascade-metrics-continuation-ko.md). 기준선116개 평균값과 GT 재사용을 검증하고, own SS archive에 의존한 connector5개 및 최종 reducer를 등록했다. 아래의 후속 연결 미완료 설명은 해당 실행 기록으로 갱신하며, 실제 신규 cascade 품질은 계속 미완료다.

목표는 각 요소가 실제로 기여하는지 판별하고, 그 근거에 맞는 하나의 파이프라인과 논문을 완성하는 것이다. 모든 요소의 효과가 양수라는 결론을 전제하지 않는다. 이전 결과의 조건이 같으면 재사용하며 지엽적인 구조 탐색은 추가하지 않는다. 아래는 등록된 작업의 완료 조건과 논문 산출물을 정리한 것이다. 이번 문서 갱신으로 새 학습 arm을 추가하지 않는다.

### 남은 작업의 우선순위와 종료 산출물

| 순서 | 남은 실행 | 종료할 때 남겨야 하는 근거 |
|---|---|---|
| 1. 최종 문맥의 모듈 대조 | Spatial / Raw semantic / Decoder-cosine 각각480 updates와 SS128×3. 완료 Uniform/no-R는 재학습하지 않음 | 기존 Fixed/Top-1을 합친7행 SS 표와 사전 정의6대조 통계 |
| 2. 실제 출력 평가 | 신규5행×동일 DEV128의 cascade, 공통 frozen Shape와 Top-1 R PBR,8metric | 같은 checkpoint의 SS/geometry/appearance 표,48대조 family 통계. 준비된 arm부터 실행하며 나머지 학습을 기다려 일괄 시작하지 않음 |
| 3. 독립 확인·재로딩 | Fixed SS seed202480/128×3, PBR R seed2021900 및 두 seed collector, 선택 후보의 VAL36 두 fresh-load | 단일 학습 운·초기 함수 하락·checkpoint composition 문제와 최종 모듈 효과를 구분하는 실제 결과 |
| 4. 최종 모델 동결 | 앞의 근거로 실제 SS/Shape/PBR 조합·checkpoint·code·평가 protocol 확정 | TEST 전에 고정한 모델과 설정. Fixed의 추가 이득이 지지되지 않으면 Top-1 R 선택 가능 |
| 5. TEST·정성 평가 | TEST508/134 source cluster에서 두 zero-shot·V1·R·Ours의10metric, 고정 사례와 실패 사례 | Main table, paired source-cluster interval, 최종 정성 그림. Ours가 R과 같으면 중복 행 제거 |
| 6. 논문 완성 | 실제 수치로 Main/Ablation/Mechanism/Supplement와 Results/Discussion 작성, 마지막에 Abstract/Conclusion 확정 | 표·claim 원장·실제 checkpoint/결과가 일치하고 compile·시각 검수를 마친 PDF |

1–3은 의존성이 허용하는 범위에서 병행한다. 실제 cascade는 각 own SS 결과가
준비되는 즉시 연결하며,4의 동결은 모듈별 대조·독립 확인·재로딩 근거 뒤에 한다.
본 학습이 끝나도2·5·6이 끝나기 전에는 실험이나 논문 전체가 완료된 것이 아니다.

## 1. 현재 근거와 잠정 선택

현재 working candidate는 **Fixed SS + frozen Shape512/1024 + Top-1 retargeting PBR**다. Fixed는 matching probability로 Top-K value를 가중 결합하는 구조다. 추가 learned Q/K attention head는 없다. 아직 최종 TEST 모델로 동결하지 않았다.

- Top-1 semantic matching+retargeting branch 대 V1의 SS 개선은 두 실제 training seed에서 +0.005245/+0.005169다. 이것은 결합 branch의 근거이며 retargeting MLP 단독 효과가 아니다.
- 공통 DEV128 cascade에서 Fixed의 F-Score는 0.844058, Top-1 R은 0.840095다. 차이 +0.003963의 탐색용 source-cluster CI95는 [+0.001011,+0.007314]다.
- 같은 비교의 source LPIPS는 0.053438 대 0.053227, DreamSim은 0.151909 대 0.148809다. 두 지표 모두 낮을수록 좋으므로 preservation 개선을 주장할 수 없다. 차이의 CI가 0을 포함한다고 비열등성이 입증되는 것도 아니다.
- Learned refinement는 SS 점추정이 더 높아도 cascade geometry 우위가 확인되지 않았다. 원래 ray-vs-voxel 비교도 ray의 품질상 필수성을 입증하지 않았다.

따라서 현재 논문의 중심 질문은 “semantic 후보를 적절히 활용하고 source feature를 적응시키면 최종 geometry와 appearance에 무엇이 달라지는가”다. SS IoU 최고점만으로 Full을 정하지 않는다.

## 2. 남은 핵심 ablation: 기존 2행 + 신규 5행

모든 신규 행은 동일 원본 multi-view LoRA 초기값, DeformNeXt split, SS global batch96, 480 updates/약5epochs, 같은 seed·학습 stream·native conditioning 계약을 사용한다. 아래 matching ladder는 이웃 행끼리 비교한다.

| 행 | Correspondence / aggregation / retargeting | 비교와 질문 | 현재 상태 |
|---|---|---|---|
| M_FIXED_SPATIAL | 원래 spatial3 / Fixed 결합 / R | 다음 raw semantic 행과 비교: semantic 탐색의 효과 | 실제 GPU proof·micro96 capacity 완료, GPU6 본 학습 진행 |
| M_FIXED_RAW | Raw semantic64 / cosine weighting / R | Spatial→Raw: semantic correspondence의 묶음 효과 | 실제 GPU proof·micro96 capacity 완료, 본 학습 대기 |
| M_FIXED_D1_COS | Decoder semantic64 / cosine weighting / R | Raw→D1 cosine: contextual decoder의 효과 | 실제 GPU proof·micro96 capacity 완료, 본 학습 대기 |
| E_FIXED_RAW | Decoder semantic64 / scalar probability weighting / R | D1 cosine→scalar: scalar matching 모듈의 효과 | SS128×3·cascade128×1 완료 |
| 실제 Top-1 R | Decoder / 실제 최상위 feature1개 / R | Fixed와 비교: 여러 후보 활용의 효과 | SS·cascade 완료, 재학습 불필요 |
| A_FIXED_UNIFORM | Fixed와 같은64개 후보·value / uniform / R | Fixed와 비교: matching score weighting의 효과 | 480 학습·DEV128×3 완료, IoU0.473613; cascade 대기 |
| A_FIXED_NO_R | Fixed와 같은 후보·weight·전달 / R 물리적 제거 | Fixed와 비교: 최종 경로에서 retargeting의 효과 | 480 학습·DEV128×3 완료, IoU0.468832; cascade 대기 |

**최종 문맥의 신규 ablation은 총 다섯 개이며 남은 본 학습은 세 개다.**
Uniform/No-R 두 개의 학습·SS 평가는 완료됐고 matching 세 개는 등록돼 있다.
각 arm의 proof/capacity/evaluation 작업을 별개의 모델로 세지 않는다.

완료된 동일 DEV128×3에서 Fixed0.473471 대 no-R0.468832의 IoU 차이는
+0.004639다. R의 직접 기여를 지지하는 점추정이지만 전체 family CI와 downstream
평가 전 필수성을 선언하지 않는다. Uniform0.473613 대 Fixed0.473471의 차이는
+0.000142이므로 score weighting의 stage 우위는 확인되지 않았다. 이는 동등성이나
효과 부재의 증명도 아니다. 각 arm의 세 실제 종료·checkpoint·GT·ID를 기존
`validate_series`로 검사하고 actual file4 SHA/size 검증까지 완료했다.

후속 `completed-three-of-six-SS-contrasts-v409`에서 실제 완료된 네 모델을 다시
위 validator로 검증하고 사전 정의6개 대조 중 가능한3개를 집계했다. 원래 전체
reducer와 동일한12,000 draw·seed20261025·source-cluster resampling·family6을
유지했으며 미완료3개를 제외해 보정 수를 줄이지 않았다. Fixed−Top-1의 보정 IoU
CI95는[−0.001642,+0.003679], Fixed−Uniform은[−0.002590,+0.002233],
Fixed−No-R는[−0.000811,+0.010633]이다. 마지막의 일반 CI95는
[+0.000565,+0.009024]지만 family 보정 후에는0을 포함한다. 따라서 현재 direct R
효과는 긍정적인 점추정과 보정 전 구간으로 한정하며 최종 필요성으로 쓰지 않는다.
나머지 matching3개·전체7행·cascade·독립 확인은 계속 남아 있다. 실제12개
noise evaluation·GT/count/ID·checkpoint binding을 확인하고 명시적 cluster 반복
계산으로 bootstrap 수치도 대조했으며 실제 file4 SHA/size 검증을 마쳤다.

세 대비의 실제 mean·일반/보정 interval을
[W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/completed-SS-three-of-six-family-v410)에
게시하고 원격 summary를 원본과 대조했다. `paper-draft-completed-SS-contrasts-v412`는
실제15쪽으로 compile했고12쪽 본문·14쪽 통계 표를 시각 검수했다. 표의3행×5개
수치를 실제 JSON과 순서·반올림까지 대조했으며36개 산출물의 actual file4 SHA/size가
일치한다. 첫 v411은 전용 캐시의9pt math font 부재로 PDF 생성이 실패했다.
수치는 유지하고 기존 표와 같은 text 표기를 써 v412에서 성공했으며 v411의28개
실패/source 산출물도 file4에 보존했다. NN source·학습·패키지는 바꾸지 않았다.

세 correspondence arm의 GPU proof/capacity도 실제 완료됐다. Native conditioning과
same-noise flow 예측이 초기 기준선과 일치하며, 동일 quick32의 개별 asset metric·GT·
seed가 V1과 정확히 같다. 초기 macro IoU는 공통0.458358이고 native global occupancy
IoU0.426715와 구별한다. 세 arm 모두 micro96/accum1의3-update를 통과했고10개
source parameter의 gradient가 update3에서 모두 양의 유한값이다. Frozen base는
변하지 않았다. 실제 peak allocated는 spatial89.343GiB/raw89.621GiB/D1-cos89.624GiB,
reserved는92.127/94.012/94.014GiB다. Capacity의 warm update 두 번은36.77–39.55초로,
전체 학습 throughput이나 이후 OOM 부재 보장은 아니다. Cold32 raw voxel mask는
보존되지 않았으므로 위 개별 metric 일치를 raw-mask bit equality로 표현하지 않는다.
실제 audit/closure는 `three-correspondence-readiness-v382`에 있다.

No-R DEV128×3의3모델·4metric과9개 noise별 diagnostic row를
[W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/A_FIXED_NO_R-SS128x3-v381)에
게시하고 원격 summary를 원본과 대조했다. Full-family/cascade/TEST 완료로 표시하지
않았다. 게시 audit와 actual file4 closure는 `NoR-SS128x3-WandB-v381`에 있다.

해석 경계:

- Spatial3 대 semantic64는 원래 spatial 모듈을 semantic retrieval로 교체하는 효과다. 후보 수까지 고정한 순수 주소 비교가 아니다.
- Raw→D1 cosine→D1 scalar를 나누어 decoder와 score를 한 번에 바꾼 효과를 따로 나누어 주장하는 오류를 피한다. Cosine→scalar는 선택 support도 바뀔 수 있다.
- **같은 support에서 score 활용 효과는 Uniform↔Fixed**로 검증한다. 공통 posterior가 carrier/use/R에 공유되므로 순수 attention-logit 한 위치만의 효과로 부르지 않는다.
- 사용자 정의 No-score는 실제 Top-1이며 Uniform과 다르다. No-corr는 원래 spatial 대응이다. 제거한 retargeter/channel을 0인 가짜 관측으로 대체하지 않는다.
- Ray grouping은 모든 신규 행의 공통 구조다. Working Fixed에 없는 learned Q/K를 제거하는 가짜 arm은 만들지 않는다. Ray의 독립 품질 기여를 핵심 claim으로 삼지 않으면 신규 ray/voxel 학습은 필요 없다. 기존 결과와 한계는 보조 자료에 공개한다.

## 3. 학습 후 반드시 실행할 실제 출력 평가

각 신규 arm의 완료 조건은 **480 학습 → strict checkpoint load → SS DEV128×3 → 공통 Shape/PBR cascade DEV128×1 →8개 metric 및 paired 통계**다. 빠른 epoch32 IoU만으로 ablation을 종료하지 않는다.

1. SS stage: macro per-asset voxel IoU, precision, recall, F1. 학습 seed와 generation noise를 구별하고 global occupancy IoU와 혼합하지 않는다.
2. 실제 cascade: 동일128개/93 source cluster, 동일 noise·camera·GT·frozen Shape·공통 Top-1 R PBR로 신규 다섯 행을 평가한다. 기존 Fixed/Top-1 결과와 공통 GT 렌더는 재사용한다. 신규 다섯 행은 최대640개 cascade case이며 별도의 PBR ablation 재학습은 하지 않는다.
3. 여덟 metric: source LPIPS/DreamSim/CLIP, target PSNR/SSIM/M-IoU, surface CD/F-Score. 최종5개 핵심 열 외의 metric도 보조 표로 공개한다.
4. 인접 matching 대조와 Fixed 대비 Top-1/Uniform/No-R를 사전에 정한 contrast family로 집계한다. 동일 source의 여러 deformation을 독립 표본으로 세지 않고 source-cluster paired CI를 쓴다. DEV32 screening 뒤 확장한 DEV128은 탐색적 결과라고 명시한다.

현재 own checkpoint/evaluator/cascade binding과 다섯 후속 connector, SS/cascade 표 reducer가 등록돼 있다. Uniform/no-R는 실제 SS archive를 검증해 각각128개 case 계획을 전부 준비하고 등록했다. 총256 GPU case와6개 metric 단계가 대기 중이며, correspondence 세 connector는 자신의 실제480 checkpoint를 기다린다. 실제 신규 다섯 arm의 cascade/metric 결과는 아직 없으며 연결 완료를 평가 완료로 표시하지 않는다. 완료된 원본128개 평가의 입력 목록은 기존32개와 추가96개를 구분해 보관했으므로, 새128개 전체 실행에서는 두 부분이 실제 모두 들어 있는 입력 목록을 검증해야 한다.

두 control 연결 중 발견한 CPU metadata 문제는 nested SS artifact의 실제 producer
receipt와 기존 render case의 environment-map hash를 명시적으로 연결해 수정했다.
관련45개 검사를 통과했고 실제 두128-case 계획의 render 수치 설정·camera8/9/10을
원본과 대조했다. 실패 로그/빈 디렉터리·기존 작업 이력을 보존했다. NN·sampler·
renderer math 변경이나 학습 재시작은 없으며, 이 복구 자체가 품질 결과는 아니다.

### DEV640 저장 공간: 실제 사전 점검과 남은 연결

`DEV640-storage-admission-v387`에서 기존3모델×128개의 canonical receipt byte
footprint와 실제 local stat을 조사했다. 로컬 여유는 약531GiB, 파일서버4는 약166TiB다.
동일 asset에서3모델 중 큰 footprint를 택한 신규5개 arm의 경험적 합은 약229GiB다.
이를2배하고 case마다128MiB를 더한 보수적 projection은 약538GiB이며, 별도로
미완료/예정 checkpoint budget과80GiB workspace를 합치면 현재 로컬 공간을 넘는다.
이는 미래 출력의 실제 최소 필요량이나 hard bound가 아니지만, 전체 누적을 그대로
허용하는 계획에는 headroom을 인증하지 못했다. 실제 case별 admission 검사는 계속 유지한다.

완료 case의 CPU cache 회수 실행·등록을 v392에서 연결했다. 최종 metric reducer는
`geometry/prediction.npz`를 다시 읽으므로 이 파일과 native guide·scored pixel·JSON·
checkpoint는 유지한다. 회수 후보는 `geometry/geometry-evaluation.npz`,
`PBR/prediction.npz`, `PBR/geometry-evaluation.npz` 세 로컬 복사본뿐이다.
`dev_cascade_cache_retirement.py`의 선택 규칙과 관련51개 검사 이후 실제
`retire_dev_cascade_cache.py` CPU 실행을 구현했고 관련47개 검사를 통과했다.
각 case closure와 원래 GPU job의 실제 성공·native child 종료를 확인한 뒤 fresh
remote SHA/size, 미완료 reader/open FD, local inode를 재검사하고 intent/journal을
file4에 보존한다. 640개 CPU 후속 작업과 실제 context/policy를 등록했다.
기존262개 case/metric descriptor가 원본 placement와 정확히 같음을 대조했고
실제 remote hasher·두 queue의 reader scan도 통과했다. NN source·GPU job·v375
connector를 바꾸지 않았다. **실제 case와 cache 회수는 아직 시작되지 않았으므로
회수된 공간이나 실제 first-case integration 성공을 주장하지 않는다.**
첫 실제 회수와 저장 공간 추이를 확인하는 것은 계속 남아 있다.
실행 policy/등록 receipt는 `DEV640-bounded-local-cache-v392`에 있다.
기존 checkpoint cache reader도 두 queue의 미생성 future bundle에 보수적으로
거절했으므로 이를 우회해 중간 checkpoint를 삭제하지 않았다.

평가 reference는 source camera9/10에서 M1 geometry+원래 material의 GT, target camera8에서 편집 목표 GT를 사용한다. CD는 공통 normalization의20k surface samples, F-Score threshold0.02로 계산하며 결과별 ICP나 scale fitting은 하지 않는다. M-IoU는2D alpha mask다. 현재 입력 camera 평가를 새 시점 일반화로 설명하지 않는다.

## 4. 독립 확인과 파이프라인 선정

남은 독립 학습은 선택 Fixed의 SS seed202 한 개와 PBR R seed202다. PBR vanilla
seed202의1900 업데이트·여섯 validation·저장은 완료됐다. V1/R SS 두 seed도 완료돼
반복하지 않는다. 모든 ablation의 seed를 늘리지 않는다.

관측 시 GPU0–3은 완료된 vanilla에 이어 필수 PBR R seed202를 자동으로 시작했다.
GPU7은 Fixed seed202, GPU6은 Spatial 대조의 실제 본 학습을 진행 중이다.
이후 Raw/D1-cos 학습과 평가가 등록돼 있다. 실제 작업은 승인된 GPU0/1/2/3/6/7만
사용한다. Cascade case는 SS 우선순위를 유지하면서 두 queue의 가용 GPU를 사용한다.

Fixed seed202의 실제 첫 epoch96이 완료됐다. 동일 seed202 quick32·noise·GT에서
macro IoU는 V1 **0.465671**, Top-1 R **0.464190**, Fixed **0.463678**이다.
Fixed−V1은−0.001993, Fixed−R은−0.000512다. 두 대비 모두 개선19/하락13이고
median 차이는 각각+0.004203/+0.000280이므로 mean과 median의 방향을 구분한다.
세 모델의 초기 per-asset 기록과 처음96개 assets/noise/timestep/global trace를
실제 대조했다. 초기 함수 하락의 근거는 없지만 첫 epoch의 평균 우위도 없다.
약속한480 업데이트와 DEV128×3를 유지하며 독립 확인 완료나 최종 성능으로 쓰지 않는다.
실제 source snapshot/audit는 `selected-Fixed-seed202-epoch96-v388`, 원격 수치까지
검증한 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/selected-Fixed-seed202-epoch96-v389)는
`selected-Fixed-seed202-epoch96-WandB-v389`에 보존했다.

후속 실제 epoch192에서 같은 seed202 quick32 macro IoU는 V1 **0.466631**,
Top-1 R **0.462070**, Fixed **0.465424**다. Fixed−R은+0.003354로 바뀌었고
개선22/하락10·median+0.001912다. Fixed−V1은−0.001207, 개선14/하락18·
median−0.002838로 여전히 평균 우위가 없다. V1과의 평균 격차는96의−0.001993보다
줄었지만 이를 최종 개선으로 해석하지 않는다. 세 모델의 첫192 실제 asset/noise/
timestep/global trace가 같다. 새 reusable epoch reader는 기존 두 actual epoch96
readout의 전체 macro cell과96 trace에 정확히 일치함을 확인한 뒤 사용했다.
실제 snapshot/closure는 `Fixed202-actual-epoch192-v400`, 원격 summary를 검증한
[W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/Fixed202-epoch192-v401)는
`Fixed202-epoch192-WandB-v401`에 있다. 최종 ablation/TEST table에는 넣지 않았다.

Spatial seed101의 첫 epoch96도 완료됐다. 같은 seed101 quick32의 macro IoU는
V1 **0.463801**, Top-1 R **0.461619**, Fixed **0.461789**, Spatial **0.461504**다.
Spatial−Fixed는−0.000285, Spatial−V1은−0.002297이며 처음96개 실제 학습 trace와
초기 개별 metric·GT·seed는 같다. 아직 semantic retrieval의 최종 우위나 무효를
판정하지 않는다. `Spatial101-first-epoch96-v396`에 실제 prefix와 audit를 보존했다.

이후 같은 seed101 quick32의 실제 epoch192는 V1 **0.467447**, Top-1 R
**0.460882**, Fixed **0.463634**, Spatial **0.465800**이다.
Spatial−Fixed는+0.002166, Spatial−Top-1 R은+0.004918이지만 Spatial−V1은
−0.001647이다. 첫192 실제 학습 입력·noise·timestep/global trace가 같음을
검증했다. 이 중간 점추정은 semantic의 우위를 지지하지 않으며 최종 열위·무효도
입증하지 않는다. 약속한480/128×3와 실제 cascade 비교를 유지한다. 실제 readout은
`Spatial101-actual-epoch192-v400`, 원격 수치 검증까지 마친
[W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/Spatial101-epoch192-v401)는
`Spatial101-epoch192-WandB-v401`에 실제 file4 SHA/size와 함께 보존했다.
최종 ablation/TEST 표에는 넣지 않았다.

완료된 PBR vanilla seed202는4개 rank 모두1900 업데이트·LoRA optimizer360개·
frozen/DDP completion proof를 통과했다. 각 rank의 실제1900 trace와24개 원래 PNG의
GT pixel identity, 여섯 validation을 재확인했다. 고정 GT shape·target4에서 pooled
PSNR **26.057314→26.862564**, foreground PSNR **14.946447→15.751697**,
full-frame LPIPS **0.010104→0.008700**, crop LPIPS **0.095142→0.086446**다.
Global24·DDP4·rank micro6·accum1이며 마지막10 update compute median은
12.895–12.898초, peak allocated43.66GiB/reserved48.46GiB 이하였다.
이는 source preservation이나 generated geometry의 평가가 아니다. 저장된 base
tensor의 전체4모델 대조는 기존 two-seed collector가 R 완료 후 수행한다.
`PBR202-V1-completed1900-readout-v393`와 범위를 분리한
[W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/PBR202-V1-complete-Spatial101-epoch96-v397)에
완료 stage와 Spatial 중간 결과를 보존했다.

진행 중인 R seed202도 초기4개 입력·noise·GT/camera·예측 latent와 초기 GT PNG
pixel이 같은 seed202 V1과 정확히 같다. 네 rank 각각의 첫32 학습 trace도 같음을
확인했다. 이는 실제 live prefix 확인이며 R1900·두 seed paired collector 완료가 아니다.

선정 순서:

1. 최종 문맥의 ablation 및 downstream geometry/appearance를 함께 판독한다.
2. Working Fixed의 독립 seed 결과와 선택 checkpoint의 fresh-load/composition 정합을 확인한다. 실제 seed별 결과와 평균을 분리해 보고한다.
3. CD/F-Score와 LPIPS/DreamSim을 primary로 유지한다. 임의 가중합으로 승자를 만들지 않는다. Appearance 악화와 geometry 개선이 함께 있으면 trade-off를 보고한다.
4. 근거 없는 learned refinement나 개별 요소의 필수성 주장은 제외한다. Fixed가 R보다 추가 이점을 뒷받침하지 못하면 R을 최종 선택할 수 있다. 최종 구조가 바뀌면 그 구조에 유효한 대조만 남긴다.
5. 실제 checkpoint/config/code·split·camera/RGB·sampler·noise·stage composition·metric implementation·통계 family를 동결한다.

## 5. 최종 TEST508: 실제4/5개 pipeline,10개 metric

134 source cluster의 TEST508은 동결 뒤 한 번의 최종 비교로 사용한다. TEST 결과로 architecture를 다시 고르지 않는다.

| Main table 행 | 입력/역할 |
|---|---|
| Zero-shot source-only4 | 변형 전 target camera의 source view를 포함한 원본 RGB4장 |
| Zero-shot edited4 | 편집 target RGB1장+원본 source RGB3장 |
| Vanilla V1 | DeformNeXt vanilla fine-tuning |
| Top-1 R | 강한 semantic+retargeting 기준선 |
| Final Ours | 선택한 실제 조합. R과 같으면 중복 행 제거 |

- Source preservation: LPIPS↓, DreamSim↓, CLIP↑.
- Global naturalness: FID↓, KID↓.
- Target alignment: PSNR↑, SSIM↑, M-IoU↑, CD↓, F-Score↑.

FID/KID는 작은 ablation마다 반복하지 않고 최종 공통 render population에서 측정한다. 현재 protocol은 모델별508×3=1524 images, official Inception2048 FP32/TF32 off, KID100 subsets×1000·seed42017이며 signed KID를 자르지 않는다. FID/KID는 global descriptive 값으로 보고하고,8개 per-asset metric의 selected-vs-V1/R 대비는 사전 고정한 family의 보정 source-cluster interval로 보고한다.

원본/GT/최종 checkpoint는 보존한다. 큰 중간 guide는 downstream 의존성과 실제 file4 복제·hash를 확인한 뒤 소유한 중간물만 rolling 관리한다. 저장 공간 부족으로 평가 표본 수를 조용히 줄이지 않는다.

## 6. 논문 표·그림·writing 산출물

| 산출물 | 구체적인 구성 | 현재와 남은 작업 |
|---|---|---|
| Main table | 위4/5개 모델×10 metric, 목적별3개 column group | TEST 실제 값 전부 남음 |
| Ablation table | Panel A: spatial→raw→decoder cosine→scalar. Panel B: Top-1/Uniform/no-R 대 Fixed | Fixed/Top-1은 SS·cascade 실제 값; Uniform/no-R는 SS만 완료; 나머지 미측정 |
| Mechanism figure/table | Retrieval coverage→유효 후보 probability mass→같은 사례의 출력 변화 | 실제 native ray 그림·본문 삽입 완료; matched Uniform 결과와 최종 출력 연결은 남음 |
| Qualitative figure | 같은 asset/view의 GT·두 zero-shot·V1·R·Ours, 성공과 실패 | [Metadata-only2개 사례·conditional lower-median 규칙](2026-10-25-final-qualitative-selection-protocol-ko.md)은 사전 고정 완료. 최종 TEST 이미지·조건부 사례·figure/caption은 미완료 |
| Supplement | 모든8metric, 기존 ray/member/learned 음의 결과, 두 seed, protocol·실패 사례 | 기존 완료 결과 반영·최종 정합 검수 |

Ablation 본문 열은 `Correspondence / Weighting / R | SS IoU / CD / F-Score / LPIPS / DreamSim`으로 구성한다. 기존12모델 원래 문맥의 수치를 신규 최종 문맥의 행에 옮기지 않는다. Matching-aware weighting과 추가 learned attention을 구분한다.

Writing 순서는 problem/method·protocol 정리 → DEV ablation/기전 → 독립 재현 → TEST main results → discussion → abstract/conclusion 확정이다. 실제 native ray 그림을 삽입한13page/24reference 개발 초안을 compile·검수했으며, 남은 작업은 실제 최종 표·정성 결과·claim을 넣고 PDF를 검수하는 것이다. 현재 contribution의 표현은 다음 범위가 적절하다.

> We aggregate uncertain semantic correspondence evidence at the camera-ray level and adapt source conditioning while retaining the native target-conditioning stream. Matched ablations evaluate retrieval, score-weighted aggregation, and retargeting separately; geometric alignment and source appearance preservation are reported independently.

이는 설계와 검증 범위의 설명이다. “모든 모듈이 필수”, “learned attention이 일관되게 개선”, “모든 metric 우세”, “완전한 target 보존”은 현재 결과로 쓰지 않는다. Metric 개선이 없는 요소는 구조 설명 또는 음의 결과로 남기고 독립 성능 기여에서 제외한다.

### Claim을 결정하는 기준

| 주장 후보 | 필요한 직접 대조·평가 | 결과에 따른 본문 표현 |
|---|---|---|
| Semantic correspondence가 spatial 전달보다 유리 | Spatial→Raw, 공통 cascade의 CD/F-Score 및 preservation | 지지되는 metric과 범위만 명시. 후보 수3→64도 함께 바뀌므로 순수 주소 효과라고 쓰지 않음 |
| Matching decoder가 raw feature보다 유리 | Raw→D1 cosine | Decoder 표현의 추가 기여. Scalar head 효과와 분리 |
| Scalar matching score가 cosine보다 유리 | D1 cosine→D1 scalar | Scoring/selection 모듈의 효과. 후보 support가 달라질 수 있음을 명시 |
| Matching-aware weighting이 후보 활용을 개선 | 동일 support의 Uniform↔Fixed; Top-1↔Fixed | 두 대비를 함께 보고, carrier/use/R가 공유하는 score-weighted 경로의 효과로 기술 |
| Retargeting이 최종 조합에서 필요 | Fixed no-R↔Fixed | 결합 baseline의 기존 향상과 구분해 R의 직접 기여를 판정 |
| Ray-wise 구성 자체가 더 높은 품질을 보장 | 기존 ray↔voxel의 실제 matched 결과 | 현재 근거로 보장하지 않음. 새로운 최종-context ray 대조를 하지 않으면 구조·계산 공유 특성으로만 설명 |
| 최종 시스템이 편집 목표를 더 잘 따르며 source appearance도 유지 | 동결 TEST에서 Ours↔V1/R, geometry와 preservation을 각각 평가 | Geometry 향상과 appearance trade-off를 분리. CI가0을 포함하는 것을 동등성·비열등성으로 표현하지 않음 |

현재 Fixed는 learned Q/K refinement가 없는 확률 가중 집계다. 따라서 이를 “학습된 QKV attention의 우위”로 쓰지 않는다. 추가 learned attention의 기존 음의 결과는 supplement와 discussion에 보존한다. 이 구성에서 실제 입증되는 요소를 중심으로 기여를 줄일 수 있어야 한다.

### 실제 writing 순서와 완료 조건

1. **Method:** RGB4장/camera, target native stream과 source adaptation, member 기반 semantic 후보, ray 단위 evidence 공유, score-weighted 전달과 R의 정확한 수식을 코드와 맞춘다. No-match는 native source feature를 유지하며 GT visibility를 inference gate로 사용하지 않는 계약을 설명한다.
2. **Experimental setup:** DeformNeXt split·초기 multi-view LoRA checkpoint·동일 학습 stream·update·global batch·noise·고정 Shape/PBR·metric reference·cluster 통계를 표/본문에 명시한다. Clean80k는 초기 모델의 학습 이력이며 deformation 실험의 데이터가 아니다.
3. **Ablation/Mechanism:** 실제7행의 matched 결과와6개 사전 대조를 설명한다. 기존 retrieval coverage, Top-1 miss/Top-K hit, GT 후보 probability mass 관측은 활용 가능성의 근거이며 최종 성능의 대체 증거가 아니다. 실제 ray/candidate 그림과 동일 사례의 성공·실패 출력을 함께 넣는다.
4. **Main Results:** TEST의10개 metric을 빠짐없이 보고하고, 중심 서술은 CD/F-Score와 LPIPS/DreamSim에 둔다. PSNR/SSIM/M-IoU·CLIP은 보조 해석, FID/KID는 공통 rendering 분포 비교다. FID/KID만으로3D 물리적 타당성이나 절대적인 자연스러움을 주장하지 않는다.
5. **Discussion/Limitations:** 큰 변형·가림·matching 실패, geometry/preservation trade-off, 입력 camera 평가의 범위와 learned refinement의 불확실성을 다룬다.
6. **Abstract/Conclusion:** 최종 표의 직접 근거가 있는 기여만 남긴다. 표·caption·claim 원장·checkpoint와 결과 JSON을 대조하고 실제 PDF를 컴파일·시각 검수하면 writing을 완료한다.

## 7. 이번 범위에서 추가하지 않을 것과 종료 기준

새 backbone/head 탐색, gate/temperature grid, 모든 SS×PBR 조합, target-only, clean80k 재학습, 모든 ablation의 다중 seed, 최종 모델에 없는 query 제거는 추가하지 않는다. Matching coverage·native precision 등 완료된 진단을 반복하지 않는다.

**남은 본 학습은 correspondence3개 + 선택 SS seed202 + 필수 PBR R seed202의 완료다.**
완료된 Uniform/no-R 학습·SS 평가와 PBR V1 seed202는 재실행하지 않는다. 이후
신규5개 대조의 downstream 평가·실제 cache 회수, 독립 확인·fresh load,
모델/protocol freeze, TEST와 논문 마무리가 남아 있다. Arm 수와 proof/capacity/
evaluation/CPU cache job 수를 구별한다. 종료 조건은 선택된 실제 pipeline의 효과·
모듈별 근거·한계와 Main/Ablation/Mechanism/Qualitative 결과 및 논문 문장이 일치하는 것이다.

## 근거

- [완료된 DEV128 실제8metric과 선택 근거](2026-10-24-dev128-complete-fixed-controls-and-independent-confirmation-ko.md)
- [최종 문맥3개 correspondence 구현·등록과 첫 epoch의 한계](2026-10-24-shared-fixed-correspondence-implementation-and-queue-ko.md)
- [상세 필수 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)
- [Claim-evidence 원장](../paper/claim-evidence-ledger.md)
- [실제 ablation 표 소스](../paper/tables/final-ablation-template.tex)
- [논문 개발 초안](../paper/retargeting-evidence-draft.tex)
