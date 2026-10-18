# Semantic matching + Top-K + matching-aware attention 모델 탐색

목표는 exact patch Top1 정확도를 높이는 것이 아니라, 여러 semantic 후보에서 생성에 유용한 feature를 전달해 SS 성능을 개선하는 것이다. 이전 C1의 개선은 기준선으로 보존하며, Top-K 자체의 추가 효과와 동일시하지 않는다. 새 namespace `rgb_matching_v2/semantic-topk-search-v1`에서 실행하고 과거317개 queue와 저장본을 변경하지 않는다.

## 고정 조건과 첫 비교

DeformNeXt world-v4 train9149/validation508, 실제 사용자 SS4210 base+LoRA, frozen Omega epoch7을 유지한다. 새 run은 seed101, global96, DDP2/micro48/accumulation1,5epochs/480updates다. GPU0/1과2/3을 사용한다. 매 epoch 같은32개 quick voxel metric/GT-Pred를 W&B에 남긴다. Final은 raw480, 기존 fixed128/93source clusters/서로 다른3noise,12steps/CFG3/generation8로 비교한다. Development 비교이고 미사용 test라고 표현하지 않는다.

첫 wave는 같은 R9 head와 같은64개 후보에서 `top1-reference`와 `soft-topk`를 독립 학습한다. Rejection과 retargeter는 둘 다 없다. Temperature는 head-train 내부 calibration의 고정 첫64 asset에서 alpha-valid query만 사용하고, 고정 grid의 평균 entropy 실효 후보 수가4에 가장 가까운 값으로 선택한다. GT·exact-match label·validation 성능은 선택에 사용하지 않는다. 이 기준은 collapse를 피하기 위한 실행 설정이지 성능 개선 증거가 아니다.

## 세 구성요소의 실제 사용과 효과를 구분

| 요소 | 사용 검증 | 성능 검증 |
| --- | --- | --- |
| Matching decoder | 실제 후보 context가 query/score를 바꾸는지, 학습 가능한 경우 gradient와 weight 변화 검사 | 동일 후보·같은 budget에서 contextual decoder와 pairwise/no-context 비교. Top1 대 soft를 decoder ON/OFF로 부르지 않음 |
| Matching score | score를 바꾸면 candidate weight와 전달 feature가 바뀌는지, entropy·실효K·gradient 검사 | Top1 대 softTopK, 그리고 학습 attention의 score-prior ON/OFF. 사용자 No-score는 Top1을 뜻하므로 prior-OFF를 같은 이름으로 부르지 않음 |
| Correspondence | 실제 semantic candidate 주소와 source value의 연결 검사 | Native spatial 기준 및 같은 후보/support를 보존하는 주소 개입. Wrong-address만으로 semantic 후보 검색 전체의 효과를 판정하지 않음 |

현재 cache의 `spatial_indices`는 GT target depth를 unproject해 만들어졌다. 이를 inference용 No-corr에 넣지 않는다. 원래 Pixal3D의 native camera projection을 spatial 기준으로 사용한다. Same-support/address 개입과 native spatial 비교는 서로 다른 질문이며 함께 보고한다.

## 후속 모델 탐색

Frozen score의 soft weighting 이후에는 생성 loss를 통해 후보의 유용성을 학습하는 attention을 비교한다. 초기 native 출력을 정확히 보존하고, frozen candidate/value 계약과 CFG·BF16·training/inference parity를 먼저 검사한다. 성공한 C1 retargeter와 결합한 모델은 같은 static-attention C1과 비교해 adapter와 attention의 효과를 분리한다. Decoder context와 matching-score prior를 각각 제거해 실제 기여를 확인하며, no-query channel을 물리적으로 유지한다. 세부 새 구조·arm은 구현 전 계약과 검사 결과에 추가 기록한다.

고정 결과에서 vanilla V1 및 matched Top1 대비 source-cluster CI와 noise별 방향을 확인하고, 개선 후보는 추가 training seed로 재검증한다. 후보를 선택한 development panel의 효과와 독립 seed의 재현을 구분한다. 작은 quick32 개선이나 attention이 넓어졌다는 사실만으로 모델을 채택하지 않는다. 음의 결과도 보존하고 효과를 강제로 양수로 만들지 않는다. SS에서 근거를 얻기 전에는 PBR 전체 ladder로 확대하지 않는다.

이 문서의 작성·config 생성은 학습 완료를 뜻하지 않는다. 첫 두 run만 완료해도 decoder/score/correspondence 전체 검증과 최종 모델 탐색이 끝난 것은 아니다.

## 구현한 후속 구조와 고정 비교

새 task attention은 원래 R9 D1-Q2C decoder와 scalar matching head를 `4f8771f2…`에서 초기화한다. Omega backbone/descriptor와 처음 검색한64개 후보 주소는 고정하고, decoder 및 score head는 생성 loss로 학습한다. Native DINO target query와 contextual Omega query를 각각128차원으로 projection하고 합친 query로 source DINO candidate의 content attention을 구한다. Matching score/temperature를 attention-logit prior로 더하며, native DINO value만 가중합한다. 기존 zero-output C1 retargeter와 fusion gate를 사용해 초기 native 출력을 보존한다. Extra deformation-query channel과 exact-match rejection은 없다.

| Arm | Context decoder | Matching-score prior | Value 후보 / Retargeter |
| --- | --- | --- | --- |
| top1-reference | Frozen D1 | Argmax 선택 | Semantic64 중1개 / 없음 |
| soft-topk | Frozen D1 | Frozen soft weighting | 같은 Semantic64 / 없음 |
| static-c1 | Frozen D1 | Frozen soft weighting | 같은 Semantic64 / C1 |
| task-full | 생성 loss로 학습하는 D1 | 생성 loss로 학습하는 scalar score | 같은 Semantic64 / C1 |
| task-no-decoder | 물리적 D0, context block 없음 | 같은 초기 scalar head를 학습 | 같은 Semantic64 / C1 |
| task-no-prior | 같은 학습 D1 유지 | 사용하지 않음, scalar head freeze | 같은 Semantic64 / C1 |
| task-native-spatial | 같은 학습 D1 block | 같은 scalar head를 학습 | **원래 native ProjGrid의 source camera3개** / C1 |

`task-no-prior`에서도 decoder output이 content-attention query에 들어가므로 decoder ON/OFF와 혼동하지 않는다. 이 arm은 사용자 정의의 Top1 No-score와 다른 prior ablation이다. Decoder 제거 시 나머지 attention/retargeter 초기 weight는 같고 context block 자체가 없다. Task 학습에서는 cached matching logit 대신 현재 decoder/head의 score를 계산하며, cache는 고정 semantic proposal 주소와 validity에 사용한다.

첫 wave soft temperature는 calibration 실효K4에 가장 가까운 grid 값, 후속 static/task 공통 temperature는 실효K8에 가장 가까운 grid 값이다. 두 선택 모두 GT/validation SS 성능을 쓰지 않는다. Static-C1→task-full이 learned attention 경로의 전체 추가 효과이고, task-full→no-decoder/no-prior가 각각의 기여다.

Native-spatial arm은 같은 카메라 행렬·native anchor 변환·ProjGrid로 각 source view의 실제 DINO/Omega feature를3D voxel에 sample한다. Target RGB alpha만 추가 경로의 support에 사용하며 target-depth spatial_indices는 로드하지 않는다. Uniform view weight와 zero retargeter에서는 원래 native4view 평균을 복원하는 수치 검사를 둔다. Native control은 자연스러운3개 spatial view 후보와64개 semantic patch 후보를 비교하므로 고정K 주소 하나만의 인과 비교라고 표현하지 않는다. 같은 candidate/support를 보존하는 주소 교란은 별도 진단으로 이 한계를 보완한다.

모든 새 task arm은 동일한5epoch 조건을 먼저 실제 native/backward/memory proof로 검사한 뒤 학습한다. Decoder·score·task projection의 실제 gradient, attention 집중도·실효K를 기록한다. Decoder/score/correspondence 각각의 효과와 모델 채택은 fixed-final 생성 결과를 얻은 뒤 판단한다. 개선 후보의 추가 seed 확인과 실제 W&B/file4 대조는 여전히 필수 후속이다.

Task attention의 주소 대조는 attention forward의 **key·score·이미 계산한 weight를 유지하고 value만** 각 query의 유효 후보 안에서 절반 길이 순환한다. Full 모델의 동일 saved weight/common32/한 noise로 normal과 value-address 교란을 비교하고 normal이 final128의 같은 사례 count/seed/GT hash를 exact 재현하는지 검사한다. Native spatial 대조를 대체하지 않으며, candidate key도 함께 바꾸는 input-roll과 구분한다. Hook은 예외에서도 제거하고 원본 weight/input을 수정하지 않는다.

Native spatial의3개 후보는 semantic64와 같은 candidate tensor budget을 기준으로 query chunk를 확대한다. `chunk_size64`의 semantic 처리는64query×64candidate,spatial 처리는최대1365query×3candidate다. 작은 CUDA kernel을 반복 실행하는 비용을 줄이는 변경이며, 출력·gradient의 chunk 동등성을 별도로 검사한다. 실제 GPU 속도·memory PASS는 각 arm 실행 전2update probe에서 확인한다.

새 attention projection의 공통 초기화는 CPU generator만 사용하고 CPU RNG를 복원한다. `torch.manual_seed`로 CUDA rank별 RNG까지 다시 초기화하지 않으며, 부모 CPU 상태와 CUDA seed 함수 비호출을 검사한다. 이 검사는 task arm의 실제 학습 시작 전에 적용했다. 서로 다른 구조의 전체 stochastic training trajectory가 bit-exact라는 주장은 하지 않는다.

Native-spatial의 최종 구현에서는 target DINO/Omega query와 추가경로 support를 semantic arm과 동일한 inferred target-patch index로 gather한다. Source value/key만 원래 native camera projection으로 sample한다. Target query의 보간 방식과 alpha support까지 함께 달라지는 불필요한 차이를 제거했다. Source3 대 semantic64의 자연스러운 후보 수 차이는 남으므로 고정K 주소-only 비교라고 주장하지 않는다. 앞선 CPU prototype 기록은 당시 구현의 증거로 보존하고, 최종 구현은 별도 CPU/GPU proof로 확인한다.

추가로 `static-top1-c1` 대조군을 동일5epoch로 학습한다. 위 표의 Top1 reference는 retargeter가 없으므로 C1과 비교하면 retargeter 효과가 섞인다. 새 대조군은 static-C1과 같은 R9/frozen decoder,64개 후보,같은 초기 retargeter/seed/optimizer를 유지하고 matching-score argmax의 feature 하나만 retargeter에 전달한다. Uniform 평균이 아니다. Static-C1 대 이 대조군이 retargeter를 고정한 soft evidence의 비교이며, task-full 대 이 대조군은 learned-head/attention 전체 경로까지 함께 바뀐다. 기존7개 fixed-final 보고서와 별도의 matched-retargeter Top1 보고서를 모두 얻은 뒤 추가 seed 후보를 선택한다. 추가 학습은 마지막 native-spatial arm과 병렬 배치하고, 아직 pending인 seed202 Top1 baseline은 그 다음으로 이동한다.

추가 seed의 후보는 위 두 fixed-final 보고서와 value-address 진단이 끝난 뒤 선택한다. Semantic Top-K/weighted attention을 쓰는5개 모델(soft,static-C1,task-full,no-decoder,no-prior) 중 V1·retargeter 없는 Top1·같은 retargeter의 Top1 대비 source-cluster CI가 양수이고3noise 모두 같은 개선 방향인 모델에서 microIoU가 가장 높은 후보를 재검증한다. Retargeter 없는 soft arm에서는 두 Top1 대조군이 같다. Retargeter만의 이득으로 semantic Top-K 경로의 성공을 선언하지 않기 위한 기준이며 실제 후보 선정 전에 고정한다. Decoder/prior 제거 모델이 이기면 그 결과도 허용하며 full을 미리 정하지 않는다. Native-spatial/Top1 대조군 및 이전 C1의 점수도 함께 기록해 더 강한 대조군을 숨기지 않는다. 조건을 만족하는 후보가 없으면 음성 결과를 보존하고 설계 분석을 계속한다.

선정 후보는 seed202에서 SS4210부터 새로480updates를 학습하며, 동시 실행하는 Top1 seed202 기준선과 master port를 분리한다. 같은 development128/93source clusters/3noise를 재사용하므로 이는 optimization seed 재현이며 독립 test 일반화 검증이 아니다. 두 training seed의 source-cluster CI는 각각 보고하고 두 seed를 섞어 population training-seed CI를 만들지 않는다. Seed202 기준선은 retargeter 없는Top1이므로 전체 조합의 재현을 검사한다. 같은 retargeter의Top1 대비 ablation은 seed101에 한정되며 그 특정 효과까지 두 training seed에서 재현했다고 표현하지 않는다. 이 자동 연결의 완료는 cache/BF16/CFG·weight·W&B/file4 검증과 최종 모델 채택의 완료를 뜻하지 않는다.

Zero-shot source-only4M0(camera08포함)와 edited1M1+3M0도 새 표에 유지한다. 기존 noise0는 일부 explicit protocol/GT hash 기록이 없어서 새 namespace에서 동일128/seed/원래SS4210을 다시 평가한다. 원래 noise0의 sample/source-cluster/seed/TP·FP·FN을 exact replay해야 하며, 새 실제 GT hash를 기존 noise1/2 및 현재 최종128의 hash와 대조한다. Legacy JSON에 누락된 hash를 채우지 않는다. 이 두 noise0 평가는 각 GPU pair의 최종 noise2가 실행될 때 남는 단일 GPU에서 병렬 실행한다.

Train1790source와 validation132source의 교집합은0이다. Development128/93source와 source가 겹치지 않는 validation121개/39source를 추가 평가 패널로 사전 예약했다. 기존3noise seed map에121개 모두 존재하므로 cache 재구축은 필요 없다. 후보를128개 결과로 정한 뒤 이 추가 패널에서 V1·Top1/C1 기준 및 후보를 비교한다. 이는 같은 validation split 안의 추가 SS-generation 평가이며, 과거 matching/flow 진단까지 전혀 접근하지 않았던 독립 test dataset이라고 주장하지 않는다.

추가121 평가는 seed101의8개 비교군과 V1/이전C1/두zero-shot,seed202의선정후보/Top1기준선(총14개 saved-model 조건)을3noise로 평가한다. 추가 학습은 없다. 우선순위를 학습과 원래 final128보다 낮게 두고 남는 허용 GPU에 배치한다. Micro voxel IoU/F1을 유지하고 case macro IoU는 보조로만 보고한다. 후보 선택을 변경하지 않은 상태에서39source의 CI를 별도로 계산하며, raw checkpoint·실제GT hash·source·seed·4RGB입력·optimizer0update를 검사한다.

Saved-weight 정합 검사는 원래C1뿐 아니라Top1/soft및새task/no-decoder/no-prior/native-spatial에 적용한다. 각 model의 fixed128 첫2개 active 사례에서 실제 public camera/ProjGrid 경로의cached 대 online proposal/current head,BF16,train/eval및CFG p0/p1/mixed를 확인한다. Task에서는예전cached logit을바꿔도현재학습head가그대로사용되는지,spatial에서는semantic cache주소를바꿔도원래투영경로가유지되는지검사한다. 초기head 전체가frozen이라고잘못가정하지않고실제로고정된state만비교한다. 별도로실제4RGB의FP32DINO를다시인코딩해FP16저장native token과비교한다(relativeRMS≤0.005,mean cosine≥0.9999). 이검사는0SGD의2case정합검사이며전체asset성능증거로대체하지않는다.

Full 이외의 후보가 선택되어도 그 saved model에서 value-address 진단을 수행한다. Frozen soft/static-C1도 원래 attention weight 계산을 마친 aggregation 경계에서 각 query의 유효 candidate value만 순환한다. 따라서 후보 주소·key·score·weight·support뿐 아니라 query마다 검색된 value의 multiset도 보존한다. 전체 source feature map을 먼저 교란하는 기존 R9 진단과 다르며 두 개입을 혼용하지 않는다. Normal common32가 해당 모델의 final128/noise0에서 같은 사례·GT hash·voxel count를 exact replay해야 하고, 잘못 연결한 value와의 paired source-cluster CI를 보고한다. CI가0을 포함하면 semantic 후보 검색 전체가 무효라고 해석하지 않는다.

선정이task-full이면 이미 계획한 동일 raw480/common32 value-address 평가를 재사용한다. 다른semantic 후보면 남는 허용 GPU에서 두 평가를 추가하며 학습을 다시 하지 않는다. 추가 seed 보고서는 선정 모델의 이 진단까지 완료된 뒤 생성한다. 진단의 실행 여부와 성능 기여 여부를 별도로 기록하고, positive 효과를 미리 가정하지 않는다.

첫 Top1/soft fixed-final 이후에는 frozen value가 실제로 다른지와 작은 fusion 강도 가설도 진단한다. 같은128개에서 descriptor/observed-alpha/native-projection index만 읽는 post-hoc CPU 분석이며,추가121개나 hidden depth를 사용하지 않는다. 별도로 같은 soft raw480 denoiser·conditioner에서 원래soft/score-argmax Top1과 원래gate/pre-tanh parameter4배를 common32/한noise로 생성한다. Normal은final128의같은사례count·seed·source·GT hash를exact replay해야 하고,전부0SGD다. Diagnostic parameter/arm은예외에서도복원하며checkpoint를쓰지않는다. Task module/retargeter가있는모델에는이static-B3진단을적용하지않는다.

## 전체 결과의 W&B 게시

`semantic_topk_campaign_wandb.py`는8개development arm·기존V1/B3/C1·두zero-shot을같은128개/3noise로묶는다. 후보가선정되면해당모델의추가training seed와reserved121의모든14개조건까지기다린다. 후보가없으면음성결과를보존하며추가candidate학습을만들지않는다. 이게시가모델탐색목표의완료나모델채택을뜻하지않는다.

후보선정job이`render→publish→readback`의CPU job3개를추가한다. 실제saved-weight/cache/CFG검사와모든epoch의W&B/file4 closure가준비돼야실행한다. Epoch4/5의중간집계CPU job2개와fresh Omega7검사를포함한62개queue는선정결과에따라65개(후보없음),115개(Full선정),117개(다른후보선정)가된다. 추가되는것은검증·보고작업이며모델학습횟수와비교budget은같다.

Static-C1/task-full의epoch4·5 quick32비교는각각두모델의actual W&B/raw-checkpoint file4 closure가생긴뒤자동실행한다. 기존portable compiler와source hash를고정한CPU driver가같은case·noise·GT를집계하고새JSON/Markdown의actual file4 SHA도대조한다. 전체128×3 fixed-final비교와구분하며,학습을마치기전에best epoch를고르는데사용하지않는다. 기계별driver와실행계약은ignored results의`provenance/automatic-wave2-epoch-summaries-v1-contract.json`에보관한다.

기존saved-conditioner검사는fresh DINO와cached Omega key 이후의proposal/head경로를검증한다. 별도`verify_omega_feature_cache.py`로실제원본RGB2건을선택한frozen Omega7에다시넣는검사도수행한다. Local epoch7파일은canonical file4와SHA/크기가같고,실제base파일SHA는epoch7 checkpoint가저장한training base SHA와같아야한다. 원래training config의attention/FFN LoRA rank·alpha도대조한다.

Fresh Omega검사는각view의FRAME/GLOBAL L12성분을독립적으로비교한다(relativeRMS≤0.005,mean cosine≥0.9999). FP16저장경계를거친fresh proposal의exact여부와set overlap은별도로보고하며정답Top1정확도로해석하지않는다. 실제같은RGBA byte·camera를확인하고camera/depth head·SS생성·optimizer update는실행하지않는다. 첫두active사례를그대로쓰며,허용GPU가비었을때만실행한다. 게시job은이결과의actual file4 closure도기다린다. CPU guard 통과는아직GPU검사완료가아니다.

아직시작하지않은`task-no-decoder`/`task-no-prior`학습도이검사의성공을기다리도록연결했다. 현재학습중인두run은5에폭까지그대로진행한다. 기존training base hash와canonical epoch7 hash의실제대조는완료됐고,원본RGB 재인코딩은별도의대기중인검증이다.

Development128과additional121의표·metric prefix·전체사례PDF를분리한다. 그림은noise0의기존저장결과를원래순서대로사용하며새추론을하지않는다. 모든방법의실제GT pixel·3D hash·source·sampling seed를대조한다. 각PDF는8사례당두방법그룹을보여주며마지막121개panel의빈칸에는다른사례를채우지않는다.

W&B에서는숫자·표·PNG와결과artifact를기록하고,별도빈download cache에서다시내려받아대조한다. Checkpoint binary는canonical file4에보관한다. `full_campaign_complete=false`와`model_adopted=false`를유지하며,각training seed의조건부CI와같은retargeter의Top1비교가seed101에만있다는범위를명시한다. 게시준비코드와CPU검사의통과는실제모든arm의완료나성능향상증거가아니다.

네 생성 평가와CPU summary는학습완료후남는허용GPU를사용하는낮은우선순위의evaluation resource다. Exclusive training과겹쳐메모리를소비하지않고새training arm도추가하지않는다. 결과는원인분석과후속가설에사용하며,기존개발128선정기준이나추가121의예약을변경하지않는다. 양의inference-dose결과도새모델의학습·fixed-final 검증없이채택하지않는다.
