# Semantic Top-K 첫 fixed-final SS 비교

첫 비교의 결론은 **frozen matching score의 soft Top-K만으로는 Top-1 대비 추가 이득을 확인하지 못했다**는 것이다. 두 모델 모두 vanilla V1보다 약0.19pp 높지만, soft와Top-1의 차이는0.006pp이고 source-cluster CI가0을 포함한다. 이 결과는 이후 learned attention·retargeter 조합의 실패를 뜻하지 않는다. 전체 모델 선정과 decoder·score·correspondence 기여 검증은 진행 중이다.

## 같은 조건의 최종 수치

새 두 모델은 사용자 SS4210 complete base+LoRA에서 seed101,480updates/약5epochs,global96,DDP2,micro48/rank,accumulation1로 학습했다. DeformNeXt world-v4 train9149/validation508을 사용하며 Omega epoch7과 원래 R9 topology head는 고정했다. Clean80k를 새 deformation 학습에 사용하지 않았다. Final은 raw480,동일128개/93source clusters/서로 다른3sampling noise,12steps/CFG3/generation batch8이다. 아래 IoU/F1은 정합된 target SS latent를 decode한 occupancy64³의 micro metric이다.

| 모델 | 입력 / 추가 경로 | Micro IoU | Micro F1 |
| --- | --- | ---: | ---: |
| Zero source-only | M0 RGB4장, camera08포함 | 0.380711 | 0.551471 |
| Zero edited-RGB | M1 RGB1장+M0 RGB3장 | 0.419275 | 0.590830 |
| 이전 V1 seed101 | Native multi-view fine-tuning | 0.432650 | 0.603986 |
| 새 Top-1 | Semantic64에서 matching-score argmax feature1개 | 0.434529 | 0.605814 |
| 새 soft Top-K | 같은 후보/score,temperature0.02 | 0.434589 | 0.605873 |
| 이전 B3 seed101 | 같은 frozen matcher,temperature0.005 | 0.434499 | 0.605785 |
| 이전 C1 seed101 | Temperature0.005+retargeter | **0.442193** | **0.613223** |

두 zero-shot 모두 desired M1 GT에 평가했다. Source-only의 네 장은 camera08/09/10/11의 M0이며 source M0 재구성 점수가 아니다. Edited 입력의 첫 장은 depth가 아닌 RGB다.

| 같은128개·3noise의 비교 | IoU 차이,pp | 95% source-cluster CI,pp |
| --- | ---: | --- |
| 새 Top-1−V1 | +0.188 | [+0.012,+0.371] |
| 새 soft−V1 | +0.194 | [+0.024,+0.365] |
| 새 soft−새 Top-1 | **+0.006** | **[-0.039,+0.050]** |
| 새 soft−이전 B3 | +0.009 | [-0.032,+0.051] |
| 이전 C1−새 soft | +0.760 | [+0.381,+1.137] |

CI는 한 training seed의 현재 weight에 조건부인 source-cluster bootstrap4000회 결과다. 모든noise와 같은 source의 사례를 cluster 안에 유지했다. Exploratory 비교이며 multiplicity 보정이나 training-seed 모집단 CI가 아니다. 같은 seed/budget이라고 서로 다른 구조의 전체 stochastic trajectory가 bit-exact라고 주장하지 않는다.

## 무엇을 확인했고 무엇이 남았나

Matching score로 실제 Top-1 feature를 고른 경로와 같은 score의 soft weighting을 비교했다. Uniform 평균을 No-score라고 부르지 않았다. 현재 관측된 것은 두 경로의 SS 성능이 거의 같다는 것이다. 두 경로 모두 frozen D1 decoder를 사용하므로 이 표로 decoder ON/OFF 효과를 판단할 수 없다. Score prior를 제거한 learned attention의 효과도 아직 판단하지 않는다.

Temperature0.005의 기존 B3와0.02의 새 soft도 차이가 명확하지 않았다. Entropy를 늘렸다는 사실만으로 생성 품질이 좋아졌다고 주장할 수 없다. 이전 C1이 더 높다는 사실은 보존하지만, C1−Top-1에는 retargeter 차이가 섞여 있으므로 Top-K 자체의 증거가 아니다. 같은 retargeter를 유지한 Top-1 대조군이 필요하다.

마지막 quick32/한noise에서는 Top-1 IoU0.438094,soft0.436963으로 순서가 반대였다. 고정128/3noise에서는 위의 거의 같은 점수가 나왔다. Quick panel로 best epoch나 모델을 다시 고르지 않았고, 두 모델 모두 마지막 raw480을 비교했다.

학습과 추론 경로는 실제 저장 weight로 검사했다. 두 모델 각각의 frozen base701개 tensor가 SS4210과 value/dtype까지 같고360개 LoRA tensor는 모두 갱신됐다. 전체 denoiser·conditioner·optimizer moment는 finite이며361개 optimizer state 모두480step이다. 처음 native condition 및 초기 quick IoU는 두 모델 모두 동일했다.

두 실제 active 사례에서 cached/online candidate index와 evidence/logit이 exact이고 최종 global/projection condition 오차는0이다. BF16 train/inference 및 CFG p0/p1/mixed도 통과했다. 실제4RGB를 FP32 DINO로 다시 인코딩한 값과 FP16 native-token cache의 relative RMS는0.0001963,mean cosine1.0이었다. 이는2case 동작 정합 검사이며 전체 asset의 correct-match 또는 품질 향상 증거가 아니다.

다섯 epoch의 actual W&B 숫자/image 다운로드와 actual-file4 raw checkpoint SHA/크기 검사를 완료했다. 두 zero-shot noise0는 새 explicit protocol/GT hash 기록으로 원래128개 count/seed/source를 exact replay했다. 모든 paired 최종 비교에서384개의 actual target hash를 확인했다. 학습은약17.2초/update,rank당 peak allocated 약47.1GiB였고 OOM은 없었다.

## 이어서 실행하는 비교

Static-C1과 task-full은 GPU0/1·2/3에서 각각 native/backward 및 micro48 memory proof를 거쳐 학습한다. Task-full은 원래 D1 contextual decoder와 scalar matching head,추가 content attention을 SS 생성 loss로 학습한다. 뒤이어 physical no-decoder,score-prior OFF,native spatial 및 같은 retargeter의 Top-1을 비교한다. Full을 미리 채택하지 않는다.

선정 후보는 V1·retargeter 없는Top-1·같은 retargeter의Top-1 모두에 대해 양의 source-cluster CI와3noise의 같은 개선 방향을 요구한다. 후보의 same-pool value-address 진단·추가 trainingseed202·예약한 source-disjoint121개 평가 및 최종 채택 검증은 남아 있다. 이번 첫 wave로 전체 목표가 완료된 것은 아니다.

실제 file4 결과와 증거의 공통 경로는 다음과 같다.

```text
/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/
  firstwave-fixed-final-v1/report.json
  firstwave-fixed-final-v1/report.md
  functional-QA-v1/s101-{top1-reference,soft-topk}/summary.json
  final-weight-audits-v1/train-{top1-reference,soft-topk}/audit.json
  provenance/firstwave-trained-QA-v1-file4.json
  zero-shot-replay-summary-v1.json
```

## 첫128개에서 확인한 feature 진단

완료된 같은128개의 target RGB alpha·native DINO value·고정 matching score와 원래 projection index만 CPU에서 읽었다. Depth·topology label·SS latent·추가121개는 열지 않았으며 학습이나 새 생성 update도 없다. Training 내부 calibration과 validation의 분포는 구분한다. 아래는 validation의9,883개 유효 query를 사용한 관측이다.

| Temperature | 평균 최대 weight | Entropy 실효K | Top1 대비 native value relative RMS | Top1 대비 evidence cosine |
| --- | ---: | ---: | ---: | ---: |
| 0.005 | 0.9333 | 1.231 | 3.59% | 0.99936 |
| 0.02 | 0.7614 | 2.208 | 7.33% | 0.99731 |

Weighting은 실제로 넓어졌고 value도 변했다. 방향이 비슷하다는 사실을 feature가 동일하다는 주장으로 확대하지 않는다. Native16³ voxel에서 alpha/support가 활성인 비율은23.99%이고,고정된 실제 soft gate0.04321에서 `0.75*g*(E_soft−E_top1)`의 전체 voxel/channel RMS는0.001158이었다. 이는 같은 gate와 native feature를 가정한 analytic 차이이며,따로 학습한 두 모델의 전체 condition 차이 또는 생성 품질 효과가 아니다. BF16 양자화가 원인이라는 인과 주장도 아직 하지 않는다.

세 noise의 count를 사례별로 합친 IoU에서는70개가 좋아지고58개가 나빠졌다. 중앙값 차이는+0.0145pp,10~90분위는−0.206~+0.214pp였다. Projected value 변화량과 사례별 gain의 descriptive Spearman은−0.047로,변화가 클수록 좋아지는 뚜렷한 관계를 관측하지 못했다. Shared-source 사례를 독립 표본의 유의성 검사로 취급하지 않았다.

현재 근거는 **soft feature가 전혀 변하지 않은 구현 오류**보다 **유사한 후보 feature의 작은 전달 차이와 사례별 손익의 상쇄**에 부합한다. 생성에 유용한 후보를 고르는 learned attention이 필요한지,또는 작은 fusion 강도가 병목인지의 단독 원인은 아직 분리하지 못했다. 같은 저장 soft denoiser/head에서 Top1 feature 대 soft,원래 gate 대 pre-tanh parameter4배를 common32/한noise로 재평가하도록 추가했다. 실제 tanh gate 값을 기록하며새 학습은 없다. Normal이 원래 final128의 같은32개를 exact replay해야 한다. 이 진단은 본 모델 선택이나 fixed-final128×3를 대체하지 않는다.

실제 진단 JSON과 전체128개 산점도·분포 그림은 공통 file4 경로의 `firstwave-static-value-diagnosis-v1/`에 있다. 생성 counterfactual의 대기열과 결과는 `firstwave-fusion-dose-v1/`에 기록한다. 후자는 아직 계획된 진단이며 현재 성능 향상 증거가 아니다.

## 진행 중인 static-C1 / task-full의 첫 에폭

두 후속 모델도 같은SS4210·seed101·global96/DDP2/micro48/accumulation1에서 시작했다. 초기32개 전체 per-asset count와 metric은native참조와exact였고,이전C1/새static-C1/task-full의두rank 첫actual training loss dictionary도같았다. 이는현재 초기성능하락이나첫 minibatch의RNG손상을시사하지않으며,전체학습trajectory가bit-exact라는주장은아니다.

아래는**같은epoch1/raw96·32개·한noise**다. 위의final128×3 표와섞어순위를비교하지않는다.

| 모델 | Epoch1 quick IoU | Quick F1 |
| --- | ---: | ---: |
| 이전 V1 | 0.432148 | 0.603496 |
| 이전 B3,T0.005 | 0.434742 | 0.606022 |
| 이전 C1,T0.005 | 0.436335 | 0.607567 |
| 새 Top1 | 0.435134 | 0.606402 |
| 새 soft,T0.02 | 0.434979 | 0.606251 |
| Static-C1,T0.05 | 0.438047 | 0.609225 |
| Task-full,T0.05 | 0.438333 | 0.609501 |

| Epoch1 비교 | IoU 차이,pp | 95% source-cluster CI,pp |
| --- | ---: | --- |
| Task-full−static-C1 | +0.029 | [-0.057,+0.117] |
| Task-full−이전C1 | +0.200 | [+0.060,+0.376] |
| Static-C1−이전C1 | +0.171 | [+0.042,+0.320] |
| Task-full−V1 | +0.619 | [-0.293,+1.457] |

일곱모델의동일32source·seed·GT hash·target voxel count를대조했다. 새두모델은초기보다개선됐지만,task-full의추가경로가static-C1보다좋다는증거는아직불충분하다. 이전C1대비양의차이도현재quick panel/epoch에조건부인exploratory 결과이며,temperature의단독일반화효과나마지막epoch의우월성을뜻하지않는다. 성능에따라epoch를다시고르거나Full을미리채택하지않는다.

실제SS 본학습update11과51에서는두rank의matching decoder·scalar score·task attention gradient가모두양수였다. Native/backward preflight의임시retargeter출력perturbation과달리본학습로그에서확인한값이다. 초기step1의이세gradient가0인것은native출력을보존하는zero fusion/output 초기화에따르며이후정상활성화됐다. 이관측은학습사용의증거이며각모듈의품질기여는No-decoder/No-prior/native-spatial 및선정모델진단으로추가검증한다.

새두모델epoch1의실제W&B수치/image와raw checkpoint의file4 SHA/크기검사를완료했다. 중간비교JSON/Markdown은공통경로의 `wave2-epoch1-comparison-v1/`에있다. 전체5epoch·fixed-final128×3·추가seed/평가패널·최종채택검증은아직남아있다.

## W&B의 완료된 첫 비교와 전체 사례 그림

[첫 최종 비교 W&B run](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-semantic-topk-firstwave-v1)에7개모델/baseline의final128×3표,paired contrast표,같은사례의GT–Pred비교와feature진단그림을기록했다. Youngwoojeon 계정/entity에서완료됐으며서버의44개숫자·두표·두PNG와38개artifact파일을실제로다시내려받아확인했다. PNG와artifact는원본과byte까지일치한다. Run의 `full_campaign_complete=false`와 `selected_model=not_selected`도확인했다. 이는첫wave보고의완료이며전체학습·모델채택의완료가아니다.

`firstwave-wandb-v1/presentation/`에는원래평가순서128개전체를8개씩담은16페이지PDF와첫8개의PNG가있다. 모든방법에서sample ID·seed·source cluster·3D GT hash/shape와실제GT projection pixel이같은지검사했다. 그림은기존SSoccupancy의2D max-projection을그대로표시하며새추론이나새3D metric이아니다. 물리적projection axis이름을추정하지않았고성능에따라사례를고르지않았다.

같은폴더의PDF를실제reader로열어16페이지와첫/마지막사례1·128의경계를확인했고PNG도시각적으로검토했다. 독립W&B대조는 `firstwave-wandb-v1/remote-readback/audit.json`에있다. 이게시작업은CPU에서수행하고GPU/system metadata수집을비활성화했으며모델checkpoint binary는W&B에올리지않았다. Data·checkpoint의canonical 저장소는file4다.

## 진행 중인 두 번째 에폭과 같은32개 zero-shot

Raw192/epoch2도같은quick32·한noise로비교했다. Zero-shot은완료된noise0/fixed128의같은32개를원래순서대로뽑아count를합친값이다. 새추론을한것은아니다. Edited-RGB zero-shot의32개전체count·seed·source·3D GT hash는일곱fine-tuning모델의native초기32개와exact였다. 아래zero-shot은위final128×3의다른집계값과혼용하지않는다.

| 모델 / 같은32개 기준 | Epoch2 또는zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1,epoch2 | 0.434444 | 0.605731 |
| 이전 C1,T0.005,epoch2 | 0.440602 | 0.611692 |
| Static-C1,T0.05,epoch2 | 0.440030 | 0.611140 |
| Task-full,T0.05,epoch2 | 0.440274 | 0.611375 |

Full−static-C1은+0.024pp,CI[−0.197,+0.271]pp였다. Full−이전C1은−0.033pp,CI[−0.295,+0.267]pp로현재둘의우열도불명확하다. Epoch1에서보였던이전C1대비점수우위가epoch2에서는유지되지않았다. 이두시점만으로과적합이나학습실패의원인을확정하지않으며,계획한5epoch/last-raw480을유지한다.

Full의epoch1→2 점수는43.833%→44.027%,static은43.805%→44.003%로올랐다. 그러나같은32source의temporal CI도0을포함한다. 적은quick panel에서의상승을일반적인개선이나best-epoch선택의증거로쓰지않는다. Extra learned attention 경로의기여는아직확인되지않았다.

Zero edited−zero source는같은desired M1 GT에서+3.172pp,CI[+1.111,+5.505]pp였다. Source-only는camera08을포함한deformation전RGB4장이고edited는targetRGB1장과sourceRGB3장이다. 이비교는target변형단서가도움이됨을보여주지만source M0재구성평가는아니다. Full−V1과Full−zero edited의현재점수차이는각각+0.583pp/+1.356pp이며,이quick32에서두CI는모두0을포함한다.

두새모델epoch2의actual W&B수치/image 및raw192 checkpoint file4 SHA/크기검사를완료했다. 이단계결과와모든9개방법의전체표는 `wave2-epoch2-comparison-v2/`에있다. Portable compiler는 `experiments/deformnext_retargeting/summarize_semantic_epoch.py --root <search-root> --epoch 2`다. Source/seed/actualGT32검사·같은epoch강제·micro count집계후새결과만쓰며,기존epoch1보고서를덮어쓰지않는다.

## 세 번째 에폭: learned matching 경로의 추가 이득은 아직 불명확

Raw288/epoch3에서도같은32개·한noise·SS64 GT를사용했다. Source·seed·GT hash와native초기/edited-zero참조의일치를다시검사했다. 아래는중간quick평가이며첫표의final128×3집계와별개다.

| 같은32개 기준 | Epoch3 또는zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1,epoch3 | 0.431872 | 0.603227 |
| 이전 C1,T0.005,epoch3 | 0.443624 | 0.614597 |
| Static-C1,T0.05,epoch3 | 0.441943 | 0.612982 |
| Task-full,T0.05,epoch3 | 0.441403 | 0.612463 |

Full−static은−0.054pp,95% source-cluster CI[−0.386,+0.241]pp였다. Full−이전C1은−0.222pp,CI[−0.594,+0.168]pp다. Full의초기보존과실제gradient활성화는확인됐지만,현재세에폭에서learned matching 경로의추가품질기여는확인되지않았다. Epoch2→3의점수상승폭은Full+0.113pp/static+0.191pp이며두temporal CI도0을포함한다.

Full−V1의현재점수차이는+0.953pp,CI[−0.238,+2.345]pp다. 이차이는Full의상승과V1의quick점수변동을함께반영하므로Full단독의학습효율개선으로해석하지않는다. 마지막raw480/fixed128×3와같은retargeter의Top1,decoder/score/native-spatial제거대조를유지한다.

두모델epoch3의actual W&B수치/image와raw288 checkpoint file4 SHA/크기대조가모두완료됐다. Epochaudit는이시점16/45개PASS,errors=[]였다. 기계판독전체9개방법/대조/temporal 결과는 `wave2-epoch3-comparison-v2/`에있다. 학습은계속진행중이며최종모델은선정하지않았다.

추가로기존128개descriptor/projection진단에서semantic 경로의공유정도를집계했다. 현재nearest-projection구조에서는9,882개active 2D query가125,770개native16³ conditioning cell에전달돼query당평균12.727개cell이feature/attention/retargeter출력을공유한다. 이는occupied GT voxel의개수가아니며최종native+semantic조건전체가같다는뜻도아니다. 원래native multiview와anchor조건은계속변한다.

이관측은2D semantic query가3D위치별단서를충분히사용하는지확인할근거이며실패원인의인과증거는아니다. 계획한비교에서추가이득을얻지못하면같은semantic64후보·decoder·score·retargeter를고정하고native-voxel RGB query만바꾸는후속검사를고려한다. 현재새arm을추가한것은아니다. 원래spatial3후보대조는후보자체와개수도달라query만의효과를분리하지못한다. 세부derived관측은 `provenance/semantic-query-reuse-diagnosis-v1.json`에보관했다.

## 네 번째 에폭: 두 C1 모델이 함께 개선

같은quick32·한noise의raw384평가를자동집계했다. 실제두epoch04의W&B/image/raw-checkpoint file4 closure 이후에기존compiler가실행됐고,결과JSON/Markdown과두closure의actual file4 SHA도대조했다. Job은종료코드0으로완료됐으며학습은5에폭으로계속진행됐다.

| 같은32개 기준 | Epoch4 또는zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1,epoch4 | 0.434300 | 0.605592 |
| 이전 C1,T0.005,epoch4 | 0.450249 | 0.620926 |
| Static-C1,T0.05,epoch4 | 0.449461 | 0.620176 |
| Task-full,T0.05,epoch4 | 0.449138 | 0.619869 |

Full−static은−0.032pp,95% source-cluster CI[−0.448,+0.492]pp다. Full−이전C1은−0.111pp,CI[−0.549,+0.430]pp이며static−이전C1도−0.079pp,CI[−0.236,+0.075]pp다. 현재셋의우열은분명하지않고learned matching 경로의추가품질기여도확인되지않았다.

Full−V1은+1.484pp,CI[+0.190,+2.835]pp,static−V1은+1.516pp,CI[+0.122,+2.960]pp였다. 이는현재whole-path의quick평가결과이며decoder/score 학습의기여를따로입증하지않는다. Full의zero-edited대비차이는+2.242pp지만해당CI[−0.184,+4.807]pp는0을포함한다. 다른대조와다른분산을가진CI를혼용하지않는다.

Epoch3→4는static+0.752pp,CI[+0.005,+1.456]pp,Full+0.773pp,CI[+0.021,+1.606]pp였다. 같은run의최적화가진행되는증거이며모듈단독효과나best-epoch선택의근거는아니다. Fixed raw480/final128×3와남은ablation을유지한다.

전체9개방법·paired검사·temporal결과는 `wave2-epoch4-comparison-v2/`에있다. 원격대조는 `provenance/automatic-epoch4-summary-v1-file4.json`의5개파일/282,687bytes와일치했다. Epochaudit는18/45개PASS,errors=[]였다. 아직최종모델을선정하지않았다.
