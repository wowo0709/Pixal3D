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


## 다섯 번째 에폭 완료: 추가 학습 경로의 이득은 아직 확인되지 않음

두 모델 모두 계획한 raw480/5에폭을 완료했다. 같은 32개·한 noise의 마지막 quick 평가와 실제 W&B/image/raw-checkpoint의 file4 대조까지 완료했다. 아래 표는 final128×3 결과와 별개다.

| 같은 32개 기준 | Epoch5 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch5 | 0.434837 | 0.606114 |
| 이전 C1, T0.005, epoch5 | 0.453849 | 0.624341 |
| Static-C1, T0.05, epoch5 | 0.449857 | 0.620554 |
| Task-full, T0.05, epoch5 | 0.448606 | 0.619362 |

Full−static은 −0.125pp, 95% source-cluster CI [−0.633,+0.353]pp였다. 이전 C1 대비 static은 −0.399pp, CI [−0.649,−0.167]pp, Full은 −0.524pp, CI [−1.089,−0.001]pp였다. 이 quick panel에서 temperature 완화와 task 학습 경로가 이전 C1을 개선했다고 볼 수 없다. Full−V1은 +1.377pp지만 CI [−0.065,+2.859]pp는 0을 포함한다. 마지막 epoch와 평가 프로토콜을 유지하며, 중간에 좋았던 epoch를 골라 결과를 교체하지 않는다.

실제 final checkpoint에서는 두 모델 모두 frozen base 701개 tensor가 시작점과 같고 LoRA 360개가 갱신됐으며 모든 model/optimizer tensor가 finite였다. Static은 decoder 18개와 score 6개가 그대로였고, Full은 decoder 18개·score 6개·task attention 4개가 모두 갱신됐다. 이는 구현이 실제 학습에 사용됐다는 증거다. 품질 기여는 별도의 No-decoder/No-prior/원래 spatial/같은 retargeter Top1 대조로 확인한다. 두 모델의 입력 채널은 2048이며 deformation-query 추가 채널은 없다.

저장 모델의 실제 RGB/cache/BF16/CFG 검증은 각각 두 사례에서 통과했다. Static-C1의 기존 QA 상태 이름은 `PASS_CURRENT_C1_CACHE_AMP_CFG_CONTRACT`, task 모델은 `PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT`다. 최종 게시 도구가 실제 recipe에 따라 해당 상태를 엄격히 구분하도록 수정했다. 이 수정은 보고 경로에만 적용되며 학습과 checkpoint는 바꾸지 않는다.

Omega7도 별도로 실제 RGB 두 사례를 다시 인코딩했다. 선택한 base/epoch7 weight의 SHA와 strict adapter load를 확인했고, 4개 view의 FRAME/GLOBAL feature를 각각 비교했다. FP16 저장 경계에서는 전체 feature와 Top64 proposal 순서가 cache와 exact였다. FP32 대비 relative RMS의 최대값은 약 0.00020773이며 저장 양자화 범위였다. 해당 두 사례에서 checkpoint·전처리·feature cache 경로의 일치를 확인한 것이며 전체 데이터 품질이나 SS 개선의 증거는 아니다.

전체 9개 방법의 quick32 결과는 `wave2-epoch5-comparison-v2/`, 원격 대조는 `provenance/automatic-epoch5-summary-v1-file4.json`에 있다. Epoch 검증 20/45개와 네 모델의 최종 tensor 검증이 통과했고 오류는 없었다. 다음 단계는 static/Full의 fixed128×3 평가 완료와 나머지 제거 대조다. 최종 모델을 선정하지 않았다.


## 네 모델의 fixed-final128×3 완료와 같은 가중치 진단

Static-C1과 task-full의 마지막 raw480 checkpoint를 고정 128개·93개 source·서로 다른 3개 sampling noise로 평가했다. Source·seed·실제 64³ GT hash·GT count를 모든 방법에서 대조했으며, noise0의 해당 32개는 epoch5 quick 생성 결과와 exact였다. 아래 집계는 앞의 quick32 표를 대체하는 해당 checkpoint의 고정 최종 평가다.

| 모델 / fixed128×3 | Micro IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.380711 | 0.551471 |
| Zero edited1M1+source3M0 | 0.419275 | 0.590830 |
| 이전 V1 | 0.432650 | 0.603986 |
| Top1, no retargeter | 0.434529 | 0.605814 |
| Soft TopK T0.02, no retargeter | 0.434589 | 0.605873 |
| 이전 C1, frozen matching T0.005 + retargeter | 0.442193 | 0.613223 |
| 새 static-C1, frozen matching T0.05 + retargeter | 0.440938 | 0.612016 |
| Task-full, learned decoder/score/content attention + retargeter | 0.439236 | 0.610374 |

| Paired 대조 | IoU 차이, pp | 95% source-cluster CI, pp |
| --- | ---: | --- |
| Static-C1−V1 | +0.829 | [+0.430,+1.208] |
| Full−V1 | +0.659 | [+0.202,+1.067] |
| Static-C1−no-R Top1 | +0.641 | [+0.250,+1.017] |
| Full−no-R Top1 | +0.471 | [+0.038,+0.844] |
| Static-C1−이전 C1 | −0.125 | [−0.231,−0.039] |
| Full−이전 C1 | −0.296 | [−0.566,−0.074] |
| Full−static-C1 | −0.170 | [−0.395,+0.011] |

현재 두 새 모델 모두 V1 대비 개선됐지만 이전 C1을 넘지 못했다. Full의 추가 학습 경로가 static보다 좋다는 증거도 없다. 이 CI는 training seed101에 조건부인 탐색 결과이며 다중 비교를 보정하지 않았다. 서로 다른 training seed의 일반화나 TopK 자체의 추가 효과를 입증하지 않는다. 같은 retargeter를 유지한 Top1, No-decoder, No-prior, 원래 spatial 후보 대조가 남아 있다. 이전 C1을 포함한 실제 최고 모델을 기록하며 Full을 자동 채택하지 않는다.

별도로 같은 soft raw480 가중치를 고정한 32개·한 noise 검사에서 soft TopK/Top1의 정상 gate IoU는 0.436963/0.437417이었다. Soft−Top1은 −0.045pp, CI [−0.139,+0.032]pp다. Pre-tanh fusion parameter를 4배로 바꾸면 각각 0.429815/0.428853으로 낮아졌다. Soft의 증폭 전후 차이는 −0.715pp, CI [−1.488,+0.001]pp, Top1은 −0.856pp, CI [−1.639,−0.106]pp였다. 따라서 이번 결과는 단순한 고정 fusion 증폭을 개선책으로 지지하지 않는다. 새 gate 학습률이나 다른 구조를 재학습한 검사는 아니다. 두 경우 모두 더 큰 gate에서 soft가 Top1보다 좋다는 유의한 증거도 얻지 못했다.

Full의 별도 same-weight value-address 검사에서는 key·score·attention weight·후보와 support·query별 source value 집합을 그대로 두고 value 연결만 바꿨다. 실제 호출에서 valid slot 160,704개를 재배치했다. 정상 결과 0.448606과 연결 변경 결과 0.448251의 차이는 +0.036pp, CI [−0.645,+0.770]pp로 0을 포함한다. 이 32개 검사에서는 구체적인 value 연결의 품질 기여를 확인하지 못했다. 모든 semantic matching 설계가 무효라는 결론이나 원래 spatial 후보와의 동등성을 뜻하지 않는다. 정상 결과는 같은 checkpoint의 fixed128/noise0 subset과 exact였다.

기계 판독 결과는 `wave2-fixed-final-v1/`, `firstwave-fusion-dose-v1/summary.json`, `task-value-address-v1/summary.json`에 있다. GPU가 비는 동안 기존에 예정한 진단 두 건의 배치만 조정했고 명령·데이터·checkpoint·결과 경로는 바꾸지 않았다. `provenance/value-address-idle-gpu-scheduling-v1.json`에 기록했다. 이어서 No-decoder를 GPU0·1, No-prior를 GPU2·3에서 시작했다. 각각 native/backward·2-update memory 검증 이후 동일한 fresh SS4210/5에폭 예산으로 학습한다. 목표와 최종 모델 선정은 계속 진행 중이다.


## Decoder·score-prior 제거 대조의 첫 에폭

두 제거 대조 모두 raw96/epoch1 검증을 마쳤고 학습을 계속 진행 중이다. 두 rank의 실제 update11과51에서 No-decoder는 decoder gradient가0이고 score/content-attention gradient가 양수였다. No-prior는 score gradient가0이고 decoder/content-attention gradient가 양수였다. 초기 quick32의 모든 count·source·noise·GT hash와 첫 minibatch loss는 기존 C1/native 초기와 같았다. 의도한 경로의 사용과 초기 보존을 확인한 것이며 품질 기여와 구분한다.

같은32개·32source·한noise·raw96으로 완료된 기존 모델과 비교했다. SS quick sampler는12step/CFG3이며 batch8이다. Guidance는quick config의별도필드가아니라 `quick_metrics.py`의실제sampler호출상수3.0을확인하고source hash를기록했다. 새모델의첫epoch W&B 네수치/실제PNG·optimizer상태·rawcheckpoint의actual file4검증도완료됐다. W&B history가아직표시되지않은첫조회에서는대기했고,다음조회에정상일치해학습변경없이완료됐다.

| 같은32개 기준 | Epoch1 또는zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1,epoch1 | 0.432148 | 0.603496 |
| 이전 C1 T0.005,epoch1 | 0.436335 | 0.607567 |
| Static-C1 T0.05,epoch1 | 0.438047 | 0.609225 |
| Task-full,epoch1 | 0.438333 | 0.609501 |
| Task-no-decoder,epoch1 | 0.438482 | 0.609645 |
| Task-no-prior,epoch1 | 0.439132 | 0.610274 |

Full−No-decoder는 −0.015pp,95% source-cluster CI [−0.635,+0.524]pp였다. Full−No-prior는 −0.080pp,CI [−0.527,+0.324]pp였다. 첫에폭에서는decoder와scalar-score prior의추가품질기여를확인하지못했다. CI가0을포함하므로해당모듈이유해하거나동등하다고확정하지도않는다. No-prior의점수가조금높다는이유로모델을선정하거나epoch를줄이지않는다.

No-decoder−V1은 +0.633pp,CI [−0.581,+1.912]pp,No-prior−V1은 +0.698pp,CI [−0.407,+1.785]pp였다. 두CI는0을포함한다. 이비교는32개·한noise의중간경향이며앞의fixed128×3/마지막epoch점수와섞지않는다. 조건부탐색CI이며다중비교보정이나training-seed population CI는아니다.

전체11개방법과paired대조는 `wave3-epoch1-comparison-v1/`에있다. 원래5epoch/raw480과나머지native-spatial·같은retargeter의Top1·추가seed/121개검증기준을유지한다. 기존에완료된여섯모델의noise0 첫8개GT–Pred그림도 `wave2-fixed-final-v1/visual-review-first8-v1/`에보관했다. 첫8개는원래순서이며성능으로고르지않았다. 전체128개의실제GT pixel/source/noise/3D hash를대조했고투영도가비슷해도실제3D IoU는다를수있으므로그림만으로품질이나실패원인을판정하지않는다.


## 제거 대조의 두 번째 에폭: Full 대비 개선

두 모델의 raw192/epoch2 검증, 실제 W&B 네 수치·PNG와 optimizer/raw-checkpoint file4 대조가 완료됐다. 같은 32개·32source·한 noise로 이전 모델의 동일 epoch와 비교했다. 전체 epoch 검증은 24/45개 완료됐고 두 학습은 3에폭으로 계속 진행 중이다.

| 같은 32개 기준 | Epoch2 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch2 | 0.434444 | 0.605731 |
| 이전 C1 T0.005, epoch2 | 0.440602 | 0.611692 |
| Static-C1 T0.05, epoch2 | 0.440030 | 0.611140 |
| Task-full, epoch2 | 0.440274 | 0.611375 |
| Task-no-decoder, epoch2 | 0.447879 | 0.618669 |
| Task-no-prior, epoch2 | 0.445496 | 0.616392 |

모듈을 더한 방향인 Full−No-decoder는 −0.761pp, 95% source-cluster CI [−1.505,−0.109]pp였다. Full−No-prior는 −0.522pp, CI [−1.065,−0.006]pp였다. 이 epoch의 조건부 quick32에서는 두 경로를 각각 제거한 결과가 Full보다 좋았다. 두 번째 대조의 경계는 0에 가깝다. 다중 비교를 보정하지 않은 중간 탐색 CI이므로 보편적인 decoder/score 유해성이나 최종 우열로 확대하지 않는다.

No-decoder−이전 C1은 +0.728pp, CI [+0.169,+1.322]pp, No-prior−이전 C1은 +0.489pp, CI [+0.061,+0.897]pp였다. No-decoder−No-prior는 +0.238pp지만 CI [−0.004,+0.522]pp는 0을 포함한다. 현재 32개 점수는 No-decoder가 가장 높지만 두 제거 모델 사이의 우열도 아직 확정하지 않는다.

V1 대비 No-decoder의 차이는 +1.344pp, CI [−0.177,+2.991]pp, No-prior는 +1.105pp, CI [−0.320,+2.577]pp였다. 두 CI는 0을 포함한다. Full/C1 대조보다 V1 대조의 CI가 넓은 것은 서로 다른 paired 오차 분포를 반영하며, 다른 대조의 CI를 대신 적용하지 않는다. 최종 V1·두 Top1 대조의 선정 기준은 그대로다.

같은 run의 epoch1→2 변화는 No-decoder +0.940pp, CI [−0.522,+2.347]pp, No-prior +0.636pp, CI [−0.817,+2.073]pp였다. 점수 상승은 관측했지만 두 temporal CI도 0을 포함한다. 추가 joint optimization의 변화이며 모듈 단독 효과나 best-epoch 선택의 근거로 쓰지 않는다. 앞서 Full의 초기 C1 대비 우위가 마지막 epoch까지 유지되지 않았던 점도 고려해, 계획한 5epoch/raw480/fixed128×3를 유지한다.

No-decoder는 raw semantic Top64 후보와 scalar matching score/content attention·retargeter를 계속 사용한다. No-prior도 decoder context/content attention·retargeter를 유지하므로 두 결과를 semantic matching이나 matching-aware attention 전체를 제거한 것으로 해석하지 않는다. 구체적인 대응 연결의 기여는 저장 모델의 별도 value-address 진단과 원래 spatial/같은 retargeter Top1 대조로 확인한다.

전체 11개 방법, paired contrast와 temporal 결과는 `wave3-epoch2-comparison-v1/`에 있다. 현재 가장 높은 중간 점수를 최종 모델로 채택하지 않았고 추가 seed/121개 평가도 아직 선택 전이다. 데이터·학습 구조·예산은 변경하지 않았다.


## 제거 대조의 세 번째 에폭: 두 번째 에폭 우위의 불확실성

두 모델의 raw288/epoch3 검증과 실제 W&B 네 수치·PNG·optimizer/raw-checkpoint file4 대조가 완료됐다. 같은 32개·32source·한 noise로 비교했으며 전체 epoch 검증은 26/45개 완료됐다. 학습은 4에폭으로 계속 진행 중이다.

| 같은 32개 기준 | Epoch3 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch3 | 0.431872 | 0.603227 |
| 이전 C1 T0.005, epoch3 | 0.443624 | 0.614597 |
| Static-C1 T0.05, epoch3 | 0.441943 | 0.612982 |
| Task-full, epoch3 | 0.441403 | 0.612463 |
| Task-no-decoder, epoch3 | 0.444019 | 0.614977 |
| Task-no-prior, epoch3 | 0.443037 | 0.614034 |

Full−No-decoder는 −0.262pp, 95% source-cluster CI [−0.717,+0.182]pp, Full−No-prior는 −0.163pp, CI [−0.605,+0.268]pp였다. 두 제거 모델의 점수는 Full보다 높지만, epoch2에서 관측한 0을 제외하는 CI는 epoch3에서 유지되지 않았다. 두 시점만으로 decoder/score 경로의 보편적인 유해성이나 특정 제거 모델의 최종 우월성을 확정하지 않는다.

No-decoder−이전 C1은 +0.040pp, CI [−0.290,+0.396]pp, No-prior−이전 C1은 −0.059pp, CI [−0.415,+0.295]pp였다. No-decoder−No-prior도 +0.098pp, CI [−0.126,+0.322]pp로 우열이 불확실하다. V1 대비 점수 차이는 각각 +1.215pp/+1.116pp지만 CI [−0.040,+2.719]pp/[−0.163,+2.728]pp는 모두 0을 포함한다.

같은 run의 epoch2→3 변화는 No-decoder −0.386pp, CI [−0.902,+0.221]pp, No-prior −0.246pp, CI [−0.877,+0.410]pp였다. 점수 하락을 관측했지만 두 temporal CI가 0을 포함하며, 이 quick32만으로 과적합·최적 epoch·수렴 실패의 원인을 판정하지 않는다. Epoch2만 골라 최종 결과로 대체하지 않고 원래 5epoch/raw480/fixed128×3와 남은 대조를 유지한다.

전체 11개 방법과 paired/temporal 결과는 `wave3-epoch3-comparison-v1/`에 있다. 이 결과는 training seed101에 조건부인 탐색 중간 근거이며 다중 비교 보정, independent test 또는 training-seed population CI가 아니다. 현재 최종 모델은 선정하지 않았다.


## Matching head의 입력 상태와 저장된 matching 지표의 범위

진행 중인 학습을 변경하지 않고 현재 initializer `4f8771f2…`의 실제 metadata와 기존 lineage 기록·source를 대조했다. `P0-RAW/M0-L12/D1-Q2C`에서 `M0-L12`는 layer12 하나의 score를 쓰는 multi-layer ablation 이름이다. 물체 상태 M0의 RGB만으로 학습했다는 뜻이 아니다. Cache producer는 `target_anchor`와 source RGB3장을 넣고, topology teacher는 M1을 rasterize한 barycentric attribute로 M0 rest point를 얻어 source M0 view에 투영한다. 따라서 단순한 “M0-only training→M1 inference” 설명을 현재 head의 실패 원인으로 단정하지 않는다.

실제 initializer는 topology-visible-listwise-v1, 7epoch/62,545 optimizer updates, 선언된 train9,143/validation508이다. 저장된 마지막 원래 matching validation에서는 visible query22,893개 중 Top64에 GT positive가 하나 이상 있는 query가14,619개, proposal miss가8,274개였다. 해당 protocol의 exact-patch Top64 query recall은63.858%다. Raw cosine Top1은529/22,893=2.311%, 학습한 Q2C/scalar head의 Top1은4,851/22,893=21.190%였다.

이는 원래 initializer의 저장 기록이며 현재 SS joint-training 이후 decoder/score의 matching 성능을 새로 측정한 값이 아니다. H3의 Top256/calibration 지표와 head·budget·평가 분모가 다르므로 대체하거나 순수 K 효과로 비교하지 않는다. Top1 개선에는 학습한 decoder와 scalar head가 함께 바뀌었으므로 그 차이를 decoder 단독 효과라고 부르지도 않는다.

이 head는 먼저 frozen RAW FRAME cosine으로64개를 고른 뒤 decoder/score로 재순위를 매긴다. 정확한 GT patch가 pool 밖인 query는 재순위만으로 복구되지 않지만, 근처 semantic feature가 SS 생성에 유용할 가능성은 별개다. Exact Top1/patch recall을 현재 SS 모델의 채택 또는 기각 기준으로 사용하지 않는다. Scalar head의 listwise loss는 pool 안에 visible positive가 있는 query에 적용되며, logit과 attention 집중도를 보정된 correctness probability로 간주하지 않는다.

검토의 actual checkpoint SHA, 이전 기록과 일치한 원래 count, source hash와 범위는 `provenance/current-head-protocol-domain-review-v1.json`에 있다. 새 matching 평가·head 재학습·후보 확대·threshold 변경을 한 것은 아니며, 현재 SS arm들의 학습과 기존 최종 선정 기준을 유지한다.
