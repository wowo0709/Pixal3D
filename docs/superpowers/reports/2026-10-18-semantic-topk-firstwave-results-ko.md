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


## 제거 대조의 네 번째 에폭: Full과 다시 비슷한 수준

두 모델의 raw384/epoch4 검증과 W&B 네 수치·PNG·optimizer/raw-checkpoint의 실제 file4 대조가 완료됐다. 전체 epoch 검증은 28/45개 완료됐고 두 학습은 마지막 5에폭으로 진행 중이다. 아래는 같은 32개·32source·한 noise의 동일 epoch 비교다.

| 같은 32개 기준 | Epoch4 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch4 | 0.434300 | 0.605592 |
| 이전 C1 T0.005, epoch4 | 0.450249 | 0.620926 |
| Static-C1 T0.05, epoch4 | 0.449461 | 0.620176 |
| Task-full, epoch4 | 0.449138 | 0.619869 |
| Task-no-decoder, epoch4 | 0.447982 | 0.618768 |
| Task-no-prior, epoch4 | 0.448719 | 0.619470 |

Full−No-decoder는 +0.116pp, 95% source-cluster CI [−0.415,+0.682]pp, Full−No-prior는 +0.042pp, CI [−0.453,+0.662]pp였다. 두 비교 모두 0을 포함하며, 앞선 epoch2의 제거 모델 우위가 일관되게 유지된 것은 아니다. 이 결과만으로 decoder/score를 포함하거나 제거하는 최종 결정을 내리지 않는다.

이전 C1 대비 No-decoder는 −0.227pp, CI [−0.581,+0.207]pp, No-prior는 −0.153pp, CI [−0.497,+0.190]pp였다. No-decoder−No-prior도 −0.074pp, CI [−0.272,+0.132]pp로 우열이 불확실하다. V1 대비 No-decoder는 +1.368pp, CI [−0.021,+2.923]pp, No-prior는 +1.442pp, CI [+0.017,+3.017]pp였다. 두 모델 간 직접 비교는 불확실하므로 V1 대조의 CI 경계가 0의 서로 다른 쪽이라는 이유만으로 No-prior가 No-decoder보다 우수하다고 해석하지 않는다.

Epoch3→4 점수 변화는 No-decoder +0.396pp, CI [−0.133,+0.920]pp, No-prior +0.568pp, CI [−0.067,+1.241]pp였다. 두 temporal CI도 0을 포함한다. Joint optimization의 중간 변화이며 안정적인 수렴·best epoch·모듈 단독 기여를 입증하지 않는다.

전체 11개 방법과 paired/temporal 결과는 `wave3-epoch4-comparison-v1/`에 있다. Training seed101에 조건부인 탐색 CI이며 다중 비교를 보정하지 않았다. 마지막 raw480/fixed128×3, 원래 spatial·같은 retargeter Top1 대조, 추가 seed/121개 검증과 최종 모델 선정은 남아 있다. 현재 학습 구조나 예산은 바꾸지 않았다.


## 제거 대조의 다섯 번째 에폭과 실제 attention 사용

No-decoder/No-prior의 5epoch·raw480 학습, 마지막 W&B 네 수치·PNG와 optimizer/raw-checkpoint의 실제 file4 대조가 완료됐다. 같은 32개·32source·한 noise의 마지막 epoch 결과는 다음과 같다. 전체128×3 최종 평가는 아직 진행 중이며 이 표로 모델을 선정하지 않는다.

| 같은 32개 기준 | Epoch5 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch5 | 0.434837 | 0.606114 |
| 이전 C1 T0.005, epoch5 | 0.453849 | 0.624341 |
| Static-C1 T0.05, epoch5 | 0.449857 | 0.620554 |
| Task-full, epoch5 | 0.448606 | 0.619362 |
| Task-no-decoder, epoch5 | 0.445840 | 0.616721 |
| Task-no-prior, epoch5 | 0.444225 | 0.615175 |

Full−No-decoder는 +0.277pp, 95% source-cluster CI [−0.528,+1.108]pp, Full−No-prior는 +0.438pp, CI [−0.531,+1.473]pp였다. 두 CI가 0을 포함하며 decoder/score-prior의 추가 성능 기여는 이 quick32에서 확정되지 않았다. Epoch2의 제거 모델 우위를 마지막 결과로 대체하거나 에폭을 골라 채택하지 않는다. 두 제거 모델의 epoch4→5 변화도 각각 −0.214pp, CI [−0.746,+0.541]pp와 −0.449pp, CI [−1.220,+0.427]pp로 불확실하다.

No-decoder−이전 V1은 +1.100pp, CI [−0.430,+2.890]pp, No-prior−이전 V1은 +0.939pp, CI [−0.612,+2.752]pp였다. 이전 C1 대비는 각각 −0.801pp, CI [−1.570,+0.022]pp와 −0.962pp, CI [−1.920,+0.035]pp였다. 이 탐색 CI는 한 training seed·한 sampling noise에 조건부이며 다중 비교를 보정하지 않았다. 모든 paired/temporal 결과와 실제 입력 기록은 `wave3-epoch5-comparison-v1/`에 있다.

두 저장 모델의 실제 RGB/cache·BF16·train/eval·CFG 정합 검사도 통과했다. 같은 첫 두 active 사례의 150개 유효 query에서 No-decoder의 평균 최대 weight는0.25757, entropy는2.67659nats, 평균 entropy 실효K는16.3902였다. No-prior는 각각0.03490,4.05895nats,58.2368이었다. 두 경로 모두 Top1으로 붕괴하지 않고 여러 source candidate value를 실제 사용한다. Invalid zero row는 제외했으며 전체 데이터 coverage나 생성 성능 기여를 뜻하지 않는다.

두 모델은 decoder 유무도 다르므로 이 통계 차이를 scalar-prior 단독 효과라고 해석하지 않는다. Full/static-C1의 동일 사례 저장-weight 통계는 별도 검사로 보완한다. No-prior는 사용자 정의의 Top1 No-score 대조군이 아니며, 그 대조는 예정된 `static-top1-c1` 학습이다. 남은 native spatial·동일 retargeter Top1·추가 seed/121개와 최종 선정 기준은 유지한다.


## 여섯 모델의 fixed-final128×3 비교

No-decoder/No-prior의 마지막 raw480·128개/93source·서로 다른3개 sampling noise 평가까지 완료됐다. 여섯 모델 모두 실제2-rank/480update·같은 native 초기 출력·매 epoch 검증·마지막 quick32의 final128 재현을 확인했다. 실제 최종 tensor/optimizer와 30개 epoch의 W&B/file4 대조도 통과했다. 아래는 quick32가 아닌 고정 최종128×3의 SS voxel metric이다.

| Model | Micro IoU | Micro F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.380711 | 0.551471 |
| Zero edited1M1+source3M0 | 0.419275 | 0.590830 |
| 이전 V1 | 0.432650 | 0.603986 |
| 이전 C1 T0.005 | 0.442193 | 0.613223 |
| Static-C1 T0.05 | 0.440938 | 0.612016 |
| Task-full | 0.439236 | 0.610374 |
| Task-no-decoder | 0.441023 | 0.612097 |
| Task-no-prior | 0.440340 | 0.611439 |

No-decoder−V1는 +0.837pp, 95% source-cluster CI [+0.405,+1.276]pp, No-prior−V1는 +0.769pp, CI [+0.361,+1.183]pp였다. 둘 다3noise에서 V1보다 높았다. Retargeter 없는 Top1 대비도 각각 +0.649pp, CI [+0.236,+1.074]pp와 +0.581pp, CI [+0.203,+0.971]pp였다. 같은 retargeter의 Top1 결과는 아직 없으므로 이 개선을 Top-K만의 효과라고 분리하지 않는다.

Full−No-decoder는 −0.179pp, CI [−0.498,+0.109]pp, Full−No-prior는 −0.110pp, CI [−0.398,+0.121]pp였다. Full에 decoder 또는 scalar-prior를 더한 성능 이득은 이번 fixed-final에서도 확인되지 않았다. Decoder 제거 쪽이3noise 모두 높은 점수를 냈지만 CI가0을 포함하므로 decoder의 보편적 유해성이나 제거 모델의 확정 우월성을 주장하지 않는다.

현재 최고점은 기존 C1이다. No-decoder−기존 C1은 −0.117pp, CI [−0.339,+0.119]pp, No-prior−기존 C1은 −0.185pp, CI [−0.432,+0.043]pp로 두 차이는 불확실하다. No-decoder와 static-C1의 차이도 +0.008pp, CI [−0.223,+0.263]pp였다. 새 학습 모듈이 늘어났다는 이유로 Full을 채택하거나, 점수 차이가 불확실한 후보를 승자로 확정하지 않는다.

같은 두 실제 active RGB 사례/150query의 저장 attention 검사는 네 모델 모두 통과했다. 평균 최대 weight/entropy 실효K는 static-C1 0.5398/5.5745, Full 0.3641/12.1724, No-decoder 0.2576/16.3902, No-prior 0.0349/58.2368이었다. 실제 matching score/content attention이 여러 후보에 가중치를 부여해 value를 전달하는 경로는 작동한다. 이 두 사례의 집중도와 성능 순서를 연결해 분포 확산을 실패 원인으로 단정하지 않는다. 세부 entropy·case별 값·현재 content scale과 recipe/checkpoint hash는 실제 QA 및 비교 JSON에 있다.

첫8개 사례의 GT–Pred도 원래 순서/noise0에서 V1·기존 C1·static-C1·Full·두 제거 모델을 나란히 생성해 직접 확인했다. 예를 들어 case3은 기존 C1 0.660 대 No-decoder/No-prior 0.647/0.629, case5는 기존 C1 0.415 대0.432/0.439, case8은 Full 0.551 대0.531/0.531이었다. 모델별 개선 방향이 사례마다 다르다. 실제GT pixel·source·noise·3D hash는128개 모두 같았고, 투영 그림이 비슷해도 실제3D IoU 차이를 대신 설명하지 못한다. 새 추론이나 좋은 사례 선택은 하지 않았다.

전체11개 방법·paired CI·학습 감사·입력 hash는 `wave3-fixed-final-v1/report.json`과 `report.md`, 원래 첫8개 비교는 같은 경로의 `visual-review-first8-v1/`에 보관한다. 이는 한 training seed에 조건부인 exploratory development 결과이며 다중 비교 보정이나 independent test 결과가 아니다. 원래 spatial correspondence 및 같은 retargeter Top1 대조가 이제 실제 학습을 시작했다. 두 결과와 기존 C1을 포함한6후보의 고정 선정 기준, 이후 seed202·reserved121·실제 W&B 게시/재검증은 남아 있다. 아직 최종 모델을 채택하지 않았다.


## 원래 spatial·동일 retargeter Top1 대조의 첫 에폭

두 새 대조군의 raw96/epoch1 validation, 실제 W&B 네 수치·PNG·optimizer/raw-checkpoint file4 대조가 완료됐다. 같은32개·32source·한 noise로 모든 기존 모델의 같은 epoch와 비교했다. 전체 epoch 검증은32/45개 완료됐으며 두 모델은2에폭을 학습 중이다. 아래는 fixed-final128×3와 구분하는 quick32 결과다.

| 같은 32개 기준 | Epoch1 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch1 | 0.432148 | 0.603496 |
| 이전 C1 T0.005, epoch1 | 0.436335 | 0.607567 |
| Static-C1 T0.05, epoch1 | 0.438047 | 0.609225 |
| Task-full, epoch1 | 0.438333 | 0.609501 |
| Task-no-decoder, epoch1 | 0.438482 | 0.609645 |
| Task-no-prior, epoch1 | 0.439132 | 0.610274 |
| Task-native-spatial, epoch1 | 0.439621 | 0.610745 |
| Static-Top1+C1, epoch1 | 0.437187 | 0.608393 |

Full−native-spatial은 −0.129pp, 95% source-cluster CI [−0.378,+0.144]pp였다. 이 시점의 semantic64 경로가 원래 카메라 spatial3 경로보다 좋다는 근거는 없다. 후보 수와 값의 sampling 방식도 다르므로 주소만의 인과 대조라고 부르지 않는다. 별도 동일 후보/support의 value-address 진단을 대체하지 않는다.

같은 retargeter의 soft Top-K−Top1인 static-C1−Static-Top1+C1은 +0.086pp, CI [−0.069,+0.243]pp였다. Full−Static-Top1+C1도 +0.115pp, CI [−0.035,+0.303]pp였다. 두 차이가0을 포함해 weighted multi-candidate evidence나 학습 matching 경로의 추가 이득을 아직 확정하지 않는다. 기존 C1 T0.005−Static-Top1+C1은 −0.085pp, CI [−0.177,−0.009]pp였지만 첫 epoch·한 noise에 조건부인 작은 차이이므로 최종 우열이나 Top-K의 보편적 유해성으로 확대하지 않는다.

Native-spatial−V1은 +0.747pp, CI [−0.261,+1.653]pp, Static-Top1+C1−V1은 +0.504pp, CI [−0.460,+1.393]pp였다. Native-spatial−기존 C1은 +0.329pp, CI [+0.009,+0.694]pp로 경계가0에 가깝다. 비교군별 paired 오차가 다르므로 한 대조의 CI를 다른 대조에 적용하지 않는다. 모두 다중 비교를 보정하지 않은 탐색 CI다.

두 대조군은 원래 SS4210/seed101에서 새로 시작했고, 실제32개 초기 생성 record와 두 rank의 첫 minibatch loss dictionary가 기존 C1의 초기 상태와 정확히 같았다. Global96/DDP2/micro48/accumulation1의 별도2update memory probe를 통과한 뒤 benchmark optimizer를 재사용하지 않고 본 학습을 시작했다. Native 경로의 step11·51에서 decoder/score/content-attention gradient가 두 rank 모두 양수이며, 두 모델의 retargeter 출력도 활성이다. 이는 경로 사용의 근거이며 성능 기여 증거와 구분한다.

전체13개 방법·paired contrast·선택된 epoch record/recipe hash·실제 file4 closure는 `wave4-epoch1-comparison-v1/`에 있다. 첫 epoch를 기준으로 arm을 중단하거나 후보를 선정하지 않는다. 두 모델 모두5epoch/raw480 및 fixed128×3까지 진행하고, 기존 C1을 포함한6후보의 동일 Top1+C1 대조 기준과 추가 seed/121개 검증을 유지한다.


## 원래 spatial·동일 retargeter Top1 대조의 두 번째 에폭

두 새 대조군의 raw192/epoch2 validation과 실제 W&B 네 수치·PNG·optimizer/raw-checkpoint file4 대조가 완료됐다. 전체 epoch 검증은34/45개 완료됐고 두 모델은3에폭을 학습 중이다. 아래는 같은32개·32source·한 noise의 동일 epoch 비교이며, fixed-final128×3와 구분한다.

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
| Task-native-spatial, epoch2 | 0.444871 | 0.615793 |
| Static-Top1+C1, epoch2 | 0.440365 | 0.611463 |

Full−native-spatial은 −0.460pp, 95% source-cluster CI [−1.010,+0.040]pp였다. Native 쪽 점수가 높지만 이 CI도0을 포함하며 semantic 대응의 추가 이득이 입증된 것은 아니다. Native3개 camera sample과 semantic64개 patch의 자연스러운 후보 수 차이를 명시한다.

Static-C1−Static-Top1+C1은 −0.033pp, CI [−0.153,+0.072]pp, 기존 C1 T0.005−Static-Top1+C1은 +0.024pp, CI [−0.035,+0.075]pp였다. Full−Static-Top1+C1도 −0.009pp, CI [−0.291,+0.319]pp로 불확실하다. 첫 epoch에서 기존 C1이 Top1+C1보다 낮았던 작은 차이가 같은 방향으로 유지된 것도 아니다. 고정 score의 soft evidence가 같은 retargeter의 Top1보다 추가로 좋다는 근거는 아직 없다.

No-decoder−Static-Top1+C1은 +0.751pp, CI [+0.183,+1.361]pp, No-prior−Static-Top1+C1은 +0.513pp, CI [+0.065,+0.932]pp였다. 이 epoch의 조건부 quick32에서는 두 제거 모델이 같은 retargeter의 Top1보다 높았다. 이 결과를 마지막 epoch나 추가 training seed의 재현으로 확대하지 않는다. 같은 두 모델의 마지막 raw480 평가와 혼동하거나 epoch2만 골라 최종 대조를 대체하지 않는다.

Native-spatial−V1은 +1.043pp, CI [−0.352,+2.498]pp, Static-Top1+C1−V1은 +0.592pp, CI [−0.579,+1.890]pp로 모두0을 포함했다. 같은 run의 epoch1→2 변화도 native-spatial +0.525pp, CI [−0.885,+1.840]pp, Static-Top1+C1 +0.318pp, CI [−1.220,+1.822]pp였다. 점수 상승을 관측했지만 두 temporal CI는0을 포함하며, 이는 joint optimization의 변화이지 특정 모듈의 단독 효과가 아니다.

전체13개 방법·paired/temporal 결과와 실제 입력은 `wave4-epoch2-comparison-v1/`에 있다. 집계기에 같은 run의 이전 epoch 비교를 추가했고, 첫 epoch 당시 source는 원래 file4 archive에 보존했다. 모든 CI는 training seed101·한 sampling noise에 조건부인 탐색 결과이며 다중 비교를 보정하지 않았다. 원래5epoch/raw480·전체128×3·6후보 선정·추가 seed/121개 검증을 유지한다. 모델·학습률·데이터·microbatch를 변경하지 않았다.


## SS matching key와 native value의 이미지 격자 호환 검사

현재2개 학습은 유지한 채, Omega의 주소를 DINO value에 적용할 때 전처리 crop/patch 순서가 달라지는지 CPU에서 확인했다. 기존 saved-QA의 첫 두 실제 사례/8개 RGBA에 대해 cache producer의 `load_rgba`와 native dataset의 `_load_image` 함수 자체를 실행했다. 두 경로의512×512 RGB tensor가8개 모두 bit-exact였고 최대 pixel 차이는0이었다. 두 loader는 같은 LANCZOS resize와 alpha 곱을 사용하며, SS native extractor의 이후 transform은 공간 crop/resize 없이 정규화만 한다.

실제 installed Hugging Face `DINOv3ViTEmbeddings`와 Omega `PatchEmbed`의 forward도 patch16/512 입력의 좌표 marker로 비교했다. Probe의1채널 patch-average weight에서32×32의1,024개 index가 양쪽 모두 같은 row-major 순서를 정확히 보였다. Native 쪽 CLS/register prefix는 실제 코드처럼 제외했다. 이 marker는 좌표 convention 검사이며 pretrained feature 또는 성능 증거가 아니다.

실제8개 input tensor hash·PNG hash·함수/embedding source hash와 범위는 `provenance/shared-ss-image-grid-v1.json`에 있다. 기존 fresh DINO/Omega feature 검사를 보완하는 input/격자 검사로, 전체 dataset의3D 정합·GT matching 정확도·SS 생성 품질이나 PBR1024를 검증했다는 뜻은 아니다. GPU·optimizer update·모델·cache 변경은 없었다. 현재512 SS key/value 경로에서 다른 crop이나 token 순서가 원인이라는 근거는 확인되지 않았다.


## 원래 spatial·동일 retargeter Top1 대조의 세 번째 에폭

두 대조군의 raw288/epoch3 validation과 실제 W&B 네 수치·PNG·optimizer/raw-checkpoint file4 대조가 완료됐다. 전체 epoch 검증은36/45개 완료됐고 두 모델은4에폭을 학습 중이다. 아래는 같은32개·32source·한 noise의 같은 epoch 결과이며 fixed-final128×3와 구분한다.

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
| Task-native-spatial, epoch3 | 0.441475 | 0.612533 |
| Static-Top1+C1, epoch3 | 0.441788 | 0.612833 |

기존 C1 T0.005−Static-Top1+C1은 +0.184pp, 95% source-cluster CI [+0.027,+0.356]pp였다. 이 epoch의 조건부 quick32에서는 같은 retargeter에서 soft matching evidence를 사용하는 기존 C1 쪽이 높았다. 첫 epoch의 작은 음의 차이, 두 번째 epoch의0을 포함하는 CI와 방향이 달라진 점을 함께 기록한다. 한 epoch를 골라 최종 개선이나 안정적인 Top-K 효과로 확정하지 않는다.

Static-C1 T0.05−Static-Top1+C1은 +0.016pp, CI [−0.231,+0.293]pp, Full−Static-Top1+C1은 −0.038pp, CI [−0.396,+0.331]pp였다. No-decoder−Static-Top1+C1은 +0.223pp, CI [−0.066,+0.516]pp, No-prior−Static-Top1+C1은 +0.125pp, CI [−0.139,+0.401]pp였다. 두 제거 모델의 epoch2 우위가0을 제외하는 CI로 유지된 것은 아니다.

Full−native-spatial은 −0.007pp, CI [−0.618,+0.530]pp로 거의 같은 점수와 불확실한 차이를 보였다. Native3개의 camera sample과 semantic64개의 patch 차이를 명시하며, semantic 주소만의 인과 효과로 해석하지 않는다. Native-spatial−V1은 +0.960pp, CI [−0.258,+2.393]pp, Static-Top1+C1−V1은 +0.992pp, CI [−0.193,+2.497]pp로 모두0을 포함했다.

같은 run의 epoch2→3 변화는 native-spatial −0.340pp, CI [−1.163,+0.515]pp, Static-Top1+C1 +0.142pp, CI [−0.423,+0.763]pp였다. 두 temporal CI가0을 포함하므로 이 변화만으로 과적합·수렴 실패나 최적 epoch를 판정하지 않는다.

전체13개 방법·paired/temporal 결과와 실제 입력은 `wave4-epoch3-comparison-v1/`에 있다. 모두 한 training seed·한 sampling noise에 조건부인 탐색 결과이며 다중 비교를 보정하지 않았다. 학습·모델·데이터 조건을 바꾸지 않고5epoch/raw480 및 fixed128×3, 기존 C1을 포함한6후보 선정과 추가 seed/121개 검증을 유지한다. 아직 최종 모델을 채택하지 않았다.

## 저장된 Full의 matching prior와 content-attention 변화

기존 saved-QA의 첫 두 실제 사례·150개 active query에서 initializer와 raw480 Full의 저장 weight를 CPU FP32로 비교했다. 원래 semantic64 주소·alpha-valid support·DINO value·Omega FRAME 입력을 유지하고 matching decoder/score prior 및 content-attention의 출력을 구분했다. 학습·생성 평가·GPU 사용·GT matching label 접근은 없었다. 이는 두 사례의 feature 진단이며 전체 dataset의 matching 정확도나 SS 성능 기여를 판정하지 않는다.

| 같은150 query | 평균 실효 후보 수 | 평균 최대 가중치 |
| --- | ---: | ---: |
| Frozen initializer score, T0.05 | 5.57454 | 0.53980 |
| Trained decoder/score prior만, T0.05 | 12.93452 | 0.35556 |
| 같은 trained prior + trained content-attention | 12.17337 | 0.36410 |

Trained prior와 initializer prior의 argmax는42.67% query에서 달랐다. 이는 두 모델의 선택 불일치율이며 정답 Top1 accuracy가 아니다. Decoder state의 relative L2 변화는3.705%, scalar head는2.170%였고, contextual query의 initializer 대비 relative RMS 변화는 두 사례에서0.534/0.420, cosine은0.866/0.916이었다. 이 정도의 parameter 변화가 실제 score/feature 변화 없이 무시할 만큼 작다고 가정할 수는 없다.

가중합한 DINO value의 reference 대비 relative RMS 차이는 trained-prior 대 initializer-prior에서11.12%, content-attention을 추가한 결과 대 같은 trained-prior에서4.01%였다. 서로 다른 reference를 쓰는 순차 대조이므로 두 수치를 더하거나 인과 기여율로 바꾸지 않는다. 현재 두 사례의 broadening은 content-attention만이 아니라 decoder/score prior의 변화에서 크게 나타났고, content 항은 평균 실효K를 오히려 조금 줄였다. 이 관측만으로 matching 학습이 유해하다거나 Full의 SS 성능 차이 원인을 확정하지 않는다.

Cached frozen score와 initializer의 fresh score로 만든 value 가중합의 relative RMS 차이는2.61e-7이었다. 실제 consumer는 source value를 FP32로 올린 뒤 logit을 cast하므로 cached score도 FP32로 비교했다. 앞선 BF16 GPU parity 검사를 이 CPU 계산으로 대체하지 않는다. FP32 total-attention 실효K12.17337과 기존 BF16 saved-QA의12.17240도 서로 다른 계산 경계의 기술통계다.

정확한 입력/weight/source hash와 결과는 `matching-prior-drift-two-case-v2/summary.json`에 있다. 분석 초안v1의 불필요한 cached-logit FP16 cast는 별도 superseded 기록으로 보존하고 보고에는v2만 사용한다. 실제 학습 코드·cache·checkpoint는 바꾸지 않았다. 모든 현재 fixed-final 비교가 끝난 뒤에도 유효한 후보가 없으면, matcher를 고정한 채 content-attention만 학습하는 분리를 다음 설계의 우선 가설로 둔다. 이 가설을 아직 구현·학습·검증된 모델로 부르거나 현재8arm/선정 기준을 변경하지 않는다.

## 같은 prior 진단의 고정128 전체 확인과 saved-component 대조

위 두 사례 진단을 원래 순서의128개·93source·9,883개 active query 전체로 확장했다. CPU84/85 두 코어에서 FP32 saved-weight inference만 실행했고, GPU 학습·원래8arm·추가121개 패널은 바꾸지 않았다. Feature 가중합의 오차/기준 제곱합을 case별로 누적해 큰 candidate-value tensor를 전체 panel에 대해 보관하지 않았다. 첫 두 사례 통계는 기존 직접 계산과 최대2.21e-7 차이로 재현했다.

| 고정128의 active-query 평균 | 실효 후보 수 | 최대 가중치 |
| --- | ---: | ---: |
| Frozen initializer prior, T0.005 | 1.23147 | 0.93332 |
| Frozen initializer prior, T0.05 | 4.67355 | 0.56625 |
| Trained decoder + initializer scalar head, T0.05 | 7.40808 | 0.52135 |
| Initializer decoder + trained scalar head, T0.05 | 7.44122 | 0.53820 |
| Trained decoder + trained scalar head, T0.05 | 10.66050 | 0.39080 |
| 같은 trained prior + trained content-attention | 10.40178 | 0.39436 |

Initializer prior 대비 trained prior의 argmax 불일치율은45.32%, 가중합 DINO value의 relative RMS 차이는12.35%였다. 같은 trained prior에 content-attention을 더한 대조는 각각15.68%,3.66%였다. 전체 Full attention과 frozen T0.05의 value 차이는13.28%다. 이는 score/feature 사용의 변화이며 matching accuracy나 SS 성능 차이가 아니다. Cached score와 fresh initializer score의 value 차이는3.41e-7로, 이 panel에서 old cached matcher와 initializer 불일치를 큰 변화의 설명으로 삼을 근거는 없다.

저장된 decoder만 trained weight로 바꾼 대조의 initializer 대비 argmax 불일치율은48.41%, value 차이는13.39%였다. Scalar head만 trained weight로 바꾼 대조는21.68%,10.14%였다. 두 component가 모두 가중합을 바꾸며, 둘을 함께 바꾼 수치는 단독 변화의 합이 아니다. 이 hybrid들은 **원래/학습된 저장 component를 추론 시 교체한 진단**으로, 각각을 독립 학습한 ablation이나 SS 품질의 인과 효과로 해석하지 않는다.

결과는 `matching-prior-drift-panel128-v1/summary.json`과 `matching-prior-drift-panel128-hybrid-v1/summary.json`에 있다. Hybrid 계산은 기존128 진단의 원래5분포·5대조 통계를 최대 차이0으로 재현했다. 실제 file4의 initializer·saved-conditioner·manifest·128개 RGB/feature/proposal 입력387파일(약3.86GB)을 별도로 SHA256 대조했고, hybrid가 완전히 같은 입력/weight를 사용했는지도 연결한다. 전체 panel에서의 matching-prior drift가 확인됐지만, 현재 최종 모델 채택이나 학습의 유해성 판정은 여전히 별도의 생성 평가·대조군·추가 seed 근거를 요구한다.

## 원래 spatial·동일 retargeter Top1 대조의 네 번째 에폭

두 대조군의 raw384/epoch4 validation과 실제 W&B 네 수치·PNG·optimizer/raw-checkpoint file4 대조가 완료됐다. 전체 epoch 검증은38/45개 완료됐고 두 모델은 마지막5에폭을 학습 중이다. 검증기가 W&B에 epoch row가 반영되기 전에 조회해 일시적으로 `StopIteration`을 기록했지만, 같은 run/checkpoint를 다시 조회해 두 epoch 모두 통과했다. 학습·checkpoint를 재시작하거나 결과를 재업로드하지 않았다.

| 같은32개·32source·한 noise | Epoch4 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch4 | 0.434300 | 0.605592 |
| 이전 C1 T0.005, epoch4 | 0.450249 | 0.620926 |
| Static-C1 T0.05, epoch4 | 0.449461 | 0.620176 |
| Task-full, epoch4 | 0.449138 | 0.619869 |
| Task-no-decoder, epoch4 | 0.447982 | 0.618768 |
| Task-no-prior, epoch4 | 0.448719 | 0.619470 |
| Task-native-spatial, epoch4 | 0.442147 | 0.613179 |
| Static-Top1+C1, epoch4 | 0.449885 | 0.620581 |

기존 C1 T0.005−Static-Top1+C1은 +0.036pp, 95% source-cluster CI [−0.038,+0.113]pp였다. 세 번째 에폭의 양수 CI가 네 번째 에폭에서도 유지된 것은 아니다. Static-C1 T0.05−Static-Top1+C1은 −0.042pp, CI [−0.192,+0.106]pp, Full−Static-Top1+C1은 −0.075pp, CI [−0.541,+0.482]pp였다. No-decoder와 No-prior도 같은 Top1+C1보다 각각 −0.190pp [−0.556,+0.249],−0.117pp [−0.482,+0.247]로 불확실했다. 이 epoch에서 weighted multi-candidate evidence의 추가 이득을 확정하지 않는다.

Full−native-spatial은 +0.699pp, CI [−0.029,+1.574]pp로 semantic 쪽 point estimate가 높지만0을 포함했다. Native3개 camera sample 대 semantic64개 patch라는 자연스러운 후보 수 차이를 유지하며, 주소만의 인과 효과라고 부르지 않는다. Static-Top1+C1−V1은 +1.559pp, CI [+0.093,+3.095]pp였다. 그러나 Top1+C1−retargeter 없는 Top1은 +1.359pp, CI [−0.136,+2.904]pp로, 앞 대조의 양수 CI를 retargeter 단독 효과로 옮겨 해석하지 않는다. Native-spatial−V1은 +0.785pp, CI [−0.637,+2.337]pp였다.

같은 run의 epoch3→4 변화는 native-spatial +0.067pp, CI [−0.337,+0.529]pp, Static-Top1+C1 +0.810pp, CI [+0.119,+1.473]pp였다. Top1+C1의 이 구간 개선은 관측되지만 joint optimization의 변화이며 특정 모듈의 단독 효과나5에폭 최종 수렴을 입증하지 않는다.

전체13개 방법·paired/temporal 결과와 실제 입력은 `wave4-epoch4-comparison-v1/`에 있다. 한 training seed·한 sampling noise에 조건부이고 다중 비교를 보정하지 않은 중간 결과다. 모델·데이터·학습률·microbatch·원래5에폭 budget을 변경하지 않았으며 fixed128×3,6후보 선정,추가 seed/121개 검증과 최종 채택은 계속 남아 있다.

## 원래 spatial·동일 retargeter Top1 대조의 다섯 번째 에폭

두 대조군 모두480 update/5epoch 학습과 마지막 validation을 완료했다. 실제 W&B 네 수치·PNG·optimizer/raw-checkpoint file4 대조도 두 epoch 모두 통과해 현재8개 development 학습의40epoch 검증이 완료됐다. 전체 target45개 중 남은5개는 아직 시작하지 않은 seed202 Top1 기준선이다. 새 두 모델의 최종 frozen-base/LoRA/conditioner 검증도 완료됐고, fixed128×3 생성 평가는 GPU0~3에서 병렬 진행 중이다. 아래는 마지막 epoch의 common32/32source/한 noise 결과로, 아직 두 대조군의128×3 결과가 아니다.

| 같은32개 기준 | Epoch5 또는 zero-shot IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.394997 | 0.566305 |
| Zero edited1M1+source3M0 | 0.426715 | 0.598179 |
| 이전 V1, epoch5 | 0.434837 | 0.606114 |
| 이전 C1 T0.005, epoch5 | 0.453849 | 0.624341 |
| Static-C1 T0.05, epoch5 | 0.449857 | 0.620554 |
| Task-full, epoch5 | 0.448606 | 0.619362 |
| Task-no-decoder, epoch5 | 0.445840 | 0.616721 |
| Task-no-prior, epoch5 | 0.444225 | 0.615175 |
| Task-native-spatial, epoch5 | 0.444720 | 0.615649 |
| Static-Top1+C1, epoch5 | 0.454022 | 0.624505 |

기존 C1 T0.005−Static-Top1+C1은 −0.017pp, 95% source-cluster CI [−0.224,+0.171]pp였다. 이 panel에서 둘은 거의 같은 점수이며, epoch3에서 보인 양수 CI를 마지막 epoch의 추가 이득으로 재현하지 못했다. Static-C1 T0.05−Static-Top1+C1은 −0.417pp, CI [−0.726,−0.150]pp, Full−Static-Top1+C1은 −0.542pp, CI [−0.948,−0.164]pp였다. No-decoder는 −0.818pp [−1.515,−0.091], No-prior는 −0.980pp [−1.900,−0.056]로, 이32개·한 noise 평가에서는 넓은 soft/학습 attention 경로들이 matched Top1+C1보다 낮았다. 전체128×3와 추가 seed의 결론으로 확대하지 않는다.

Static-Top1+C1−retargeter 없는 Top1은 +1.593pp, CI [+0.096,+3.211]pp, Static-Top1+C1−V1은 +1.919pp, CI [+0.432,+3.439]pp였다. 앞 대조는 같은 frozen matcher의 score-argmax evidence에서 retargeter를 추가한 조건부 결과다. 두 run의 training noise stream이 bit-exact라는 주장은 하지 않는다. Zero-edited RGB 기준선 대비도 +2.731pp, CI [+0.256,+5.326]pp였으며 source4M0와 edited1M1+3M0를 구분해 보고한다.

Full−native-spatial은 +0.389pp, CI [−0.375,+1.259]pp로 불확실했다. Native-spatial−기존 C1은 −0.913pp, CI [−1.722,−0.115]pp지만 두 모델의 head 학습/attention도 달라 주소만의 효과로 부르지 않는다. 같은 run의 epoch4→5 변화는 native-spatial +0.257pp [−0.355,+0.796], Static-Top1+C1 +0.414pp [−0.115,+1.071]로 모두0을 포함했다. 이를 추가 epoch의 효과나 최적 epoch 선정 근거로 쓰지 않는다.

학습 후반의 실제 기록도 확인했다. Full의 step451에서 decoder/score/content gradient가 모두 양수이고, No-decoder는 decoder만0, No-prior는 scalar score만0이었다. 새 native-spatial도 네 번의 validation 이후 step401에서 두 rank의 세 gradient가 모두 양수였다. 이는 후반에도 의도한 경로가 학습됐다는 근거이며 성능 기여 증거와 구분한다. 학습 중 최대 allocated memory는 native-spatial 두 rank59.75/59.39GiB, Top1+C1 두 rank48.00/48.00GiB였다.

전체13개 방법·paired/temporal 비교와 입력은 `wave4-epoch5-comparison-v1/`에 있다. 같은 training seed101의 탐색 CI이며 다중 비교를 보정하지 않았다. 학습 budget·microbatch·학습률·데이터를 변경하지 않았고, 원래 fixed128×3/6후보 선정·조건부 추가 seed/121개·실제 전체 W&B readback·모델 채택은 아직 남아 있다.

## 첫 round: 실제128개·3noise 최종 평가와 선정

이 절이 앞 절들의 중간 진행 상황을 대체한다. 새8개 구조는 모두5epoch/480update를 완료했고, 각 모델의 fixed128/93source·3noise 최종 생성 평가, 실제 saved-weight cache/BF16/CFG 검사, 전체40epoch W&B 수치·이미지 및 optimizer/file4 검사, 최종 frozen base·LoRA·conditioner tensor 검사가 완료됐다. 추가 optimization seed의 학습과 전체 campaign 원격 게시 확인은 아직 완료되지 않았다.

| 방법 | 최종 micro IoU | Voxel F1 |
| --- | ---: | ---: |
| Zero source4M0 | 0.380711 | 0.551471 |
| Zero edited1M1+source3M0 | 0.419275 | 0.590830 |
| V1 | 0.432650 | 0.603986 |
| Frozen semantic Top1, retargeter 없음 | 0.434529 | 0.605814 |
| Frozen semantic soft-Top64, retargeter 없음 | 0.434589 | 0.605873 |
| 기존 C1, frozen Top64 T0.005 | 0.442193 | 0.613223 |
| Static-C1, frozen Top64 T0.05 | 0.440938 | 0.612016 |
| Task-full | 0.439236 | 0.610374 |
| Task-no-decoder | 0.441023 | 0.612097 |
| Task-no-prior | 0.440340 | 0.611439 |
| Task-native-spatial | 0.441701 | 0.612750 |
| Frozen semantic Top1+C1 | 0.442757 | 0.613765 |

같은 retargeter의 Top1을 기준으로 기존 C1 T0.005는 −0.056pp,95% source-cluster CI[−0.131,+0.007]pp,static T0.05는 −0.182pp,[−0.308,−0.079]pp,Task-full은 −0.352pp,[−0.624,−0.127]pp였다. Retargeter 없는 soft-Top64−Top1은 +0.006pp,[−0.039,+0.050]pp다. 따라서 현재 soft/학습 Top-K가 Top1보다 추가 개선을 만든다는 근거는 없다. Top1+C1−retargeter 없는 Top1은 +0.823pp,[+0.427,+1.222]pp로,retargeter가 다른 대조를 Top-K 자체의 효과로 해석해서는 안 된다. 이 두 run의 전체 training noise stream이 bit-exact라고 주장하지 않는다.

실제 decoder 효과 대조 Full−No-decoder는 −0.179pp,[−0.498,+0.109]pp,scalar matching-score prior 효과 Full−No-prior는 −0.110pp,[−0.398,+0.121]pp,semantic correspondence 경로 대조 Full−native-spatial은 −0.246pp,[−0.660,+0.091]pp였다. 세 대조 모두0을 포함하므로 모듈의 생성 품질 기여가 입증되지 않았다. No-prior는 score prior를 끈 content attention이며,사용자가 정의한 Top1 no-score arm과 다르다. Native-spatial은 원래 camera projection의3개 후보,semantic은64개 patch 후보여서 주소만 바꾸는 인과 실험은 아니다. Gradient와 saved tensor 변화 검사는 의도한 경로의 사용을 확인했으며,품질 개선의 증거와 구분한다.

원본 `seven-arm-final-report.json`, `matched-retargeter-top1-report.json`, `candidate-selection-v1.json`의 실제 선정 결과는 `NO_PROMISING_COMBINATION_YET`, `selected_candidate:null`이다. 최고 점수는 Top1+C1이며 semantic Top-K utility 모델로 채택하지 않는다. CI는 한 training seed101에 조건부인 탐색 결과이며 다중 비교를 보정하지 않았다. 양수였던 중간 epoch를 선택해 이 최종 결론을 바꾸지 않는다.

동일 noise0·원래 순서의 첫8개 asset에서8개 학습 모델과 두 zero-shot의 GT/Pred 투영 이미지를 직접 확인했다. Source-only zero-shot이 학습 모델보다 나은 사례와 방법별 우열이 달라지는 사례가 모두 있었다. 투영 silhouette가 비슷해도3D IoU가 다를 수 있어 이 그림만으로 hidden-depth 오류나 축 정합 오류를 단정하지 않는다. 실제 PNG/PDF와 검토 기록은 `wave4-noise0-first8-v1/`에 보존했고 관련26개 파일의 실제 file4 SHA 검증이 통과했다.

후속 실험은 [frozen matcher utility 계획](2026-10-19-frozen-matcher-utility-attention-followup-ko.md)에 따른다. Frozen decoder/score와 초기 가중합을 보존하고 zero-start utility attention만 추가해 검사한다. 원래 음성 결과는 보존하며,새 모델도 같은 retargeter의 Top1과 V1을 넘어서는지 raw480/fixed128×3로 판단한다.

## 실제 semantic candidate의 RGB 위치 확인

`matching-address-first8-v1/`의 PNG8개와 PDF는 원래 fixed128의 첫8개 asset을 그대로 사용한다. 각 asset의 foreground query를 row-major 순서의 네 mid-quantile에서 선택하고,frozen matching-score Top1·ranks2~4와 완료된 Task-full raw480의 최종 attention Top1을 같은 target M1·source M0 RGB canvas에 표시했다. 이는 현재 학습 중인 frozen-utility 모델의 결과가 아니다. CPU2개만 사용했고 SGD·추가 생성·GT correspondence label·예약121개를 사용하지 않았다.

622개 active query의 실제 선택이 기존 saved128 진단과 일치하고,공유하는 transport/match/target RGB24개 file fingerprint도 기존 canonical 기록과 일치했다. 실제로 확인한 도끼·침대·책장 세 그림에서는 같은 넓은 부위에 모이는 후보와 반복 구조·가림에 따른 위치 모호성이 함께 보였다. 세 그림의 정성 관찰을 전체 데이터의 정답 매칭 정확도나 decoder/score 변경의 품질 인과 효과로 해석하지 않는다. Source-camera/row-major patch 주소를 직접 확인하는 자료이며,생성 품질 효과는 같은 retargeter의 Top1 및 native-spatial 대조와 구분해 평가한다.
