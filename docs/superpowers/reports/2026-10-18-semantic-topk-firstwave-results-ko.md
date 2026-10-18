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
