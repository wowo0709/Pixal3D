# Member source-ray: 실패 원인에 따른 다음 설계와 검증

2026-10-21 UTC. `wowo0709/feature-retargeting`.
목표는 semantic matching + matching-aware attention + deformation-aware
feature retargeting의 **추가 생성 품질 이득**을 확인하는 것이다. 실행 개수,
matching coverage 또는 V1 대비 개선만으로 이 목표를 달성했다고 판단하지 않는다.

## 현재 결과가 지지하는 것

같은 128개 validation asset, 93개 source cluster, seed101 학습의 최종480
update를 비교했다. Sampling noise 3회는 독립 학습 seed가 아니다.
모든 IoU는 TP/FP/FN을 합산한 micro IoU다.

| 모델 | 최종 voxel IoU | V1 대비 차이 |
|---|---:|---:|
| V1 | 0.432650 | — |
| 기존 평균-query spatial | 0.437457 | +0.004807 |
| member Top-1 + R | 0.436263 | +0.003613 |
| member posterior 가중합 + R | 0.436649 | +0.003999 |
| member scalar attention + R | 0.435829 | +0.003179 |

Member 가중합 − Top-1은 +0.000386,
source-cluster bootstrap CI95는 [−0.000233, +0.000962]다.
Scalar attention − 가중합은 **−0.000820**, CI95
**[−0.001566, −0.000079]**다. 여러 비교를 보정하지 않은 개발 평가다.
V1 대비 개선과 별개로, 현재 scalar attention을 효과적인 추가 모듈로
채택할 근거가 없다. 기존 spatial 대비 member 세 모델의 CI도 0을 포함한다.

Matching 자체는 다른 결과다. 최초32개 source cluster를 제외한72개 asset에서
member posterior의 Exact Top-64는 52.26%에서73.02%로 개선됐다.
이 결과는 후보 coverage 개선이며, 최종 feature 조합이나 생성 품질 이득은 아니다.
Matching의 정답/오답 진단과 생성기의 feature 의존도 진단을 섞지 않는다.

원본 근거는 result root의 다음 closed artifact다.

- `native-prior-source-ray-v1/member-posterior-disjoint-clusters-v1`
- `member-posterior-source-ray-v1/comparison-v1`
- `member-posterior-source-ray-v1/per-arm-final128x3-v1/{top1,weighted,attention}`

Result root는 `rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/`
`anchor-ray-composition-v1/feature-readout-v1`이다. 각 디렉터리의 실제 file4
SHA256/size receipt가 해당 수치와 입력을 보존한다.

## 원인으로 배제한 것과 아직 확인할 것

1. **Source-use gate 붕괴는 현재 설명이 아니다.** 활성 ray에서 평균 use는
   Top-1 0.999927, 가중합 0.999938, scalar attention 0.999937이다.
   전체 ray 평균 약0.235는 inactive ray의0을 포함하므로 gate 값으로 오해하면 안 된다.
2. **후보 누락만 해결하면 충분하다는 가설은 지지되지 않는다.** Member별 후보
   coverage는 개선됐지만 새 semantic readout의 downstream 추가 이득은 확인되지 않았다.
3. **현재 query 구성에는 정보 희석 가능성이 있다.** 실제32개 입력의 캐시 진단에서
   utility의 전체 ray 평균에 들어가는 voxel 중 입력 target alpha 바깥 비율은
   평균0.5134였다. 이는 GT surface visibility가 아니다. 관측 patch와 배경의
   혼합이 문제인지, 학습 모델의 품질에 영향을 주는지는 실제 NN 대조가 필요하다.
4. **Scalar attention은 과거 feature readout과 다른 모듈이다.** 확률을 재가중하는
   것만으로는 과거4-head Q/K/V/output 변환과 gate 밖의 direct feature correction을
   재현하지 않는다. 이를 맞춘 NoDQ/DQ와 물리적 Top-1/spatial 대조가 필요하다.
5. **Wrong-value 진단은 제한적인 의존도 검사다.** Member 세 모델에서32개
   saved-weight wrong-address 진단은 일관된 악화를 보이지 않았다. 같은 후보 내
   valid feature 재배치이므로 완전히 무관한 source 제거와 같지 않다. 또한 GT 대응
   정확도, 재학습 NoCorr 또는 decoder의 독립 효과를 입증하지 않는다.

## 이미 연결한 실험과 의사결정

| 확인할 원인 | 고정하는 조건 | 실제 대조 | 결과에 따른 다음 설계 |
|---|---|---|---|
| Full-ray utility의 배경 혼합 | 같은 최종 NoDQ weight, 후보·prior·Top-1·R target context·source-use·write 영역 | full-ray utility vs observed-input utility | 유효하면 query 입력만 바꾸는 학습 대조를 우선한다. 음성이면 frozen-weight 분포 변경의 한계도 함께 해석한다. |
| 학습형 readout이 matching prior를 훼손하는가 | 같은 최종 NoDQ weight와 원래 candidate/Top-1/source-use | 원래 QKV vs matching-prior-only vs readout-logprior 제거 | Prior-only가 낫다면 content mixing의 query/regularization을 먼저 수정한다. Prior 제거가 악화되면 matching score의 실제 readout 의존도를 확인한다. |
| 추가 feature branch가 유효한가 | 같은 최종 NoDQ weight와 원래 Top-1/R target/source-use | 원래 QKV vs feature correction0 | 보정 제거가 낫거나 같으면 feature 전달 경로의 학습 목적과 크기를 먼저 검토한다. 이는 재학습 Top-1 arm과 구분한다. |
| 확률 조합만으로 부족한가 | source-camera ray, member64, matching 초기화와 native training protocol | scalar attention vs 실제 QKV/direct NoDQ | QKV가 matched control보다 나을 때 feature 변환의 기여를 확인한다. |
| Top-K 조합이 필요한가 | 같은 V/output, R, source-use, seed/budget | 실제 correspondence Top-1 feature vs QKV NoDQ | Top-1을 이기지 못하면 Top-K attention 필요성을 주장하지 않는다. |
| Semantic correspondence가 필요한가 | 같은4-head QKV/score/R | semantic64 vs 원래 spatial3 correspondence | Spatial을 이기지 못하면 semantic matching의 생성 품질 이득을 주장하지 않는다. |
| Deformation query가 필요한가 | 같은19개 기존 tensor와 초기 출력, native carrier | NoDQ vs 물리적2048-channel DQ 추가 | 입력 차원과 gradient가 맞는 대조에서만 DQ 효과를 분리한다. |
| Source geometry 개선이 실제 영상으로 전달되는가 | 원래 Shape512/1024 및 고정 V1 PBR | 동일32개 full cascade | Target/source camera의 영상·geometry metric과 paired contrast를 함께 확인한다. |

DQ를 제거한 arm에는 해당 channel/projection이 실제로 없다.0으로 채워 같은
정보라고 해석하지 않는다. No-score arm은 실제 Top-1 feature이며 uniform weights가
아니다. No-corr arm은 spatial correspondence이며 source feature 제거가 아니다.

Query 대조는 한 ray의 모든 voxel로 보정을 전달하는 native projection을 유지한다.
입력 foreground를 query 계산에 쓰는 것과 표면 가시성으로 write를 막는 것은 다르다.
Target RGB/native feature carrier는 유지하고 source feature 보정만 학습한다.
대응이 없을 때에도 native four-view prior를 보존하며 target-only fallback을 만들지 않는다.

Readout 세 대조는 query 진단의 실제 정상32개 결과를 재사용한다. 각 대조와 정상
forward의 candidate/prior/Top-1/R target/source-use SHA가 같아야 집계한다.
`without-readout-logprior`는 feature readout의 log prior만 제거하며 기존 Top-1
carrier와 frozen decoder는 유지한다. 사용자가 정의한 No-score 학습 arm의 대체가 아니다.
CPU에서는 원래 four-head 확률을 같은 forward의 Q/K/score로 정확히 재구성하고,
의도한 확률/feature만 변경되는지, 결손 입력 fallback, 가중치 보존 및 예외 시
hook 복구를 검증했다. 실제 GPU 진단 결과는 아직 별도로 필요하다.

## 다음 설계 변경의 범위

추가 epoch나 넓은 조합 탐색을 먼저 늘리지 않는다. 위 대조가 지목한 원인을 한 번에
하나씩 변경하고 같은480-update budget으로 비교한다. 우선순위는 utility query의
정보 보존, 실제 QKV feature 전달, 마지막으로 물리적 deformation query다.

과거 Direct 재현은 target-patch ray query와 R96/optimizer/sampler stream까지 맞춘
별도 conditional arm으로 준비돼 있다. Source-camera ray 이식과 동일 실험으로
취급하지 않는다. 과거 Direct의 수치상 최고 결과도 matched R 대비 CI가0을 포함하므로
이미 입증된 attention 이득이라는 가정을 하지 않는다.

긍정 후보가 나오면 같은 후보·K·feature readout·R을 유지하는 decoder 유무 대조가
추가로 필요하다. 현재 semantic-vs-spatial 또는 Top-1-vs-Top-K 비교만으로 decoder의
독립 효과를 주장하지 않는다. 후보 확인용 독립 training seed202도 별도로 필요하다.

## 생성 품질과 최종 채택

32개 개발 full-cascade 평가는 PBR 학습 효과와 SS 전달 효과를 분리한다.
고정 PBR의 M1 geometry + 원래 source material reference는 source camera에서의
adaptation 평가이며 undeformed M0 RGB와의 직접 보존 평가가 아니다.
각 metric의 reference와 camera 역할을 명시하고 LPIPS/DreamSim/CLIP 하나의
전체 평균을 source preservation 전체로 부르지 않는다.

최종508개 평가는 개발 단계에서 구조·checkpoint를 고정한 후 실행한다. 사용자가
지정한 source preservation(LPIPS/DreamSim/CLIP), target alignment
(PSNR/SSIM/M-IoU/CD/F-Score), global naturalness(FID/KID)를 보고한다.
SS의 IoU 개선 하나로 full 모델을 채택하지 않으며 test508을 설계 선택에 쓰지 않는다.

## 실행 복구의 과학적 범위

DQ의 FFN24 capacity OOM은 FFN12로 분할해 해결했고 실제 micro96/accum1로3회
업데이트를 통과했다. Spatial의 OOM은 동일 연산에서 checkpoint에 보관하는 frozen
feature row만 줄여 해결했다. 출력·native query gradient·head parameter gradient를
CPU에서 정확히 비교했고, 실제 micro96/accum1 capacity도 통과했다.
둘 모두 실패한 임시 update나 optimizer를 본 학습에 이어 쓰지 않는다.

원래 FAILED/HELD 기록과 source/config는 보존하고, 한 번도 시작하지 않은 후속
학습·평가에 새 job alias를 추가했다. 이 과정의 RAW metric proof에서 config receipt
경로 오류가 한 차례 있었고, 모델 생성 이전의 preflight 실패로 별도 보존했다.
정확한 receipt로 새 proof를 실행해 통과한 뒤 DQ 본 학습을 시작했다.
이 복구는 모델 성능 개선의 증거가 아니다.
