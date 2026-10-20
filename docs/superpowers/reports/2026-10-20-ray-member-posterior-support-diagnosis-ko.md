# Ray 안의 target patch를 보존하는 matching 진단

2026-10-20 UTC. 브랜치 `wowo0709/feature-retargeting`.
이 기록의 수치는 validation 대응 관계 진단이며 SS/PBR 생성 품질 결과가 아니다.

## 확인한 병목

기존 source-camera ray query는 관측된 target Omega patch를 먼저 평균낸다.
여러 target patch를 지나는 ray에서는 이 평균이 서로 다른 대응 모드를 지울 수 있다.
이를 검증하기 위해 동일한 32개 source cluster, 50,493개 관측 가능
`(source-camera ray, target patch)` 관계에서 네 방식을 실제 실행했다.

| 후보 및 순위 구성 | Exact Top-1 | Exact Top-16 | Exact Top-64 | 주변 1-patch Top-64 |
|---|---:|---:|---:|---:|
| 평균 query RAW64 → 기존 D1+score | 8.56% | 38.18% | 50.76% | 80.41% |
| member별 RAW cosine 최댓값으로64 선택 → 평균 query D1 | 8.91% | 38.95% | 51.22% | 82.85% |
| **member별 D1 확률 → ray에서 혼합 → 최종64** | **11.19%** | **59.00%** | **73.36%** | **95.59%** |
| 기존 평균 RAW64 유지 → member별 D1 확률만 혼합 | 10.24% | 43.19% | 50.76% | 80.41% |

새 혼합 방식의 Exact Top-64 개선은 +22.6031 percentage point,
source-cluster paired bootstrap 95% CI는 [+15.9945, +30.1202] pp다.
Top-16 개선은 +20.8187 pp, CI [+15.5021, +26.5782] pp다.
4,000 draw, seed42017이며 여러 비교에 대한 보정은 하지 않았다.
단순 RAW 최댓값 union의 Exact Top-64 개선 신뢰구간은 0을 포함한다.

따라서 모든 candidate score를 평균 query에서 구하는 방식에는 손실이 있다.
기존64 안에서 순위만 바꾸는 방식도 개선되지만, 그 밖의 정답을 회복할 수 없다.
이 결과는 학습된 attention의 독립적인 생성 품질 이득을 증명하지 않는다.

## 실제 구현과 정합

같은 ray의 서로 다른 관측 target patch만 모으고, 반복 voxel은 중복 표를 주지 않는다.
각 patch에서 기존 frozen D1 decoder와 score로 RAW64 확률을 구한다.
정규화한 확률 분포를 ray에서 동일 가중치로 혼합한 뒤 최종64를 선택하고 재정규화한다.
잘린 확률 질량도 별도로 기록한다. 한 ray의 최종 확률과 evidence는 각각 하나다.
native source-camera projection은 해당 ray의 모든 voxel에 동일하게 전달한다.
표면 가시성에 따른 voxel write mask는 추가하지 않았다.

모든 후보·확률 forward를 마친 다음 GT topology label을 열었다.
원래 baseline은 32개 각각의 정수 count와 부동소수 확률 질량을 이전 결과와
정확히 재현했다. 첫 asset의 실제 conditioner 후보 주소와 확률도 정확히 일치했다.
CPU 진단 prototype을 device 공통 학습용 helper로 옮긴 뒤 실제32 전체에서
주소·확률·context·retained mass·member count와 대응 평가 count가 모두 정확히 일치했다.
CUDA 수치·메모리·NN 학습까지 검증됐다는 뜻은 아니다.

세 readout은 같은 초기 공통 weight, posterior64, source-use, retargeter를 사용한다.

1. Top-1: 매칭된 **하나의 correspondence feature**를 사용한다. No-score 정의다.
2. Weighted: matching posterior로 Top-64 native feature를 가중합한다.
3. Attention: 같은 posterior를 log prior로 사용하고 target ray query와
   member D1 context를 이용한 학습형 Q/K attention을 추가한다.

source correction은 기존 native-prior 실험과 같은
`NativeMV4 + sum_v Proj_v[u*g*(E-original_S_v)+u*R]/4`다.
새 경로가 0이면 원래4-view global/projection/negative를 유지한다.
미관측 query 또는 source 부족도 correction을 0으로 두며 native prior를 유지한다.
No-DQ retargeter 입력은 실제2048 channel이다.
No-corr는 별도로 준비된 원래 spatial3-view correspondence arm을 비교한다.

22개 CPU 검사가 통과했다. 실제 첫 validation asset의 cached DINO와 원래
ProjGrid/camera를 사용한 세 readout에서도 초기 native prior 유지, source와
evidence가 같을 때의 정확한 cancellation, no-query/no-source fallback,
matching 및 source-use를 고정한 값 교란에 대한 반응을 확인했다.
기존 saved-value 진단이 요구하는 V2 인터페이스 상속을 빠뜨린 첫 CPU 통합 실패는
보존했고, 수정 후 세 readout에 대한 회귀 검사와 새 v2 cached 검사가 통과했다.

## 이어서 확인할 생성 실험

원래의 평균 query 네 arm과 PBR 학습 설정을 보존하고 새 세 arm을 별도 family로 둔다.
SS4210에서 fresh seed101, native MV4 carrier, global96, GPU1/micro96/accumulation1,
480 update를 공통으로 사용한다. 각 arm의 실제 초기32 예측 일치와 native BF16 FFN
정합, micro96의3-update 메모리·gradient 검사를 통과해야 본 학습을 실행한다.
micro96 적합성은 아직 GPU에서 확인 전이다.

초기 및 매96 step validation32, 최종128×3 sampling noise와 값 교란32를 비교한다.
동일 예산의 평균 query readout 및 spatial arm과의 source-cluster 비교도 포함한다.
실제 source preservation/rendering metric, 독립 학습 seed, PBR 확장과 최종 test508은
별도로 완료해야 한다. 최종 full 모델은 아직 채택하지 않았다.

원본 진단·실행 소스·확정 결과의 실제 file4 SHA/size receipt는
`native-prior-source-ray-v1/member-proposal-audit-v1`과
`native-prior-source-ray-v1/member-production-parity-v1`에 보관한다.
새 학습 family 경로는 `member-posterior-source-ray-v1`이다.

## 처음32개와 겹치지 않는 source cluster에서의 확인

세 학습 arm과 설정을 고정해 큐에 넣은 다음, 미리 정의된 validation128 패널에서
최초 진단의32개 source cluster를 전부 제외했다. 남은72개 asset/61개 cluster를
원래 순서대로 사용했으며 label이나 점수로 asset을 선택하지 않았다.
별도 test508은 사용하지 않았다.

| 방식 | Exact Top-1 | Exact Top-16 | Exact Top-64 | 주변 1-patch Top-64 |
|---|---:|---:|---:|---:|
| 기존 평균 query | 8.41% | 38.20% | 52.26% | 82.86% |
| 고정한 member posterior | **9.92%** | **58.29%** | **73.02%** | **96.49%** |

109,682개 관측 가능 ray/patch 관계에서 사전에 정한 primary Exact Top-64 개선은
+20.7618 pp, source-cluster bootstrap CI는 [+16.5887, +24.9655] pp다.
Cluster에 속하는 asset들의 count를 먼저 합쳐 cluster 단위로4,000번 resample했다.
이 추가 결과는 query 구성의 개선이 최초32개에만 국한되지 않음을 지지한다.
Validation 개발 확인이며 독립 test 또는 생성 품질 이득으로 해석하지 않는다.
실행 소스·고정 패널·72개 record·실제 file4 SHA receipt는
`native-prior-source-ray-v1/member-posterior-disjoint-clusters-v1`에 있다.

## 생성물 metric을 위한 예측 보존

현재 새 SS 최종 평가는 occupancy를 저장하지 않는다. 별도의 subset archive entry는
그 sampler를 그대로 재실행해 최종128 평가의 처음32개와 모든 per-asset record를
정확히 비교한다. Decode한 실제 예측 occupancy의 TP/FP/FN 및 GT/noise identity도
검증하고, predicted latent와 occupancy를 저장한다. 원래128개 aggregate와32개
aggregate를 같다고 요구하지 않는다. 원래 기록에 없는 prediction-grid hash까지
일치가 증명됐다고 주장하지 않는다. 후속 generation은 저장한 예측만 입력으로 읽는다.
이 경로의6개 CPU 검사가 통과했다.

모델 수를 늘린 여섯/eight-metric 집계도 기존32개×3모델 실제 생성물에 적용했다.
원래 means, paired source-cluster contrast/CI, per-asset 수치가 모두 정확히 보존됐다.
새 SS의 GPU archive와 생성물 metric 측정은 실제 학습 완료 후 별도로 실행해야 한다.

## 기존 Direct와의 구조 구분

새 SS attention은 `TaskCandidateAttention`으로 후보 확률을 학습해 재가중한다.
기존 수치상 최고 Direct의 `RayFeatureReadout`은4-head Q/K/V와 zero-init output으로
feature 자체를 변환한다. 초기7-epoch utility를 불러오고, 변환한 feature의 bounded
correction을 semantic gate 밖의 retargeting residual로 직접 전달한다.
원래 Top-1 carrier는 유지한다. 두 readout을 동일한 모듈의 재현이라고 해석하면 안 된다.

Query도 기존 Direct의 edited-target patch ray와 새 후보의 source-camera ray group이
다르다. 현재 일곱 SS arm은 새 query 및 확률 조합의 대조로 보존한다.
기존 feature 변환·직접 전달 경로와 맞춘 대조, 그 다음의 물리적 DQ 추가 검증은
별도로 필요하다. 대기 중인 PBR Direct는 기존 feature 변환 경로를 사용한다.
실제 source/config 대조와 SHA receipt는 `source-ray-cascade32-v1/readout-lineage-audit-v1`에 있다.

## Four-head feature 경로의 source-ray 이식: CPU core 검증

`MemberFeatureSourceRayConditioner`는 기존 `RayFeatureReadout` 클래스를 그대로
호출한다. Member posterior의 같은64개 후보에서 실제 Top-1 feature를 carrier로
유지하고,4-head Q/K/V의 변환 feature를 R 입력과 별도 bounded residual로 전달한다.
보정은 semantic fusion gate 밖에 있으며 source-use 판단은 원래 Top-1을 사용한다.
이 arm에는 DQ 입력이나 projection이 없고, R 입력은 실제2048 channel이다.

새 feature output이0이면 cold native MV4 correction도0이고, 이미 활성화한
Top-1/R/source-use 대조의 출력과도 정확히 같다. 두 기존 correction 경로가0인
첫 backward에서도 새 output weight에 gradient가 들어간다. Toy gradient 검사에서
semantic gate와 R을 갱신하지 않고도3 update 이내에9개 head tensor 모두 gradient를
받았다. 관련 기존 검사까지20개 CPU 검사가7.71초에 통과했다.

실제 validation asset의 cached DINO/Omega, 원래 카메라와 ProjGrid를 사용한
556개 활성 source ray에서도 초기 correction0, 활성 Top-1/R 대조 일치,
source/query 결손 시 correction0을 확인했다. 기존 Direct의 완전한7-epoch/112-update
utility5개 tensor를 SHA 검증 후 정확히 읽었다. 이 utility는 학습 초기화이며,
향후 joint head 학습에서 고정하도록 설계한 것은 아니다.

새 saved-weight 진단은4-head의 projected V만 재배치한다. 같은 forward의
matching/4-head weights, Top-1 carrier와 source-use를 정확히 유지한 상태에서
실제 ray projection 결과가 바뀌는 것을 확인했다. 정적 Top-1 aggregation만
재배치하는 기존 진단으로 변환 feature branch까지 검증했다고 주장하지 않는다.

이것은 source-camera query에 이전 feature 연산·직접 전달 경로를 이식한 CPU core다.
기존 Direct의 target-patch query 또는 R96 optimizer/sampler warmup 재현은 아니다.
실제 native SS denoiser 초기32 예측/gradient, GPU BF16 정합과 micro96 capacity,
별도 trainer/runtime 연결 및 학습은 아직 남아 있다. 기존7개 SS arm과 진행 중인
PBR 설정·큐는 변경하지 않았다. 이 core는 아직 GPU 학습 큐에 추가하지 않았다.
별도 물리적 DQ arm은 이 No-DQ 경로의 실제 NN 검증 이후 순차로 추가해야 한다.

원본 실행 결과와 소스는 `member-feature-source-ray-v1/CPU-core-v1`에 보관한다.

## Four-head No-DQ runtime 연결

별도 trainer와 native proof/capacity/science/evaluation entry를 추가했다.
기존의7개 SS arm, 데이터/카메라/latent 처리, native LoRA/AdamW/AMP/sampler 및
PBR 소스·설정은 보존한다. 새 arm은 원래 SS4210에서 fresh seed101,
global96/micro96/accumulation1,480 update, 초기 및 매96 step validation32를 사용한다.
추가된 학습 tensor는 기존 source/use10개에 실제 QKV head9개를 더한19개다.

Native proof는 실제 SS loss에서 gate/R/LoRA 및 새 zero-output weight의 첫 gradient,
활성화 후 source tensor19개의 gradient, 결손 입력 fallback, 실제32 예측의 native V1
동일성과 BF16 FFN chunk 정합을 요구한다. Capacity는 같은 모델의 실제96개 입력으로
3 update를 수행하고379개 Adam state의 step3, 모든 source gradient와 frozen weight
정합 및 FFN의 실제 최대 expansion batch를 검사한다. 이 검사가 통과해야 본 학습한다.
검사용 가중치나 optimizer state는 과학 실험의 초기화로 재사용하지 않는다.

최종 평가는128개×3 sampling noise와 projected-V 교란32개다. 같은 forward의 실제
Q/K/score에서 원래4-head 확률을 재구성해 정확히 비교하고, 원래 Top-1 feature와
source-use도 확인한다. 이 교란은4-head feature 전달의 진단이며 재학습한 NoCorr는
아니다. 정상 평가를 기존 member3개 및 mean/spatial4개 arm과 paired source-cluster
bootstrap으로 비교한다.3개 sampling noise는 독립 학습 seed가 아니다.

새 runtime/진단과 관련 기존 코드의40개 CPU 검사가7.68초에 통과했다.
이는 GPU native proof 또는 capacity 통과가 아니다. 별도 실행 그래프는 native,
capacity, science, 정상 평가3개, 값 교란, 비교 집계, W&B 게시의9개 작업으로 구성한다.
GPU 작업은 진행 중인 PBR R-only 완료 후 기존 SS arm과 함께 배정하며, healthy PBR을
중단하거나 승인되지 않은 GPU를 사용하지 않는다. 물리적 DQ를 추가한 다음 arm과
새 생성물의 rendering metric 및 독립 학습 seed 검증은 이후 별도로 필요하다.
