# Prior-only의 별도 source 검증과 실제 attention 분포

2026-10-21 13:43 UTC. 새 학습 arm이나 loss 변경 없이 기존 학습의 종료,
사전 연결된 GPU 진단과 CPU saved-head 검사를 추적했다.

## Pilot의 prior-only 이득은 뚜렷하게 재현되지 않았다

기존 pilot32와 source cluster가 겹치지 않는 72개 asset/61개 cluster에서,
동일 NoDQ480 checkpoint의 원래 readout과 matching-prior-only readout을
비교했다. Pilot의 outcome을 이용해 72개 중 유리한 asset을 고르지 않았다.
원래 candidate, decoder, Top-1 carrier, projected V/output, R target
context와 source-use 경로는 유지했다. Baseline은 이미 끝난 동일 noise의
128개 평가에서 해당 72개를 추출해 재사용했다.

| Sampling noise | 기존 learned attention IoU | Prior-only IoU | 차이 |
|---|---:|---:|---:|
| 0 | 0.435771 | 0.435888 | +0.000117 |
| 1 | 0.432986 | 0.433925 | +0.000939 |
| 2 | 0.435876 | 0.435183 | −0.000692 |
| 3회 pooled | 0.434871 | 0.434997 | +0.000126 |

Primary IoU는 TP/FP/FN 합산이다. Source-cluster bootstrap 4,000회의
paired 95% 구간은 **[−0.002005, +0.002203]**다. 전체 216쌍의 GT hash와
noise/asset 순서를 검증했다. Training seed는 1개이며 noise 3개를 독립
학습 반복으로 취급하지 않는다. 이 72개는 개발 validation의 부분 집합이며
untouched test가 아니다. 여러 진단에 대한 보정은 하지 않았다.

실제 native forward에서 prior-only의 유효 후보 수는 약9.74였고 기존
learned attention은 약4.81이었다. 후보를 더 넓게 조합하도록 바꾸는 데는
성공했지만 생성 품질 이득은 확인되지 않았다. 원래 pilot32의 +0.002739를
일반화된 효과로 주장할 수 없다. 그렇다고 bounded 학습 arm의 최종 결과나
모든 종류의 matching-aware attention을 이 진단으로 부정하지도 않는다.

이 진단은 고정된 learned model의 inference readout 변경이다. 사용자가
정의한 No-score(실제 Top-1 feature) 재학습 arm의 대체가 아니다.

## Bounded 제약은 동작했지만, 그것만으로 이득을 만들지는 못했다

같은 192-step에 저장된 bounded/unbounded head를 CPU FP32에서 읽었다.
원래 quick32 전체의 같은 입력/FRAME token/alpha/보정된 projection map과
native source-ray 연산으로 21,786개 활성 source ray를 구성했다. 두 저장
checkpoint의 frozen Q2C parameter가 같음을 확인했다.

| CPU saved-head 검사 | Matching prior | Unbounded192 | Bounded192 |
|---|---:|---:|---:|
| 평균 유효 후보 수, exp(entropy) | 10.8534 | 6.0692 | 9.6029 |
| 평균 KL(head || prior) | 0 | 1.09764 | 0.16697 |

실제 bounded 후보 간 상대 log-odds 보정 범위의 최대값은 1.999992였으며,
확률 질량이 `1e-20`보다 큰 support에서 선언한 상한2와 FP32 오차 범위를
통과했다. 같은 V/output을 재사용한 head-only prior counterfactual도
계산했다. 그 출력 차이의 query-weighted chunk RMS 평균은 bounded에서
0.005306, unbounded에서0.010144였다. 이는 전체 voxel/global RMS나
R/use까지 재실행한 native 품질 intervention 수치가 아니다.

앞선 GPU quick32에서 bounded192의 IoU 차이는 기존 NoDQ192 대비
−0.000747, 구간[−0.002946, +0.001383]였다. 따라서 “제약이 구현되지 않아
실패했다”는 설명은 이 검사와 맞지 않는다. 과도한 집중을 줄이는 것만으로는
지금까지 품질 개선이 확인되지 않았다. Planned480과128×3 평가를 유지한다.

CPU 검사는 cached FP16 DINO를 FP32로 projection했다. Live CUDA BF16의
분포를 측정한 것이 아니며, 480-step의 별도72개 GPU 통계와 직접적인 시간
추세로 비교하지 않는다. GT geometry나 보이지 않는 M1 source RGB는 읽지
않았다. 실제 input100개/850,765,040 byte의 로컬 SHA256·크기가 file4 원본과
같음을 읽기 전용으로 확인했다. 두 CPU core에서 약64초가 걸렸고 학습이나
optimizer update는 없었다.

## Physical Top-1 학습 종료

Top-1의 실제480-step 학습, checkpoint3종 저장, native child exit0 및
file4의16개 최종 파일 SHA/크기 검증을 확인했다. 같은 quick32/noise의
NoDQ480 IoU0.442574 대비 Top-1은0.437548이었다. 차이는−0.005026,
paired 95% 구간은[−0.012104,+0.002129]다. Top-K의 point estimate가
높지만 아직 효과 확정은 아니다. 별도128×3 최종 평가가 GPU0에서 시작됐다.
Spatial/DQ/bounded 학습은 중단하지 않았다.

## 저장 오류와 복구

Prior72의 세 GPU 평가는 모두 exit0 및 자체 file4 검증을 통과했다.
원래 CPU summary는 계산과 W&B 업로드 후 SDK debug-core.log symlink를
archive에 포함하면서 실패했다. 링크가 `/root/.cache/wandb/logs/`를 가리켜
파일서버의 rvi가 읽을 수 없었다. 실패 job/state/log는 보존했다. 별도 CPU
successor가 세 closed 결과로 bootstrap을 재계산해 정확히 같음을 확인한 뒤
원본 comparison/logging byte를 변경하지 않고 과학적 결과만 새 경로에
검증 저장했다. GPU 재실행이나 원격 권한 변경은 없었다.

CPU head 검사도 계산 완료 뒤 archival 목록 작성에서 production 입력이
고정 output root 바깥에 있어 종료됐다. 원래 worker/결과/log를 보존한
독립 finalizer가 실제 canonical 입력100개의 SHA/크기를 조회하고 output
범위를 분리해 저장했다. 계산을 반복하거나 raw audit를 수정하지 않았다.
두 저장 문제를 NN 학습 실패로 취급하지 않는다.

## 다음 설계의 판단 근거

현재 근거는 학습/캐시가 동작하지 않는다는 설명과, attention을 넓히면
충분하다는 설명을 모두 약화한다. Candidate coverage 개선, transformed
feature branch의 존재, weight 재분배가 생성기의 실제 geometry/appearance
적응에 필요한 정보를 주는지는 구분해 판단해야 한다.

다음 판단은 이미 연결된 physical Top-1/spatial/DQ/bounded의 최종 대조,
source/target별 cascade metric, PBR 학습 및 과거 target-patch query/R96
설계의 faithful reproduction에 근거한다. 현재 source-camera ray 이식을
과거 target-patch query 재현으로 부르지 않는다. 새 bound나 temperature
sweep, 무조건적인 epoch 연장을 추가하지 않는다.

Source adaptation의 학습 목표를 바꿀 필요가 생기면, 확보된 M1 camera9·10
RGB를 이용한 부분 oracle로 입력 충돌과 adaptation 여력을 먼저 분리할 수
있다. 이 추가 GT 입력 진단이나 train-only feature teacher는 아직 구현·queue하지
않았다. Camera11 GT RGB가 없어 full four-M1 oracle로 부를 수 없다.
향후 새 학습 목적을 쓰면 R-only 및 실제 Top-1/spatial 대조에도 같은 조건을
적용해야 하며, inference에 GT source를 넣는 competitive 모델로 만들지 않는다.

최종 모델은 채택하지 않았다. SS/PBR 비교, 독립 학습 seed 확인, 최종508개와
사용자 지정10개 지표의 범위는 그대로 유지한다.

- [Prior72×3 실제 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-readout-prior-disjoint72x3-s101-v1)
- [CPU192 head 분포 및 근거](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-bounded-vs-unbounded-head192-cpu32-s101-v1)
- [Top-1 최종 quick32 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-member-native-controls-top1-final480-s101-v1)

Canonical result root는
`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`이다.
실제 근거는 `historical-query-reproduction-v1/prior-disjoint-confirmation-v1/summary-canonical-v2`,
`bounded-prior-member-source-ray-v1/checkpoint192-readout-mechanism-CPU32-v1`,
`member-feature-controls-v1/paired-top1-final480-observation-v1`에 보존했다.
