# Semantic ray attention 및 생성 shape PBR 평가 진행 기록

2026-10-20 08:42 UTC 기준. 작업 브랜치는 `wowo0709/feature-retargeting`이며,
실제 데이터·설정·체크포인트·결과는 file4의 `retargeting-v1/rgb_matching_v2/semantic-topk-search-v1` 아래에 보관한다.
로컬 파일은 해당 file4 결과의 캐시다. 사용 GPU는 물리 GPU 0–3으로 제한한다.

## 현재 성능 판단

동일한 128개 validation asset, 93개 source cluster, 3개 sampling noise에서의
SS 평가다. 새로운 full 후보의 학습 seed는 101 하나이며, sampling noise 세 번을
학습 seed 세 번으로 해석하지 않는다. Retargeter 자체의 별도 두 학습 seed 검증은
기존 `2026-10-19-retargeter-two-seed-effect-ko.md`에 남아 있다.

| 구성 | SS voxel IoU |
|---|---:|
| Vanilla V1 | 43.2650% |
| 같은 예산의 Top-1 feature + retargeting 기준 | 44.0294% |
| 이전 positive retargeting 기준 | 44.2757% |
| Ray feature attention | 43.9242% |
| Ray feature attention + physical deformation query | 44.2249% |
| Direct retargeting 조합 | **44.3692%** |
| Scalar Top-K + retargeting | 44.0329% |
| Candidate별 paired correction + reliability | 44.1322% |
| Bounded task ray attention + paired correction | 44.1089% |

Direct−Vanilla는 +1.1042 percentage point, source-cluster paired bootstrap 95% CI는
[+0.6120, +1.6301] pp다. 그러나 Direct−같은 예산 retargeting은 +0.3399 pp,
CI [−0.0556, +0.7493] pp이며, 이전 positive retargeting 대비도
+0.0935 pp, CI [−0.2694, +0.4836] pp다. 따라서 현재 결과는 retargeting을 포함한
fine-tuning의 이득을 지지하지만, semantic matching/attention의 독립적인 추가 이득이나
최종 full 모델 채택을 증명하지 못한다. 두 retargeting 기준을 모두 유지한다.

## 현재 학습하는 attention

`TaskRay369`의 SS 480/480 step 학습, native32 및 최종128×3 평가가 완료됐다.
공통 R96에서 추가 384 update, GPU 2·3 DDP, global batch 96, microbatch 48/rank,
accumulation 1이다. 최근 실제 update는 약 18.7–19.1초, peak 할당 메모리는
rank별 약 51 GiB였다. 관찰된 정상 학습을 상태 확인 시간 초과 때문에 재시작하지 않는다.

- 학습된 semantic matcher/decoder, score prior 및 reliability head는 고정한다.
- 각 target patch ray 그룹에 하나의 query를 만들고, 해당 그룹 voxel이 query를 공유한다.
- 전체 Top-64 후보 주소·feature·score를 유지하고, 4-head Q/K attention으로 후보를 조합한다.
- 실제 RAW Omega 차이 2048 channel을 포함한 4096-channel query를 사용한다.
- 고정 semantic prior에 대한 logit 보정은 ±1, KL 가중치는 0.001이다.
- 462 step의 전용 `task-ray-training.jsonl`에서 유효 후보 수 평균 5.0018,
  query/key gradient L1 각각 0.09052/0.00909를 확인했다.

일반 conditioner 로그의 Top-1 appearance 경로 통계와 추가 attention의 통계를 구분한다.
마지막96 update의 실제 추가 attention 유효 K 평균은 5.6633, confidence 평균은
0.43293이었다. 현재 TaskRay에서 Top-K attention은 candidate별 deformation correction을
혼합하며, 공통 appearance carrier는 Top-1을 유지한다. Appearance feature 자체의
추가 readout은 앞선 feature/Direct arm에서 별도로 비교했다. 이 실험은 matcher 자체를
새로 학습하는 실험과도 구별한다.

최종 TaskRay IoU는 44.1089%, matched R 대비 +0.0796 pp,
source-cluster CI [−0.0837, +0.2417] pp다. Task attention이 없는 paired correction 대비는
−0.0232 pp, CI [−0.1072, +0.0452] pp다. 이전 positive R 대비도 −0.1668 pp,
CI [−0.4818, +0.1426] pp다. 추가 attention의 이득이 입증되지 않았으며 이 후보를
full로 채택하거나 추가 epoch를 자동 실행하지 않는다. 기존 summary를 보존하고,
historical R을 추가한 `final128x3-strong-references-v2`에서 원래 비교 수치가
모두 정확히 유지됨을 확인했다.

## 32개 생성 결과와 여덟 metric

현재 32×3 모델의 cascade 비교는 SS 모델만 달리하고 Shape512/1024 및
V1 PBR762를 고정한다. 이는 SS 변화가 최종 생성물에 주는 영향에 대한 비교다.
학습된 PBR conditioner까지 포함한 full-stack 비교는 별도로 준비한다.

Dense asset의 원본 생성 mesh는 유한한 9,187,516 vertex / 18,722,964 face였고,
native rasterizer가 CUDA illegal-memory-access로 실패했다. OOM으로 단정하지 않는다.
새 프로세스의 전체 원본에서도 실패를 재현했고 native `Mesh.simplify`의 1M face 예산은
네 depth layer 렌더링을 통과했다. 실패 기록과 성공한 원본 생성물은 보존한다.

새 `render_cascade_artifact_v2.py`는 모든 모델·asset에 동일한 1M face 렌더링 예산을
적용한다. 원본 decoder 좌표에서 simplify한 뒤 기존 카메라/scale을 적용하며,
원본 prediction 및 CD/F-Score용 surface는 바꾸지 않는다. GT나 metric을 이용한
출력별 정렬·축소·제외는 없다. Native GLB export와 face 예산만 같으며,
GLB 전체 remesh 과정을 재현한다고 주장하지 않는다. 기존 네 생성물은 재사용하고
32×3 전체를 이 공통 렌더링 프로토콜로 처리하도록 연결했다.

요청된 metric은 LPIPS, DreamSim, CLIP, PSNR, SSIM, M-IoU, CD, F-Score다.
GT-only foreground crop 보조 점수도 별도 기록한다. 공식 DreamSim ensemble과
별도의 원본 OpenAI CLIP ViT-B/16을 작업 전용 dependency/weight 경로에 고정했고,
공식 전처리와 실제 GPU identity/changed-image 검사를 통과했다. DreamSim의 학습된
72개 LoRA tensor도 확인했다. 단일 asset 9개 image pair 검사는 수치 구현 검증이며
전체 성능 증거가 아니다. 32개 전체 scorer·source-cluster CI·W&B 집계는 예약되어 있다.
FID/KID는 충분한 최종 이미지 집합에서 측정해야 하며 아직 측정하지 않았다.

## PBR continuation과 생성 support 연결

동일한 V1 step762에서 step1905까지 추가 1143 update로 Vanilla, retargeting-only,
Direct 세 arm을 비교하도록 준비했다. global batch24, GPU4, micro6/rank, accumulation1이다.
자기 모델의 native DDP4 capacity 검사가 끝나기 전에는 본 학습을 시작하지 않는다.
Retargeting-only는 실제 C1/3072-channel 구조이며, Direct의 추가 channel을 0으로
채워 대체하지 않는다. Native microbatch gradient/frozen-state 검증과 초기32개
전체 record/metric 및 34개 PNG의 V1 일치 검사는 완료했다.

생성된 shape의 실제 grid64 support에서 online voxel→ray 주소를 계산하고,
native training fusion을 사용하는 추론 hook을 구현했다. 32개 기존 cache와 주소가
일치했고, GT latent/geometry 경로 없이 실제 생성 support 3161 token에서 동작했다.
Native 12-step PBR sampler에서 R-zero와 Direct-zero 출력이 native와 정확히 같았다.
명시적 진단용 gate 변화는 실제 sampler 출력에 전달됐다. 이는 구현 검증이며
학습된 모델의 성능 향상을 뜻하지 않는다.

Native texture sampler는 CPU에서 noise를 만들기 때문에 hook은 CPU와 CUDA RNG를
모두 보존한다. 현재 full-cascade native sampler의 guidance는 1.0이며,
GT-shape 고정 PBR stage 평가의 guidance3과 평가 범위를 구분한다.

추가한 `replay_pbr_on_generated_shape.py`는 저장된 실제 generated shape latent를
사용해 새 PBR checkpoint로 texture를 추론하고 native texture VAE까지 디코딩한다.
Shape decoder는 texture guide hierarchy를 얻는 데 사용한다. 원본 raw mesh와
geometry 평가를 보존하고, decoder voxel address의 순서만 정확한 정수 좌표 join으로
맞춘다. 가까운 점 보간이나 shape 보정은 허용하지 않는다. PBR noise는 모델별로
초기화가 끝난 뒤 동일 seed로 새로 추출하는 별도 replay 프로토콜이다.
CPU 테스트 10개가 통과했고, 실제 native/R-zero/Direct-zero export 비교도 완료했다.
첫 시도에서 Vanilla는 성공했지만 R-only 설정을 Direct prototype factory에 전달해
R 생성 단계가 실패했다. 새 v2 plan은 factory에 Direct constructor 설정을 전달하고,
생성되는 실제 C1 설정이 기존 native R 설정과 정확히 같음을 확인한다. Native 수치
소스·weight·sampler는 수정하지 않았고 첫 실패와 성공한 Vanilla 결과도 보존했다.
수정된 세 arm은 실제 12-step sampler와 native texture VAE를 거쳐, 원본 geometry뿐
아니라 새 texture latent 및 약120만 voxel의 attribute까지 모두 정확히 일치했다.
각 arm은 약64초, peak16.26–16.31 GiB였다. 결과와 actual file4 SHA/size 검증은
`generated-PBR-replay-export-v2`에 남겼다. 이 검증은 초기 zero 경로의 정확성에 대한
증거이며, 학습된 step1905의 최종 생성물 평가까지 완료된 상태로 해석하면 안 된다.

## 남은 실험과 최종 평가

1. TaskRay480 최종128×3와 두 retargeting 기준 비교는 완료했다. 현재 설정의
   추가 이득은 입증되지 않았다. Direct의 수치상 최고 결과와 두 R 기준을 유지한다.
2. 32×3 공통 렌더링 및 여덟 metric을 완료한다.
3. PBR 자체 DDP4 capacity 검사 후 세 arm의 같은 예산 continuation을 진행한다.
4. 학습된 PBR checkpoint를 실제 generated shape에서 비교하고, GT-shape 고정 stage와
   full-stack 결과를 구분한다. 세 PBR1905 학습·stage summary의 실제 file4 receipt와
   zero-export GPU 검증을 기다리는 자동 후속 추론을 연결했다. SS variant별 shape와
   PBR을 함께 바꾸는 96건, 동일한 generated V1 shape에 R/Direct PBR만 추가로 적용하는
   64건을 준비하며, 후자의 Vanilla는 앞의 결과를 재사용한다. 160개 metadata 경로를
   검사했다. 이 단계는 추론 연결이며, 학습된160개 출력의 생성 및 후속 렌더링·여덟 metric
   집계까지 완료됐다는 뜻은 아니다. 긍정적인 후보에는 독립 학습 seed와 모듈별 대조를 추가한다.
5. No-score는 Top-1 correspondence feature, No-corr는 native spatial correspondence,
   No-DQ는 channel/projection의 물리적 제거로 정의한다. Retargeting-only 다음에
   deformation-aware query를 더하는 두 arm을 사용하며 target-only 및 2×2는 추가하지 않는다.

별도 legacy test508이 존재하고 train9149/validation508과 sample/source cluster/source UID가
겹치지 않는 것은 metadata로 확인했다. 그러나 이 test는 아직 depth-target 및 legacy latent
schema여서 현재 RGB/world-v4 모델의 최종 test로 바로 쓸 수 없다. Canonical file4의 기존
RGB/cache 존재 여부를 조사하고 test 전용 전처리를 해야 한다. 후보를 고르기 전에
test 모델 점수를 열지 않는다. Validation32/128/508은 이미 노출된 개발 데이터다.

최종 모델 채택, full-stack 평가, 독립 학습 seed, 올바르게 전처리된 별도 test508,
충분한 이미지 수의 FID/KID까지 남아 있으므로 전체 실험 목표는 계속 진행 중이다.
