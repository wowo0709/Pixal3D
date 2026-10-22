# Native SS 첫 epoch와 physical Top1 control 실제 진행

2026-10-22 05:55 UTC 확인. 네 학습은 모두 world1·global96·micro96·accumulation1,
같은 TRAIN9149·seed101·original multi-view SS4210+fresh optimizer다. 학습 시작 전
동일 DEV32의 native global occupancy IoU는 네 모델 모두 0.4267154667이었다.

## 실제 96-update 결과

| 모델 | Native global voxel IoU | Voxel F1 | 같은 step vanilla 대비 IoU |
|---|---:|---:|---:|
| Vanilla | 0.433978862 | 0.605279301 | — |
| R_feature | 0.431369326 | 0.602736580 | −0.002609536 |
| R_raw_topk | 0.430619123 | 0.602003868 | −0.003359739 |
| R_top1_only | 첫 96-update 검증 대기 | — | — |

Vanilla는 학습 중 validation뿐 아니라, 저장한 complete denoiser/LoRA를 독립 GPU
프로세스에서 다시 불러온 결과도 32개 asset·seed·GT hash·TP/FP/FN·전체 metric과
정확히 일치했다. 재평가는 optimizer update0, 실제 child returncode0이며 file4 SHA
검증을 마쳤다. 두 R 모델의 독립 GPU replay는 큐에 있고 네 GPU의 본 학습이 끝나
자원이 비면 실행된다. In-process 결과와 독립 replay 완료를 구분한다.

세 모델의 실제 첫 96개 training step에서 asset IDs·noisy latent·timestep·native
global condition hash가 모두 일치했다. 해당 prefix를 immutable 파일로 저장하고
파일서버4 SHA 검증을 마쳤다. 비동기 current step을 서로 비교한 결과가 아니다.

Asset별 mean IoU의 paired source-cluster bootstrap 결과는 다음과 같다. 위 표의
native global occupancy IoU와 집계 방식이 다르므로 수치 차이를 혼용하지 않는다.

| 대비 | Mean asset IoU 차이 | CI95 |
|---|---:|---|
| R_feature − vanilla | −0.002693426 | [−0.010530021, +0.004145385] |
| R_raw_topk − vanilla | −0.002829539 | [−0.010754897, +0.004214157] |
| R_raw_topk − R_feature | −0.000136113 | [−0.001159088, +0.001087442] |

한 generation noise, 재사용 DEV32, 첫 epoch의 보정하지 않은 탐색적 CI다. 현재
점 추정치는 vanilla가 좋지만, 추가 모듈의 확정적 악화나 최종 성능을 뜻하지 않는다.
최종480·DEV128×3 generation noises·독립 training seed·후속 cascade가 남아 있다.

## 구현·학습 경로에서 실제 확인한 것

두 R 모델의 96-update source checkpoint를 CPU에서 실제 읽어 native recipe와 own
schema를 검사했다. Frozen matcher의 decoder/score·buffer와 일곱 head reference anchor는
초기 값과 정확히 같다. Learned head 아홉 tensor는 두 모델 모두 값이 변경됐다.
따라서 head가 optimizer에서 누락되어 가중치가 전혀 갱신되지 않는 상황은 아니다.

R_feature의 query/key weight relative L2 변화는 약1.65%/1.46%, R_raw_topk는
약1.74%/1.54%다. 이는 파라미터 변화이며 attention posterior가 유용한 후보를 선택했다는
증거는 아니다. 실제 posterior·R feature·consumer delta 계측은 별도 inference 진단으로
연결돼 있다. Scalar fusion scale이 약0.00837이라는 이유만으로 R 전달이 꺼졌다고
해석해서도 안 된다. 현재 구현의 R residual은 그 scalar gate 밖으로 전달된다.

## Physical Top1 control은 실제 본 학습 중

R_feature에도 learned Top-K feature가 있으므로 이를 NoScore로 해석하지 않는다.
새 R_top1_only는 semantic posterior의 argmax feature 하나만 gather하고 R/use/fusion
열 tensor만 둔다. Q/K/V/output/score utility head는 없다. Uniform attention이 아니다.
원래 source-camera ray grouping·native carrier·missing-match fallback을 유지한다.

다음 검증은 CPU 계획이 아니라 실제 GPU 2에서 수행한 결과다.

* Native global/projection/negative field와 same-noise velocity/loss가 원래 native와 같다.
* 공통 R/use/fusion 열 개 초기 tensor가 R19와 같고, constructor 이후 네 RNG state 및
  초기 LoRA가 기존 vanilla proof와 일치한다.
* Initial DEV32 validation 뒤 실제 micro96 세 update를 수행했다. 원래 LoRA360+
  source10의 370개 optimizer state가 step3이고, 열 source gradient가 모두 finite positive다.
* Frozen parameter/buffer가 변하지 않고, 실제 세 batch의 asset/noise/timestep/global
  trace가 vanilla capacity와 같다. 이 임시 모델/optimizer는 본 학습에 이어 쓰지 않았다.
* Peak allocated88.4994 GiB, reserved93.4395 GiB였다. Steady compute는 약36초/update다.

자체 admission 통과 후 fresh model로 GPU2 본 학습을 시작했다. 초기 validation IoU도
0.4267154667과 일치한다. 기존 세 학습의 코드·config·optimizer를 고치거나 재시작하지 않았다.
새 데이터나 Omega/matching cache를 만들지 않았다. 새로운 CPU matching 학습도 없다.

최종 해석은 Top1+R 대 vanilla, learned Top-K feature 추가 대 Top1+R, raw weighted
appearance 추가 대 learned feature로 분리한다. Native spatial NoCorr와 decoder의 독립
효과는 이 네 모델만으로 주장하지 않는다.

## 실행과 보존

GPU0 R_feature, GPU1 vanilla, GPU2 R_top1_only, GPU3 R_raw_topk. 모두 micro96,
accumulation1이며 현재까지 OOM은 관측되지 않았다. 최근 compute는 각각 약48초,
34.6초,36.3초,47.4초/update다. Source19의 peak allocated는90.6991 GiB다.

기존 GPU2의 broad cascade case는 자기 작업을 정상 종료한 뒤 새 control proof/학습에
자리를 넘겼다. 나머지 기존 case·render·metric job은 큐에 보존돼 있다. 네 본 학습이
GPU를 점유하므로 일부 독립 replay/diagnostic/cascade가 기다리는 상태이며 실패가 아니다.

본 실험과 평가의 canonical root:
`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/`
`ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/`
`source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/native-flow-cold3-v1/`.
세 모델의 checkpoint evaluation은 `checkpoint-evaluation-v2`, Top1은
`physical-Top1-control-v1`, first-epoch CPU evidence는
`checkpoint-evaluation-v2/epoch96-CPU-evidence-v1`에 있다.

W&B project: `youngwoojeon-unist/pixal3d-deformnext-retargeting`.
새 control run ID는 `ss-native-Top1-only-cold-s101-science-v1`이다.
전체 목표는 아직 진행 중이며 native PBR 개선, component 검증, 독립 seed 및 최종
508개·10 metric 평가까지 완료된 상태가 아니다.
