# Generated PBR 8-metric와 R-mediated 7에폭 결과

2026-10-22 UTC. 개발 32 source cluster, 학습 seed 101, generation noise 1회 결과다.
독립 학습 반복이나 final test508 결과가 아니며, 반복 개발 비교의 신뢰구간은 다중 비교 보정 전이다.

## 현재 결론

Full-R은 Full-V1보다 target8 PSNR과 DreamSim이 개선됐다. Source9/10의 LPIPS·DreamSim 개선은
확인되지 않았다. SS와 PBR이 함께 달라지는 full-stack 결과이므로 R 모듈 하나의 효과로 해석하면 안 된다.
동일 geometry와 동일 초기 texture noise를 쓰는 PBR-only 비교에서는 R의 이득이 입증되지 않았고,
Direct는 target PSNR 및 source DreamSim을 악화시켰다.

SS의 새 R-mediated 두 설계는 실제 7에폭을 완료했다. Raw Top-K가 기존 bounded 모델보다
매칭 지표와 source feature fit을 일부 개선했으나, native IoU 차이는 불확실하다.
현재 모델을 채택하거나 단순히 에폭을 늘릴 근거는 부족하다.

## 실제 generated support에서의 PBR 결과

160개 prediction, 160개 render/shared-reference 평가, 공식 DreamSim·CLIP 480 view가 완료됐다.
Shape512/1024는 지정된 원본 LoRA checkpoint를 사용했다. 모든 모델의 reference는
**변형된 M1 geometry에 원래 source material을 적용한 동일 RGB/alpha**다.
Source 카메라 reference도 M0 이미지가 아니라 이 M1 reference이므로 material 보존과 edit 적응을 함께 평가한다.

| 모델 | target PSNR ↑ | target SSIM ↑ | target M-IoU ↑ | source LPIPS ↓ | source DreamSim ↓ | source CLIP ↑ | raw CD ↓ | raw F-Score ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-V1 | 24.2682 | .95049 | .88096 | .053055 | .155995 | .940614 | .012458 | .849527 |
| Full-R | 24.7413 | .95167 | .88493 | .052596 | .154441 | .943435 | .012115 | .861474 |
| Full-Direct | 24.2567 | .95087 | .88987 | .053431 | .155872 | .942331 | .012264 | .856702 |
| PBR-only R | 24.3004 | .95045 | .88099 | .053184 | .157008 | .940912 | .012458 | .849527 |
| PBR-only Direct | 23.9970 | .94970 | .88099 | .053497 | .161607 | .940248 | .012458 | .849527 |

CD는 source bbox 좌표계, surface 20k/20k, F-Score threshold .02, ICP 없음이다.
PBR-only 두 모델은 Full-V1 geometry를 그대로 사용하므로 CD/F-Score가 정확히 같다.
렌더는 각 shape의 기존 native predicted LOD를 공유하며, GT shape/latent로 생성하지 않았다.

주요 paired source-cluster bootstrap 95% CI:

| 비교 | 지표/방향 | 평균 차이 | 95% CI |
|---|---|---:|---|
| Full-R − Full-V1 | target PSNR ↑ | +.47307 dB | [+.22162, +.76412] |
| Full-R − Full-V1 | target DreamSim ↓ | −.009154 | [−.016979, −.002630] |
| Full-R − Full-V1 | source LPIPS ↓ | −.000458 | [−.002088, +.000948] |
| Full-R − Full-V1 | source DreamSim ↓ | −.001554 | [−.008205, +.005087] |
| PBR-only Direct − Full-V1 | target PSNR ↑ | −.27122 dB | [−.58329, −.00646] |
| PBR-only Direct − Full-V1 | source DreamSim ↓ | +.005611 | [+.000240, +.012538] |

PBR-only R의 개선은 확인되지 않았다. 이 결과는 이전 GT-shape stage 평가의 작은 개발 패널 추세와
실제 generated shape에서의 추세를 구분해야 한다는 점도 보여준다.
FID/KID는 이 32개로 추정하지 않고, 모델 선택 후 final508의 계획된 view pool에서 평가한다.

완료 근거는 file4 `rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/trained-PBR-generated32-render-v2/target-source-eight-metric-summary-v1/{audit.json,statistics.json,file4-closure.json}`이다.
[W&B: 8 metric, paired CI, GT/Pred/Diff](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr1905-generated32-target-source-eight-metrics-v1).

## 새 SS R-mediated 모델의 고정 112-step 결과

두 모델 모두 source-camera ray-wise query, member Top-K, decoder, score prior, bound 1,
물리적 NoDQ 구조를 유지했다. Direct feature bypass를 제거해 실제 전달 residual이 R 출력과
같음을 native forward마다 검사했다. 초기 native32 global/projection/negative/RNG는 원본과 정확히 같았다.
Top1+feature와 raw probability-weighted Top-K+feature를 R appearance 입력으로 비교했다.

조건은 TRAIN1536, source cluster disjoint, 7에폭/112 updates, 동일 batch 순서·19 tensor 초기값,
global/microbatch 96, accumulation 1이다. Denoiser는 frozen이며 **full SS LoRA fine-tuning이 아니다**.
학습 GPU peak는 각각 31.68/32.27 GiB, update 평균 compute는 11.41/10.44초였다.
두 실행은 실제 terminal 0과 file4 SHA closure를 남겼다.

| 모델 | native IoU ↑ | exact ray-set Top1 ↑ | exact GT mass ↑ | source foreground MSE ratio ↓ |
|---|---:|---:|---:|---:|
| Bounded contrast | .430592 | .228752 | .177564 | .992084 |
| R: Top1+feature | .430335 | .233342 | .179174 | .991497 |
| R: raw Top-K+feature | .431005 | .242152 | .181623 | .991476 |

Raw Top-K − bounded IoU +.000413, CI [−.001359, +.002278].
Raw Top-K − Top1+feature IoU +.000670, CI [−.000841, +.002003].
Exact Top1/GT-mass 차이도 CI가 0을 포함한다. Near1 GT mass에서는 raw Top-K − bounded
+.008144, CI [+.000999, +.015649]이나, 이는 정확한 3D correspondence나 native 품질의 입증이 아니다.
Source foreground 오차의 약 .85%만 줄었고 correction 방향 cosine도 약 .092다.
정보 전달 경로 수정은 feature fit을 소폭 개선했으나, reconstruction 향상을 충분히 설명하지 못한다.

원인은 확정하지 않는다. Feature-teacher objective와 실제 SS task 사이의 차이, 부족한 source 적응,
동결된 native denoiser/global context 등이 후보이나 현재 비교로 분리하지 못했다.
Global context가 네 view의 CLS/register 평균으로 그대로 유지된다는 것은 코드로 확인했지만,
그것이 병목이라는 인과 증거는 없다. GT visibility gate나 target-only 입력을 새로 도입하지 않았다.

완료 근거는 위 feature-readout 경로의
`source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/matching-feature-native-summary-CPU-v2/{audit.json,file4-closure.json}`이다.
[W&B: matching/feature/native 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-R-mediated-teacher112-matching-native32-v1).

## 다음 비교와 운영 수정

원본 four-RGB zero-shot 두 조건(source4 M0, edited-target M1+source3 M0)과 위 다섯 모델을
같은 8-metric 표로 묶는다. 기존 zero-shot 생성·렌더 작업을 재사용하며, 새 GPU 생성 작업을 중복 추가하지 않는다.
새 CPU join은 generation lineage, source cluster, 실제 camera archive SHA, raw GT/alpha,
GT-only crop, 공식 metric registry/weights/precision, PBR isolation의 geometry·초기 noise를 검증한다.
기존 다섯 모델의 통계도 재계산해 정확히 재현해야 한다. Original cascade와 PBR replay의 texture noise stream은
동일하지 않으므로 zero-shot 비교는 deployment 비교이며 학습/모듈의 단독 인과 증거가 아니다.

Zero-shot 작업 64개와 기존 후속 평가를 먼저 완료하도록 **아직 시작하지 않은 작업의 우선순위**를 조정한다.
진행 중인 정상 작업은 중단하지 않고, 다른 기존 평가 범위도 유지한다.
그 뒤 데이터 분리 검증, 실제 SS/PBR joint task 학습, 독립 seed, cascade 및 final508 검증이 남는다.
기존 plain-contrast source72에서 내려진 HOLD는 유지하며 새 모델의 개선으로 바꿔 적지 않는다.

운영 중 이전 render activator가 늦게 중복 166개를 추가한 문제가 있었다. 이미 완료된 새 160-case
render와 prediction/weights/raw geometry/camera/shared reference의 동일성을 검사하고 file4에 기록한 뒤,
시작 전 중복 render 109개에 HOLD gate를 추가했다. 먼저 실패한 51개와 dependency-held 6개 이력은 보존했다.
기존 명령을 완료했다고 처리하거나 queue-state를 수정하지 않았다. Canonical raw 결과는 file4에 남아 있다.
새 480-view metric 및 8-metric 집계는 실제로 완료됐다.

새 CPU join의 회귀 검증은 8개다. 잘못된 four-RGB/원본 LoRA lineage, GT support 유입,
metric precision 혼합, 다른 pixel에 붙은 score, cross-model GT 차이, isolation noise 차이를 거부하고,
정상 synthetic seven-model join이 기존 통계를 재현하는지 확인한다. Synthetic receipt는 원격 file4 검증의 대체 증거가 아니다.
