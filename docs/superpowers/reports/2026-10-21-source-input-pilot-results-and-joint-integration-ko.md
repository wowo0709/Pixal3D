# Source 보정 방향 진단, 입력 설계 결과와 native SS 연결

2026-10-21 UTC. 아래 수치는 완료되어 file4 SHA256/size 검증을 통과한 결과다.
새 source-input 설계는 개발 단계이며 최종 full 모델을 채택하지 않았다.

## 실패 원인에서 바꾼 부분

NoDQ SS480의 고정 32개 source feature replay에서 source foreground 보정의
정답 변화 방향 cosine은 0.02191이었다. 보정 크기는 정답 변화 norm의 0.971배인데,
foreground feature MSE는 무보정 대비 1.901배였다. 입력 foreground의 90.74%에
보정이 도달하고 정답 변화 에너지의 86.20%를 지원하는 조건에서도 나타났다.
따라서 단순히 보정 크기나 지원 영역을 늘리는 실험보다 **보정 방향을 학습할 수
있는 입력 표현**을 먼저 검증했다. 이 replay는 literal DINO feature 진단이며,
task conditioning code를 배우는 full SS 모델의 오류나 생성 품질 저하를 단독으로
증명하지 않는다.

기존 retargeter는 `[E + head_delta, T]`를 받았다. `E`는 실제 Top1 correspondence,
`T`는 target context다. 새 SourceContrast는 같은 19개 trainable tensor를 유지하며
`[E + head_delta - S, T]`를 받는다. `S`는 해당 **source-camera ray의 원래 source
feature**이며, retrieved candidate나 0으로 대체하지 않는다. 초기 zero-output
retargeter는 그대로 유지한다. 이것은 RAW Omega 2048 deformation-query arm과
다르며, query grouping이나 frozen matching decoder를 변경하지 않는다.

비교용 SourceReference는 원래 입력에 zero-initialized source projection을 추가한
20-tensor 구조다. warm parent의 함수를 유지하는 장점이 있지만, SourceContrast는
zero-output 상태에서만 구성하고 자기 checkpoint를 strict load해야 한다.

## 동일 teacher supervision을 적용한 실제 결과

TRAIN1536의 고유 source cluster 1536개에 대해 7 epochs/112 updates,
global96/micro96/accumulation1로 학습했다. 세 모델의 모든 minibatch 순서가 같다.
추가 TRAIN M1 source9/10 feature가 supervision이며 source11 정답을 만들지 않았다.
원래 multi-view SS4210 denoiser는 고정했다. 매 epoch 실제 RGB/BF16 native SS를
평가했으며, teacher label이나 GT correspondence를 inference에 넣지 않았다.

| source 모듈 | trainable tensor | native SS IoU, 고정32 |
|---|---:|---:|
| 초기 native SS4210 | — | 0.42671547 |
| Original | 19 | 0.42828020 |
| SourceReference | 20 | 0.42936351 |
| SourceContrast | 19 | 0.43052553 |

SourceContrast−Original은 +0.00224533, source-cluster paired bootstrap 95% CI는
[+0.00056446, +0.00407330]이다. SourceContrast−SourceReference는 +0.00116202,
CI [+0.00030370, +0.00201180]이다. 4000 draws, seed42017을 사용했다.
이는 32 assets/32 clusters, generation noise 1개, training seed 1개의 탐색 결과이고
다중 비교 보정을 하지 않았다. V1 full fine-tuning을 이겼다는 결과가 아니며,
semantic matching/TopK/attention/score/decoder 각각의 효과도 입증하지 않는다.

Feature fitting 자체도 아직 작다. Original/Reference/Contrast 순서로 epoch7 TRAIN
평균 MSE는 무보정 baseline의 0.98939/0.98724/0.98610배다. 이는 epoch 내 변하는
가중치의 training 평균이다. 별도 VAL foreground MSE 비율은
0.99444/0.99416/0.99249, 방향 cosine은 0.07521/0.08223/0.08935다.
Feature replay는 cached target feature를 사용하고 native 생성은 live RGB/BF16을
사용하므로 두 결과를 같은 측정으로 취급하지 않는다.

## 완료된 큰 패널의 기존 SS 결론

아래는 별도 full SS480, 128 assets/93 source clusters × generation noise 3개의
완료 결과다. 위 frozen-denoiser teacher pilot과 직접 순위를 비교하면 안 된다.

| arm | pooled IoU | NoDQ 대비 gain | paired CI95 |
|---|---:|---:|---|
| NoDQ | 0.43880999 | — | — |
| native spatial correspondence | 0.44087903 | +0.00206904 | [−0.00453179, +0.00922399] |
| RAW2048 deformation query | 0.43824305 | −0.00056694 | [−0.00231051, +0.00108991] |
| bounded matching-prior residual | 0.43928187 | +0.00047187 | [−0.00123289, +0.00198570] |

현재 NoDQ 대비 spatial, DQ, bounded prior의 추가 이득은 확인되지 않았다.
bounded arm은 historical V1 reference보다 +0.00663180이지만 query/retargeter lineage가
다르므로 이를 bounded attention의 효과로 해석할 수 없다. 고정 가중치 wrong-value
개입 또한 retrained NoCorr 또는 NoScore ablation을 대신하지 않는다.

Teacher optimizer에서는 작은 Q/K gradient의 상당 부분이 Adam epsilon 1e-8보다
작았다. 그러나 기존 full SS480 Q/K의 median RMS는 약 7.7e-7/1.3e-6으로 충분히
컸다. 따라서 teacher의 epsilon 감쇠를 기존 SS ablation 실패 원인으로 확대하지
않는다. epsilon 변경이나 loss scaling 실험은 실행·예약하지 않았다.

## 다음 판단과 구현 상태

1. 설계32의 source cluster를 모두 제외한 기존 72 assets/61 clusters 패널에서
   세 teacher 모델을 sampling noise 3개로 확인한다. 추가 NN 학습 없이 9개 native
   평가가 예약되었으며 현재 정상 실행 중인 4-GPU PBR의 종료 후 시작한다.
   이는 노출된 validation의 개발 확인이며 untouched final test가 아니다.
2. `source_teacher_initialization.py`와 `source_input_joint_runtime.py`로 native SS
   로딩 경로를 추가했다. 기존 trainer가 historical utility를 로드한 **후** 전체
   teacher state를 로드하고, 그 뒤 원래 optimizer를 생성한다. 역순이면 학습한
   score selector가 덮어써질 수 있으므로 순서를 고정했다. 19 tensors, frozen matcher,
   schema marker, 파일 hash, 112-update TRAIN audit와 전체 state 일치를 검사한다.
3. 실제 두 teacher checkpoint의 CPU tensor roundtrip, frozen matcher, RNG 보존과
   16개 CPU regression check가 통과했다. GPU native SS gradient/micro96 capacity와
   warm-source inference 동등성 검사는 아직 실행하지 않았다. warm teacher에
   cold native identity를 주장하는 실행은 거부한다. 새 full SS 학습도 아직 예약하지
   않았다. 기존 PBR/queue/registry는 변경하지 않았다.
4. 72개 확인에서 이득이 유지되면 자기 native gradient/capacity 검사를 거쳐
   **동일하게 teacher를 학습한 Original19와 SourceContrast19**를 TRAIN9149의 같은
   native SS480 budget으로 비교한다. 동일 SS4210 base+LoRA, global96/micro96,
   AdamW/AMP/sampler/epoch validation을 유지한다. 추가 teacher supervision은
   결과표에 명시한다. 기존 cold Original만 비교해 입력 설계 효과라 부르지 않는다.
5. 최종128×3, 독립 training seed, PBR stage/full cascade와 source/target metric을
   거친 뒤 채택한다. matching/score/decoder의 효과는 성공한 backbone 위의 실제
   retrained controls로 분리한다. NoScore는 실제 Top1 feature, NoCorr는 원래 spatial
   correspondence, NoDQ는 채널·projection 제거라는 정의를 유지한다.

현재 PBR Direct는 common762 prefix에서 total1905로 이어지는 학습이며
world4/global24/micro6/accumulation1이다. 이 보고서 시점에는 실행 중이므로
PBR 결과를 선결론내리지 않는다. final508/full cascade/독립 seed와 요청한
LPIPS·DreamSim·CLIP, PSNR·SSIM·M-IoU·CD·F-Score, 공식 shared-Inception
FID·KID 평가 범위는 유지한다. 32개에서 FID/KID 최종 결론을 내리지 않는다.

## 추적 가능한 결과

Campaign의 `feature-readout-v1/` 아래 canonical 결과:

- `failure-driven-diagnostics-v1/source-projection-alignment-CPU32-v1/`
- `failure-driven-diagnostics-v1/final-controls-and-teacher-optimizer-v1/`
- `bounded-prior-member-source-ray-v1/final128x3-comparison-v1/`
- `source-feature-teacher-pilot-v1/source-contrast-input-pilot-v1/three-pilot-comparison-v1/`
- `source-feature-teacher-pilot-v1/native-joint-loader-readiness-v1/`
- `source-feature-teacher-pilot-v1/source-disjoint72-confirmation-v1/prelaunch-v1/`

각 완료 결과의 `file4-closure.json`이 실제 원격 파일의 hash와 size를 검증한다.
Pilot 비교는 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-source-contrast-three-pilot-native-feature32-v1),
bounded 최종 비교는 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-bounded-member-source-final128x3-s101-v1)에 기록했다.
