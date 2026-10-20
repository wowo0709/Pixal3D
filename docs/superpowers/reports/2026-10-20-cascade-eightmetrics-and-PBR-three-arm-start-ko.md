# SS cascade 여덟 metric 및 PBR 동일 예산 continuation 시작

2026-10-20 09:41 UTC 기준. 브랜치 `wowo0709/feature-retargeting`.
물리 GPU 0–3만 사용한다. 원본 실험과 실패 결과는 보존한다.

## 완료한 32개 생성물 평가

32개 validation asset, source cluster32개, 카메라8/9/10의 96개 view를
각 모델에서 비교했다. SS만 V1/retargeting/Direct로 바꾸었으며,
Shape512/Shape1024와 **V1 PBR762는 고정**했다. 아래 표는 학습된 PBR conditioner까지
포함한 최종 full 모델의 평가가 아니다. 학습 seed101 하나, sampling repeat1회인
개발 panel이며, source-cluster bootstrap4000회로 차이의 95% CI를 계산했다.

| Metric | V1 | Retargeting | Direct |
|---|---:|---:|---:|
| LPIPS ↓ | 0.049296 | 0.049617 | 0.049214 |
| DreamSim ↓ | 0.142520 | 0.143855 | 0.139390 |
| CLIP image cosine ↑ | 0.943445 | 0.943101 | 0.943773 |
| PSNR ↑ | 24.0131 | 24.0112 | 24.0688 |
| SSIM ↑ | 0.945699 | 0.944869 | 0.945815 |
| M-IoU ↑ | 84.5057% | 84.8806% | 84.7644% |
| CD ↓ | 0.012458 | 0.012115 | 0.012264 |
| F-Score ↑ | 84.9527% | 86.1474% | 85.6702% |

모든 arm이 동일한 original-material M1 GT pixel, renderer, 조명, 카메라와 AO seed를
사용했다. Native `Mesh.simplify`의 동일한1M face 예산을 적용했지만, 원본 prediction과
CD/F-Score용 surface는 보존했다. CD는 source bbox 최대 축 길이로 정규화한 좌표계에서
계산했으며, F-Score threshold는0.02, surface sampling은 양쪽20,000 point다.
GT-only crop DreamSim/CLIP은 별도 보조 점수다. FID/KID는 아직 측정하지 않았다.

Direct는 V1 대비 여덟 primary metric의 수치상 방향이 모두 좋지만, **모든 차이의 CI가
0을 포함**한다. Direct−retargeting의 LPIPS는 −0.000403 [−0.001429,+0.000510],
DreamSim은 −0.004465 [−0.011940,+0.002520], CLIP은 +0.000673
[−0.003966,+0.005272]다. Geometry에서는 retargeting의 수치가 Direct보다 좋다.
따라서 matching/attention의 독립적인 추가 이득이 확인됐다고 해석하지 않는다.

Retargeting−V1의 F-Score는 +1.1947 pp, CI [+0.1558,+2.4322] pp다. 이는 하나의
개발 panel에서 여러 metric을 비교한 탐색적 결과이며, 다중 비교를 보정한 최종 검증이나
새로운 독립 학습 seed 증거로 대체하지 않는다. 기존 SS128×3 결과 및 별도 R 두 학습 seed
검증과 함께 보존한다. Bounded TaskRay 최종 SS IoU44.1089%의 추가 이득도 미확인이다.

실제 결과는 `feature-readout-v1/full-cascade32-render-v2/eight-metric-summary`에 있으며,
[W&B의 전체 metric·CI·32개 이미지·per-asset table](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-cascade32-eightmetrics-render1m-v2-s101-20261020)에 로깅했다.

## PBR 실제 학습 시작

R-only 자체 DDP4 capacity 검사가 완료됐다. Native 최대 크기31,900 token asset을
micro6/rank로 사용해 global24, accumulation1, 초기 V1 prediction/loss 일치,
optimizer360→366 state, trainable R6, frozen NN/matcher 불변, validation 후 다시
full microbatch update, 4 rank의 post-DDP initial R30 일치와 update 후 동기화를 확인했다.
Peak 메모리는 최대49.0206 GiB였다. 이 검사의 네 update는 폐기한 profile이며,
과학 실험 학습 budget에 포함하지 않는다.

세 arm의 공통 시작은 V1 step762, 종료는1905다. NN Adam moment와 sampler/clip state를
이어 받고 추가1143 update를 수행한다. global24, GPU4, micro6/rank, accumulation1,
NN LR5e-6, 추가 R/모듈 LR1e-4와 warmup5를 사용한다. Eligible PBR train9139개를
세 arm에서 동일하게 사용한다. Vanilla → retargeting-only → Direct 순서로 이어진다.
새 모듈은 각자의 초기 출력 보존·gradient·frozen-state·capacity 검사를 거쳤다.

09:41 UTC의 Vanilla는789/1905 step으로 실제 실행 중이었다. 첫 보통 크기 batch의
관찰값은 compute 약12초/update, data load 약2.2초, 할당 memory 약43.4 GiB였다.
최대 token capacity batch는 compute 약39초/update였다. 아직 짧은 초기 구간이므로
전체 학습 ETA나 수렴 여부를 확정하지 않는다. 매 epoch validation은 기존 native
프로토콜을 유지한다. Healthy process를 관찰 timeout만으로 재시작하지 않는다.

## 학습 후 두 종류의 생성물 비교 연결

학습된1905 checkpoint를 실제 generated shape support에서 사용할 native replay 경로는
이미 sampler와 texture VAE까지 검증했다. GT sparse support를 사용하지 않는다.
이후 160개 추론을 예약할 activation은 실제 세 학습 및 stage summary receipt를 기다린다.

- Full-stack: SS와 PBR을 각각 V1/R/Direct로 짝지은32×3=96건.
- PBR isolation: 같은 generated V1 shape에서 R/Direct PBR을 적용한 추가64건.
  Vanilla는 full-stack의 V1 결과를 재사용한다. 실제 sparse support와 noise field를
  세 모델에서 동일하게 검증한다.

새 render activation은160개 실제 prediction이 모두 완료된 뒤, 동일한1M face native
렌더링과 기존 shared GT reference를 연결한다. 각 panel에 공식 GPU DreamSim/CLIP
288개 image pair, 여덟 metric, source-cluster CI,32개 비교 이미지, W&B 로깅을 연결했다.
Full-stack과 PBR isolation의 결과는 별개로 집계한다. 학습된 결과는 아직 생성되지 않았다.

평가 코드 검증에는 실제 generated shape에서 나온 native/R-zero/Direct-zero의
정확히 같은10개 NPZ array를 사용했다. 동일 출력을 중복 렌더링하는 대신 native Vanilla
렌더링을 명시적으로 재사용했다. 실제 여덟 metric 차이는 모두0이었으며, metric을
덮어쓰지 않았다. 이는 한 asset의 초기 경로 검증이며 학습 성능 증거가 아니다.
통계 함수는 불균형 source cluster와 비정상 pairing의 CPU 테스트4개를 통과했고,
완료한32×3 실데이터의 기존 여덟 metric mean/CI를 다시 계산해 최대 차이0을 확인했다.

## SS64 개선이 native Shape 입력에 전달되는지 확인

추가 GPU 추론 없이 위32개 asset의 실제 SS archive를 읽어 native2× max pooling을
검사했다. NumPy로 계산한96개 support와 token 수가 실제 `native_shape_support` 및
생성 기록과 정확히 같았다. 아래32-grid target은 동일 GT64를 pooling한 진단값이며,
Shape512 VAE의 별도 encoder label이라고 해석하지 않는다.

| 동일32개 panel | SS64 IoU | Native input32 IoU | 최종 F-Score |
|---|---:|---:|---:|
| V1 | 43.4837% | 61.0379% | 84.9527% |
| Retargeting | 44.6554% | 62.5877% | 86.1474% |
| Direct | 44.6784% | 62.1542% | 85.6702% |

Direct−R은64-grid에서 +0.0230 pp지만32-grid에서 −0.4335 pp다.
후자의 CI는 [−1.0121,+0.2480] pp로0을 포함한다. 숫자상 native32와 최종 surface
순위가 일치하지만 이 사실만으로 인과관계를 확정하지 않는다. SS128×3의44.3692%
최고값과 이32개 panel의44.6784%를 같은 모집단의 수치로 혼동하지 않는다.

두 모델 사이 fine-grid 차이가 있는 parent block 중 약70–73%는 max pooling 후 같은
occupancy가 된다. 이는 accuracy gain의70–73%가 사라진다는 뜻이 아니다.
실제로 R−V1의32-grid 개선은 +1.5498 pp [ +0.0616,+3.0995 ] pp로 유지됐으며,
모델 간 pooled support가 완전히 같은 asset은0/32였다. 따라서 pooling이 개선을
전부 없앤다는 설명은 지지되지 않는다. Full 선택에는 SS64뿐 아니라 native 입력의
변화와 최종 CD/F-Score/렌더링 점수를 함께 사용해야 한다.
[실제 pooling 진단과 CI도 W&B에 기록했다](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-cascade32-nativepool-support-s101-20261020).

## Test508 준비 상태

6242개 canonical 입력의 실제 file4 SHA256·byte 일치와 test 전용 RGB/world-frame
metadata 변환은 완료됐다. 기존 train/validation은 변경하지 않았다.
첫 고정 test asset의 native Shape/PBR VAE 인코딩도 완료됐고, 두 support가12932개
token으로 정확히 같았다. 나머지 test latent와 SS/projection/Omega/DINO/matching cache,
registration QA는 준비해야 한다. Test에서 head를 학습하거나 후보 모델의 품질 점수를
확인하지 않았다.

전체508개 Shape/PBR world latent는 네 독립 shard와 coverage 검사로 연결했다.
현재 PBR 세 학습의 완료 receipt 이후 실행하여 DDP 학습 GPU와 겹치지 않으며,
기존 첫 asset의 올바른 인코딩을 재사용한다. Token 수가 큰 test asset도 제외하지 않는다.
SS/projection 및 입력 feature/matching cache는 후속 작업으로 남아 있다.

최종 full 모델 선택, matching decoder/score/correspondence별 독립 효과,
새 독립 학습 seed 확인, 올바른 별도 test508과 FID/KID 평가까지 남아 있으므로
전체 목표는 계속 진행 중이다.
