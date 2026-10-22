# Native PBR source-ray 이식: sparse geometry와 전달 경로

2026-10-22 UTC. 기존 네 SS native 학습을 유지하면서 다음 PBR 단계의 필수 경로를 구현했다.
현재 작업은 PBR 학습 완료나 성능 향상 결과가 아니다.

## 필요한 구조 차이

현재 SS source-ray trainer는 dense16 grid, 1024차원 value, 고정4096 query를 요구한다.
PBR은 native64 grid의 가변 sparse support와 DINO1024+NAF1024 value를 사용한다.
따라서 SS guard를 풀거나 NAF 절반을 버려 이식하지 않는다. 기존 SS 및 PBR 학습 코드는 수정하지 않았다.

새 `pbr_source_ray_geometry.py`는 다음을 구현한다.

* 원래 회전된 ProjGrid 좌표와 camera 변환으로 실제 sparse row의 source-camera ray를 구한다.
* 가변 batch와 interleaved coordinate 순서를 유지하며 실제 ray member만 평균한다.
  Padding0을 관측 feature로 계산하지 않고 빈 ray는 invalid다.
* Source32patch map의 DINO/NAF2048 전부를 native bilinear sampler로 전달한다.
  Native MV4 carrier에 이미 포함된 own-source 항을 빼고 세 source correction의 합을4로 나눈다.
* Query frustum membership과 feature write를 구분한다. Projection은 원래 border sampling을 유지하며
  surface visibility/depth gate를 추가하지 않는다. Zero semantic gate도 R residual을 막지 않는다.
* Sparse row만 sample하여 dense64³×2048 correction tensor의 materialization을 피한다.
  Channel chunk는 보간의 임시 메모리를 제한하며 microbatch accumulation이 아니다.

## 검증 범위

CPU 검사에서 native dense projection을 실제 sparse 좌표로 gather한 reference와 새 결과 및
gradient를 비교한다. 서로 다른 camera/mesh scale, batch별 row 수, interleaved 순서,
양쪽 DINO/NAF, channel chunk127/2048, exact-zero replacement, native border delivery를 포함한다.
실제 입력 contract는 SS-width cache, 다른 RGB hash, 다른 support/scale, camera 순서 및 invalid patch를 거부한다.
두 관련 test 파일의 CPU 검사9개가5.57초에 통과했다. GPU를 사용한 결과가 아니다.

별도 GPU engineering entry `probe_sparse_pbr_source_geometry.py`도 구현했다. 선언한 실제 TRAIN
PBR latent·transport·camera snapshot을 읽고 native64 dense reference와 sparse projection을 비교한다.
실제 GPU 실행 전에는 GPU parity 완료로 적지 않는다. Reference는 F32 atol2e-5/rtol1e-5이며
실제 DINO/NAF별 max error와 RMS를 기록한다. Same-source zero delta는 정확히0이어야 한다.

이 검사는 denoiser·encoder·learned R·attention·optimizer를 생성하지 않는다.
통과해도 PBR 학습 admission이 아니다. Learned source2048 module, native sparse trainer 연결,
원본 complete PBR LoRA의 초기32개 동등성, 실제 native flow gradient, 요청 global24에서의
GPU 수/microbatch capacity, fixed rendering validation 및 matched control은 별도로 필요하다.
현재 SS의 native480·128×3 비교와 독립 training seed, 후속 cascade·final508/10metric 범위를 유지한다.
