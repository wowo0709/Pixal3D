# Ray 학습 시작 검사와 기존 비교군 epoch3

2026-10-19 UTC. 새 ray-group5개 본 학습은 GPU0~3 전용 큐에서 대기 중이다. 기존 voxel-query/softprior 학습과 scheduler/storage PID를 직접 확인해 정상 실행을 확인했다. 새 ray 모델의 성능 결과는 아직 없다.

## 시작 직후 확인하는 항목

각 ray arm에 CPU-only startup audit을 연결한다. 실제 main의 rank0/1 첫3 update, 실제2-update DDP benchmark,초기 common32 예측을 저장해 검사한다. Global96/DDP2/micro48/acc1,finite loss/compute/memory,실제 native-conditioning zero-start,현재13개 numerical component SHA256,준비된 model/data/trainer/validation recipe 동등성을 확인한다. 초기32개별 GT hash·generation noise·TP/FP/FN·4개 voxel metric은 V1 및 기존 Top1-R의 학습 전 결과와 정확히 같아야 한다. 이 검사는 학습 완료나 향후 OOM 부재를 보장하지 않는다.

신규 verifier 회귀 검사2개를 통과했다. Aggregate IoU가 같더라도 GT/noise/count/order 변경을 거부하고,rank 누락·accumulation 변경·비정상 메모리/시간 값을 거부한다. 추가로 실제 기존V/P의 initial32와 두 rank 첫3 update 자료에도 검사를 적용해 통과했다. 새 ray run 자료가 아닌 historical-input 검증이라는 범위를 구분한다.

## 기존 arm의 닫힌 epoch3 결과

모두 같은32개/동일 noise0/raw288이다. 각각의 epoch audit·W&B4 numeric/PNG readback·file4 SHA closure가 완료된 결과다. Source-cluster95% CI이며 새 ray-group 성능이나 최종128×3 결과가 아니다.

| 기존 arm | IoU | vs V1, pp | vs Top1-R, pp | Top1-R 대비95% CI, pp |
| --- | ---: | ---: | ---: | --- |
| 개별 voxel query | 0.444113 | +1.224 | +0.233 | [+0.057, +0.431] |
| anchor-patch query + softprior T0.05 | 0.440996 | +0.912 | -0.079 | [-0.266, +0.138] |

개별 voxel query는 이 중간32개에서 Top1-R보다 높지만,V1 대비 CI는0을 포함한다. Softprior arm은 Top1-R보다 높은 효과를 확인하지 못했다. 어떤 결과도 최신 지시의 camera-ray group query 실험으로 재분류하지 않는다. 사전에 정한 raw480/dev128×3와 training-seed 재현 전에는 모델을 채택하지 않는다. 기존 query의 자동 seed202 확장은 계속 보류/retire 상태다.

동일 common32에서 zero-shot source-only는0.3949965510,edited-RGB는0.4267154667이다. 각각4개M0 RGB와1개M1+3개M0 RGB이며,둘 다 같은 desired M1 GT로 평가한다.

## Query 입력의 추가 점검

실제 common32의 camera/edited-RGBA/16cubed native grid만 사용한 geometry 검사에서,target frustum 밖이면서 clamped target patch를 통해 observed로 잘못 포함된 voxel은0개였다. 이 검사는32개에 한정한 입력 검사이며 전체9149개나 GT geometry의 품질 검사로 확장해서 주장하지 않는다. Depth/mesh/GT correspondence는 query 입력에 쓰지 않았다.

기존 source/학습 numerical 파일은 바꾸지 않았다. Startup audit은 GPU를 사용하거나 optimizer update를 수행하지 않는다.
