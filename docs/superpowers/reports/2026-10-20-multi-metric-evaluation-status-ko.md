# 현재 평가 범위와 전체 편집 품질 평가 보강

Voxel IoU는 SS64의 점유 구조를 빠르게 비교하는 지표다. 색·재질 보존, 상세 표면, 최종 rendering 품질을 대신하지 못한다. 앞선 사용자의 빠른 SS/PBR 검증 요청에 맞춰 SS에서는 voxel 지표, PBR에서는 rendering 지표를 실행했지만, SS IoU만으로 전체 모듈의 효용이나 최종 모델의 우열을 판정해서는 안 된다. 새 attention 후보도 SS IoU가 비슷하다는 이유만으로 PBR 효과까지 없다고 결론 내리지 않는다.

## 실제 계산된 PBR 지표

기존 저장 RGB 모델의 동일 step762, 동일32개, 동일 noise·GT target Shape·입력 anchor camera08 평가다. 새 task-ray attention의 본학습 결과가 아니다. 원본은 canonical `semantic-topk-search-v1/pbr-matched32-v1/{evaluations,six-model-summary-v2}`다.

| 기존 모델 | Foreground PSNR↑ | Crop SSIM↑ | Target-crop LPIPS↓ |
|---|---:|---:|---:|
| Zero-shot:undeformed RGB4 |15.06174|.914208|.128776|
| Zero-shot:edited RGB1+undeformed RGB3 |15.29189|.921306|.122529|
| Vanilla V1 |15.49508|.920779|.121535|
| B1 retargeting |15.49518|.920714|.121453|
| B2 learned-score Top1 |15.56750|.921666|.120846|
| B3 score TopK 가중합 |15.56732|.921568|.120803|

B2/B3와 V1의 차이는 작고 B3가 B2보다 모든 지표에서 우월하지 않다. 이 표에는 GT geometry를 고정한 texture-stage 결과만 있다. 해당 alpha IoU가 약1이라는 사실은 geometry 복원에 성공했다는 증거가 아니다. Source-cluster 비교와 replay 한계는 원래 summary를 유지한다. 개발32개·생성 noise1·training seed1의 결과를 최종 test 결과로 사용하지 않는다.

## 전체 평가에 적용할 구분

| 평가 목적 | 지표 | 기준 및 범위 |
|---|---|---|
| Source preservation |LPIPS↓,DreamSim↓,CLIP similarity↑|원본 identity/material을 유지한 paired textured M1 target render와 비교|
| Target alignment:영상 |PSNR↑,SSIM↑,2D M-IoU↑|동일 target camera·renderer·background·조명에서 GT M1 RGB/mask와 비교|
| Target alignment:geometry |CD↓,surface F-Score↑|예측 최종 mesh와 GT target mesh;동일 source-defined 정규화;출력별 ICP·recenter·rescale 없음|
| Global naturalness |FID↓,KID↓|생성/GT target render 분포;표본 수와 source 상관을 명시하고 작은 개발 패널의 주 선택 지표로 쓰지 않음|

Source preservation의 기준을 undeformed M0 이미지로 단순 치환하지 않는다. 의도한 deformation을 pixel 오류로 벌점화하기 때문이다. 이 기준은 원래 `anchor-hybrid-prestudy/docs/superpowers/reports_cvpr/README.md`의 Metrics 절에도 명시되어 있다. GT target render는 평가 label이며 추가 모델 입력이 아니다. 기존 PBR LPIPS는 GT latent-decoded target crop과 비교한 제한된 texture-stage 진단이며, 그 수치를 전체 cascade의 논문용 LPIPS로 재명명하지 않는다.

Mesh CD/F-Score에서는 기존 소규모 실험과 맞춰 surface sample20,000개, symmetric Chamfer-L1(양방향 nearest distance 평균의 평균), F-Score threshold.02를 사용할 계획이다. 정규화·sampling seed·거리 단위와 성공/실패 denominator를 실행 전에 고정한다. 기존 voxel F1을 surface F-Score로 사용하지 않는다. 카메라별 지표는 deformation 내에서 먼저 평균한 뒤 deformation-weighted/source-UID-weighted 평균을 함께 보고하고, paired CI는 source geometry cluster로 resample한다. 입력 camera 평가와 held-out camera 평가는 구분한다.

## 준비 상태와 남은 구현

실제 source code를 확인한 결과 PBR PSNR·SSIM·LPIPS는 구현·측정되어 있다. DreamSim·CLIP scoring, 최종 cascade mesh의 CD/F-Score, 해당 mesh의2D M-IoU는 현재 실험 경로에 아직 연결되지 않았다. 현재 환경에는 LPIPS와 open_clip이 있지만 DreamSim package와 고정한 DreamSim/CLIP 평가 weight는 아직 확인하지 못했다. 준비되지 않은 지표를 완료 또는0으로 표기하지 않는다.

사용자가 ACL을 변경한 Shape5124210/Shape102416850은 두 config·원본 checkpoint를 확보하고 SHA256 및 실제 strict CPU LoRA reconstruction을 검증했다. 이를 고정한 full-cascade 생성 경로를 구현하고, 실제 mesh 및 raw multi-view RGB/alpha를 보관해 지표를 계산해야 한다. GT Shape로 고정한 PBR stage 평가를 이 full-cascade 평가로 대체했다고 주장하지 않는다. Zero-shot4M0,zero-shot1M1+3M0,V1,matchedR,matching 후보를 같은 입력·noise·카메라에서 비교하며 사용자가 제외한 target-only arm은 추가하지 않는다.

새 task-ray attention의 자체 초기 생성·DDP2 capacity·R96→480 학습 연결도 남아 있다. 기존 정상 학습과 평가는 중단하지 않으며, 평가 범위 보강을 SS 후보 학습 대신 수행 완료한 것으로 처리하지 않는다. 전체 목표는 진행 중이다.
