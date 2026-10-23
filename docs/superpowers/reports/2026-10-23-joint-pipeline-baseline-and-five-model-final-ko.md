SS 다섯 모델 최종 DEV128×3 결과와 실제 joint baseline 평가 연결

다섯 모델의 native480 checkpoint를 동일한 validation 128개·source cluster 93개·서로 다른 generation noise 세 개로 평가했다. 학습 seed는 모두 101 한 개이므로 generation noise 세 개를 독립 학습 세 번으로 해석하지 않는다. 기존 모델 세 개의 통계가 실제 최종 평가에서 재현됐고, 두 projected 비교군의 평가도 모두 완료됐다.

| 모델 | Mean-asset IoU, 3-noise 평균 | Native pooled IoU, 3-noise 평균 |
|---|---:|---:|
| Vanilla V1 | 0.4673550 | 0.4319794 |
| Top1+R | 0.4726002 | 0.4370479 |
| Top1 + learned V | 0.4722178 | 0.4376690 |
| Fixed Top-K + learned V | 0.4727452 | 0.4382791 |
| Full learned attention + R | 0.4720782 | 0.4370457 |

Primary statistic은 asset별 IoU를 noise 세 개에 대해 평균한 뒤 asset 평균을 구한 값이다. Native pooled statistic은 큰 occupancy 집합의 voxel 수가 더 많이 반영되는 별도 지표다. 같은 V1이 두 방식에서 각각 0.46736과 0.43198이므로, 0.43대에서 0.47대로 숫자가 달라진 것을 전부 모델 개선으로 설명하면 안 된다. 다른 패널·noise·과거 데이터 정합 버전의 수치도 직접 섞지 않는다.

현재 finite protocol의 두 pipeline 비교와 열 module 비교 family 크기를 유지한 paired source-cluster bootstrap 4000회 결과는 다음과 같다. 이 데이터는 반복적으로 확인하는 DEV이므로 독립 TEST의 확증 결과로 해석하지 않는다.

| 비교 | Mean-asset IoU 차이 | Family를 반영한 95% 구간 |
|---|---:|---:|
| Full − V1 | +0.0047231 | [+0.0002880, +0.0093939] |
| Full − Top1+R | −0.0005221 | [−0.0033393, +0.0025674] |
| Top1-V − Top1+R | −0.0003825 | [−0.0041348, +0.0034789] |
| Fixed Top-K − Top1-V | +0.0005274 | [−0.0009548, +0.0019376] |
| Full − Fixed Top-K | −0.0006670 | [−0.0022991, +0.0007773] |

따라서 이 DEV 프로토콜에서는 Full의 V1 대비 상승을 관측했지만, 강한 Top1+R보다 더 좋다고 결론 내릴 수 없다. Fixed Top-K의 평균이 가장 높다는 것만으로 최종 모델을 고르지 않는다. V transform·Top-K 확장·learned QK attention 각각의 추가 이득을 SS IoU에서 입증하지 못했다. Semantic correspondence·matching decoder·scalar score·ray query unit·member proposal·retargeting 제거 비교는 남은 핵심 arm과 실제 rendering 지표를 통해 확인해야 한다.

실제 통계는 `projected-controls-v163/final-evaluation-v164/five-model-summary-family10-v2/audit.json`의 파일서버4 receipt에 보존됐다. W&B에도 [같은 다섯 모델 최종 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-unified-projected-five-model128x3-family10-s101-v165)가 기록됐다.

기존 SS isolation은 각 SS 모델의 예측을 공통 V1 PBR로 전달하고, PBR isolation은 공통 V1 geometry에서 PBR 네 모델을 비교한다. 이 둘만으로 SS Top1+R과 PBR Top1+R을 함께 사용한 결과를 추론할 수는 없다. 최종 pipeline 비교의 강한 baseline을 실제 측정하기 위해 이 조합 평가를 추가로 연결했다. 새 학습 arm은 없다.

추가 작업은 Top1+R SS의 기존 예측 geometry·native guide 32개에서 Top1+R PBR 텍스처를 생성하는 32개 사례와 coverage/official metric/summary 세 작업, 총 35개다. V1 SS+V1 PBR의 32개 seed-reset replay는 기존 v174 작업을 재사용한다. 합쳐서 실제 두 조합의 64개 사례, target8/source9·10의 192개 영상에 PSNR·SSIM·M-IoU·LPIPS·DreamSim·CLIP·CD·F-Score를 집계한다. FID/KID는 최종 TEST508 평가에 남긴다.

모든 조합의 텍스처는 seed 20261020으로 같은 원래 sampler를 다시 시작한다. 원래 cascade의 Shape sampling 후 RNG로 만든 V1 텍스처와 seed-reset retargeting 텍스처를 비교하지 않는다. 같은 geometry support에서는 실제 texture noise byte identity를 확인한다. 서로 다른 SS가 만든 support에서는 noise tensor 크기도 달라질 수 있어 byte identity를 주장하지 않고, 같은 seed-reset sampling 규칙을 적용한다.

새 `joint_pipeline_summary`는 실제 SS bundle의 NN/source/config binding, raw480 SS occupancy archive, 원래 Shape5124210·Shape102416850의 config/weight, 실제 PBR1900 NN/source pair, 세 owned child의 성공·drain, 생성 geometry/native guide 보존, image/GT/mask/camera/crop/official-score 정합을 검사한다. 최종 후보가 기존 두 baseline과 같으면 모델을 복제하거나 새로운 후보인 것처럼 꾸미지 않는다. 독립 후보라면 그 정확한 조합도 실제로 측정해야 한다.

새 통계 contract의 26개 테스트와 관련 기존 검증을 합쳐 84개 CPU 테스트가 통과했다. 실제 완료된 V1/Top1+R SS archive와 두 PBR의 네 rank 완료 기록, 원래 Shape weight의 현재 byte hash를 확인했다. 기존 두 SS 모델의 첫 실제 geometry 사례도 새 interface에 맞는 것을 확인했다. 이후 전체 기존 descriptor를 그대로 유지하고 새 35개만 append했으며, dispatcher가 전부 pending으로 인식했다. 새 GPU 작업 모두 v172 resource permit을 요구하므로 raw PBR의 GPU 네 장 확보를 방해하지 않는다.

Canonical 경로는 기존 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/joint-retarget-SS-PBR-native32-v176`이다. 실제 파일서버4 SHA256/size 검증은 prelaunch 134개 파일, queue 계획 2개, 적용 기록 1개, 연결 후 실제 입력 compatibility 검증 1개에 완료했다. GPU replay 결과와 32개 전체의 joint 품질 개선은 아직 관측하지 않았다.

현재 GPU0–3의 C_RAW/C_SPATIAL/C_D1_COS/C_DECODER_NO_R와 GPU6·7의 FULL_QK_ONLY/FULL_VOXEL_Q가 계속 진행 중이다. 이 핵심 비교와 실제 stage/joint cascade 결과를 기다리며, 임의의 추가 sweep이나 성능이 확인되지 않은 모듈을 필수 요소로 고정하지 않는다. 남은 요구는 선택된 정확한 조합의 DEV 측정, 독립 seed 확인, actual validation36/tail4 parity, 현재 finite12에 맞는 최종 model/population/metric freeze 구현 및 TEST508의 10개 metric·두 zero-shot 대조군 평가다. 최종 모델은 아직 확정하지 않았다.
