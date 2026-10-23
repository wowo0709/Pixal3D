# 핵심 모듈 대조군 구현·실행 연결과 첫 epoch 관측

기준 시각: 2026-10-23 14:30 UTC. 브랜치: `wowo0709/feature-retargeting`.

이 기록은 `2026-10-23-unified-module-pipeline-core-experiments-ko.md`의 후속이다. 핵심 arm 수는 12개로 유지했다. 새로운 탐색 arm이나 지엽 sweep을 추가하지 않았다. 기존 지엽 대기 작업 36개는 계속 보류한다.

## 1. 이번에 완료한 준비

남은 네 설계 대조군의 구현과 학습 연결을 완료했다. 각 arm의 실제 GPU 검증·학습 결과는 아직 없다.

| Arm | 설계 차이 | 학습 가능한 source tensor 수 |
|---|---|---:|
| `FULL_QK_ONLY` | attention에 더하는 matching prior만 제거. 후보, D1 문맥, Top-1 carrier는 유지 | 19 |
| `FULL_VOXEL_Q` | 같은 ray 후보·문맥을 쓰되 voxel별 QKV를 계산하고, raw aggregate와 transformed feature를 ray 단위로 모은 후 동일한 R에 입력 | 19 |
| `FULL_MEAN_MATCH` | distinct member들의 posterior를 섞는 대신 같은 관측 member descriptor를 먼저 평균해 proposal/decoder 수행. attention query와 전달 경로는 유지 | 19 |
| `C_DECODER_NO_R` | 실제 Top-1 feature/use/fusion을 유지하고 R module·parameter를 물리적으로 제거 | 5 |

QK-only는 scalar matcher를 없애는 arm이 아니다. attention 내부에 prior를 넣는 효과를 분리한다. 기존 사용자 정의의 no-score/Top-1 arm은 correspondence feature 하나를 실제 선택하는 `C_DECODER`로 유지한다.

voxel-query arm은 checkpoint 안에서 point별 후보 feature를 읽어 memory를 제한한다. 두 readout 출력을 모두 평균한 뒤 기존 R을 적용하므로, query 단위 비교에서 R의 위치까지 바뀌지 않는다. 전체 dense point×K activation을 외부 학습 그래프에 저장하지 않는다. 이 구현의 실제 microbatch 96 용량은 GPU capacity 검사로 확인해야 한다.

네 arm은 공통 초기 source parameter를 실제 reference constructor와 비교한다. full 계열은 같은 historical utility를 로드하며, matcher는 frozen/eval 상태를 유지한다. 매 forward에서 R-only 전달 또는 no-R residual 부재를 검사한다. 실제 학습 전 native fields, velocity/loss, 전체 학습 asset 순서, noise/timestep/global trace를 확인하는 gate를 연결했다. 각 arm은 자체 GPU 초기 검사와 full-batch 3-update 검사, 기준 arm과의 CPU 비교를 통과해야 480-step 학습을 시작한다.

관련 configuration·물리적 parameter 제거·feature pooling 수학·checkpoint identity 검사 30개와, 12-model 통계/짝짓기 검사 11개가 통과했다. CPU에서 neural forward나 학습을 실행하지 않았다. 이 검사는 실제 GPU memory/backward 검증을 대신하지 않는다.

## 2. 최종 SS 평가 연결

이미 연결했던 projected 두 arm 외에 RAW/spatial/D1-cosine 세 arm과 위 네 arm의 최종 평가도 연결했다.

- 각 모델의 실제 480-step 종료와 child session 종료를 확인한 다음 자체 config·LoRA/source checkpoint pair를 봉인한다.
- 각 모델을 동일한 128개 개발 asset, 같은 생성 noise 3개로 평가한다.
- 같은 forward에서 실제 SS latent·occupancy를 함께 보관해 후속 Shape/PBR cascade가 사용할 수 있게 한다. 별도의 archive32 SS 재생성은 추가하지 않았다.
- 새 결과와 기존 V1, Top-1+R, full 후보 결과를 모아 12개 모델·10개 모듈 비교·2개 전체 비교를 계산한다.
- asset·source cluster·GT·seed·checkpoint step·source tensor 수가 맞지 않으면 통계 생성을 거부한다. 기존 세 모델의 집계값도 재현됐는지 확인한다.

최종 통계는 asset별 생성 noise 평균 후 source-cluster bootstrap으로 계산한다. module family 10개와 pipeline family 2개를 유지한다. generation noise 3개를 독립 학습 seed 3개로 취급하지 않는다. 통계와 비교 이미지를 W&B에 기록하도록 연결했다.

## 3. 첫 epoch 결과

새 projected 두 모델이 epoch 1/step 96을 마쳤다. 아래는 같은 개발 asset 32개, 같은 GT·생성 seed에서 얻은 값이다. 학습 전 다섯 모델의 asset별 검증 record와 집계값은 정확히 같다.

| Arm | 학습 전 mean asset IoU | Epoch 1 mean asset IoU | Epoch 1 pooled-count IoU |
|---|---:|---:|---:|
| V1 | 0.458358 | **0.463801** | **0.433979** |
| Top-1+R (`C_DECODER`) | 0.458358 | 0.461619 | 0.430991 |
| Full 후보 (`R_raw_topk`) | 0.458358 | 0.460971 | 0.430619 |
| Top-1+V (`C_TOP1_V`) | 0.458358 | 0.461015 | 0.430606 |
| 고정 Top-K (`C_FIXED_TOPK`) | 0.458358 | 0.460818 | 0.430996 |

주 지표는 mean asset IoU다. 고정 Top-K는 Top-1+V보다 pooled IoU는 높지만 mean IoU는 낮다. 따라서 이전 진행 설명의 “Top-K 소폭 우세”는 pooled 값에 한정되며, 주 지표의 우세로 해석하면 안 된다. 현재 첫 epoch에서는 V1이 가장 높고, Top-K/attention의 추가 이득을 입증하지 못했다.

이 수치는 최종 128개×3-noise 평가와 다른 평가 패널이다. 과거 `.44` pooled와 `.47` mean 차이, 또는 32개와 128개 패널 차이를 모듈이 만든 성능 상승으로 설명하지 않는다. 첫 epoch만으로 최종 열위나 추가 학습의 성공을 단정하지 않는다.

## 4. 남은 필수 작업

현재 GPU 6·7의 projected SS 학습과 GPU 0–3의 PBR feature 학습은 유지한다. 일곱 개 후속 대조군은 GPU 검증을 통과한 뒤 학습·최종 평가로 이어진다. 실행 전 연결 상태를 실제 결과로 표시하지 않는다.

다음 우선 작업은 새 SS 결과를 공통 trained V1 PBR에 연결하는 생성 geometry 기반 full-cascade 평가다. 기존의 legacy14·native8 전체 완료를 요구하는 경로에 묶이면 보류한 지엽 실험이 다시 필요해지므로, 핵심 모델 결과만 요구하는 새 연결이 필요하다. SS 차이를 비교할 때 Shape512/1024와 PBR은 고정한다. PBR 차이를 비교할 때 생성 geometry와 native Shape guides를 고정한다.

후속 평가는 source-view LPIPS/DreamSim/CLIP, target-view PSNR/SSIM/M-IoU와 CD/F-Score를 포함한다. source-view의 참조가 변형 M1 geometry와 원래 material을 사용한 렌더인지, 원래 M0 RGB인지 명시해야 한다. 기존 `cascade_view_roles`는 전자를 사용한다. 이를 설명 없이 M0와의 직접 보존 지표로 바꾸지 않는다. FID/KID와 TEST 508개는 구조·checkpoint 동결 이후 평가한다.

아직 새 full-cascade 향상, 독립 학습 seed 202 확인, 최종 TEST, 최종 모델 채택은 완료되지 않았다. 모듈별 증거와 실제 완성 파이프라인 품질을 확인할 때까지 전체 실험 목표를 유지한다.
