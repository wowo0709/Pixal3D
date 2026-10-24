# SS IoU 집계 기준과 실제 개선 폭 정정

2026-10-24. 동일한 실제 prediction/GT count에서 micro와 macro를 다시 계산했다. 새로운 학습·추론·전처리는 하지 않았다. 기존 원래 SS summary 코드의 primary는 **mean asset voxel IoU, averaged over three paired generation noises**이며 이를 유지한다. 새 micro readout은 집계 차이를 설명하는 descriptive 진단이다.

## 0.43대와 0.47대를 구조 개선으로 연결하면 안 된다

같은 V1 checkpoint의 같은 DEV128개·93 source cluster·생성 noise3개가 다음 두 값을 가진다.

- Case macro IoU: **0.46735504**. 각 deformation case의 IoU를 동등하게 평균한다.
- Noise별 global micro IoU의 평균: **0.43197940**. 각 noise의128개 case에서 TP와 union을 모두 합친 IoU를 구한 뒤 noise3개를 평균한다.

이 둘의 차이0.03537564는 학습이나 모듈의 이득이 아니다. 같은 checkpoint와 같은 count에 대한 집계 차이다. 과거 학습 로그의0.43대와 최종 표의0.47대를 직접 이어서 모델 성능이4%p 올랐다고 해석하는 설명은 수정해야 한다. Panel32/128, 생성 noise 수와 epoch 차이도 따로 유지해야 한다.

## 같은 조건·집계로 비교한 실제 endpoint

전부 seed101,480 update, 동일 DEV128×생성 noise3다. Micro는 위의 noise별 global micro 평균이다.

| 구조 | Case macro IoU | Global micro IoU |
|---|---:|---:|
| Vanilla V1 | 0.46735504 | 0.43197940 |
| Spatial + retargeting | 0.47124225 | 0.43399700 |
| Raw semantic Top-1 + retargeting | 0.47235972 | 0.43691318 |
| Decoder + cosine Top-1 + retargeting | **0.47300814** | 0.43814652 |
| Decoder + learned-score Top-1 + retargeting | 0.47260025 | 0.43704787 |
| Decoder Top-1, retargeting 제거 | 0.46867128 | 0.43354423 |
| Fixed Top-K | 0.47274517 | **0.43827915** |
| Full | 0.47207818 | 0.43704567 |
| Full에서 additive attention prior 제거 | 0.47244621 | 0.43761540 |

현재 최고 macro 구성의 V1 대비 실제 차이는 **macro +0.00565310, micro +0.00616712**다. 기존 Full−V1도 macro +0.00472314, micro +0.00506627로 양의 point difference다. Retargeting 추가의 conditional 차이도 양의 방향이지만, 기존 모듈 family 보정 CI의 한계와 독립 training seed1개라는 범위는 그대로다. 이번 집계 audit이 유의성이나 필요성의 새 증거를 만들지는 않는다.

소수 모델의 순위는 집계에 따라 달라진다. Macro에서는 decoder-cosine Top-1, micro에서는 fixed Top-K가 가장 높지만 차이가 작다. 기대한 모듈이 이기는 집계를 사후 선택하지 않는다. Prespecified macro와 기존 source-cluster 통계를 primary로 유지한다. 모든 noise의 count를 한꺼번에 pool한 또 다른 micro도 artifact에 별도로 기록했으며 noise별 micro 평균과 혼용하지 않는다.

## 새 consistent-evidence fixed arm의 첫 epoch

아래는 동일 DEV32·생성 noise1,96 update다. 위 DEV128×3 endpoint와 직접 비교하지 않는다. 네 arm의 처음96개 training trace에서 실제 asset ID/x_t/t/global hash가 정확히 같고 평가 source·GT voxel representation·seed도 일치했다.

| 구조 | Logged micro IoU | Case macro IoU |
|---|---:|---:|
| V1 | 0.43397886 | 0.46380091 |
| Decoder Top-1 + retargeting | 0.43099057 | 0.46161876 |
| 기존 Full | 0.43061912 | 0.46097138 |
| E_FIXED_RAW | 0.43167609 | 0.46178885 |

새 fixed arm은 초기 micro0.42671547/macro0.45835792에서 개선됐으나 첫 epoch에 V1을 넘지 않았다. 기존 Full보다 소폭 높다는 중간 point difference를 최종 개선으로 주장하지 않는다.480 update와 prespecified DEV128×3, 이후 learned-attention matched arm이 필요하다.

실제 증거는 file4의 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/SS-IoU-aggregation-and-epoch96-v193/`에 있다. 같은 실제 final128×3 count, epoch0/96의32개 case, 동일 training prefix96와 audit·canonical closure를 보관했다. 원본 prediction, checkpoint, 기존 metric 값과 집계 코드는 변경하지 않았다.
