# Shared-evidence fixed SS: 같은288-step 개발 평가

2026-10-24. `E_FIXED_RAW`의 세 번째 epoch 검증이 완료돼, 같은 DEV32·생성 noise1·training seed101의288-update 결과로 비교했다. 최종 DEV128×noise3 결과와 직접 비교하지 않는다.

| 구성 | Mean case IoU | Global micro IoU |
|---|---:|---:|
| V1 | 0.46897292 | 0.43361953 |
| Decoder learned-score Top-1 + retargeting | 0.46799147 | 0.43060914 |
| 기존 Full | 0.46649383 | 0.43150360 |
| 새 shared-evidence fixed | **0.47155197** | **0.43611780** |

새 fixed 구성은 이32개에서 V1보다 case macro +0.00257906, micro +0.00249827이다. 기존 Full보다도 두 집계가 높다. 첫96·192 update에서는 V1보다 낮았지만 기록된288-step 검증에서 평균 순위가 바뀌었다.

## 현재까지의 같은 조건 추이

| 구성 | Macro @96 | Macro @192 | Macro @288 |
|---|---:|---:|---:|
| V1 | 0.46380091 | 0.46744735 | 0.46897292 |
| Top-1 + retargeting | 0.46161876 | 0.46088226 | 0.46799147 |
| 기존 Full | 0.46097138 | 0.46459021 | 0.46649383 |
| Shared-evidence fixed | 0.46178885 | 0.46363435 | 0.47155197 |

추가 NN 실행 없이 이미 완료된10개 원래 control의 같은 step도 확인했다. 이번 비교에 포함된11개 모델 중 case macro는 E_FIXED_RAW가 가장 높았고, global micro는 raw semantic Top-1인 C_RAW0.43924526이 가장 높았다. C_RAW의 case macro는0.47073808이었다. 두 집계를 모두 보존하며 관측 후 순위를 유리하게 바꾸기 위해 primary 집계를 변경하지 않는다.

실제 첫288개 training trace의 asset ID·noisy latent·timestep·global conditioning이 동일했다.32개 평가의 source·GT voxel hash·generation seed도 같았고 초기 TP/FP/FN은 동일했다. Immutable control log는 기존 file4 receipt로 확인했다. 실행 중인 E model은 완료된 validation row와 학습 prefix snapshot을 별도로 고정했다.

## 해석과 남은 비교

현재 결과는 전달 경로 통일과 identity-V raw Top-K를 사용하는 조합을 계속 검증할 근거다. E_FIXED_RAW에는 learned Q/K attention이나 learned V/output projection이 없다. 기존 Full과의 차이는 여러 설계 요소를 포함하므로 이 비교만으로 “전달 경로 통일 하나의 효과” 또는 “학습된 attention의 효과”를 분리했다고 주장하지 않는다.

같은 raw evidence와 identity V를 유지하는 E_FIXED_RAW 대 E_LEARNED_RAW의 matched 비교가 별도로 예정돼 있다. 480-update 최종 DEV128×noise3, 원래 미완료 control, 독립 training seed, 실제 생성 geometry·source/target rendering 평가를 완료해야 한다. 중간32개 순위를 최종 model adoption으로 사용하지 않는다. 새 유의성 검정이나 추가 epoch를 이번 결과만으로 시작하지 않았다.

실제 증거는 file4 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/consistent-fixed-epoch288-matched-v197/`와 `consistent-fixed-epoch288-all-available-controls-v197/`에 있다. 전체11개 모델의0/96/192/288 실제 기록·두 집계와 입력 검증을 보존했다. [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-consistent-fixed-epoch288-all-controls-v197)에 scalar28개·표·추이 그림·artifact를 게시한 뒤 실제 remote finished 상태와 값을 확인했다.
