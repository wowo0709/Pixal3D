# PBR 3-epoch 추이와 완료된 SS attention gradient 검사

2026-10-24 05:27 UTC. 실행 중이던 native PBR 모델의 실제 1,140-step 검증이 끝난 뒤 기존 세 모델의 같은 step과 비교했다. 별도로 완료된 Full SS의 원래 gradient log를 확인했다. 학습·모델 source와 기존 checkpoint는 수정하지 않았다.

## PBR: Top-K 조합의 평균 순위가 바뀌었다

모두 같은 seed101·네 rank·global batch24·rank당 micro6·accumulation1이며, 아래는 같은 GT shape와 target-camera8을 사용하는 고정4개 sample의 stage 진단이다. Source preservation이나 생성 geometry의 성능은 측정하지 않는다. Foreground PSNR은 foreground-weighted MSE를 pool한 다음 log를 취한 값이며 asset별 dB 평균과 다르다.

| 구성 | Foreground PSNR ↑ | Crop LPIPS ↓ | Crop SSIM ↑ |
|---|---:|---:|---:|
| V1 | 15.611604 | 0.08823027 | 0.95273672 |
| Top-1 + retargeting | 16.045404 | 0.07874926 | 0.95630522 |
| Top-1 + feature head | 16.032318 | 0.07887620 | 0.95640717 |
| Raw Top-K + feature head | **16.107360** | **0.07812583** | **0.95681232** |

Raw Top-K 조합은 Top-1+retargeting보다 foreground PSNR +0.06196dB, crop LPIPS −0.00062342, crop SSIM +0.00050710이다. 760 step에서는 Raw Top-K의 crop LPIPS0.08665431이 Top-1+retargeting0.08610232보다 나빴으나, 1,140 step에서는 평균 순위가 바뀌었다. 같은 step의 학습·검증 조건으로 확인한 새 결과다.

| 구성 | Crop LPIPS @380 | @760 | @1,140 |
|---|---:|---:|---:|
| V1 | 0.09242722 | 0.09007769 | 0.08823027 |
| Top-1 + retargeting | 0.09369044 | 0.08610232 | 0.07874926 |
| Top-1 + feature head | 0.09485977 | 0.08751500 | 0.07887620 |
| Raw Top-K + feature head | 0.09474101 | 0.08665431 | 0.07812583 |

**이 평균 개선을 일반적인 Top-K/attention 효과의 입증으로 해석하면 안 된다.** Top-1+retargeting 대비 sample별 crop LPIPS 차이는 −0.00459290, +0.00210109, +0.00011699, −0.00011888로 2개 개선·2개 악화다. 평균 차이는 첫 sample의 영향이 크다. Raw Top-K 대 Top-1+feature 비교에서도 3개 개선·1개 악화지만 그중 두 차이는 매우 작다. 독립 학습 seed도 하나다.

또한 Raw Top-K 대 Top-1+retargeting은 raw aggregation과 feature head 등 여러 차이를 포함한다. 이 비교 하나로 learned matching-aware attention만의 효과를 분리했다고 주장하지 않는다. 현재는 조합의 수렴 추이를 보는 결과다.

## 실제 정합 검사

네 모델의 네 rank별 첫1,140개 실제 학습 trace에서 asset ID·noisy latent·timestep·global conditioning row가 정확히 같았다. 각 step의 고정4개 평가에서도 RGB input·GT PBR latent·GT shape condition·camera·sample·source·noise identity가 같았다. 초기 prediction PBR latent hash도 네 모델에서 같았다. Optimizer, LoRA, 원래 multiview initializer 등 핵심 학습 설정은 동일했다.

완료된 세 모델의 원본 log는 기존 file4 SHA256/size receipt로 확인했다. 실행 중인 Raw 모델의 append log 전체를 닫힌 artifact라고 취급하지 않고 실제 선택한 validation row와 네 rank의 같은1,140-step prefix를 별도로 고정했다. 세 artifact 약4.2MB를 실제 file4로 복사한 뒤 remote SHA256와 size를 확인했다. 검증 후 원래 Raw 모델은 같은 PID/birth로 1,141 step 이후 학습을 재개했다.

## SS Full: attention으로 gradient는 전달됐다

완료된 Full SS의 실제 `source-native-gradients.jsonl`에는 step10부터480까지 매10 step, 총48개 기록이 있다. 네 readout branch는 모든48개 지점에서 finite positive aggregate gradient를 기록했다. Frozen matcher의 기록은 모두0이다.

| Branch | 비영 gradient 지점 | L1 중앙값 | 마지막 L1 |
|---|---:|---:|---:|
| Query | 48/48 | 0.08326090 | 0.12098950 |
| Key | 48/48 | 0.06151084 | 0.09511088 |
| Value | 48/48 | 0.14171548 | 0.27892336 |
| Output | 48/48 | 0.27823210 | 0.35078514 |
| Frozen matcher | 0/48 | 0 | 0 |

이 값은 각 layer parameter의 `abs(grad)` 합이며 parameter 수로 normalize한 값이 아니다. 서로 다른 layer나 모델의 gradient 크기를 그대로 비교해 learning rate를 결정하지 않는다. 이 검사로 attention branch의 완전한 gradient 단절을 실패 이유로 삼기는 어렵다. 좋은 correspondence posterior를 학습했는지, 각 parameter가 얼마나 바뀌었는지, attention이 유용한지, 추가 epoch가 도움이 되는지는 이 기록만으로 입증되지 않는다. 기록하지 않은 step의 gradient도 추정하지 않는다.

## 다음 실행과 채택 범위

현재 학습은 원래 계획대로 유지한다. Raw PBR1,900 update를 완료하고, 이미 연결한 동일한 생성 V1 geometry의 네 PBR 구성 DEV32 비교로 source·target 지표를 확인한다. 현재 decoder-cosine SS + Top-1 PBR의 joint DEV32도 실제 큐에 연결돼 있다. Raw PBR이 생성 geometry를 사용한 DEV32에서도 유력하면 해당 실제 SS/PBR 조합을 같은 joint protocol로 평가한 뒤 채택을 판단한다. 고정4개 stage 평균만으로 새 최종 조합을 고정하지 않는다.

SS의 fixed/learned shared-evidence와 원래 두 control, 독립 seed202 확인, 최종 freeze 이후 실제 TEST508 및 10개 metric은 계속 남아 있다. 추가 epoch나 새로운 architecture arm을 이번 결과만으로 추가하지 않았다.

실제 file4 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `PBR-epoch1140-four-arm-fixed4-matched-v195/`, `PBR-epoch1140-four-arm-WandB-v195/`, `completed-Full-attention-gradient-audit-v195/`에 있다. PBR scalar16개·표2개·artifact는 [W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/pbr-native-fixed4-epoch1140-matched-v195)에 게시한 뒤 실제 remote finished 상태와 값을 확인했다. 이번 변경은 문서와 외부 CPU readout뿐이며 새로운 NN forward나 학습 update를 실행하지 않았다.
