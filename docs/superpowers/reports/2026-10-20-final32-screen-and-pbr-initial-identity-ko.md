# SS final480의32 사례 결과와 PBR 초기 동등성

## 현재 결과

새 input-side attention은 고정32 사례에서 matched R 대조군보다 소폭 높지만, 추가 이득은 아직 확립되지 않았다. 이전 positive R보다도 낮다. 따라서 이 후보를 full model로 채택하지 않는다. 아래 수치는 모두32개 사례의 TP/FP/FN을 합친 IoU이며, 최종128×3 결과를 대신하지 않는다.

| 모델 | Total training update | 고정32 IoU |
|---|---:|---:|
| Zero-shot source4M0 | 0 | 0.39499655 |
| Zero-shot edited1M1 + source3M0 | 0 | 0.42671547 |
| 기존 vanilla V1 | 480 | 0.43483711 |
| 기존 positive Top1 + R | 480 | 0.45402236 |
| 새 matched R control | 480 | 0.44655414 |
| 새 input-side feature attention | 480 | 0.44819734 |

새 attention−matched R은 **+0.1643pp**, source-cluster paired95% CI는 **[-0.0705,+0.4486]pp**다. 새 attention−기존 positive R은 **-0.5825pp**, CI는 **[-1.2141,-0.0775]pp**다. 새 R control도 기존 positive R보다 -0.7468pp였다. 이 개발 패널에서는 약한 새 대조군만 이겼다는 이유로 성공이라 해석하면 안 된다.

두 새 모델은 old R step96의 NN·conditioner·Adam을 공통으로 이어받았고 post-constructor stage RNG도 같게 시작했다. 기존 continuous R480은 step96 이후의 noise/CFG stream이 같지 않다. Model/data/LoRA/LR/global batch 및 common Adam 복원 검사는 통과했다. 그렇더라도 RNG 차이가 성능 차이의 충분한 원인이라고 단정하지 않는다. 고정128×3 비교와 적절한 독립 training seed가 남아 있다.

`epoch480-matched-v3`의 CPU helper가 intermediate용 맥락 flag를 상속해 historical V1/R 비교에도 `not_equal_training_step_comparison=True`를 기록한 것은 잘못된 metadata다. V1/R은 실제 total480으로 예산이 같다. Zero-shot만 total0이다. Count와 CI는 바뀌지 않았으며 `final480-screen32-context-v1/audit.json`에 올바른 예산 및 training trajectory 구분을 별도로 기록했다. 이전 report의 per-asset mean 설명도 수정했다. 실제 metric은 처음부터 pooled count였다.

## PBR 시작 성능 하락 검사

Direct physical-DQ 모델에 기존 PBR V1 step762의 complete NN base+LoRA를 넣고 실제32개 generation·decode·render를 끝냈다. Native process는 exit0으로 종료했다.

- 모든32개 per-asset record가 기존 V1과 정확히 같았다. RGB, 실제 noise, GT PBR, GT Shape, source cluster 및 seed hash도 포함한다.
- Aggregate rendering metric이 모두 정확히 같았다. Foreground PSNR은15.4950775882dB, target-crop LPIPS는0.1215346221, SSIM은0.9207790308이다.
- Case PNG32개, aggregate panel, actual-input panel, 총34개 파일이 byte exact였다. GT와 예측 이미지 모두 포함한다.
- 추가 attention output projection과 DQ projection은0으로 초기화했고, RAW Omega 입력 채널은 실제로 존재한다. 입력값을0으로 채운 것이 아니다.

초기 생성 성능을 그대로 보존했다는 증거다. 학습 후 개선이나 full-cascade 성능의 증거는 아니다. GT target Shape를 고정하는 PBR stage 검사라는 범위도 유지한다.

## 이어지는 실험

R control과 input-side attention의 scientific 학습 및 checkpoint file4 closure는 끝났다. Physical-DQ와 direct feature retargeting 모두 실제 world2/global96/micro48/accumulation1 fullbatch 검사를 통과했고, 두 rank의376개 optimizer state와 새10개 tensor의 학습 및 DDP 일치를 확인했다. Capacity peak는 각각 약50.06/50.34GiB였고 두 receipt 모두 실제 file4와 일치했다. GPU0/1에서는 physical-DQ, GPU2/3에서는 direct retargeting 본 학습 프로세스가 시작됐다. Initial96의32 사례 validation과 actual optimizer update를 확인하기 전에는 본 학습의 warmup 완료나 첫 에폭 결과를 주장하지 않는다.

이후 두 실제 scientific run의 initial96 검증과 첫5 update까지 확인했다. 초기32개 per-asset record는 공통 R96과 정확히 같았고, 첫 실제 update97의 두 rank별 loss 및 모든 timestep-bin MSE도 R/F/DQ/direct 네 모델에서 각각 정확히 같았다. 이 검사는 추가 모듈 생성으로 공통 학습 RNG나 초기 condition이 바뀌지 않았는지 확인한다. 이후 actual DQ projection과 feature head에 비영 gradient가 관측됐고, steady update는 약18~19초, peak allocated는 약50~50.4GiB였다. 이는 초기 구간 수치이며 향후 모든 batch의 OOM 보장은 아니다.

두 새 run의 W&B 원격 metric을 읽었고, 실제 GT/prediction 시각화 PNG를 각각 다운로드해 server summary의 SHA256과 대조했다. 두 초기 panel 모두21,873 bytes, SHA256 `09984f01504484ea6dd983b32082fa7a150bcbdc6065471a6c8d85a4fd722981`로 일치했다. SS panel은 voxel의2D max projection이다. 이 PNG와 TP/FP/FN 일치만으로 전체3D predicted voxel mask의 bitwise equality를 주장하지 않는다.

최종 비교에는 기존 positive R, vanilla V1, 두 zero-shot, matched R 및 각 새 후보를 포함한다. Saved-weight value/score/decoder/DQ 경로 진단을 통해 feature 전달 여부와 품질 기여를 구분한다. PBR은 initial32 동등성 조건을 충족했고, SS 주요 결과 이후 실제 paired capacity에 따라 동일762→1905 예산의 두 arm을 연결한 상태다.

Canonical 증거 root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`이다.

- `epoch480-matched-v3`: matched final32 결과와 actual file4 receipt.
- `final480-screen32-context-v1`: strong R 비교 및 budget/aggregation 설명 수정 기록.
- `direct-retargeting-v1/PBR-joint-continuation-v1/initial32-parity-v1`:32개 record·34개 PNG exact 및 actual file4 receipt.
- `deformation-query-v1/capacity-v1`: 실제 DDP2/micro48 검사 및 actual file4 receipt.
- `direct-retargeting-v1/capacity-v1`: direct 경로의 별도 DDP2/micro48 검사 및 actual file4 receipt.
- `physical-query-startup-v1`: 두 actual main의 초기96 검증, 두 rank의 첫5 update·첫 loss/bin 동등성 및 W&B PNG readback과 actual file4 receipt.

성능 개선·full model 채택·전체 실험 완료는 아직 선언하지 않는다.
