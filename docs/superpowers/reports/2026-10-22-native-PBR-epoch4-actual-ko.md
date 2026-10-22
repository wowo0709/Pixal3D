# PBR vanilla 네 번째 epoch 실제 검증

PBR vanilla가1520/1900 update와 네 번째 epoch 저장·고정4개 렌더 검증을
완료하고 학습을 재개했다. 물리GPU0~3/world4/global24/micro6/accumulation1을
유지했다. Native parent238065/birth1792670094.76의 관찰 작업은 원래 프로세스와
plan/mode/output을 매번 확인한 뒤 실제 validation row와 checkpoint file4 receipt가
모두 존재할 때만 snapshot을 만들었고, 실제 exit0으로 종료했다.

## 경향과 집계 방식

`evaluate_pbr_streaming.aggregate_metrics`는 full-frame PSNR을 네 이미지의
MSE를 평균한 뒤 계산한다. 자산별 PSNR(dB)의 산술평균과 다르다. 기존에 보고한
26.xx 수치는 이 **pooled-MSE PSNR**이며, 아래에 두 집계를 함께 표시했다.
LPIPS와 crop LPIPS는 네 자산의 산술평균이다. 학습·metric 구현이나 원래 수치는
수정하지 않았다. 네 이미지의 해상도는 같으며 target camera8의 GT Shape 조건이다.

| Update | Pooled-MSE PSNR ↑ | Mean asset PSNR(dB) ↑ | Full-frame LPIPS ↓ | Foreground crop LPIPS ↓ |
|---|---:|---:|---:|---:|
| 0 | 26.0573 | 29.4633 | .010105 | .095126 |
| 380 | 26.3444 | 29.6719 | .009688 | .092427 |
| 760 | 26.6023 | 29.8789 | .009273 | .090078 |
| 1140 | 26.7225 | 29.9656 | .008977 | .088230 |
| 1520 | 26.7444 | 29.9652 | .008884 | .087684 |

1140→1520의 pooled PSNR은+.02190dB, mean-asset PSNR은−.00040dB로
사실상 같다. LPIPS는−.0000934, crop LPIPS는−.0005467이다. Per-asset PSNR의
해당 epoch 변화는+.0581/−.0560/−.0014/−.0023dB로, 모든 자산이 계속 좋아진
것은 아니다. PSNR 두 집계의 변화 방향이 다를 수 있는 이유는 MSE와 log의
집계 순서 차이다.

0→1520 per-asset PSNR 차이는+.3665/+1.6515/−.0009/−.0098dB다.
초기 대비 개선은 여전히 주로 두 자산에서 나타난다. 이 네 개 panel만으로 전체
데이터의 포화·과적합·early-stop 필요성이나 source module의 효과를 판정하지 않는다.
사전 계획의 다섯 epoch/1900 update를 유지한다.

## 실제 보관과 실행 상태

0/380/760/1140/1520의 모든4개 case에서 sample/source cluster·conditioning hash·
RGB·noise·GT PBR latent·GT Shape condition·camera가 동일함을 확인했다.
GT/Pred/Diff PNG25개를 포함한 snapshot32개4,273,619 bytes를 actual file4
SHA/size와 대조했다. 실제1520 NN/Adam/config/descriptor archive receipt도 있다.
관찰 작업 자체의 완료 근거5개2,813 bytes도 실제 file4에 보관했다.

최근10 update compute 중앙값은 네 rank 모두 약13.25초다. Rank0 peak
allocated43.6595GiB/reserved46.6152GiB, 나머지 rank peak allocated43.42GiB 이하,
reserved48.4609GiB다. 저장·검증 이후 실제1524 update까지 학습 재개를 확인했다.
재시작이나 batch 변경은 하지 않았다.

Canonical root는 PBR-native-science-runtime-v1의
`vanilla-epoch1520-actual-evidence-v1/`과 `epoch1520-observer-v1/`이다.

이 결과는 GT-shape-conditioned target stage 진단이다. Source preservation이나
생성 geometry/full-cascade 성능이 아니며 alpha IoU를 geometry 개선으로 해석하지
않는다. 마지막1900의 실제 terminal·네 rank 완료·canonical checkpoint를 확인한 뒤
native SS480 소비 경로 진단과 나머지 PBR source 세 모델, isolation/full-cascade,
독립 학습 seed 및 최종508/10 metric을 계속 진행한다. Full 모델은 아직 채택하지 않았다.
