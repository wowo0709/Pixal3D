# SS와 PBR에서 matching 효과를 구분하기 위한 현재 증거

SS에서 추가 attention의 이득이 확인되지 않았다는 결과를 PBR에 일반화할 수는 없다. SS64의 거친 점유 구조보다 PBR의 색·재질 복원에서 source correspondence가 더 중요할 수 있다는 것은 검증할 가설이다. 현재 기록이 PBR의 우월한 반응을 입증하지는 않는다.

## SS에서 확인한 범위

같은128개×3 생성 noise에서 기존 ray Full IoU0.433994, score 가중합0.434168, Top-1 0.434027이다. Full−가중합 및 Full−Top-1의 source-cluster paired95% CI는 모두0을 포함한다. 학습 seed는1개이며 추가 attention 효과를 확인하지 못했다.

다만 correspondence가 SS에 전혀 영향을 주지 않는다는 뜻은 아니다. 별도 저장 Full480의 탐색32개 value-address control에서는 정상0.436073, 주소 교란0.431890이었다. 정상−교란 +0.41833pp의 CI는[+0.00860,+0.78817]pp다. 이는 해당 모델·32개에서의 조건부 민감도이며 Top-K 조합의 baseline 대비 이득이나 일반적인 causal winner의 증거는 아니다. 이전 Top-1+R의 증분은 별도2개 학습 seed에서 확인했지만 새 query/carrier에서도 같은 이득이 유지된다고 가정하지 않는다.

## 기존 RGB PBR의 실제 quick panel

다음은 모두 edited M1 RGB cam08 + M0 RGB cams09/10/11, 같은 GT target Shape, global24/world3/forward micro4/accumulation2의 기존 RGB 학습 기록이다. 잘못된 과거 depth 입력 run은 제외했다. **Flow-loss 기록의 `bounded panel (16 assets)`와 달리 렌더 quick panel은 실제2개 asset이다.** 네 arm의 `quick_panel_ids`가 같음을 확인했다.

| 기존 arm / 같은 epoch2·step762 | Foreground PSNR↑ | Target-crop LPIPS↓ |
| --- | ---: | ---: |
| Vanilla V1 | 20.227840 | 0.099360 |
| B1 | 20.221546 | 0.099154 |
| B2 learned-score Top-1 | 20.052774 | 0.099169 |
| B3 score 기반 Top-K 가중합 | 20.077900 | 0.099143 |

B3−B2의 PSNR 차이는+0.02513dB다. LPIPS 차이는-0.00002644이고 B3−V1의 PSNR은-0.14994dB다. 두 샘플의 이 작은 차이를 attention 효과나 PBR stage의 우월성으로 판정하지 않는다. 이 B3는 기존 score 가중합이며 새 target-ray content attention의 PBR 학습 결과가 아니다.

V1의 epoch0→5 foreground PSNR은21.08750→19.82899dB, target-crop LPIPS는0.095410→0.103389였다. Vanilla에서도 quick 생성 품질이 나빠지므로 matching 모듈만으로 이 경향을 설명할 수 없다. 과적합, 최적화 또는 평가 조건 중 어느 원인인지는 이2개 기록만으로 분리되지 않는다. Training loss 감소와 생성 품질 개선을 동일시하지 않는다.

이 표의 LPIPS는 **GT target rendering과의 기존 crop 진단값**이다. 사용자가 정의한 논문용 source-preservation LPIPS로 이름을 바꾸거나 사용하지 않는다. 현재 지표는 입력 anchor camera의 GT-shape-conditioned texture stage이며 novel-view 또는 전체 cascade 결과가 아니다. Alpha IoU1.0도 고정한 GT geometry의 결과다.

원래 RGB 기록은 `rgb_v1/runs_world_v4/pbr1024/v1/` 및 `rgb_v1/runs_r9_v5/pbr1024/{b1,b2,b3}/`다. 각 run의 config/validation/progress를 복사한 감사는 canonical `rgb_matching_v2/semantic-topk-search-v1/pbr-training-trend-review-v1/`에 있다. 실제 file4의13개 파일/116,068bytes를 SHA256·크기로 검증했다.

## PBR 확대를 판단할 다음 비교

SS winner가 나올 때까지 PBR의 모든 검사를 보류할 필요는 없다. 우선 기존 저장 epoch2 weights를 공통32개에서 다시 평가해2개 panel의 우연인지 확인한다. 같은 input·noise·GT Shape·sampling·render 설정을 고정하고 source별 불확실성을 보고한다. 원격에 네 arm의 동일 step762 NN 및 필요한 conditioner가 존재함을 read-only로 확인했으며 evict됐던 V1 local checkpoint 복원을 시작했다. 이 단계는 새 PBR fine-tuning이 아니다.

이 anchor 검사는 경향 확인용이다. 실제 새 attention을 PBR에 학습할 때는 동일 geometry에서 Top-1·고정 score 가중합·content attention을 비교하고, 입력 camera08을 제외한 held-out target views와 source가 보유한 정보의 평가 영역을 사용해야 한다. 초기 parity와 retrieval-value sensitivity를 먼저 확인하고 작은 고정 budget에서 검증한다. 모든 입력·학습·캐시는 GT conditioning 없이 구성하며, geometry를 평가용으로 고정했다는 범위를 명시한다.

현재 두 SS canonical arm은 계속 진행 중이고 target-ray utility auxiliary7epochs는 완료했다. 새 SS joint continuation 경로는 원래 epoch1의360 NN LoRA +6 R Adam state를 실제로 복원하고 utility5+gate를 추가한372-state GPU profile을 통과했다. 이는 버려지는 micro2의3개 임시 update이며 DDP2/micro48 본실험이나 새 PBR 학습 완료를 뜻하지 않는다.
