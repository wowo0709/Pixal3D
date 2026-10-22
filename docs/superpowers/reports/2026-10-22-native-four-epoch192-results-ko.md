# Native SS 네 모델의192-update 중간 비교

원본 MV SS4210+fresh Adam, TRAIN9149, 학습 seed101,
world1/global96/micro96/accumulation1의 실제192 update를 비교했다.
초기32개 IoU는 네 모델 모두0.4267154667이고, 첫192개의 실제 asset·noise·timestep·
global condition trace가 같다. 학습을 중단하거나 validation 성능에 따라 재시작하지 않았다.

| 모델 | Global occupancy IoU | Mean asset IoU |
|---|---:|---:|
| Vanilla V1 | .435746760 | .467447349 |
| Physical Top1+R | .427218192 | .460882256 |
| Learned feature+R | .435143188 | .465662464 |
| Raw Top-K+feature+R | .432707077 | .464590209 |

Global count와 asset 평균은 다른 집계이므로 열을 섞지 않는다.
32개 asset의 TP/FP/FN으로 각 asset의 precision/recall/F1/IoU와 pooled metric을
다시 계산해 실제 기록과 일치하는지 확인했다. Source cluster 재표집4000회,
32개 source cluster·생성 noise1회의 mean asset IoU 비교는 다음과 같다.
단위는 **percentage point**이며 다중 비교 보정 전95% CI다.

| 비교 | IoU 차이 (%p) | Source-cluster CI95 (%p) |
|---|---:|---|
| Top1+R − V1 | −.656509 | [−1.447721, +.064759] |
| Feature+R − V1 | −.178488 | [−.906493, +.470406] |
| Raw Top-K+feature+R − V1 | −.285714 | [−1.017006, +.354922] |
| Feature+R − Top1+R | **+.478021** | **[+.157758, +.869086]** |
| Raw Top-K+feature+R − Top1+R | **+.370795** | **[+.069774, +.724639]** |
| Raw Top-K+feature+R − feature+R | −.107226 | [−.248859, +.026233] |

## 해석

첫96-update 비교에서는 두 head 모델과 Top1+R의 차이가 불확실했다.
192에서는 두 Top-K/feature 경로가 물리적 Top1+R보다 높은 중간 결과가 나왔다.
Feature+R은96의.431369에서192의.435143으로 개선됐고,
Top1+R은.430991에서.427218로 하락했다. 따라서 현재 차이는 feature 경로의
향상과 Top1 경로의 하락이 함께 만든 것이다. Attention이 V1을 넘었다거나,
모든 학습 시점에서 개선됐다는 뜻은 아니다.

이 비교는 공통 ray grouping 및 R 위에 feature readout을 더한 모델의 차이다.
Decoder와 matching score는 공통 frozen matcher이므로 그 둘의 독립 효과를
입증하지 않는다. Semantic-vs-spatial, ray grouping 자체의 효과도 분리하지 않았다.
Raw weighted appearance의 추가 이득은 현재 feature+R 대비 확인되지 않았다.

V1 대비 세 source 모델의 CI는 모두0을 포함한다. 작은 개발 panel을 반복 사용한
중간·다중 비교이며, 독립 학습 seed도 하나다. 최종480 update·128개×3 generation noise,
독립 TRAIN seed202 및 cascade의 target/source rendering 지표에서 재확인한다.
192의 양수 CI를 근거로 학습 예산을 바꾸거나 best epoch를 채택하지 않는다.

## 실제 근거와 한계

Canonical result는 file4 `rgb_matching_v2/semantic-topk-search-v1/`
`ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/`
`source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/native-flow-cold3-v1/`
`checkpoint-evaluation-v2/epoch192-CPU-evidence-v1/`이다.

`actual-inputs-v1`은 실제 네 모델의 NN/source/optimizer192 파일, 원래 config와
validation192 snapshot, 같은 첫192 trace 및 CPU8개 검사 결과를 보존한다.
24개 파일11,470,289,779byte를 실제 file4 SHA/size와 대조했다.
`four-native-paired32-v1/audit.json`은 그 입력에서 계산한 paired 통계로,
별도로 실제 file4와 대조했다. 현재 학습 로그 전체를 고정 파일인 것처럼 hash하지 않고
완료된192행과 trace prefix만 snapshot했다.

`r_native_paired_epoch.py`는 실제 step·asset·source·GT hash/shape/표현·생성 seed와
occupancy count를 검증한다. 같은 평균이라도 다른 GT/noise나 잘못된 pooled metric은
거부한다. 이 CPU 집계는192 저장 weight의 독립 GPU replay를 대신하지 않는다.
새 GPU 생성이나 CPU neural training은 하지 않았다. Model 채택과 전체 목표 완료는 미정이다.
