# Utility scale16 최종 결과와 후속 본 학습 시작

2026-10-19 07:02 UTC. Scale16 후보는5에폭/raw480을 완료했으나, retargeter가 같은 Top-1 모델을 넘지 못해 채택하지 않았다. 실제 최종 checkpoint·5개 epoch·W&B 기록·saved GPU cache/AMP/CFG·128개/3noise 평가를 확인한 결과다.

| 모델 | 원래128개·93 source·3 생성 noise의 voxel IoU |
|---|---:|
| Source-only4M0 zero-shot | 0.380711 |
| Edited1M1+3M0 zero-shot | 0.419275 |
| Vanilla V1 | 0.432650 |
| Semantic Top-1, retargeter 없음 | 0.434529 |
| 같은 retargeter의 semantic Top-1 | 0.442757 |
| Static C1, score T0.005 | 0.442193 |
| Utility scale4 | 0.442268 |
| Utility scale16 | 0.442247 |

Scale16 대 같은 retargeter Top-1 차이는 **−0.051044 pp**, paired source-cluster95% CI **[−0.134041,+0.016946] pp**다. Static C1 T0.005 대비는+0.005402 pp, CI[−0.036054,+0.041347] pp이며, scale4 대비는−0.002157 pp, CI[−0.039233,+0.031817] pp다. Vanilla 대비+0.959672 pp는 양성이지만, 같은 retargeter 기준을 넘지 않아 Top-K attention의 추가 이득으로 해석하지 않는다. 이 후보의 seed202 재학습은 실행하지 않았다.

학습이 연결되지 않은 모델은 아니다. Frozen base701개와 matcher24개는 유지되고 LoRA360개 및 utility5개 tensor가 갱신됐다. Optimizer371개 상태가 모두 step480 및 finite였다. 실제 최종 GPU conditioning/cache/AMP/CFG 검증과 모든5개 epoch의 실제 W&B4metric/PNG·file4 대조를 통과했다. 초기 output도 원래 모델과 같았다.

실제 raw480/128개/9,883query CPU FP32 진단에서 utility가 prior의 argmax를 바꾼 비율은2.874%였고, 합성 value의 상대 RMS 변화는2.771%였다. 유효 후보 수는 여전히1.232였다. Utility logit span 중앙값3.535, prior의 Top-1/2 gap 중앙값7.228이며,69.62% query는 prior gap이 전체 utility span보다 컸다. Cached/fresh prior 차이는 약1e-6 수준이었다. Query/context/key Adam moment의 epsilon damping 중앙값은0.848/0.916/0.946이었다. 모두 학습되고 있지만 prior 집중이 남아 있다는 근거이며, 이 수치만으로 품질 저하의 단일 원인을 확정하지 않는다.

같은 raw480/common32/noise0의 value-address 진단은 정상 대응이 잘못 연결한 값보다+0.587917 pp였지만 CI[−0.080727,+1.284376] pp로0을 포함했다. Normal 결과는 실제 final128의 같은32개와 정확히 재현됐다. 이는 No-corr 학습 arm이나 매칭 정답률 평가가 아니다. No-corr은 원래 native spatial correspondence, No-score는 Top-1 feature라는 사용자 정의를 유지한다.

후속 두 모델은 계획한 조건으로 본 학습을 시작했다. `voxel-query-c1`은 GPU0/1에서 utility query만 native voxel의4-view projected DINO로 바꾸며 원래2D C1 query·semantic64·frozen head·T0.005·scale16을 유지한다. `utility-softprior-c1`은 GPU2/3에서 기존 patch query를 유지하고 score T만0.005→0.05로 바꾼다. 모두 원래 SS4210의 base+LoRA에서 fresh seed101, global96/DDP2/micro48/accumulation1, raw480으로 시작한다. 데이터·캐시를 다시 만들지 않았다.

실제 본 학습의 initial32에서 두 후보 모두 모든 per-asset 기록·GT hash/source/noise/order·네 metric이 scale16의 초기 결과와 정확히 같았다. IoU는0.42671546671444294다. 양 rank의 실제 첫 loss와 모든 timestep-bin loss도 정확히 같았다. Main loader는 검증된 workers2다. 이전2-update preflight의 가중치를 main에 사용하지 않았다. 초기 regression이 확인되지 않은 상태에서 실제 SGD를 진행 중이다.

두 모델 각각 native/DDP proof,5에폭 validation·한 번만 실행하는 epoch/final audit, 최종128개/3noise·GPU QA·value-address 진단·W&B 비교 보고서가 연결돼 있다. 최종 조건을 통과한 후보만 fresh202로 재현한다. Soft-prior는 같은 T0.05의 기존 static C1(IoU0.440938)과도 비교해야 temperature 완화만의 효과와 학습된 attention의 효과를 분리할 수 있으며, 실제 두 config와 원래 static checkpoint/평가의 호환성을 확인했다.

Scale16의 모든128개 비교 그림·수치·표는 [W&B development report](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-utility-scale16-c1-development-v1)에 게시했고, 별도 download cache로 수치/table/PNG/artifact를 재검증한 뒤 file4에 대조했다. 음성 결론도 유지한다. 후속 결과 게시를 예약하는 두 작업의 동시 append가 충돌했으나, 기존 실패 이력·backup을 보존하고 완료된 writer 뒤에 직렬화한 별도 복구 작업으로 누락된8개 게시 예약을 추가했다. 모델·학습·평가 command와 건강한 프로세스는 변경하거나 재시작하지 않았다. 현재 구성된 작업에 미해결 실패는 없고 과거 실패 기록은 남아 있다.

후속 모델이 아직 성능 향상을 보였다는 뜻은 아니다. 실제 새 모델의 decoder/Top-1/native-spatial 효과와 추가 source-disjoint121개 평가도 남아 있다. 예약121개는 이전 auxiliary-head validation에 포함됐으므로 독립 test라고 부르지 않는다. 목표는 진행 중이다.
