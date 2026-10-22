# Native SS480 attention의 실제 가중치 갱신 확인

추가 attention의 IoU 이득이 없는 원인이 단순한 가중치 미갱신인지 확인했다.
원본 SS4210에서 native SS flow objective로480 update 학습한 R_feature와
R_raw_topk의 실제 final source checkpoint/config/bundle을 canonical file4 receipt와
대조하고 own schema로 strict load했다. Denoiser는 로드하지 않았고 source module의
forward/backward 또는 optimizer update도 수행하지 않았다. CPU92/93에서 정적인
텐서 비교만 실행했다. 현재 PBR 학습과1520 관찰 프로세스는 변경하지 않았다.

비교 초기값은 기존 native96 가중치 검사와 같은 선언된 cold constructor다.
Head의 내부 초기화 seed20261021과 동일 utility initializer를 사용했다.
Current checkpoint의 frozen matcher 및7개 initial-function anchor가 초기값과
dtype/값이 정확히 같음을 검사했다. Retargeter 첫 층의 임의 초기값까지 재현하거나
전체 optimizer trajectory를 검증한 것으로 해석하지 않는다.

| 항목 | Feature+R | Raw Top-K+Feature+R |
|---|---:|---:|
| 값이 바뀐 learned attention 텐서 | 9/9 | 9/9 |
| Q matrix 상대 L2 변화 | 7.07% | 7.32% |
| K matrix 상대 L2 변화 | 5.48% | 5.64% |
| V matrix 상대 L2 변화 | 10.37% | 10.17% |
| Head output matrix L2: 초기→최종 | 0→1.8486 | 0→1.7760 |
| R output projection L2 | 2.2152 | 2.2483 |
| Semantic fusion scale tanh(g) | .04430 | .04450 |

Utility의 query/context/key 행렬도 약5% 변했고, content output의 상대 L2 변화는
45.62%/47.04%다. 이 변화 크기를 attention 확률 변화량이나 매칭 정확도라고
부르지 않는다. 특히0으로 시작한 head output이 nonzero가 된 것은 weight decay만으로
생길 수 없으므로, 전체 head가 단순히 freeze되어 있었다는 설명과 맞지 않는다.
가중치 변경이 의도한 대응 관계를 학습했는지까지 증명하는 결과는 아니다.

Frozen matching decoder/score와 reference anchor가 바뀌지 않은 것은 이 native
recipe의 의도다. Matching decoder와 score를 실제로 사용하는지, 각각 얼마나
도움이 되는지 또는 현재 source correction이 제대로 전달되는지는 가중치 비교만으로
입증되지 않는다. Fusion scale이 작다는 이유로 전체 R 경로가 닫혔다고도 볼 수 없다.
현재 R residual 전달은 그 scale만으로 결정되지 않는다.

현재 가장 타당한 다음 질문은 **학습된 head가 posterior와 R 입력을 어떻게 바꾸며,
그 변화가 native consumer 및 최종 생성에 도움이 되는가**다. 연결된 native480
normal-shadow/prior-only/Top-1 feature/wrong-V/no-transformed-V 진단이 이를 확인한다.
Inference-only 개입이며 재학습 ablation, uniform No-score 또는 spatial No-corr를
대체하지 않는다.

실제 근거는 native-flow-cold3-v1 아래
`final480-head-weight-learning-CPU-v1/{audit.json,file4-closure.json}`이다.
Audit과 실행 script2개17,423 bytes를 actual file4 SHA/size와 대조했다.
기존 SS 최종128의 성능 결론이나 epoch 선택은 바꾸지 않았으며, PBR/full-cascade·
독립 학습 seed·최종 test508/10 metric 결과는 계속 남아 있다.
