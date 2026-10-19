# Utility attention 4에폭 관측과 최종 판정의 범위

2026-10-19 06:19 UTC 확인. 현재 improved full 모델을 확정하지 않았다. 학습 seed101의 utility scale16 후보는 raw384 validation을 마치고 raw480까지 계속 학습 중이다. epoch04의 실제 optimizer, 양 rank global96/micro48/accumulation1, W&B 네 수치 및 PNG, checkpoint와 비교 결과가 file4 SHA/크기로 대조됐다.

같은 32개 사례·같은 생성 noise·동일 raw384 비교다. IoU는 3D voxel metric이다.

| 모델 | IoU |
|---|---:|
| Vanilla V1 | 0.434300 |
| 같은 C1 retargeter의 Top-1 | 0.449885 |
| Utility scale4 | 0.449995 |
| 현재 utility scale16 | 0.449735 |

Scale16의 paired source-cluster bootstrap 차이이며 단위는 percentage point다.

| 비교 기준 | IoU 차이 | 95% CI |
|---|---:|---:|
| V1 | +1.54345 | [+0.04007, +3.14483] |
| 같은 retargeter의 Top-1 | −0.01510 | [−0.11431, +0.07682] |
| Utility scale4 | −0.02601 | [−0.09014, +0.02724] |

Vanilla를 이겼다는 관측만으로 Top-K/attention의 이득이라고 해석할 수 없다. 같은 retargeter의 Top-1도 비슷한 성능이기 때문이다. Scale16은 epoch03에서 같은 Top-1보다 +0.21407 pp였지만, epoch04에서는 그 우위가 유지되지 않았다. 반복해서 관측하는 32개 개발 사례의 중간 결과이며 최종 모델 선택이나 독립 검증으로 사용하지 않는다.

앞서 완료한 scale4의 고정 raw480·128개·93 source·세 생성 noise에서 IoU는 0.442268, 같은 retargeter Top-1은 0.442757이었다. 차이는 −0.04889 pp, 95% CI [−0.13107, +0.02128]였다. 이 음성 결과도 유지한다. 과거 joint full에서 decoder 제거, score prior 제거, native spatial 대조와의 차이도 CI가 0을 포함하므로 해당 full의 모듈 향상이 입증됐다고 주장하지 않는다. Score prior 제거는 별도 진단이며, 사용자가 정의한 No-score arm은 Top-1 feature 대조다.

현재 후보는 마지막 raw480 checkpoint를 128개·93 source·세 noise에서 평가한 뒤 판정한다. 기준을 통과하면 fresh training seed202를 실행한다. 통과하지 못하면 native voxel-query 및 soft score prior 두 후보를 준비된 GPU 0/1, 2/3에서 각각 실행한다. 이 두 후보도 실제 최종 조건을 통과했을 때만 각각 fresh202로 반복하도록 연결했다. 후보별 한 번의 5에폭 학습뿐 아니라, 동일 seed의 두 Top-1 기준과 GPU conditioning/epoch 기록/최종 가중치를 함께 확인한다.

Top-1 C1 기준 모델의 seed202는 raw480 학습을 완료했다. Frozen base 701개가 정확히 유지되고 LoRA 360개가 갱신됐으며 원래 matching head 24개는 유지됐다. 최종 optimizer 366개 상태의 step480 및 다섯 epoch의 file4 대조를 통과했다. 이 모델의 128개·세 noise 생성 평가는 진행 중이다.

재검증 구현과 실제 두 후보 recipe/두 Top-1 config의 호환성 확인은 완료했다. 관련 CPU 검사 24개가 통과했다. 공용 launcher가 검증 후 갱신하는 상태 필드만 허용하고 numerical/data 설정 변경은 거부한다. 실제 file4의 `orthogonal-utility-followups-v1/provenance/orthogonal-confirmation-v2/file4-closure.json`에 41개 소스·설정 원본·실제 preflight 파일을 대조했다. 이는 모델 향상 증명이 아니다.

실제 결과의 상대 경로는 `rgb_matching_v2/semantic-topk-search-v1/utility-strength-followup-v1/early-progress-v1/epoch04-common32-with-reference-v1.json`과 같은 폴더의 `epoch04-comparison-file4.json`이다. 원본 file4 결과와 기존 checkpoint는 덮어쓰지 않았다. 추가 source-disjoint 121개는 SS 생성 결과를 아직 보지 않았지만 이전 auxiliary matching-head validation에 포함됐으므로 독립 test로 부르지 않는다. 채택 전에는 후보를 잠그고 이 추가 평가와 해당 후보의 decoder/Top-1/native-spatial 모듈 대조가 필요하다.
