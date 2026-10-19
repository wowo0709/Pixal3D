# 후속 SS 모델의 에폭별 비교

진행 중인 voxel-query와 utility-softprior의 모델·학습 config·고정 raw480 선택 규칙은 유지한다. 각 에폭의 실제 optimizer/W&B/PNG 검증과 file4 SHA256 대조가 끝난 immutable common32 snapshot만 비교한다. 비교 계산은 GPU를 쓰지 않는다.

각 후보는 동일 에폭의 V1, 같은 retargeter의 Top-1, static T0.005, static T0.05, utility scale4, utility scale16 및 두 zero-shot 결과와 비교한다. 특히 softprior 대 static T0.05는 temperature 완화만의 효과와 학습한 utility의 추가 효과를 구분하는 비교다. Voxel-query 대 static T0.05는 query와 temperature도 달라지므로 utility 하나의 효과라고 해석하지 않는다.

Sample 순서·GT voxel hash/shape·source cluster·생성 noise seed·초기 per-asset 출력의 일치를 요구한다. Voxel TP/FP/FN 합으로 IoU를 계산하고 source-cluster bootstrap CI를 함께 기록한다. 이전 에폭과의 변화도 기록한다. 32개/생성 noise 하나/학습 seed 하나의 중간 경향이며, 새 모델 채택이나 독립 test 결과로 사용하지 않는다. 최종128개/93 source/3 생성 noise 및 양성 후보의 fresh202 검증은 별도로 유지한다.

실제 비교와 snapshot 검증의 CPU 검사8개를 통과했다. 동일 voxel count를 가진 다른 GT, 다른 noise/source, 누락된 hash, 바뀐 순서가 비교에 들어가지 못하는지 확인했다. 본 학습의 forward source는 수정하지 않았다.
