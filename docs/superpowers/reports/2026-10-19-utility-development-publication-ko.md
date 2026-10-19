# Utility 후보의 최종 비교 로깅

`utility_followup_wandb.py`는 실제 완료된 fixed-final report와 value-address 진단이 file4에 대조된 뒤에만 비교를 게시한다. 학습 run과 별도인 development report이며,음성 결과도 그대로 기록한다. 원래 F1의 완료된 report는 덮어쓰거나 다시 게시하지 않는다.

현재 scale16과 조건부 native voxel-query/soft-prior 후보에 같은 절차를 적용한다. 각 후보의 원래128개·93source·세 sampling noise 집계와 paired source-cluster CI를 report 입력 SHA/크기·실제 TP/FP/FN으로 대조한다. Source-only4M0·edited1M1+3M0 두 zero-shot,V1,no-retargeter Top1,같은 C1 retargeter Top1,이전 C1/Full/spatial,해당 scale reference를 모두 유지한다. Old joint 참조와의 차이를 새 후보의 단일 모듈 제거 효과로 부르지 않는다.

로그에는 voxel IoU/F1/precision/recall,percentage-point gain/CI,common32 에폭 곡선,같은 raw480/common32의 value-address sensitivity,모든128개 case를 원래 순서로 담은 PDF와 첫8개 비교 PNG가 포함된다. Source input 이미지나 추가 모델 추론을 생성하지 않고,저장된 실제 GT/Pred max-projection pixels를 사용한다. 숫자는3D voxel metric이며,PBR rendering metric이나2D mask IoU가 아니다.

게시 전에 실제 per-asset GT hash·source/noise와 GT pixels를 대조하고,게시 뒤에는 별도 download cache로 W&B run의 수치·table·PNG·artifact를 다시 내려받아 검증한다. 이후 report·그림·실제 readback·게시 source를 file4 SHA/크기로 대조한다. Model `.pt`는 W&B artifact에 넣지 않는다. 계정/project는 기존 `youngwoojeon-unist/pixal3d-deformnext-retargeting`을 유지한다.

여섯 CPU 검사와 완료된 F1 실제 기록의 read-only 대조가 통과했다. 새 collector는 기존 F1의9개 모델 집계·모든 paired contrast·30개 에폭 curve와 동일한 address 진단을 정확히 재현했다. 이 확인에서는 새 추론·학습·게시를 수행하지 않았다. Scope는 개발 결과이며,seed202·새 모델의 module ablation·예약121개 SS generation·model adoption·전체 목표 완료를 인증하지 않는다. 예약121개가 예전 auxiliary matching-head validation에 포함됐다는 평가 범위도 명시한다.

Scale16의 게시4단계(render/publish/readback/file4 close)를 실제 최종/진단 closure 뒤에 예약한다. 조건부 두 후속 모델은 실제 본 학습이 추가된 분기에서만 후보당 게시4단계를 추가한다. 현재 scale16이 기준을 통과해 대안 학습이 생략되면 대안 report도 만들지 않는다.
