# Semantic Top-K attention 후속 모델의 재현 판정

현재 실행 중인 utility scale16 모델의 최종 성능은 아직 확인되지 않았다. 기존 scale4 모델은 vanilla V1보다 좋았지만 같은 retargeter를 쓰는 Top-1을 넘지 못했다. 중간 epoch의 양성 신호로 full 모델을 확정하지 않는다.

scale16의 고정 raw480 최종 비교가 음성이면 준비된 두 후보를 실행한다. `voxel-query-c1`은 utility의 query만 native voxel에 투영한 4-view DINO로 바꾼다. `utility-softprior-c1`은 patch query를 유지하며 frozen matching score의 temperature만 0.005에서 0.05로 바꾼다. 두 후보 모두 semantic Top64, zero-start utility scale16, 원본 frozen R9 D1/scalar head, C1 retargeter, 원본 SS4210 LoRA continuation을 유지한다. 초기 출력을 동일하게 하는 증명과 실제 DDP2/micro48 benchmark를 각 실행 전에 반복한다.

`orthogonal_utility_confirmation.py`는 두 후보의 실제 최종 결과가 닫힌 뒤, V1과 두 Top-1 기준 모두에 대해 source-cluster 95% CI의 하한이 양수이고 세 생성 noise에서 모두 향상된 후보만 fresh training seed202로 반복한다. 후보가 둘 다 조건을 통과하면 GPU 0/1과 2/3에서 각각 반복한다. 하나도 통과하지 못하면 추가 학습하지 않는다. 기존 scale16 모델이 먼저 통과하면 그 모델의 기존 confirmation 경로를 사용하고 두 후속 후보의 단계는 종료한다.

각 반복은 global96, DDP2, micro48, accumulation1, raw480을 유지한다. seed101의 실제 recipe를 읽어 seed, 출력 경로, 통신 port, 로깅 이름만 바꾼다. 두 후속 후보는 검증된 workers2를 사용하고, 기존 workers8 Top-1 모델과 비교할 때 실제 dataset/order/RNG parity proof를 요구한다. voxel query의 구현 trainer/chunk 설정은 명시적인 구조 차이로 기록한다. 실행 전후 소스 hash, 실제 epoch별 optimizer/W&B 수치 및 PNG, 최종 frozen base/LoRA/conditioner 상태, GPU cache/AMP/CFG 확인, 최종 128개·93 source·3 noise 재평가가 모두 있어야 재현 결과를 산출한다.

공용 launcher는 native/DDP 검증 후 원본 resolved recipe의 `release_status`와 `microbatch_per_gpu`를 갱신한다. 소스 guard는 실행 코드와 불변 입력의 hash를 고정하고, 두 seed101 recipe는 실제 file4에 보존한 원본과 비교하되 이 두 필드의 허용된 상태 전이만 인정한다. microbatch 값은 48만 허용하며 optimizer, 데이터, batch, step, 모델 설정의 변경은 거부한다. 검증 완료 후의 seed202 recipe에도 같은 검사를 적용한다.

두 optimization seed는 재학습 반복이다. 세 생성 noise는 같은 checkpoint의 sampling 반복이며 별도 학습 seed로 세지 않는다. V1은 고정 checkpoint를 재사용하며 seed202 V1을 새로 학습했다고 표현하지 않는다. source-cluster CI는 학습 seed 모집단의 CI가 아니다. 여러 후보를 살핀 개발 단계라는 한계도 유지한다.

이 단계가 양성이더라도 모델 채택을 완료하지 않는다. 잠근 후보의 추가 source-disjoint SS validation과 같은 구조에서의 decoder/Top-1/native-spatial 비교가 남는다. 예약된 121개는 SS 생성 결과를 아직 보지 않았지만 이전 matching head의 validation에 포함되었으므로 독립 test라고 부르지 않는다. No-score는 Top-1 feature 사용, No-corr은 native spatial correspondence 사용이라는 정의를 유지한다. 기존 joint full의 삭제 실험은 새 후보에 대한 직접적인 모듈 효과로 대체해 해석할 수 없다.
