# Source-input 모델의 두 번째 joint 학습 seed 연결

2026-10-21 UTC. 새 Original19/SourceContrast19의 joint480 학습은 아직 실행하지
않았다. 현재 준비한 seed 202 경로는 향후 개선 재현성을 확인하기 위한 실행 지원이며,
성능 향상이나 GPU 검사를 완료했다는 증거가 아니다.

## 발견한 연결 문제와 변경

본 학습의 기존 recipe validator는 training seed 101과 202를 모두 허용하고,
native warm 정합 probe는 config의 training seed로 초기화한다. 반면 disposable
micro96 capacity probe는 seed 101로 고정돼 있어 새 source-input 경로로 seed 202를
검증할 수 없었다. 기존 파일을 수정하지 않고 `source_input_joint_seed202.py`를 추가했다.

이 진입점은 config의 정수 `training_seed=202`, 실제 fresh run, 명시한 source-input
설계와 teacher 사용을 확인한다. `warm-parity`, `capacity`, `science` 단계는 기존
native warm probe와 source-input runtime을 사용한다. 각 단계의 원래 parser와
수치 recipe 검사는 그대로 수행한다.

Capacity 단계에서만 기존 seed adapter의 AST 검사를 사용해 원래 probe의 단일 seed
guard를 메모리 안에서 101→202로 바꾼다. 반대로 바꾼 AST가 원본 전체와 같아야 하며,
guard가 추가·누락되면 거부한다. 원래 파일은 변경하지 않고, 종료·실패 시 원래 함수를
복원한다. Native trainer가 config seed 202를 사용한다. Optimizer, AMP, LoRA,
global96/micro96/accumulation1, 세 번의 disposable update와 frozen weight 검사는
원래 코드가 수행한다.

Warm probe와 본 학습은 이미 config seed를 사용하므로 AST 변경이 없다. 본 학습
성공 뒤 실제 startup의 seed가 202인지 확인한다. 실행 중 config나 teacher spec이
바뀌면 성공 기록을 만들지 않는다. 각 단계 성공 뒤 별도 seed runtime 기록을 남긴다.

## 무엇을 독립적으로 검증하는가

이 경로는 **동일한 고정 teacher112에서 시작하는 두 번째 joint 학습 seed**다.
Teacher 자체를 다른 seed로 다시 학습한 결과가 아니며, sampling noise를 바꾼
실험과도 다르다. 이 한계를 runtime 기록에 명시한다.

Original19와 Contrast19는 각각 자신의 teacher112 초기화를 유지하고, 같은 joint
training seed끼리 비교해야 한다. 서로 다른 teacher 사용 여부나 다른 SS base로
input 설계의 효과를 주장하지 않는다. 학습 seed를 바꿀 때 평가 자산, GT, 카메라와
sampling noise는 고정한다. 실제 seed202 모델이 나오면 동일한 completed480 evaluator와
개발128/32 cascade 경로로 평가할 수 있다. 후속 후보 선택에서 seed별 provenance와
비교 결과를 확인하는 작업은 별도로 필요하다.

## 현재 검증 상태와 실행 조건

새 진입점 검사 20개와 기존 seed AST 검사 7개, 총 **CPU 27개가 통과**했다.
실제 원본 capacity 파일의 단일 guard와 역변환을 검사했고, mock phase를 사용해
두 설계·세 단계의 routing, seed101 startup 대체 거부, config 변경 거부와 실패 시
복원을 확인했다. Mock 출력은 실제 GPU/학습 결과로 사용하지 않는다.

실제 seed202 warm 정합, micro96 GPU capacity 및 joint480 학습은 아직 실행하지 않았다.
이미 예약된 별도72 자산 확인, native warm 검증과 저장 공간 검사를 통과한 뒤
과학적 비교가 필요한 모델에 실행해야 한다. 현재 PBR 4GPU 학습과 실제 대기열은
수정하지 않았다. 새 source-input 모델의 개선 또는 semantic attention 효과가
입증됐다는 결론도 내리지 않았다.
