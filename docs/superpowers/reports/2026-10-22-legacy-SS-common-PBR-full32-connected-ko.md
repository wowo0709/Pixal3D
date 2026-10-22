# Legacy/common-PBR DEV32 전체 비교 연결

앞서 연결한4개 prefix 작업 뒤에 나머지191개 작업을 실제 추가했다. 전체 범위는
retargeting(C1)·feature-spatial·feature-DQ의32개 geometry에 original PBR16850 및
공통 native vanilla PBR1900을 적용하는6그룹이다. Native8그룹과 합쳐14그룹·32asset의
1344개 official perceptual view를 source/target 역할별로 비교한다.

이번 추가는93개 GPU case,96개 CPU own texture raw 정리, 공식576개 view GPU 평가1개,
14그룹 CPU 집계1개다. 기존96개 geometry 입력은 읽기 전용으로 유지한다.4개의 순차
lane은 각 case의 canonical archive 검증과 local raw 정리가 끝난 뒤 다음 case를 시작한다.
원래 geometry·canonical remote raw·render pixels·panel·checkpoint는 정리 대상이 아니다.

새 retirement·summary의 synthetic CPU19개 검사와 두 worker CLI help 및 구문 검사를
통과했다. 구현과 프로토콜은0900886으로 요청한 원격 브랜치에 push했다. 실제 GPU
평가가 통과했다는 뜻은 아니다. Native8개의 기존 통계를 재현하고14개 모델의 GT pixel,
reference crop·metric registry·precision·공통 downstream lineage를 확인하도록 구성했다.

| 실제 file4 SHA256/size 검증 | Files | Bytes |
|---|---:|---:|
| Full32 prelaunch | 198 | 22,194,171 |
| Full32 queue | 2 | 8,233,248 |
| Full32 applied | 1 | 14,312 |

기존3557개 descriptor는 정확히 유지됐고 전체3748개 dependency DAG를 확인했다.
Dispatcher가 새191개를 모두 pending으로 읽었다. Queue-state나 진행 중인 학습의
checkpoint·config를 편집하지 않았다. 기존 native8그룹의 실제 완료와 engineering
prefix gate 전에는 이 평가를 시작하지 않는다. GPU는 지정한0–3만 사용한다.

현재 실제 결과는 기존 모델과 진행 중인 PBR의 중간 평가까지다. 새14그룹 비교는
아직 품질 결과가 없다. Source 보존·target 정합 tradeoff, 독립 TRAIN seed, 적절한
선택 조합 확인과 frozen TEST508의10 metric까지 계속 진행해야 하며, 이 큐 연결을
Full 채택이나 전체 실험 완료로 취급하지 않는다.
