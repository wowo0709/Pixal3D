# 실제 모델 구조를 유지하는 후속 모델 재현·평가 준비

2026-10-23. 구현 커밋 `a0d14b2`, 브랜치 `wowo0709/feature-retargeting`.

## 완료한 작업

기존 population/독립 seed 검증기는 원래 12개 arm만 허용했다. 새 E_FIXED_RAW와
E_LEARNED_RAW를 기존 Full로 바꾸어 부르지 않고 검증할 수 있도록 별도 버전을 추가했다.
기존 실행 중인 학습·config·worker·원래 12개 arm 프로토콜은 수정하지 않았다.

- 새 두 모델은 각각 source10/source17 및 고유 checkpoint/frozen-binding schema를 유지한다.
- 원래 12개 모델은 기존 검증기로 그대로 위임하며 기존 binding 구조도 그대로 반환한다.
- validation36/TEST508 실행기는 실제 해당 family의 evaluator를 사용한다. 모델 수식과
  sampler를 바꾸지 않고, 검증된 데이터 선택만 변경한다.
- 독립 seed config는 실제 seed101 모델과 데이터·모델·optimizer·수치 설정·JSON 타입을 대조한다.
  seed202 자체 proof/capacity 파일 및 같은 seed의 실제 V1 경로만 새 artifact 경로로 기록한다.
- 독립 seed 결과 수집기는 실제 NN/source/config SHA, frozen base와 360개 LoRA의 구조,
  종료된 실제 child, 전체480 학습 입력 trace 및128×3 생성 결과를 확인한다.
  seed101/202를 같은 seed처럼 정규화하지 않는다.

관련 CPU 검사 **243개 통과(새 검사71개)**. NN forward를 stub으로 대체하거나 metadata와
tensor data만 검사했으며 CPU neural forward/학습은 수행하지 않았다.

## 실제 외부 artifact 검증

`selected-followup-completion-v178`에 새 두 모델 × proof/capacity/science의
**seed202 config template 6개**를 저장했다. reference config는 recipe 전용이며 학습 결과가 아니다.
데이터와 원래 SS4210 initializer 및 기존 seed202 V1의 실제 예정 경로도 대조했다.
파일서버4 SHA256/size 확인은 **10개 파일 PASS**다.

두 모델을 모두 seed202로 학습하도록 큐에 넣지는 않았다. 실제 DEV 결과로 후보를 선택한 뒤
해당 모델과 필요한 control만 실행한다. 선택·활성화 factory 및 실제 seed202 실행은 아직 남아 있다.

새 closed reader는 완료된 실제 V1·Top1+R·Full의 NN/source/config,9개 평가 파일 및
원본 frozen weight를 읽어 다음 mean-asset IoU128×3를 그대로 재현했다.

| 실제 완료 모델 | IoU |
|---|---:|
| V1 | 0.46735504 |
| Top1+R | 0.47260025 |
| 기존 Full | 0.47207818 |

`selected-followup-actual-closed-reader-v178`의 실제 file4 closure PASS다. 이는 새 reader의
실제 입력 호환성 검사이며 새 E 모델이나 seed202 결과를 읽거나 생성한 것이 아니다.

## 학습 중간 관측

`focused-status-v178`에 실제 살아 있는 학습 parent/child와 progress/validation의 안정된
복사본을 저장했다. 13개 파일의 file4 SHA256/size PASS. 관측 당시 진행:

| GPU | arm | update / 480 |
|---:|---|---:|
| 0 | C_RAW | 200 |
| 1 | C_SPATIAL | 194 |
| 2 | C_D1_COS | 200 |
| 3 | C_DECODER_NO_R | 206 |
| 6 | FULL_QK_ONLY | 129 |
| 7 | FULL_VOXEL_Q | 38 |

동일한32개·동일 noise의 **2에폭(192 update)** mean-asset IoU:

| 구성 | IoU |
|---|---:|
| V1 | 0.46744735 |
| Top1+R | 0.46088226 |
| 기존 Full | 0.46459021 |
| C_RAW | 0.46426591 |
| C_SPATIAL | 0.46642605 |
| C_D1_COS | 0.46172169 |
| C_DECODER_NO_R | 0.46752463 |

현재 No-R은 V1과 거의 같다. 이 중간 결과로 retargeter가 불필요하다고 결론 내리지 않으며,
Top1+R 대 V1의 최종 조합 이득을 retargeter 단독 효과의 증명으로 확대하지도 않는다.
동일480 update의 최종128×3 비교가 필요하다. 새 E 두 모델의17개 검증·학습·평가 작업은
여전히 GPU 대기 중이며, 이번 준비 작업은 추가 NN 학습을 큐에 넣지 않았다.

## 전달 강도에 관한 추가 소스·수치 확인

완료된 Full의 실제480 update 기록에서 `g=tanh(fusion_gate)=0.04450051`,
Top1+R은0.04375174였다. 전달식 `native MV4 + sum Proj[u*g*(E-S)+u*R]/4`에서
`u`는0~1이므로 직접 source 대체 계수는 원래 단일 source 항 대비 각각 최대4.45%,4.38%다.

이는 **전체 feature 보정 크기의 상한이 아니다**. R에는 global fusion gate가 곱해지지 않는다.
특히 기존 Full에서는 직접 carrier가 Top1이고 attention feature는 R로 들어간다.
따라서 이 scalar만으로 attention이 죽었다거나 성능 정체의 원인이라고 결론 내릴 수 없다.
`native-source-carrier-scalar-audit-v178`에 실제 closed input hash 및 수식을 함께 기록하고
file4 SHA256/size를 확인했다.

새 E 모델도 개선이 없다면 다음 확인 대상은 실제 동일 forward의 `E-S`, R 및 native source의
전달 크기다. 그 값을 측정하기 전에 gate scale이나 matching 용량을 바꾸는 추가 학습 arm을
늘리지는 않는다. 이 감사에서는 새 NN forward나 추가 학습을 수행하지 않았다.

## 남은 완료 조건

현재 12개 arm과 이미 연결한 E 두 모델의 실제 학습/DEV 평가, 공통 geometry 및 실제 joint
SS+PBR 평가를 완료해야 한다. 결과에 따라 실제 후보를 선택하고, 필요한 seed202 학습을
활성화해 독립 재현을 확인해야 한다. 두 번의 실제 validation36 load/tail4 검사도 아직 미실행이다.
최종 selected-pipeline freeze builder/활성화, freeze 후 TEST508의10개 metric 및 두 original
zero-shot control 실행 역시 남아 있다. 새 validator나 template 준비를 실제 평가 완료로 간주하지 않는다.
