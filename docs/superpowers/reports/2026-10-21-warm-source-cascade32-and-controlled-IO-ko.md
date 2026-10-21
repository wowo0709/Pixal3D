# Warm source-input SS의 native cascade 및 역할별 평가 연결

2026-10-21 UTC. 기존 PBR Direct 학습은 GPU 0–3, world4/global24/micro6/
accumulation1을 유지한다. 새 warm SS 두 모델은 source-disjoint72 검증과 native
초기화·메모리 검증을 통과한 경우에만 학습한다. 아래는 그 실제 SS480 결과에
의존하는 평가 연결이며, 새 모델의 성능 개선을 관측했다는 결과 보고가 아니다.

## 실험이 답할 질문

앞선 frozen SS prototype에서 Contrast19−Original19의 voxel IoU 차이는
+0.002245, source-cluster bootstrap CI95는 [+0.000564, +0.004073]이었다.
32개·sampling noise 하나·학습 seed 하나의 탐색 결과다. 반면 기존 cold SS480의
NoDQ−spatial/Top1 및 DQ 효과는 큰 개발 panel에서 확정되지 않았다. 따라서
source-relative 입력을 학습한 결과가 native joint SS와 최종 출력에 전달되는지
검증한다. 같은 matching/Top-K/attention 아래 입력을 비교하므로, 이 실험만으로
matching decoder·score·attention 각 모듈의 효과를 입증하지 않는다.

두 warm 모델은 teacher 학습 예산, SS 초기 checkpoint, train/validation split,
480 updates, batch96 및 validation·sampling 설정을 맞춘다. Source72 결과가
사전 기준을 통과하지 않으면 뒤의 새 SS 학습·cascade를 실행하지 않는다.

## 추가한 cascade와 비교 조건

정상 noise0 native SS128 추론에서 이미 생성한 지정32개의 SS prediction을
CPU에서 byte 그대로 추출한다. 각 warm 모델에 이32개를 사용하므로 새 cascade는
64개다. SS를 다시 sampling하거나 GT SS로 바꾸지 않는다. Shape512·Shape1024와
PBR checkpoint, sampler, RGB 입력, source normalization, camera, GT 및
surface metric 조건은 기존 개발32 평가와 동일하다.

이 비교의 PBR은 **기존 V1 step762**로 고정한다. 현재 진행 중인 PBR Direct의
최종1905 모델을 이 표에 섞지 않는다. SS 입력 변경이 geometry와 render에 미치는
효과를 비교하기 위한 조건이다. 최종 모델 조합의 평가는 별도로 남아 있다.

| 비교 | 답할 질문 |
|---|---|
| Warm Contrast19−Warm Original19 | 같은 teacher/native 학습 예산에서 source-relative 입력의 효과 |
| Warm Original19−cold Original19 (`feature-nodq`) | 추가 teacher 초기화를 포함한 차이 |
| 두 warm 모델−기존 V1 | 기존 기준 대비 변화; 개별 모듈 효과의 대조군은 아님 |

기존 V1/cold Original19의 지정32개 case를 재사용한다. 아직 생성 대기 중인
reference case에는 실제 producer dependency를 둔다. 새 두 모델과 reference
두 모델의 공식 DreamSim/CLIP pass는 총384개 view(4×32×3)를 평가한다.

Source preservation은 원래 material의 M1 GT source camera9/10에 대해
LPIPS·DreamSim·CLIP으로 비교한다. Target alignment는 camera8의 PSNR·SSIM·
M-IoU와 동일 normalization의 CD·F-Score로 비교한다. 기존 전체 view 집계와
역할별 재집계의 수치 일치를 확인하며, GT-only crop의 보조 결과도 유지한다.
자산·camera별 GT pixel hash가 모델 간 같아야 한다. FID/KID는 개발32의 대체
수치로 만들지 않고, 고정 후보의 최종 test508 공식 Inception 평가에 남겨 둔다.

## 원격 I/O 검증 의존성 수정

최초 연결에서 종료된 `prove-remote-native-cascade-render-parity-v1`을
의존성으로 지정한 오류를 발견했다. 이 검사는 별도 실행에서 각각 native LOD를
만든 결과를 비교해 실패했다. 해당 LOD는 실행 간 bit-level 결정성을 보장하지
않으므로, 이 차이를 원격 I/O가 바꾼 픽셀이라고 해석할 수 없다.

실제로 완료된 후속 검증 `diagnose-native-LOD-prove-controlled-remote-render-v2`
는 **동일한 native LOD geometry**를 사용해 로컬/원격 렌더의 세 camera 픽셀과
metric이 일치함을 확인했다. 성공 상태는
`PASS_ACTUAL_FROZEN_NATIVE_LOD_LOCAL_REMOTE_RENDER_IO_EXACT`다.
고정 LOD는 이 I/O 검증에만 사용했다. 새 scientific case는 원래처럼 prediction당
한 번 native simplification을 수행하고, raw geometry의 CD/F-Score를 유지한다.

잘못 연결된69개는 모두 미실행 HELD였고 native output도 없었다. 이 기록과 실패
상태를 보존한 채, 성공한 후속 검증을 사용하는 **새 이름·새 output의69개 후속 작업**을
추가했다. 기존 학습, queue state, dispatcher, keeper는 변경하지 않았다.
새 worker는 기존 worker에서 성공 proof를 확인하는 부분만 달라진다.

실제 연결 경로는 `source-feature-teacher-pilot-v1/native-joint-primary480-v1/`
`cascade32-controlled-IO-v2`다. Prelaunch269개 파일과 queue snapshot3개 파일을
file4에 복사하고 SHA256·size 일치를 확인한 후 연결했다. 연결 후 전체 queue는
1,827개이며, 새69개가 모두 pending이고 선행 작업380개에 FAILED/HELD가 없음을
확인했다. 이69개는 native case64·공식 metric GPU pass1·CPU 집계/로깅4이며,
새 학습 arm을 추가한 것은 아니다.

## 결과에 따른 판단

Native SS128×3 및 같은 weight의 projected-V 대응 순열 진단으로 먼저 voxel 성능과
전달 feature의 유용성을 확인한다. Cascade에서는 target과 source의 방향이 함께
개선되는지 본다. 단계별 개선이 최종 출력에서 사라지면 그 전달 단계를 조사하며,
성능을 보고 case를 제외하지 않는다. CI는 개발 과정의 반복 비교로 해석한다.

개별 matching decoder·score·correspondence 효과를 주장하려면 해당 대조군이
추가로 필요하다. No-score는 Top-1 correspondence feature, No-corr은 원래 spatial
correspondence, −DQ는 channel 자체 제거라는 정의를 유지한다. 저장 weight의
V 순열 진단이 이 학습 ablation들을 대신하지 않는다. 독립 joint seed202 확인,
최종 SS/PBR 조합, test508 cache QA·freeze·전체 metric도 완료 범위에 남아 있다.
