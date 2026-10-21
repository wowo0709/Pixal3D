# Source adaptation 여력의 부분 RGB oracle 진단

2026-10-21. 실제 RGB/카메라 정합과 CPU 입력 어댑터 검증을 마쳤으며,
아래 GPU 비교를 준비한다. 새 학습 arm, feature teacher loss 또는 최종 모델은 아니다.

## 질문과 고정 조건

현재 attention의 후보 분산을 바꾸는 것만으로는 별도72개에서 생성 품질
이득이 확인되지 않았다. 추가 학습을 늘리기 전에 source 입력 불일치를
줄였을 때 원본 multi-view Pixal3D가 개선되는지 확인한다.

동일한 native SS64 step4210 checkpoint와 고정32개를 사용한다. Deformation
fine-tuning, conditioner, Omega 또는 matching cache는 사용하지 않는다.
Sampling12/CFG3, generation batch8, asset별 noise seed, 카메라, mesh scale와
GT SS latent는 기존 original-zero edited-RGB 평가와 같다.

| Camera | 원래 edited-RGB 입력 | 부분 oracle 입력 |
|---|---|---|
| 8 | M1 target RGB | 같은 M1 target RGB |
| 9 | M0 source RGB | 실제 M1 source RGB, 추가 GT 입력 |
| 10 | M0 source RGB | 실제 M1 source RGB, 추가 GT 입력 |
| 11 | M0 source RGB | 같은 M0 source RGB |

Camera11의 M1 RGB는 없다. 따라서 full four-M1 oracle이나 엄밀한 성능
상한이 아니다. 추가 정답 입력을 사용하므로 competitive 모델 결과, semantic
matching의 효과 또는 retargeting module의 학습 이득으로 보고하지 않는다.

## 실제 입력 정합

기존 quick32 전체를 그대로 선택했다. 파일서버4에서 raw M1 camera8 및
M0 camera9·10이 현재 export와 같은지96쌍의 byte 비교를 통과했다. 추가 M1
camera9·10을 포함한192개 파일 hash와 로컬 캐시가 일치했다. 필요한 파일은
이미 캐시에 있었고 추가 다운로드·렌더링은 없었다.

원래 native RGBA loader로32개 입력을 읽어 camera8과11 텐서의 SHA256이
교체 전후 정확히 같음을 확인했다. 새 입력은 실제 M1 camera9·10 RGB다.
CPU contact sheet도 확인했다. 카메라·scale·scoring target 등 다른 필드는
원래 dataset 객체를 유지한다. Stale transport tensor가 들어오면 실패한다.

두 CPU 테스트는 source 역할 교체, 원본 tensor 불변, camera/GT tensor 보존,
잘못된 camera 교체와 NaN 입력 및 transport 혼입 거부를 검증한다.
Native GPU 경로의 성공과32개 GT/noise hash 일치는 실제 실행 후 별도로 확인한다.

## 실행 및 해석

기존 Top-1의128×3 및 wrong-value 평가를 먼저 완료하도록 의존성을 둔다.
기존 admission lease가 열려 있을 때 GPU0~3 중 한 장에서 실행하며 PBR
4-GPU 전환을 위한 drain을 유지한다. 원래128개 zero-shot 평가의 첫32개
결과를 baseline으로 재사용한다. 예전 frame 오류가 있던 native-v2 oracle은
baseline으로 재사용하지 않는다.

두 입력 조건의 pooled voxel IoU와 source-cluster paired bootstrap을 비교한다.
32개 GT hash와 noise seed가 같고, 실제 입력 tensor가 선언한 두 view만 바꾼
것인지 검사한 뒤에 집계한다. Generic evaluator의 기본 M0 label은 oracle
annotation으로 바로잡고 원본 metadata도 보존한다.

개선되면 source feature adaptation이 모델에 유용할 가능성을 지지하지만,
그 feature를 input만으로 예측할 수 있다는 증명은 아니다. 개선이 없으면
부분 source 입력 개선만으로 충분하다고 볼 수 없고, native prior의 deformation
적응 및 source11 충돌 등의 대안 설명을 남긴다. 어느 결과든 즉시 새 teacher
학습을 여러 개 추가하지 않는다. 기존 semantic/Top-1/spatial/DQ/bounded 최종
대조와 과거 target-patch/R96 재현 결과에 함께 근거해 다음 수정을 정한다.

데이터 근거는 canonical result root의
`failure-driven-diagnostics-v1/partial-M1-source-RGB-oracle32-v1/data-preflight-v1`에
저장했다. 실제 GPU 품질 측정은 아직 완료되지 않았다.
