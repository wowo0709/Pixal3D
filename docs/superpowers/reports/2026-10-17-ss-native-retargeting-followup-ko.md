# SS native projection / retargeting 후속 학습

## 진행 근거와 원래 gate의 범위

기존 raw-transport-only B2-v2/B2-v2-R/B3-v2는 GT transport gate에서 계속 HOLD한다. 해당 gate 파일을 PASS로 덮어쓰지 않는다. 반면 공통 128개에서 **이미 학습된 C1이 V1보다 +1.218%p**, source-cluster bootstrap CI [+0.460,+2.013]%p였으므로 trainable retargeter가 포함된 경로 자체에는 양의 근거가 있다. Frozen V1에 raw feature만 더한 진단의 실패가 trainable C1의 실패를 뜻하지 않는다. 따라서 다음은 raw matching ladder와 구분한 **retargeting/interface 개선 실험**이다.

Calibration32에서 cached target self-map의 native anchor 대비 상대 RMSE는 nearest 약 6.9%, native bilinear 약 0.6%였다. GT 주소를 쓰는 V1 진단에서도 nearest IoU 0.421874 → bilinear 0.424428 → support-normalized bilinear 0.425286으로 개선됐다. Native 0.426813에는 못 미쳤다. Bilinear wrong-object는 0.402297이어서 correct source value와의 차이는 남는다. 이 결과를 SS 본 성능 개선으로 승격하지 않고, 배포 경로의 interpolation/support를 맞춘 상태에서 retargeter를 다시 학습해 검증한다.

## Arm과 순서

| Arm | Matcher / attention | Value consumer | 목적 |
| --- | --- | --- | --- |
| 기존 C1 | 기존 R9 / Top64 / T0.005 | nearest patch | 이미 저장된 5-epoch 비교 기준, 재시작하지 않음 |
| P1-native-grid | 기존 C1과 동일 | native bilinear + support normalization + C1 adapter | Projection/interface를 맞춘 재학습이 유용한지 |
| P2-matcher-grid | 7-epoch H1/H2/H3 중 calibration에서 고정한 H* / Top256 / 고정 T | P1과 같은 C1 adapter와 projection | 개선된 matcher bundle이 같은 consumer에서 유용한지 |
| 후속 C2 | 채택할 P 계열과 동일 | 동일 + 물리적 RAW2048 deformation query | Parent가 확인된 뒤 query 추가의 효과 |

P2에서는 NULL rejection을 끈다. Confidence calibration/precision-coverage는 별도로 기록하며 untrained confidence를 사용하지 않는다. 첫 비교에 projection, matcher, rejection, query의 변경을 동시에 섞지 않는다. 필요 시 P2와 같은 parent에서 rejection을 추가한 별도 arm을 정의한다. C2는 adapter+query의 순차 추가이고 retargeter/query 2×2가 아니다. Parent 선택과 query 실행 여부는 P 계열 결과와 함께 명시한다.

P1/P2는 모두 원본 mixed385 SS base+LoRA checkpoint에서 독립적으로 시작한다. 기존 C1의 denoiser/adapter를 continuation하지 않는다. Dataset, Omega epoch7, LoRA rank/LR, conditioner LR, optimizer, scheduler와 objective는 기존 SS recipe를 유지한다. Global96, 5 epochs/480 updates, epoch마다 같은 32개, final 동일 128개와 source-cluster paired 비교를 사용한다. Gate LR이나 output scale은 바꾸지 않는다.

## 실행 검증과 자원

- GPU 0~3만 사용하고, run당 GPU 한 장을 기본으로 한다. 본 학습은 GPU를 독점한다. 남는 GPU는 head/cache/평가에 병렬 사용한다.
- 실제 checkpoint에서 초기 native condition 최대 오차 0과 gate/adapter의 실제 backward를 확인한다. Native bilinear에서 rejection은 adapter를 포함한 전체 update에 적용하며 support가 0이면 native로 돌아간다.
- 각 run은 96 microbatch의 실제 2-update memory probe부터 시작한다. OOM인 경우 global96을 유지하며 가능한 최대 divisor로 줄인다. 실제 microbatch가 기존 C1의 48×2와 달라지면 stochastic training nuisance도 결과에 적는다.
- P1의 기존 R9 attention도 SS batch에서 K64 값을 펼치지 않도록 사전 집계한다. Weighted evidence는 FP32로 유지하고 원래 R9 온라인 출력과 별도 감사한다. Frozen scorer 제거에 따른 RNG 소비 차이도 보존해 adapter 초기 weight가 R9 recipe와 동일하도록 검증했다. P2도 같은 초기화와 FP32 weighted evidence를 사용한다.
- P2는 frozen H*가 확정되고 별도 namespace의 inference-only cache parity가 통과한 뒤 실행한다. GT address/visibility teacher를 배포 cache나 SS 입력에 넣지 않는다.
- 기존 중단한 SS C2/PBR B3는 재개하지 않고 checkpoint를 보존한다.

새 학습/검증 위치는 file4의 `rgb_matching_v2/retarget-native-v3/`다. Gate 또는 pilot 통과를 성능 향상으로 표현하지 않으며, final 생성 지표와 이미지가 개선돼야 채택한다. 기존 4개 quick panel은 최종 판단 기준으로 사용하지 않는다.
