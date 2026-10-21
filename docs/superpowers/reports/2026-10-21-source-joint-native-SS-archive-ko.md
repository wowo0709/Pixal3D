# 새 source-input 모델의 SS 출력과 cascade 연결

2026-10-21 UTC. 새 Original19/SourceContrast19 joint480 학습과 그 GPU 평가는
아직 실행하지 않았다. [Warm native 검증](2026-10-21-native-warm-source-proof-protocol-ko.md)과
72개 확인을 통과한 후 사용할 export 경로를 준비했다.

## Cascade가 받는 것은 실제 SS occupancy

기존 `cascade_from_ss_artifact.py`는 SS conditioner를 다시 만들지 않는다.
저장된 prediction occupancy를 읽고 원래 `native_shape_support`의 64→32 max-pooling으로
Shape512의 sparse 좌표를 만든다. 이후 기존 Shape512/Shape1024와 지정한 PBR
checkpoint를 사용한다. 따라서 새 conditioner의 구성·strict load는 SS 평가에서
보장하고, cascade에는 그 평가가 실제로 만든 occupancy를 전달해야 한다.

## 한 번의 SS128 평가에서 저장

`evaluate_source_input_joint_archive.py`는 원래 `evaluate_source_input_joint`를 실행하면서
기존 native decode capture를 사용해 128개 결과를 저장한다. Sampler는 batch8,
12 steps, CFG3이고 두 번째 inference를 수행하지 않는다. 동일한 완료480 run의
denoiser/source와 실제 checkpoint 로딩 검사는 기존 joint evaluator가 수행한다.

각 asset의 prediction/GT latent, packed SS64 occupancy, 실제 voxel grid hash,
GT/noise identity와 TP/FP/FN을 함께 기록한다. 원래 core evaluation JSON도 보존한다.
Prediction은 생성 입력이고, GT SS는 평가용이다. 이 exporter는 개발 validation128에
한정하며 final-test508 평가 wrapper를 대신하지 않는다.

## Cascade용 DEV32는 CPU에서 추출

`select_source_input_ss_archive.py`는 사전에 지정한 32개 panel과 seed map을 받아
128개 중 해당 artifact를 **byte 그대로 복사**한다. 파일·prediction·GT grid hash,
sample/seed/shape, 실제 occupancy counts를 검사한다. 순서는 지정한 panel을 따른다.
32개 지표는 실제 grid로 다시 계산하며 128개 aggregate를 재사용하지 않는다.

빈 prediction도 보존하고, 생성 입력을 GT 좌표로 대체하거나 asset을 제외하지 않는다.
원래 Shape 입력용 SS32 support와 좌표 hash도 기록한다. 이 결과는 새 inference나
재생성 결과가 아닌, 기존 native128 출력의 CPU subset임을 별도 status로 표시한다.
모델·decoder를 다시 로드하거나 GPU 추론을 반복할 필요가 없다.

## 완료된 검증과 남은 검증

- 새 archive/subset 검사 9개, 기존 decode capture 검사 6개, completed-run binding
  검사 9개를 합쳐 **CPU 24개 검사 통과**했다. Synthetic fixture는 새 모델 예측이 아니다.
- 이전 bounded-prior SS480에서 실제 생성·저장한 32개를 읽어 새 reader의 파일/grid
  hash·counts와 기존 `native_SS32_support`의 전체 좌표 hash·metric이 일치함을 확인했다.
  원본 archive를 수정하거나 새 SourceContrast 결과로 바꾸어 표기하지 않았다.
- 새 모델의 실제 joint480 학습, native128 GPU export와 이 artifact를 이용한 cascade
  생성은 아직 실행하지 않았다. 현재 진행 중인 PBR과 기존 queued job은 변경하지 않았다.
- 최종508은 기존 freeze/cache-QA/identity-disjoint guards를 유지해야 한다. 기존 bound
  population evaluator의 허용 module 목록에는 새 joint evaluator가 아직 없으므로,
  실제 후보 선정 후 별도 연결이 필요하다. 이 작업으로 final508 준비가 끝났다고 하지 않는다.

Canonical 검증 결과는 campaign `feature-readout-v1/` 아래
`source-feature-teacher-pilot-v1/native-joint-archive-readiness-v1/`에 저장했고,
`file4-closure.json`으로 실제 원격 SHA256/size를 확인했다.
