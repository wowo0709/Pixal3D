# 선택 후보의 독립 seed 실행 경로와 3에폭 중간 결과

기준 시각: 2026-10-23 22:13 UTC. 코드 커밋: `86cab6b`.

## 실제 진행 상태

GPU 0·1·2·3·6·7에서 SS 학습 6개가 계속 실행 중이다. 실제 wrapper와 NN child의 PID/birth/command를 확인했고, 현재 계획에 고정된 소스 308개 파일의 SHA256이 일치했다. 기존 학습을 수정하거나 재시작하지 않았다. 현재 로그에 CUDA OOM, traceback, nan loss 표시는 없다.

| GPU | Arm | 실제 update / 480 |
|---|---|---:|
| 0 | C_RAW: raw semantic Top-1 + R | 318 |
| 1 | C_SPATIAL: native spatial Top-1 + R | 309 |
| 2 | C_D1_COS: frozen decoder feature cosine Top-1 + R | 318 |
| 3 | C_DECODER_NO_R: decoder correspondence, R 제거 | 327 |
| 6 | FULL_QK_ONLY: attention에서 additive matching prior 제거 | 216 |
| 7 | FULL_VOXEL_Q: 개별 voxel query | 68 |

모두 world1/global96/micro96/accumulation1이다. 원래 ray 단위 query와 voxel query의 최종 품질 차이는 아직 나오지 않았다. GPU 7의 voxel query가 느리다는 사실만으로 품질 효과를 주장하지 않는다.

## 새로 확인한 3에폭 결과

동일한 DEV32 asset, noise, GT hash의 행 순서까지 대조했다. 아래는 **asset별 IoU 평균**이며, 전체 voxel count를 합친 pooled IoU와 구분한다. 128개 × 생성 noise 3개 최종 결과와 직접 섞지 않는다.

| Arm | 192 step | 288 step |
|---|---:|---:|
| Vanilla V1 | 0.46744735 | 0.46897292 |
| 기존 Full: decoder Top-K attention + R | 0.46459021 | 0.46649383 |
| Raw semantic Top-1 + R | 0.46426591 | **0.47073808** |
| Native spatial Top-1 + R | 0.46642605 | 0.46731529 |
| Decoder feature cosine Top-1 + R | 0.46172169 | 0.46703344 |
| Decoder correspondence, R 제거 | 0.46752463 | 0.46924330 |

Raw semantic은 288 step에서 V1 대비 +0.00176516, spatial 대비 +0.00342279다. 후보 탐색 자체가 무효라는 결론을 내릴 상황은 아니다. 그러나 32개·한 noise의 중간 패널이므로 semantic matching의 최종 효과를 입증했다고 쓰지 않는다. R 제거 arm도 V1과 가까워 R의 독립적인 필요성은 최종 대조가 필요하다.

기존 최종 DEV128×3에서는 V1 0.46735504, Top-1+R 0.47260025, Full 0.47207818, 고정 Top-K 조합 0.47274517이다. 세 생성 noise는 독립 학습 seed 세 개가 아니다. 학습형 attention의 추가 이득은 아직 입증되지 않았다.

## 이번에 완료한 구현

`selected_confirmation_runtime.py`는 14개 물리 모델을 기존 6개 native runtime으로 분기한다. 기존 proof/native_training/conditioner/flow loss/optimizer를 그대로 호출한다. 새 observer는 학습 연산을 추가하지 않으며 다음을 검사한다.

1. 실제 training seed는 정수 202다. 기존 seed101 설정에서 모델·데이터·LoRA·FFN chunk·학습률·학습 예산은 바뀌지 않는다. 명시한 GPU/CPU 배치 메타데이터만 별도로 기록할 수 있다.
2. 자신의 GPU cold field/negative field/velocity/loss/backward proof와 micro96 3-update capacity가 있어야 한다. 최초 source/LoRA 가중치가 서로 맞아야 하며 capacity가 업데이트한 가중치로 본 학습을 시작할 수 없다.
3. 자신의 capacity 3개 실제 asset/noise/timestep/global trace가 같은 seed의 V1과 일치해야 한다. 최초 native32 정합 검사와 frozen base 검사가 필요하다.
4. 실제 science 시작 시점의 LoRA·RNG·TRAIN 순서가 V1과 같고, 첫 5개 학습 입력도 일치해야 한다.
5. 480 update 종료 시 frozen parameter/buffer가 그대로이고, 원래 LoRA 360개와 자신의 source parameter optimizer state가 모두 480 update를 완료해야 한다.
6. shared-evidence 후보는 native runtime의 기존 검사와 observer가 같은 V1 prefix와 자신의 proof/capacity를 읽어야 한다. identity V, 동일 evidence의 retargeting 전달, 초기 prior 정합도 검사한다.

`prepare_selected_confirmation.py`는 실제 살아 있는 V1 child의 PID/birth/command 또는 실제 종료 기록을 확인한 뒤 첫 5개 입력만 고정 복사한다. 계속 늘어나는 전체 로그를 닫힌 파일로 오인하지 않는다. 가중치는 복사하지 않는다. 이후 실제 file4 SHA256/size receipt가 있는 proof/capacity/prefix만 사용해 admission plan을 만든다. GPU runtime은 아직 file4 검증이 없는 plan을 거부한다.

머신 외부 작업 경로에는 child 소유·종료·session drain·디스크 예약·file4 검증을 수행하는 `run_selected_confirmation_stage_GPU_v180.py`를 준비했다. 아직 선택 후보가 없으므로 이 worker를 큐에 넣거나 GPU에서 실행하지 않았다. 기존 V1/Top-1+R seed202 대기 작업을 중복 생성하지 않았다.

## 검증 범위

- 관련 CPU 테스트 **297개 통과**: 14개 × 3 phase 설정, 잘못된 초기 가중치·noise·reference·receipt 차단, live process 식별, 계속 append되는 prefix 복사, observer 복구와 native 반환값 보존 등. CPU neural forward는 수행하지 않았다.
- 두 public CLI의 `--help` 통과. private owned stage worker는 syntax 검사까지 수행했다.
- 실제 file4에 닫힌 **11개 seed101 모델의 proof/capacity/trace**를 새 검사로 읽었다. 완료된 archive는 실제 startup도 검증했다.
- shared-evidence 두 후보의 seed202 설정 **6개는 설정 호환성만** 확인했다. 실제 seed202/E GPU 결과를 읽거나 만들어 낸 것이 아니다.
- 준비물, 실제 검사 결과, 진행 snapshot은 서로 다른 새 폴더에 보관하고 실제 file4 SHA256/size를 대조했다.

## 다음 실행과 완료 조건

현재 6개 arm의 480 update 및 같은 DEV128×3 최종 평가를 유지한다. 이미 연결한 shared-evidence 고정/학습형 두 후보도 자신의 GPU 초기 정합·용량 검사를 거친 뒤 실행한다. 기존 후보가 Top-1 appearance를 전달하고 다른 Top-K evidence를 R에 전달하던 경로와 달리, 이 두 후보는 하나의 evidence를 carrier/use/R에 일관되게 전달한다. 효과를 가정하지 않고 대조한다.

그 결과와 실제 SS→Shape→PBR joint 평가로 후보를 정한 뒤, 필요한 후보 한 개만 seed202로 추가 학습한다. Vanilla/Top-1 기준선의 기존 seed202 작업은 재사용한다. PBR의 독립 seed 및 source/target rendering 평가, 선택 후보의 두 번 fresh-load VAL36, DEV 기반 설정 동결, 이후 TEST508 전체 cascade와 10개 metric 검증이 남아 있다. 이번 구현·테스트를 그 학습이나 최종 성능 검증의 완료로 간주하지 않는다.

File4 증거 루트는 기존 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래의 다음 폴더다.

- `focused-status-v180/`: 실제 여섯 NN 진행 상태와 paired interim 패널.
- `selected-confirmation-runtime-actual-preflight-final-v180/`: 실제 기존 11개 모델 및 설정 6개 검사.
- `selected-confirmation-runtime-preparation-v180/`: 소스 hash, 검사 기록, 미실행 private worker.

새 GPU 학습 작업 수 0, 기존 정상 NN 재시작 수 0, 선택 모델 없음, 독립 seed 검증 미완료, TEST model forward 미실행. 전체 실험 목표는 계속 진행 중이다.
