# 모듈 대조군 480 update 및 진행 상황

기준 시각: 2026-10-23 23:58 UTC. 실행 제어 구현 커밋 `50ee49ca0006c4ac24be43104aff6312575ebb57`.

## 같은 DEV32의 5에폭 결과

다음은 같은 sample ID·생성 noise·GT hash를 사용하는 asset별 IoU 평균이다. 학습 seed 1개, 생성 noise 1개인 중간 패널이며, DEV128×3 및 독립 학습 seed 확인을 대체하지 않는다.

| 모델 | 480 update IoU |
|---|---:|
| Vanilla V1 | 0.47327916 |
| Decoder correspondence, retargeting 제거 | 0.47308540 |
| Raw semantic Top-1 + retargeting | 0.47390104 |
| Decoder feature cosine Top-1 + retargeting | 0.47409496 |
| Matching decoder Top-1 + retargeting | 0.47696710 |
| 기존 Full | 0.47824306 |

Matching decoder Top-1+R와 동일 모델의 NoR 차이는 +0.00388170이다. 이번 패널에서는 retargeting의 조건부 기여를 지지한다. Decoder correspondence+R는 raw correspondence+R보다 +0.00306607이다. 이 역시 같은 32개 패널의 결과이므로 전체 평가 및 독립 seed 검증 전에 모듈의 필요성이나 일반적 우위를 확정하지 않는다.

Full−Top-1+R는 +0.00127596이나, Top-K·readout 등을 함께 바꾼 조합 차이다. 이를 학습형 attention 단독 이득으로 해석하지 않는다. 이미 완료한 DEV128×3에서는 Full 0.47207818, Top-1+R 0.47260025, Fixed Top-K 0.47274517이었다. 해당 큰 패널에서 학습형 attention의 추가 이득은 아직 입증되지 않았다.

## 실제 실행 상태

NoR와 raw semantic 학습은 실제 queue return code 0 및 native 출력 저장을 확인했다. Decoder cosine은 native 학습·최종 validation 480/480 완료 후 wrapper 저장 마무리 중이다. Spatial 대조군은 472/480, GPU 6의 additive matching prior 제거 대조군은 340/480, GPU 7의 개별 voxel query 대조군은 110/480이다. SS global/micro batch 96, accumulation 1을 유지한다.

확인한 최근 native 로그에는 CUDA OOM, traceback, nan loss 표식이 없다. GPU 3에서는 완료 후 기존 SS→원래 Shape→공통 trained V1 PBR 평가 case를 실행 중이다. GPU 0~3의 승격된 SS 작업이 실제 종료·저장되면 기존 resource guard가 raw Top-K PBR 학습에 네 GPU를 모으도록 연결되어 있다. Raw PBR은 아직 대기 중이며, 실행 중이라고 보고하지 않는다. GPU 4·5는 사용하지 않는다.

## 대기 중 MeanMatch 배치 개선

새로운 학습 arm을 추가하지 않고, 기존 MeanMatch proof/capacity/science 세 작업을 raw PBR의 실제 종료 후 GPU 0에서 실행하도록 연결했다. GPU 6·7에서 긴 voxel-query 대조군과 후속 작업 때문에 기다리는 시간을 줄이기 위한 배치 변경이다. 원래 MeanMatch numerical/data/seed/batch/480 설정은 유지하고 GPU·CPU affinity·실행 경로만 바꿨다. 기존 네 pending auxiliary descriptor만 실제 완료 확인용 CPU bridge와 seal로 바꾸었고, main queue에는 세 작업을 append했다. 기존 final128×3 및 twelve-arm summary 의존성은 유지했다.

검증한 dispatcher의 루프 끝 sleep 경계에서 dispatcher PID만 잠시 정지하고 재개했다. 정상 neural worker를 정지하거나 재시작하지 않았으며 두 dispatcher의 PID/birth를 유지했다. 실제 정지 시간은 main 약 3.56초, auxiliary 약 0.22초다. watchdog·실패 시 descriptor rollback·실제 완료 bridge를 포함한 관련 CPU 테스트 33개가 통과했다. Neural 계산 코드는 수정하지 않았다.

## 다음 판단에 필요한 결과

현재 남은 대조군의 480 update와 DEV128×3를 완료해 spatial→raw→decoder correspondence, retargeting, query 단위, score prior, Top-K 결합 방식을 같은 기준에서 비교한다. 병행 중인 실제 SS→Shape→PBR 평가에서는 source preservation 및 target alignment metric을 함께 비교한다. 공통 evidence를 retargeting과 attention에 일관되게 전달하는 두 후속 후보도 기존 queue에 대기 중이며, 아직 성능 결과가 없다. 추가 지엽적 arm을 만들지는 않았다.

최종 후보 선택, 독립 학습 seed 확인, VAL fresh-load, 설정 동결, TEST508 전체 cascade 및 10개 metric 평가는 아직 완료되지 않았다. 전체 목표는 진행 중이다.

실제 file4 SHA256/size 검증 증거는 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `focused-status-v183/`, `focused-status-v183-late/`, `mean-matching-post-raw-PBR-GPU0-v183/verification-v1/`에 보관했다.
