# 선택한 물리 모델의 최종 cascade 연결 준비

기준 시각: 2026-10-24 01:24 UTC. 구현 커밋 `e1b2f3b4f35ba90814de08ce1f02971e3093bea8`.

## 실제 학습 상태

실제 PID·birth·command를 이전의 정상 실행 handle과 대조했다. Main/aux dispatcher, raw PBR resource coalescer, 세 학습 wrapper 및 NN이 모두 같은 handle로 실행 중이다. 기존 NN을 재시작하지 않았다.

| GPU | 실험 | 실제 progress |
|---|---|---:|
| 0–3 | Raw semantic Top-K + feature retargeting PBR, DDP4 | 257/1900 |
| 6 | Full에서 additive attention matching-score prior만 제거 | 442/480 |
| 7 | Ray-wise query 대신 개별 voxel query | 143/480 |

PBR은 global24/micro6/accumulation1, 두 SS는 global96/micro96/accumulation1이다. 실제 실행 source 325개와 science config 3개의 hash가 원래 pin과 일치한다. 현재 세 NN 로그에는 OOM·Traceback·NCCL error marker가 없다. 마지막 완료 validation은 각각 PBR step0, QK-only step384, voxel-query step96이다. 이번 작업에서 새 완료 epoch나 최종 품질 결과가 추가된 것은 아니다.

앞서 관측한 같은 DEV32·384 update·noise1의 QK-only 0.47717633 대 기존 Full 0.47481289는 유지된다. 후보 ranking의 semantic matching score는 유지하고 attention logit의 additive prior만 제거한 비교다. 이 결과를 최종 DEV128×3나 독립 학습 seed의 개선으로 해석하지 않는다. 완료된 DEV128×3의 Full 0.47207818, Top-1+R 0.47260025, Fixed Top-K 0.47274517 비교도 변하지 않았다.

## 이번에 완료한 구현

기존 TEST 실행 경로는 고정된 native 네 arm과 original zero 두 arm, 과거 parent schema를 전제로 했다. 새 selected-pipeline V2 freeze가 고른 실제 SS/PBR 조합 및 두 강한 기준선은 중복 제거 후 2–3개이며, original zero 두 입력 조건을 합치면 4–5개다. 다른 구조를 기존 Full의 이름으로 바꾸지 않고 실제 binding을 사용하는 consumer를 추가했다.

`selected_final_population.py`는 실제 DEV-only freeze, 명시적으로 선택한 SS/PBR pair, 일곱 development evidence role, 원본 네 stage 및 전체508/134 source population을 확인한다. 두 original zero RGB4 control과 후보·기준선을 합치고 각 모델의 실제 full508 SS archive, sampler, noise, source cluster, GT hash와 occupancy counts를 검증한다. TEST용 상태를 DEV32 prefix로 대신하지 않는다.

`selected_final_case_plans.py`는 실제 SS archive를 원본 Shape512/Shape1024에 연결하고, 같은 native Shape guide를 사용해 선택된 native1900 PBR를 replay할 per-sample plan을 만든다. 기존 cascade/texture paired seed, sampler template, camera8/9/10 및 original-material M1 shared reference를 유지한다. GPU와 canonical output root를 명시적으로 받는다. 같은 물리 SS를 쓰는 두 조합은 동일한 actual SS spec·prediction hash와 canonical geometry owner를 선언해야 하며, 별도 Shape 생성으로 대체하지 않는다.

`selected_original_zero_cascade_population.py`는 새 parent schema에 맞는 metadata callback과 명시한 GPU membership만 바꾸고 original zero numerical entry를 실행한다. 실제 original Shape decoder가 반환한 네 guide의 좌표·logit·순서·scale을 기존 capture context로 보관한다. Visibility gate, GT conditioning, 학습 update나 sampler 변경을 추가하지 않는다. 실제 GPU 실행은 아직 수행하지 않았다.

`selected_final_case_integrity.py`는 임의의 실제 SS/PBR pair의 종료 결과를 검증한다. 자기 SS prediction, original Shape, inline original PBR, 최종 native1900 PBR checkpoint 및 mode, 모든 geometry field의 byte-identity 기록, paired texture seed, native guide capture/reuse, post-generation GT surface join, camera와 shared reference를 확인한다. Geometry reuse에는 같은 실제 SS/weights/source/sample의 완료된 owner case와 성공한 실제 child terminal 기록이 필요하다. 이 validator 자체가 process 실행이나 byte/receipt 읽기를 대신하는 것은 아니므로, 향후 worker가 실제 closed files와 종료 handle을 읽어 공급해야 한다.

## 실제 원본 가중치 경로 정합

기존 zero SS evaluator는 `/root/dev/.cache/deformnext-retargeting/checkpoints/denoiser_step0004210.pt`를 실제로 사용해 왔다. 새 freeze의 original stage 경로 규칙과 이 실제 사용 경로가 달라, 이름만 바꾸면 native loader config와의 정합이 깨질 수 있었다.

파일서버4의 사용자 지정 원본 SS/PBR checkpoint를 실제로 hash하고 기존 사용 캐시와 대조했다. SS4210은 `3b1f2e764dd1113ef9a9466a817712389febfac23b8d9c17496cfeea20356c7a`, PBR16850은 `703cb2dfc6aa1bd457053232df65a583ad028e1a9f888e20158609ff07dd9dc2`로 각각 일치했다. 빠진 canonical cache의 두 original config를 파일서버에서 가져왔고, checkpoint는 동일 filesystem의 기존 파일에 hard link를 만들어 추가 다운로드와 중복 데이터 복사를 피했다. 기존 두 Shape stage도 실제 config/weight hash를 확인했다.

Zero 모델 binding은 실제로 사용한 SS checkpoint copy와 native loader config를 그대로 유지한다. 원본과 실제 사용 copy의 lineage를 함께 기록하고 두 경로의 실제 checkpoint bytes가 같은 경우에만 허용한다. 이미 닫힌 original control freeze, 기존 NN config 또는 과거 evaluation 기록을 수정하지 않았다.

증거는 file4의 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/selected-original-stage-byte-preflight-v187/` 및 `selected-final-composition-status-v187/`에 저장했다. 두 audit의 실제 file4 SHA256/size closure를 확인했다. 전자는 원본 stage와 copy lineage의 byte 검증이며, 모델 선택이나 TEST forward의 증거가 아니다.

## 검증과 남은 조건

새 CPU 검사 305개와 관련 기존 검사 97개, 총402개가 통과했다. 모든14 SS family ×4 PBR mode의 metadata 계획과 최종 보고서 계약, 두 zero control, 잘못된 checkpoint copy/lineage, 불완전508 archive, 다른 SS/noise/GT, guide 재계산과 누락, 실패한 child, 잘못된 geometry reuse owner와 카메라 등을 검사했다. 모두 synthetic test 자료 또는 metadata/byte 검증이며 새 GPU inference 결과가 아니다.

이번 작업은 새 학습 arm이나 GPU/CPU queue job을 추가하지 않았다. 기존 QK-only·voxel-query·MeanMatch 및 shared-evidence 두 후보, 실제 joint32·추가 seed·validation36 작업은 기존 dependency로 계속 진행한다.

현재 최종 모델은 아직 선택·동결하지 않았다. 새 contract와 plan을 실제 child/session ownership, runtime plan closure, geometry group의 모든 replay consumer 종료, bounded local raw retention 및 file4 byte 검증에 연결할 일반 worker/campaign이 필요하다. 기존 고정6모델 raw retirement와 metric activation도 새 4–5개 물리 composition에 연결해야 한다. 이후 실제 DEV 평가를 통한 선택, 독립 seed 확인, validation36 두 fresh-load 및 V2 freeze를 거쳐 TEST508 full cascade와 LPIPS/DreamSim/CLIP/PSNR/SSIM/M-IoU/CD/F-Score/FID/KID를 모두 실행해야 한다. 전체 실험 목표는 진행 중이다.
