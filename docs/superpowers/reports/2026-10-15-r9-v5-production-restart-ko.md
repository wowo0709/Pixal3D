# R9 v5 데이터 구축 및 학습 재시작

기존 native V1은 재학습하지 않는다. 정합을 검증한 world-v4 DeformNeXt 렌더링/SS/Shape/PBR latent와 Omega epoch-7 cache를 재사용하고, 수정된 R9 matching teacher/head/cache 및 B1부터의 독립 fine-tuning을 새로 실행한다. 출력은 `runs_r9_v5`로 분리했다. 이전 semantic 실험의 hold는 유지한다.

## 데이터 구축 결과

| 산출물 | 수량 / 상태 |
| --- | --- |
| 전체 population | train 9,149 + validation 508 = 9,657 |
| B1 RAW FRAME Top-1 cache | 전체 9,657개 완료 |
| Topology-visible teacher | train 9,142 + validation 508 = 9,650개 인증 |
| 인증 실패 teacher | train 7개; 원래 raster 기준을 완화하지 않고 제외 사유 기록 |
| 본 SS 학습 population | 9,149개 유지 |
| 본 PBR 학습 population | 기존 32,768 token cap 이후 9,139개 유지 |
| D1 warm-up | 인증된 train teacher만 사용, fixed-final 3 epochs 실행 중 |

Teacher 실패 7개는 matching head의 supervision에서만 제외한다. 이 목록을 inference validity로 사용하거나 SS/PBR 학습 population을 변경하지 않는다. B1은 teacher/head 없이 RAW cosine Top-1만 사용한다. B2–C2는 같은 fixed-final D1 head와 hash가 일치하는 cache를 공유한다. Head warm-up 첫 epoch의 전체 508개 validation에서 Top-1은 RAW 0.3387615 → D1 0.4413975다. 분모는 visible query 36,722개이며, GLOBAL64 proposal recall은 0.9073852로 고정이다. 이 결과는 생성 품질 향상에 대한 주장이 아니다.

Matching feature attention은 원본 R8의 train-only fixed temperature **0.005**를 계승한다. 근거는 `reports_final2/2026-09-15-exg-app-cue-stage-b-r8-raw-region-fusion-results-ko.md`와 이를 계승하는 R9 설정이다. 새 cache는 attention contract까지 검사하며, online aggregation과 C2 context 계산이 같은 함수를 사용한다. 이 온도는 B1/B2 Top-1 선택에 영향을 주지 않는다. D1 topology listwise warm-up은 unscaled logits를 사용하며, 원본 controlled RGB objective와 구별한다.

D1은 query-only attention이므로 supervision에 포함되지 않는 query row의 forward를 생략했다. Loss, visible/proposal-miss/null 집계는 유지한다. Nonzero decoder/head 상태의 loss와 모든 parameter gradient parity를 CPU에서 검증했다.

## 실제 실행 설정

각 arm은 `dev-jaehyeok` 계열의 동일 mixed385 base+LoRA checkpoint에서 독립적으로 시작한다. 앞 arm의 생성 모델 weight를 이어받지 않는다. 기존 V1의 optimizer recipe와 LoRA LR/warm-up을 유지한다: SS LR 1e-5/warm-up 25, PBR LR 5e-6/warm-up 96; 추가 conditioner LR 1e-4. Native-zero residual 식과 C1/C2의 물리적 channel 차이는 이전 구현 검증 보고서와 같다.

| 항목 | SS B1 | PBR B1 |
| --- | --- | --- |
| Physical GPU | 0 | 1,2,3 |
| Global batch | 96 | 24 |
| Micro-batch / GPU | 96 | 4 |
| Accumulation packs | 1 | 2, rank당 합계 8 assets |
| 실제 probe | 2 updates, 90.14 GiB peak | micro 8 OOM, micro 4 PASS |
| 3-rank 고-token stress | 해당 없음 | 전 rank 2 updates PASS, 최대 64.54 GiB |
| Step 계산 시간 | steady 약 36초 | 고-token stress steady 약 46초; 평균 학습 시간과 구별 |
| 목표 | 5 epochs / 480 updates | 5 epochs / 1,905 updates |

PBR stress는 token 상위 24개를 사용하며 rank별 pack당 4개 이하를 확인했다. 작은 D1 warm-up을 GPU 2에서 병행한 상태에서도 통과했다. 모든 GPU 선택은 명시적이며 물리 GPU 4–7은 사용하지 않는다. B2–C2도 각각 실제 batch profile 및 exact frozen-head native-start proof를 통과해야 실행된다.

SS B1의 실제 step-zero quick panel은 voxel IoU 0.4519056615로 native V1과 동일하다. B1 RAW cache/online equality, all-arm native condition/denoiser/sample equality는 SS 32개/PBR 16개 및 대표 generation에서 검증했다. 최종 production head가 완성되면 B2–C2용 proof를 다시 실행한다.

## 학습 큐와 평가

`run_r9_pipeline.py`가 기존 SS B1과 D1 warm-up에 attach하고, PBR B1을 실행한다. D1 종료 후 frozen matching cache 전체 9,657개를 구축하고 integrity 및 실제 cache/online parity를 검증한다. 이후 각 stage가 B2 → B3 → C1 → C2 순으로 독립 실행한다.

처음 1 epoch를 bounded pilot으로 실행하고 동일 quick panel에서 step zero/동일 epoch native V1 대비 큰 하락과 nonfinite metric을 검사한다. 통과하면 같은 process의 optimizer/sampler/RNG를 유지하며 5 epochs까지 계속한다. SS B1은 controller 설치 시점에 따라 epoch-one native checkpoint resume으로 연장할 수 있다. 정상 checkpoint의 optimizer와 sampler를 복원하며 초기화부터 재시작하지 않는다. 실패하면 해당 stage에 `PILOT_HOLD.json`을 남기고 후속 arm을 멈춘다.

이 gate는 **명백한 실행 실패/급격한 성능 하락 방지용**이다. SS는 초기 IoU−0.02 및 native epoch-one IoU−0.05 중 큰 값을 하한으로, PBR은 초기 FG PSNR−1 dB/native−2 dB의 큰 값과 초기 crop LPIPS+0.03/native+0.05의 작은 값을 사용한다. 이 gate 통과를 모듈 효과의 유의한 향상으로 해석하지 않는다. 패널이 작은 경향 평가라는 한계도 유지한다.

매 epoch SS voxel IoU/precision/recall/F1, PBR FG/base-color PSNR와 crop SSIM/LPIPS, GT/Pred panel을 계산한다. PBR native V1 epoch-one checkpoint는 동일 AO seed로 재평가했으며 FG PSNR 23.92346, crop SSIM 0.94754, crop LPIPS 0.11665다. 학습은 W&B `youngwoojeon-unist/pixal3d-deformnext-retargeting`에 online logging한다. Fusion gate/coverage 및 해당 arm의 retarget residual RMS도 기록한다.

이번 수정의 직접 CPU 회귀 검사 88개가 통과했다. 기존 구현 검증의 111개 결과와 별도 실행이며 합산하지 않는다.

## 저장 및 재확인 위치

파일서버4의 공통 root:
`/file4/youngwoo/deformation/experiments/retargeting-v1`

- `topology_labels_r9_v5/`, `matches_raw_top1_v5/`: 완성본 파일서버 저장 완료.
- `q2c_r9_v5/`, `matches_r9_v5/`: head/cache 구축 완료 후 파일서버로 동기화 및 검증.
- `runs_r9_v5/{ss64,pbr1024}/{b1,b2,b3,c1,c2}/`: 새 결과.
- `runtime_diagnostics_r9_v5/production/`: pipeline-state, teacher coverage, 검증 JSON, profile 및 로그.
- `runtime_diagnostics_r9_v5/`: 메모리 probe와 DDP stress checkpoint의 검증된 archive.

실제 compute는 기존 local file4 mirror를 사용한다. 파일서버로 2분 간격 동기화하며, 완성된 checkpoint의 remote checksum을 확인한 후에만 local cache를 비운다. 파일서버 sync 실패 시 local artifact를 유지한다. 학습용 원본 weight나 유일한 결과를 삭제하지 않는다.

로컬 controller/evidence root는 `/root/dev/.cache/deformnext-retargeting/production-r9-v5`다. `pipeline-state.json`, `pipeline.log`, `storage-state.json`, `d1-warmup.log`, stage별 campaign log와 실제 run의 `progress.json`으로 상태를 확인한다.

## 후속 실행 점검: matching head의 GPU 공유 병목

본 학습은 SS 96 update(첫 epoch)와 PBR 150 update 이상까지 OOM/NaN 없이 진행했다. 한편 GPU 2의 PBR rank가 연산을 계속 점유하면서 D1 warm-up은 약 100 update에 수 분 이상 걸리는 수준으로 느려졌다. 메모리 여유만으로 공유 실행이 효율적이라고 판단한 것은 잘못이었다.

같은 실제 teacher/cache 12개를 CPU 4 threads에서 profile한 결과, 마지막 10 update 평균은 0.12668초였다. 따라서 SS/PBR 프로세스와 global batch는 그대로 두고, D1만 CPU에서 동일 seed 42017 / fixed-final 3 epochs recipe로 다시 실행했다. 이전 GPU head process는 fallback으로 일시 정지했다. CPU 작업 성공 및 기본 checkpoint/accuracy 검증 후에만 final 경로에 atomic publish하고 이전 GPU process를 종료한다. CPU 실행 실패 시 기존 GPU process를 재개한다. 이 전환은 `head-device-transition.json`, `d1-warmup-cpu.log`에 기록한다.

CPU/CUDA 부동소수점 rounding이 bitwise 동일하다고 주장하지 않는다. 최종 CPU head에 대해서도 기존 queue의 frozen-weight hash, cache/online parity 및 native-start 검증을 다시 통과해야 B2–C2가 실행된다. `train_q2c`의 device 선택 지원과 직접 관련된 R9 CPU 회귀 검사 28개가 통과했다.
