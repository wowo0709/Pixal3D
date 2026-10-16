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

## GPU 0 공유 추가 실측

사용자 제안에 따라 CPU head 학습을 보존하면서 physical GPU 0에서도 작은 head 학습 probe를 실행했다. SS micro 96이 allocator 예약 메모리를 확장한 순간 probe의 20 MiB allocation이 실패했다. 당시 GPU free는 8.75 MiB였고 SS process는 약 94.11 GiB를 점유했다. Probe 자체 PyTorch allocated는 72.12 MiB, 전체 process 점유는 744 MiB였다. **OOM은 별도 probe에서 발생했으며, SS/PBR 본 학습에는 OOM이 없고 계속 진행했다.** 단순히 head parameter 크기만으로 GPU 0 공유 가능성을 판단할 수 없다.

이 시점에 CPU head는 3번째 epoch 23,900 update로 전체 작업의 대부분을 마친 상태였다. SS micro-batch나 allocator 정책을 변경하고 head를 초기화부터 GPU에서 다시 실행하는 대신, 현재 CPU fixed-final 작업의 남은 분량을 완료하도록 유지했다. PyTorch intra-op threads는 기존 4개이고, 해당 process의 모든 기존 thread에 nice 10과 낮은 I/O 우선순위(BE 7)를 적용했다. 상세 증거는 `runtime_diagnostics_r9_v5/production/gpu0-head-sharing-profile.json`에 저장한다.

## 사용자 추가 할당: node17 GPU 6/7

사용자가 `n17.unist.info`의 physical GPU 6/7을 추가로 할당했다. SSH로 확인한 GPU UUID 전체와 kernel boot ID가 현재 compute environment와 정확히 일치했다. 즉 기존 SS/PBR도 이미 node17 GPU 0–3에서 실행 중이며, 다른 서버로 dataset/environment를 복제할 필요가 없다. GPU 6/7의 확인 당시 free memory는 각각 약 95/89 GiB였다. 기존 다른 process를 종료하지 않는다.

진행 중인 SS/PBR campaign PID를 보존한 채 lightweight pipeline controller만 교체했다. 새 controller는 `--ss-gpus 0 --pbr-gpus 1,2,3 --matching-gpus 6,7 --attach-running-campaigns`로 실행한다. Frozen matching cache의 두 worker와 cache/online parity 검증을 6/7에 배정하며, 기존 0–3의 global batch/optimizer/sampler와 학습 process는 유지한다. 추가 장치는 명시적 할당이 있을 때만 사용하고 generation GPU와 중복되지 않도록 검증한다. GPU 4/5는 사용하지 않는다.

이 시점의 CPU head는 fixed-final 3 epochs의 마지막 단계였으므로 진행된 학습을 버리지 않고 완료하도록 유지한다. 이후 cache 구축부터 dedicated GPU를 사용한다. GPU 할당 및 attach 관련 직접 CPU 검사 6개가 통과했다. 실제 rebind/UUID 확인과 PID 근거는 `runtime_diagnostics_r9_v5/production/pipeline-rebind.json`에 기록한다.

## GPU 초기화 실측 및 7-epoch head 예산 확장

추가 cache worker의 초기화 중 NVIDIA driver 대기가 발생해 기존 SS/PBR도 약 3분 정체됐다. 추가 worker만 종료하자 즉시 진행이 재개됐다. 한 장씩 분리한 CUDA probe에서 GPU 6은 15초 timeout이 재현됐고, GPU 7은 CUDA initialization 0.573초 / process 전체 2.275초로 정상 동작했다. 따라서 GPU 6을 제외하고 **GPU 7만 matching 전용**으로 사용한다. GPU reset, 다른 process 종료 또는 다른 장치 사용은 하지 않았다. 증거는 `dedicated-gpu-serial-init.json`, `gpu-init-delay.json`이다.

CPU head는 fixed-final 3 epochs / 26,940 optimizer updates를 완료했다. 전체 508개 validation의 Top-1 추이는 RAW 0.338762 → epoch1 0.441016 → epoch2 0.457355 → epoch3 0.481319다. 매 epoch 개선 중이므로 3 epochs로 수렴했다고 주장하지 않는다. CPU 작업은 이미 종료됐다.

사용자가 5–7 epochs 확장을 요청해 **총 7 epochs**를 새 production head 예산으로 정했다. 이전 3-epoch file은 weight만 포함하고 AdamW/RNG 상태가 없어 정확한 이어 학습이 불가능하다. 따라서 원래 seed 42017과 같은 objective/LR 1e-4/AdamW weight decay 0.01/Top-64/teacher population으로 GPU 7에서 처음부터 7 epochs를 실행했다. 기존 3-epoch checkpoint/history는 `q2c_r9_v5/history/d1-topology-3epoch-cpu.{pt,jsonl}`에 보존했다. GPU rounding 차이 때문에 새 epoch3 결과와 CPU epoch3 결과의 bitwise 일치를 요구하지 않는다.

새 head는 매 epoch full checkpoint(`d1-topology-final.epoch-XXXX.pt`)와 latest training state를 저장한다. Weight/optimizer 및 Python/NumPy/Torch/CUDA RNG를 함께 복원하는 `--resume-state`를 구현했으며, 실제 AdamW 두 번째 update의 parameter equality를 통해 interrupted/uninterrupted CPU 실행이 정확히 같음을 검증했다. 잘못된 데이터 hash/device/recipe와 이전 weight-only file의 resume는 거부한다. 관련 CPU 회귀 검사 90개가 통과했다.

Epoch3/5/7의 동일 508개 validation 경향을 비교하되, production weight는 사전에 정한 **epoch7 final**로 고정한다. 이 weight가 완성되고 hash/actual native-start/cache-online 검증이 통과한 뒤 B2–C2가 같은 head를 사용한다. SS/PBR B1은 head를 쓰지 않으므로 계속 학습한다. 기존 head3의 partial cache 24개는 새 weight hash와 다르므로 production으로 사용할 수 없으며 재구축한다.

Head도 W&B `youngwoojeon-unist/pixal3d-deformnext-retargeting`, run `deformnext-r9-v5-d1-head-7ep`에 loss/속도/전체 validation을 기록하도록 설정했다. 학습 체크포인트를 file4에 2분 간격으로 동기화한다. 재실행/할당 근거는 `head-7epoch-launch.json`, `pipeline-rebind.json`, `head-storage-state.json`에 보관한다. GPU 한 개가 메모리에는 충분하지만 2-GPU throughput보다 빠르다는 검증은 하지 않았다. 현재 한 개를 쓰는 직접 이유는 GPU 6의 초기화 실패다.

## GPU7 head 실제 학습 속도 및 최종 실행

GPU7의 작은 CUDA probe만으로 실제 학습 성능을 보장할 수 없었다. 실제 descriptor transfer/NPZ 읽기에서 system CPU time이 크게 늘어 초기 validation과 head update가 느려졌다. 신호 기반 Python stack dump로 source-key 전송과 NumPy array 읽기 구간을 구분했고, 서버의 THP compaction stall 누적도 확인했다. Global kernel 설정이나 다른 process를 변경하지 않았다.

Head에서는 미사용 GLOBAL 채널을 CPU에서 제거한 뒤 contiguous FRAME 값만 pinned asynchronous H2D로 전송한다. Frozen cache의 C2용 FRAME+GLOBAL 값은 그대로 유지하되 전송 버퍼를 pin한다. 같은 첫 두 실제 validation sample의 loss는 변경 전후 정확히 같았다(3.8141326904, 3.5824522972). 첫 12개 중 warm-up을 제외한 평가 평균은 약 0.021초였다. CPU 회귀 검사에는 FRAME 값 equality, GLOBAL NaN의 배제, optimizer/RNG exact resume를 포함했고 총 **91개가 통과**했다.

또한 해당 process에만 `NUMPY_MADVISE_HUGEPAGE=0`을 설정해 큰 NumPy 임시 배열의 huge-page compaction 경로를 피했다. 장치 순서는 `CUDA_DEVICE_ORDER=PCI_BUS_ID`, allocator는 검증된 `expandable_segments:True`를 쓴다. NumPy 옵션 변경 전 첫 100-update 평균은 1.0922초였고, 변경 후 같은 seed의 첫 100-update loss가 정확히 같은 상태에서 평균 0.02837초였다. 이후 100-update 구간은 0.02583 / 0.02533 / 0.02780 / 0.02732초다. 즉 실제 backward/optimizer까지 포함해 **약 36–39 updates/s**를 확인했다. 최초 500 update 기준 total seven epochs는 평가/저장 포함 약 35–45분으로 추산한다. 아직 완료 시간이 측정된 것은 아니다.

현재 production head run ID는 `deformnext-r9-v5-d1-head-7ep-final`, 로그는 `d1-warmup-gpu7-final.log`다. 이전 `deformnext-r9-v5-d1-head-7ep`는 초기화/전송 진단 중 100-update까지만 진행한 partial run이며 비교 결과로 취급하지 않는다. Current head/controller PID와 process-local 옵션은 `head-7epoch-launch.json`에 기록한다. GPU7 한 개로 현재 처리 속도를 확보했으며, 2-GPU 학습의 speedup은 측정하지 않았다. GPU6은 초기화 timeout 때문에 계속 제외한다.

이 시점 SS/PBR B1은 모두 epoch-one quality gate를 통과하고 5-epoch 목표로 계속 진행 중이다. Head 확장 때문에 V1/B1을 재학습하거나 기존 geometry/latent를 재구축하지 않는다.

## 14:50 UTC: 최종 캐시 완료와 검증 자원 변경

GPU7 head는 총 7 epochs를 완료했다. 동일한 validation 508개에서 Top-1은 epoch0–7 순서로 33.88 / 44.14 / 44.53 / 47.64 / 49.58 / 51.38 / 52.51 / 53.18%이며, validation loss도 매 epoch 감소했다. 최종 epoch7로 production cache 9,657개를 모두 생성했고 실패는 0건이다.

캐시 완성 후 새 GPU7 cache/online verifier를 초기화할 때 공통 NVIDIA driver lock 대기가 재발했다. SS/PBR의 마지막 update가 각각 14:44:18 / 14:44:37에서 멈추고, PBR rank들의 kernel wait가 `os_acquire_rwlock_read`, SS autograd가 `os_acquire_rwlock_write`로 관찰됐다. GPU7 verifier만 SIGTERM으로 종료하자 PBR은 14:48:21에, SS는 14:48:38에 update를 재개했다. 기존 SS/PBR 학습 process/optimizer/sampler 및 완성된 cache는 보존했다. OOM으로 진단하지 않았으며 GPU reset이나 다른 작업 종료는 하지 않았다.

Controller를 `--verify-cache-on-idle-ss`로 교체했다. 완성된 cache가 동일 checkpoint hash와 전체 schema 검사를 통과하면 CUDA context를 새로 만들지 않고 재사용한다. file4에 cache를 보관한 뒤 SS B1이 완료되어 GPU0이 비면 실제 bf16 cache/online parity 검증을 수행한다. 이 검증과 stage별 native-start 검증이 통과해야 B2–C2를 실행한다. GPU7에서 동일 CUDA 초기화를 반복하지 않으며 검증을 생략하지 않는다. 기존 SS/PBR은 각각 252/480, 570/1905까지 재진행을 확인했다. 원인은 GPU7의 새 context 초기화와 연관된 driver 대기로 좁혔으며 하드웨어 고장 여부는 확정하지 않았다.

직접 CPU 회귀 검사 7개가 통과했다. 새 검사는 CUDA 초기화를 금지한 상태에서 실제 cache를 읽어 current head를 허용하고 old head 및 누락 파일을 거부한다. 실행 근거는 `cache-verification-stall.json`, `pipeline-rebind-cache-recovery.json`, `pipeline-recovery-cache.log`에 보관한다.

## 2026-10-16 01:25 UTC: SS B2 완료 및 PBR DDP 완료 대기 복구

전체 matching cache 9,657개는 file4 보관 및 실제 bf16 cache/online 검증을 통과했다. SS/PBR 각 8개 category panel의 evidence 최대 절대오차는 모두 0이었다. SS production native-start 검증도 condition/denoiser/sampling 오차 0으로 통과했다. SS B1과 B2는 각각 480 updates / 5 epochs를 완료했고 B3가 진행 중이다. Final quick-panel voxel IoU는 V1 0.490831, B1 0.490866, B2 0.491964로, B2의 차이는 작아 유의한 향상으로 주장하지 않는다.

PBR B1은 10월 15일 21:06 UTC에 1,905 updates와 최종 validation/checkpoint 저장을 완료했지만 다음 arm으로 넘어가지 못했다. 원인은 pilot 연장 결정에서 rank0의 config만 `auto_continue_after_pilot=False`로 바뀌고 나머지 rank에는 extension 값만 broadcast한 것이었다. 마지막 step에서 rank0은 종료 barrier로 가지만 rank1/2는 pilot extension broadcast를 다시 호출해 collective 순서가 달라졌다. 모든 rank가 첫 결정 직후 flag를 소진하도록 수정했다. 3개 rank의 pilot/final collective 순서를 확인하는 회귀 검사와 기존 runtime/campaign 검사를 포함해 15개가 통과했다. 진행 중인 SS B3는 single GPU이므로 이 종료 문제의 영향을 받지 않는다.

최종 denoiser/conditioner/misc 파일 3개의 CPU load와 local/file4 SHA-256 일치를 확인한 뒤 완료된 PBR B1 worker를 정리했다. 학습을 재시작하거나 checkpoint를 변경하지 않았다. W&B의 기존 동일 run을 resume해 final summary와 completion 상태를 반영하고 정상 종료했다. Controller는 이제 production head를 사용한 PBR native-start 검증을 수행하고, 통과 후 B2 DDP preflight와 학습을 실행한다. 실행 근거는 `pbr-b1-completion-recovery.json`, `pbr-b1-wandb-finalized.json`이다. PBR B1의 epoch2→5 foreground PSNR은 24.006→20.881 dB로 낮아지고 crop LPIPS는 0.11443→0.09579로 개선되어, 지표가 엇갈린다. 이를 일괄적인 성능 향상으로 해석하지 않는다.
