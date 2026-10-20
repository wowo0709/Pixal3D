# Predicted availability와 candidate-paired retargeting 구현·검증

2026-10-20 UTC, `wowo0709/feature-retargeting`. [직전 실패 분석](2026-10-20-matching-confidence-visibility-and-retargeting-ko.md)을 바탕으로 신뢰도 head를 실제 학습하고, 새로운 SS conditioner와 continuation runtime을 구현했다. 생성 품질 향상이나 full 모델 채택은 아직 확인되지 않았다.

## 신뢰도 head는 실제 검증 기준을 통과했다

기존 DeformNeXt world-v4 cache를 사용했다. 새 렌더/latent/Omega/raw 다운로드는 없다. TRAIN512 / TRAIN-calibration32 / 실제 SS VAL32의 source cluster는 role 간 겹치지 않는다. VAL32는 이미 여러 분석에 노출된 개발 패널이므로 독립 최종 test로 취급하지 않는다.

목표는 “현재 native semantic Top64 안에 visible near1 correspondence가 하나라도 있는가”다. 기존 topology teacher의 patch-center visibility와 near1 label을 사용한다. Source에서 보이지 않는 경우와 proposal이 정답을 놓친 경우를 모두 negative로 학습하고, GT patch center가 background인 unknown query는 제외한다. Query가 없는 asset도 삭제하지 않는다.

| 역할 | asset | 학습/검사 ray | usable | source-invisible | proposal-miss | 제외한 unknown center |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 512 | 41,117 | 19,593 | 17,937 | 3,587 | 45 |
| TRAIN-calibration | 32 | 2,314 | 970 | 1,094 | 250 | 2 |
| SS VAL | 32 | 2,510 | 1,345 | 993 | 172 | 1 |

입력은 inference에서 얻는 native DINO query, Q2C contextual query, semantic score, candidate mask/address다. 두 query의 per-ray normalization과 12개 summary를 합친 2,060차원을 hidden64 MLP에 넣는다. Trainable tensor4개, 총131,969개 parameter다. GT/visibility label은 head 입력이 아니다. Summary mean/std는 TRAIN만 사용한다.

Epoch7, seed101, AdamW LR .001 / WD .01, microquery4096을 사전에 고정했다. **7 epoch는 작은 head의 77 update**이며 SS/PBR 모델을 7 epoch 학습한 결과가 아니다. GPU2에서 실제 peak .4323 GiB, 측정 구간26.71초였다. Wrapper 시작·입력 읽기 등 전체 wall time과 구분한다.

| 지표 | 실제 SS VAL32 |
|---|---:|
| AUROC | **.837800** |
| source-cluster CI95 | [.778165, .881623] |
| Brier ↓ | **.174240** |
| TRAIN prevalence constant의 Brier ↓ | .252235 |
| Brier 차이의 source-cluster CI95 | [−.103525, −.050394] |
| usable recall, threshold .35 | .814870 |
| unsupported false acceptance, threshold .35 | .303004 |

Threshold .35는 TRAIN-calibration의 고정19-grid balanced accuracy로 선택한 뒤 actual VAL label을 열었다. Cal AUROC는 epoch1 .7996에서 epoch7 .8384로, Brier는 .1871에서 .1630으로 개선됐다. 실제 VAL AUROC≥.65, CI 하한>.5, constant 대비 Brier 개선 및 paired-CI 상한<0이라는 사전 gate를 통과했다. 임계값으로 후보를 hard reject하는 모델을 채택한 것은 아니다. 아래 conditioner는 soft probability를 추가 보정의 강도로 사용한다.

평균 confidence는 source-invisible .2659, proposal-miss .2417, usable .5990이었다. 오판이 남아 있으므로 이를 완벽한 visibility detector로 해석하지 않는다. [W&B run](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ray-correspondence-reliability-v1-s101-e7)에 학습/검증 수치를 기록했다.

## 새 모델은 후보를 먼저 보정한 뒤 합친다

`RayConsistentRetargetingConditioner`는 기존 native Top1 appearance와 양성 기준 R을 유지한다. 각 source 후보의 appearance와 실제 RAW Omega difference를 함께 R에 넣고, 해당 후보의 geometry 보정만 score로 합친다.

\[
e_q=a_{q,i^*}+R(a_{q,i^*},t_q)
 +g_q\sum_{i=1}^{64}\alpha_{qi}
 \left[R_{geom}(a_{qi},t_q,\Delta_{qi})-R(a_{qi},t_q)\right],
\qquad \Delta_{qi}=\Omega^t_q-\Omega^s_i.
\]

- Query는 **target patch ray 단위**다. 원래 nearest projection 경로로 같은 ray의 voxel들에 전달한다. 개별 voxel feature로 query를 새로 만드는 구조가 아니다.
- Alpha는 frozen D1 matching decoder context, 학습된 utility5, semantic score prior를 사용한다. 이전에 near-mass gate를 실패한 auxiliary Q/K epoch7 가중치는 배포하지 않았다. 추가 Q는 0인 prior-preserving initializer를 유지하며, 따라서 여기의 신규 outer Q/K head 자체의 추가 효과를 입증한 것은 아니다. 학습된 utility의 query/context/key content attention은 사용한다.
- Matching7과 reliability4 tensor는 reconstruction 학습 동안 동결한다. SS gradient가 matching을 망가뜨리던 경로를 차단한다.
- Geometry projection은 실제 RAW2048 difference를 받는다. 추가 projection **가중치**를 0으로 초기화하며 관측값을 0으로 대체하지 않는다. Appearance V/O 행렬은 없다.
- 추가 보정의 R weight는 해당 backward에서 detach해 사용한다. 기존 Top1/R 경로의 R은 계속 학습한다. 초기에는 기존 R의 gradient와 함수가 보존되고, 새 projection은 첫 backward부터 학습할 수 있다.
- `g=0`의 의미는 기존 Top1/R 기준으로 돌아가는 것이다. Source feature 자체를 NULL vector로 관측한 것처럼 처리하지 않는다. 후보별 `g*alpha`와 추가 보정 없음의 `1-g`로 해석할 수 있다.
- `paired`와 `mean` composition을 코드에서 구분했다. `mean`은 비교를 위한 구현이며 아직 자체 native 재학습 결과는 없다.

## 실제 native 검증 결과

CPU checks 4개가 통과했다. NumPy 학습 입력과 Torch 추론 입력, 동결 checkpoint 로딩, SS/PBR width에서 초기 함수·공통 gradient 보존과 즉시 새 gradient, 후보 pairing의 주소 민감도 및 permutation 일관성을 검사했다. PBR width의 CPU 확인을 native PBR 통합으로 보고하지 않는다.

CUDA 배포 port는 실제 VAL32의 2,510개 known ray를 native query partition256으로 검사했다. 기존 학습 결과와 최대 probability 차이는 **.0000112653**, threshold 판정 변화는 **0개**였다. 배포 AUROC .83780015, Brier .17424057로 유지됐으며 재학습/재선택은 없었다.

실제 SS TRAIN micro2, 기존 engineering R480 기준의 최초 loss는 **.10466600209474564**로 정확히 같았고, 실제 conditioner evidence도 같았다. 새 projection의 gradient L1은 세 update에서 .42117 / .43125 / .43291이었다. Effective K는 **6.19472**로 유지됐다. NN/R/matching/reliability 상태는 그대로였다. Peak5.5719 GiB, 측정 구간38.21초다. TRAIN loss 감소를 생성 품질 향상으로 해석하지 않는다.

별도 **R96 continuation**에서는 공통 Adam366을 복원해 NN360 / R·fusion6을96→99, 신규 projection1을0→3으로 업데이트했다. 총367 optimizer state와 각 tensor의 실제 변화, matching/reliability 동결을 확인했다. Peak5.8179 GiB, micro2/world1이므로 full global96 DDP2 메모리 증명은 아니다.

실제 초기 생성도 **R96 / VAL32 / generation batch8**로 완료했다. IoU **.43718718977498344**, precision .5517037762, recall .6780659337, F1 .6083928286이 기준 R과 정확히 같았다. 32개 sample의 기존 GT/source/seed/TPFPFN/metric 필드도 모두 같았다. 전체 voxel-mask byte 동일성을 검사한 것은 아니다. Peak5.8334 GiB, 생성 측정 구간230.29초였다.

실제 global96 / DDP2 / GPU당48 / accumulation1 capacity도 GPU2/3에서 완료했다. Peak는 rank0 **49.7909 GiB**, rank1 **49.7876 GiB**, warm-up 이후 update는 약17.6초였다. 총367 state의 실제 업데이트와 양 rank 동기화를 확인했다. 첫97의 전체 loss 필드는 matched R control과 정확히 같았다: rank0 .13053171336650848, rank1 .1225043460726738. Native/wrapper 모두 종료0, 실제 file4 receipt도 닫혔다. 기존374/375/376-state 증명을 재사용하지 않았다.

Full480 학습 recipe는 공통 R96부터384 update 추가, NN LR1e-5 / R LR1e-4 / 신규 projection LR1e-4·warm5, seed101, TRAIN9149로 준비했다. 새 모델의 장기480 학습은 아직 queue에 추가하지 않았다. 대신 **첫96 update만96→192로 실행하는 유한 단계**를 연결했다. Model/data/objective/optimizer/LR schedule은 full480 recipe 그대로이며 loop 경계만192다. 완료 시 common Adam은192, 신규 projection Adam은96, scheduler position은96이어야 한다. 두 rank의 RNG·sampler와 checkpoint trio를 보존해 나머지288 update를 이어갈 수 있다.192 완료를 full480 완료로 표시하지 않는다.

이미 닫힌 초기32 생성과 실제 양 rank 첫97 parity를 재사용해 초기96 생성을 반복하지 않는다. 이 짧은 단계는 기존 panel의 첫8개·동일 seed map·batch8로 최종192 추세만 빠르게 본다. 8개 결과를 기존 R32 평균과 직접 비교하지 않으며 동일8개 baseline을 추출한다. 이후 실제32 평가와 full480·128×3 비교는 남아 있다. Full continuation 시 validation panel을32로 복원해야 한다.

시작 조건은 현재 평가들을 중단하지 않고 남는2/3 시간에 예상 단계 길이와 PBR margin이 들어가는 것이다. 실제02:29:35 판단에서 NoDQ402의 예상 잔여1,744.82초가 단계1,911.10초+margin120초보다 짧아 **학습을 시작하지 않고 defer**했다. 대신 PBR actual capacity 뒤 GPU0/1에서 첫96 update를 실행하고 기존 scalar SS는2/3에서 병렬 실행하도록 연결했다. 먼저 실행됐다면 재학습하지 않고192 checkpoint를 재사용한다. PBR main activation은 이 단계와 기존 scalar/final evaluation 완료를 기다리며, 새로운 장기480 학습은 PBR 뒤로 유지한다.

## 기존 SS 추세와 실행 속도

Geometry step384의 VAL32 IoU는 .447438, matched R은 .447489였다. 추가 이득은 확인되지 않았다. Physical Direct-NoDQ step288은 .436987이었다. 이 결과를 새 candidate-paired 모델의 성능으로 혼동하지 않는다. 최종480, 기존128×3/replay32/strong R/V1/두4-RGB zero shot 비교는 계속 진행한다.

NoDQ가 느린 별도 원인은 CPU contention이었다. Geometry는 data worker2, NoDQ는 worker0이고, 두 GPU 학습 그룹과 helper가 CPU84/85를 공유했다. 실제 유휴86/87을 확인한 뒤 **GPU0/1의 기존 NoDQ DDP worker 두 프로세스만** 모든 thread를86/87로 옮겼다. PID/생성 시각/명령, `OMP_NUM_THREADS=2` 환경변수, GPU 배정은 같으며 framework의 thread 설정은 직접 바꾸지 않았다. Helper는84/85를 유지했다. 코드·config·sampling·optimizer를 직접 바꾸거나 프로세스를 재시작하지 않았다.

| 관찰 구간 | rank0 load 중앙값 | rank1 load 중앙값 | update load+compute 중앙값 |
|---|---:|---:|---:|
| 변경 전20 update | 13.45초 | 14.22초 | 약32.50초 |
| 변경 후290–299의10 update | 3.07초 | 3.40초 | 약20.70초 |

관찰 구간의 전체 속도는 약 **1.57배**였다. 격리된 전용 benchmark는 아니다. 학습은 정상적으로 계속 진행했다. 향후 PBR0/1 worker에서도 같은 CPU 경합이 생기면 실제 상태를 확인해 같은 방식으로 분리할 수 있다. Frozen 실행 스크립트는 수정하지 않는다.

## 재현 위치와 남은 작업

Canonical root는 기존 실험 아래 `feature-readout-v1/prior-anchored-QK-v1/reliability-head-v1`이다. `fit-fixed7-v1`, `overlap-capacity-v1`, `overlap-scheduling-v1`, `overlap-restored-v1`, `deployed-port-v1`, `NoDQ-CPU-affinity-v1`과 `consistent-retargeting-v1/{SS-gradient-v1,joint-profile-v1,initial-state-v1,initial32-parity-v1,prelaunch-v1,capacity-v1,segment-plan-v1,phase1-decision-v1,segment-after-capacity-plan-v1}`을 실제 file4의 SHA256/size와 대조해 보관했다. 실제 첫96 update 학습 결과는 아직 pending이다.

다음 증거는 새 모델의 **실제 학습 후 matched R 대비 생성 품질**이다. 그 전에는 matching/retargeting의 추가 효과나 full 모델을 확정하지 않는다. 후보가 좋아지면 같은 초기 NN/R·Adam·학습 예산으로 frozen-mean, no-reliability, matched-Top1 payload, native-spatial correspondence, physical-NoDQ를 분리한다. 사용자 정의 NoScore는 uniform weights가 아닌 Top1 payload다. Native-spatial은 semantic Top64와 후보 개수/분포가 다르므로 semantic64용 reliability head를 그대로 적용해 OOD gate를 만드는 오류를 피해야 한다. Physical-NoDQ가 기존 R-only와 함수·학습 경로까지 동일하면 기존 matched R control을 재사용할 수 있다.

현재 PBR V1 대 FullDirect는 stack 전체 비교이며 matching만의 독립 효과를 보여주려면 matched R-only가 필요하다. 신규 conditioner의 native PBR 통합은 아직 수행하지 않았다. 생성 noise 반복, 개발 VAL32, head AUROC를 독립 training seed나 최종 test508, 논문의10개 full-pipeline metric으로 대체하지 않는다.

## 03:10 UTC 실행 갱신: 실제 candidate 학습과 PBR 메모리 복구

새 candidate-paired SS는 GPU0/1에서 실제 NN/R 학습을 시작했다. 첫 scientific step97에서 **두 rank의 전체 loss 항목이 matched R control과 정확히 일치**했다. Rank0 loss는 .1305317134, rank1은 .1225043461이다. 임시 capacity 모델의 결과를 재사용해 학습이 시작됐다고 표기한 것이 아니다. `scientific-startup97-v1`에 실제 학습 로그·wrapper PID/생성 시각/명령·worker CPU 배정과 함께 actual file4 증거를 닫았다. GPU worker와 DataLoader worker는 CPU86/87, helper는84/85를 사용한다. 새 학습의 생성 품질 향상은 아직 판정하지 않았다.

첫 구간192가 끝나면 원래32개 전체 패널에서 별도 native 평가를 실행한다. 같은192의 R/control, feature, DQ, Direct, geometry, NoDQ, V1, historical R 및 두4-RGB zero shot과 source-cluster paired 비교한다. 전체480 continuation도 실행 대기열에 연결했다. PBR 비교 후 같은192 checkpoint의367 Adam state, rank RNG, sampler, scheduler(last_epoch96)를 복원해193–480을 진행한다. 저장 config의 **validation만8→32로 복원**하며 학습 recipe는 유지한다. 480 평가32와128×3 및 비교도 연결했다. 과거 R의 post96 학습 stream 차이와 sampling noise 반복이 독립 학습 seed가 아니라는 제한은 그대로 남는다.

Physical NoDQ는480을 정상 완료했다. VAL32 IoU는 **.4396122**, matched R은 .4465541이었다. Geometry-only480은 .4472968이었다. 이 결과들은 새로운 candidate-paired 모델의 학습 결과가 아니다.

원래 PBR full-batch capacity는 **실제 CUDA OOM으로 실패**했다. World2/micro12는 V1과 Direct 모두, world4/micro6는 V1에서 실패했다. 실패 위치는 denoiser가 아닌 frozen DINO/NAF1024 이미지 특징 추출이었다. 이 실패를 PASS로 덮지 않고 `capacity-failure-v1`에 로그와 return code를 별도로 보관했다. PBR 성공 receipt에 걸려 있던 SS 첫 구간은 실제 실패 기록을 확인하는 새 실행 경로로 해제했다. 기존 FAILED capacity와 HELD activation 기록은 남긴다.

새로운 `frozen_projection_streaming.py`는 frozen 이미지 encoder의 single-view 계산만 asset별로 순차 수행하고, native 크기의 projection 출력에 복사한다. Multiview 평균, denoiser batch, 학습 objective, LoRA, optimizer, pretrained weight와 전처리는 유지한다. 공유 extractor 파일은 수정하지 않고 새 PBR entry에서만 process-local adapter를 설치한다. **Gradient accumulation을 추가하지 않는다.** CPU 검사2개가 통과했다. 실제 world2 capacity의 두 rank에서 native batch2와 streaming의 global `[2,5,1024]`, projection `[2,262144,2048]` 특징값은 모두 bitwise 일치했다. 아직 전체 micro12 capacity가 통과했다고 주장하지 않는다.

새 PBR 경로는 두 arm의 최대31900-token 반복 batch,4개 렌더 validation, 이후 추가 full-batch update를 다시 검사한다. World2가 실제 OOM이면 SS192 평가와 initial32 parity가 끝난 뒤 world4/micro6를 검사한다. 별도 initial32 평가에서 기존32개 기록·metric·34개 PNG가 모두 같은지도 다시 확인한다. 실제 두-arm capacity와 initial32가 통과한 경우에만 scalar SS/남은 SS 평가를 해제하고, 기존에 예정한 PBR762→1905 두-arm 학습·3개 평가·paired 요약을 새 경로로 연결한다. 신규 candidate-paired conditioner의 PBR 포팅과 이 기존 FullDirect PBR 비교는 별개다.
