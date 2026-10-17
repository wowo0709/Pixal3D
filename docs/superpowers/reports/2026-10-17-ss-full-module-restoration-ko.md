# SS 전체 모듈 비교 복원과 실행 범위

원래 [matching v2 계획](2026-10-17-ss-matching-v2-experiment-plan-ko.md)의 핵심은 Top-1, rejection, matching-aware attention, retargeting adapter, deformation query를 분리하는 것이다. Frozen V1에서 GT feature 주입도 개선되지 않아 필수 B 세 arm이 HOLD됐고, 실행이 P1/P2와 retarget-only query 비교로 이동했다. P1/P2는 유효한 interface 진단이지만 이 다섯 비교를 대체하지 못한다. 사용자가 지적한 범위 축소를 다음과 같이 복원한다.

## 전체 실험과 상태

| 구분 | 실험 | 상태/역할 |
| --- | --- | --- |
| 공통 기준 | Source-only RGB 4장 / edited-RGB 1+3장 zero-shot, V1, 기존 B1/B2/B3/C1 | 기존 checkpoint 공통128 재평가 완료; V1 재학습 없음 |
| 원인 진단 | Native/current/GT 주소, semantic 경로 OFF/wrong correspondence, adapter OFF, projection 보간/support, source/target 비중 | 완료 자료 보존; GT 지원 영역과 주소의 confound를 구분 |
| Matching head | 같은 protocol의 Top64 H0 replay, Top256 H1, confidence H2, feature-task H3 | 7 epochs 완료; train-internal calibration으로 H3/T0.1 고정 |
| Interface 진단 | P1: R9 + native bilinear + adapter / P2: H3 + 같은 consumer | 진행 중; 새 main ladder와 별도로 보고 |
| SS 본 비교 | 아래의 다섯 arm | native projection에서 실행 복원 |
| 후속 | 작은 gain의 training-seed 재현, 필요한 동일 sampler 진단, SS 채택 후 PBR 확대 판단 | 최종 생성 결과에 따라 결정 |

P1/P2를 포함하면 새 SS run은 총 일곱 개다. 현재 P1/P2는 같은 run의 저장 상태부터 이어가며, 나머지 다섯 개는 독립 원본 initialization이다. 이전 SS C2와 PBR B3는 재개하지 않는다. Retarget-only query-study의 미실행 두 학습은 아래 C1/C2로 대체하고 기존 예약·provenance를 보존한다. 별도의 두 query run을 중복 실행하지 않는다.

## 복원한 다섯 arm

| 새 run ID | 원래 계획 대응 | Correspondence value | Confidence/rejection | Adapter | RAW query |
| --- | --- | --- | --- | --- | --- |
| b2-top1 | B2-v2 | Learned Top-1 하나 | OFF | 없음 | 채널 없음 |
| b2-top1-r | B2-v2-R | 같은 Top-1 | 고정 H3 p_use | 없음 | 채널 없음 |
| b3-attention-r | B3-v2 | 같은 후보/score의 weighted attention | 같은 p_use | 없음 | 채널 없음 |
| c1-retarget-r | C1-v2 | 동일 attention | 동일 | 추가 | 채널 없음 |
| c2-query-r | C2-v2 | 동일 attention | 동일 | 동일 | 물리적 RAW2048 추가 |

V1은 native spatial correspondence 기준이다. No-score 비교는 b2-top1-r 대 b3-attention-r이며 uniform 평균을 사용하지 않는다. C1에는 deformation channel 자체가 없고 C2에만 추가한다. Retargeter/query 2×2가 아니라 두 개의 추가 arm이다.

모든 arm은 같은 frozen H3 checkpoint/hash, candidate set, FP32 cached Top1/soft evidence, temperature0.1, native bilinear/support-normalized projection을 쓴다. R arm의 threshold0.276316212는 기존 train calibration에서 잠근 값이다. Validation 결과로 다시 조절하지 않는다. Teacher address/visibility/GT deformation은 inference 입력에 넣지 않는다.

과거 nearest raw-transport gate의 HOLD와 수치를 그대로 남긴다. 이 연구의 config-ready PASS는 실행 준비를 뜻하며 성능 PASS가 아니다. Negative frozen GT probe는 주입 문제의 증거이지만, LoRA를 함께 학습하는 모듈별 비교를 생략하는 근거로 사용하지 않는다. 기존 C1의 공통128 세-noise 이득도 있으므로 adapter가 raw 경로의 약점을 회복하는지 다섯 arm에서 직접 확인한다. 주입 구조가 실패한다는 가설도 최종 결과로 명시할 수 있어야 한다.

## 학습·자원

- DeformNeXt RGB/world-v4 train9149/val508, frozen Omega epoch7, 기존 정합 latent와 cache를 재사용한다. Clean80k를 추가하지 않는다.
- 원본 mixed385 complete base+LoRA checkpoint에서 각 arm을 독립적으로 시작한다. 구조의 순서는 weight continuation을 뜻하지 않으므로 고정한 arm은 병렬 학습할 수 있다.
- 480 updates/약5 epochs, global96, 2-GPU DDP, rank당 micro48, accumulation1. 원래 optimizer/LoRA/conditioner LR/warmup/CFG dropout을 유지한다.
- GPU0·1에는 P2 → b2-top1 → b3-attention-r → c2-query-r, GPU2·3에는 P1 → b2-top1-r → c1-retarget-r를 배치한다. 각 run의 final 세-noise 평가를 마치면 같은 pair의 다음 학습을 실행한다. 생성 평가는 학습 GPU와 겹치지 않는다. 마지막 학습 중 다른 pair는 평가와 진단에 사용한다.
- 실제 native-start/gradient 증명과 2-update per-rank memory probe를 통과한 run만 본 학습으로 진행한다. 메모리·속도·loss·gate/update RMS·metric·이미지를 W&B와 file4에 남긴다.

P1/P2는 사용자 중단 당시 step103/56 checkpoint에서 optimizer/scheduler/sampler까지 복원했다. Native checkpoint에는 rank별 RNG가 없어 bitwise uninterrupted replay라고 주장하지 않는다. 현재 설정의 steady-state는 약17.2초/update, 각 rank peak47.75GiB였다.

## 비교·완료 조건

매 epoch 공통32의 voxel IoU/F1/precision/recall과 GT/Pred/Diff를 확인한다. 최종은 같은128개, sampling12/CFG3, sample별 고정 noise 세 개다. 다섯 인접 대비와 각 arm의 V1/기존 C1 대비 차이를 source cluster bootstrap으로 계산한다. 같은 source의 variant/noise를 독립 표본으로 세지 않는다.

Zero-shot 두 조건과 기존 B1/B2/B3도 추가 noise 두 개를 생성해 최종 세-noise 표에 함께 넣는다. V1/기존 C1은 이미 세 noise가 있다. 이 보충은 저장된 checkpoint의 평가 열 개이며 재학습이 아니다. GPU pair가 비면 실행하도록 queue에서 자원을 예약하며 기존 학습과 겹치지 않는다. 최종 집계는 baseline 일곱 개와 새 run 일곱 개, 총14개 구성을 같은128×3 조건으로 다룬다. Source-only의 네 M0 RGB도 동일 M1 target과 비교하는 입력 효과 진단임을 명시한다.

핵심 대비는 `b2-top1−V1`, `b2-top1-r−b2-top1`, `b3-attention-r−b2-top1-r`, `c1-retarget-r−b3-attention-r`, `c2-query-r−c1-retarget-r`다. P1/P2와 기존 C1도 최종 비교에 포함한다. Head Top1 향상만으로 SS module 효과를 입증하지 않는다. H2 confidence loss는 ranking branch에 detach되어 있어 H2−H1 순위 향상을 inference rejection의 인과 효과로 설명하지 않으며, 실제 SS R 대조로 판단한다.

Full은 C2로 미리 확정하지 않는다. 다섯 대비를 모두 보고 best final 구성과 확인되지 않은 효과를 기록한다. 근소한 차이는 training-seed 재현을 추가한다. 개발용128을 선택에도 사용한 결과는 unseen test 성능으로 부르지 않는다. 과거 voxel P1/G2의 cleanup 개선도 source utilization 성공과 구분한다. 기존 보고서는 두 실험 모두 DEV projected-source 효용을 입증하지 못했다고 명시했다.

완료 시 원래 계획 항목별 결과/미확인 가설, 같은-panel final 표, 학습 경향, GT/Pred 이미지, head/cache/checkpoint provenance, 실제 시간·메모리 및 PBR 확대 근거를 한 리포트에 정리한다. 실행 중이거나 config만 준비된 상태를 완료로 표시하지 않는다.

`summarize_ss_module_study`를 마지막 CPU job으로 연결한다. 완료 marker뿐 아니라 두 rank의1~480 연속 update, 여섯 epoch 검증, 초기 generation parity, native/physical-query gradient 증명과 final checkpoint를 확인해야 primary report를 만든다. 이 report의 `primary_matrix_complete`는 최초 비교 행렬의 완료이며 추가 seed/최종 채택/PBR 또는 전체 goal 완료를 뜻하지 않는다. 자동 산출물은 file4와 W&B artifact로 보존하고 최종 채택과 필요한 후속 실험은 실제 결과를 검토해 결정한다.
