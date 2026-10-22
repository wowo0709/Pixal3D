# Native spatial3에서 attention의 조건부 기여 확인

후보 탐색과 feature 집계 중 어느 단계가 병목인지 구분하기 위해, 이미 학습된
`feature-spatial` SS480의 inference-only 진단을 추가한다. 새 SS 학습이나 새로운
matching cache 구축은 하지 않는다. 현재 PBR 학습과 예정된 historical 재현·PBR
후속 arm은 유지한다.

## 진단의 근거

후기 source-ray11개 arm은 모두 실제480 update/global96/micro96/accumulation1을
완료했고, 최종128개·93 source cluster·3생성 noise의33개 평가가 있다.
현재 `eleven-source-ray-SS-final-coverage-audit-v1/`에서5280 update의 finite loss·
terminal0·33개 평가의 GT/seed/모델 binding·32개 cascade prefix를 대조했다.
관련211개 metadata 파일13,399,408bytes를 실제 file4에서 읽어 SHA256/크기를
검증했다. 대형 checkpoint tensor를 다시 load하거나 재학습한 검사는 아니다.

| 후기 source-ray arm | 자산별 평균 IoU (%) | TP/FP/FN 합산 IoU (%) |
|---|---:|---:|
| Mean Top-1 | 47.1106 | 43.6165 |
| Mean weighted | 47.0863 | 43.6019 |
| Mean attention | 47.0906 | 43.6005 |
| Mean spatial | 47.1740 | 43.7457 |
| Member Top-1 | 47.1130 | 43.6263 |
| Member weighted | 47.1256 | 43.6649 |
| Member attention | 47.0531 | 43.5829 |
| Feature NoDQ | 47.2589 | 43.8810 |
| Feature DQ | 47.2315 | 43.8243 |
| Feature Top-1 | 47.2303 | 43.7880 |
| Feature spatial | **47.6667** | **44.0879** |

이11개는 과거 semantic seven-arm 보고서에 포함된11개 구성과 다른 계열이다.
동일 숫자로 혼동하지 않는다. 생성 noise3회도 독립 학습 seed3회가 아니다.
이전128 평가에는 prediction voxel hash가 없으므로,32 archive의 count/GT/seed
재현을 원래128 전체 occupancy의 bit-exact 증명으로 확대하지 않는다.

`feature-spatial`은 후보가 원래 projection의3개이며, semantic64 후보를 쓰지 않는다.
Mean spatial 대비 feature-spatial 조합의 탐색적 평균 IoU 차이는+.4927%p였다.
두 stack은 query/context·carrier 집계와 feature 변환·전달도 달라, 이것을 QKV 하나의
효과라고 할 수 없다. Semantic family의 원인 분석에 강한 spatial 기준을 남겨야 한다.

기존 source-ray SS 단일 GPU 학습의 실측 median compute는 mean-spatial39.65초,
feature-spatial51.42초/update였다. 동일480 학습을 먼저 추가하는 대신 저장된 모델의
집계 의존성을 확인한다. 새 GPU 진단의 실제 실행 시간은 아직 측정하지 않았다.

## 고정할 것과 바꿀 것

모든 실행은 같은 Spatial480 NN/source, native3 후보·D1 matching prior·Top-1 carrier,
source-use·fusion gate·R target context를 유지한다. Head의 원래 forward에서 나온
Q/K/V/score를 캡처해 원래 확률을 정확히 재구성한 뒤 다음 조건을 적용한다.

| 조건 | 바꾸는 부분 |
|---|---|
| Normal | 원래 head 출력을 그대로 반환 |
| Matching-prior-only | 추가 QK/utility 재가중을 없애고 원래 비균등 matching prior로 V 집계 |
| Top1-feature-only | Matching prior의 Top-1 correspondence 하나의 transformed V 사용 |
| Zero-transformed-value | Head의 learned feature correction만0; R 네트워크와 원래 Top-1 입력 유지 |
| Wrong-projected-V | 후보3개 안에서 projected V 대응만 순환; 원래 확률·Top-1 carrier는 유지 |

이 구조의 raw appearance carrier는 원래 Top-1이며 feature-head 확률로 바꾸지 않는다.
변경한 feature는 기존 R 입력과 direct feature residual 경로에 그대로 전달된다.
R의 **파라미터와 target context**를 고정하는 것이며, head 변경으로 계산된 R 출력은
달라질 수 있다. Zero-transformed-value를 R-off라고 부르지 않는다. Prior-only는
uniform NoScore가 아니며, Top1 control도 재학습한 NoScore arm이 아니다.
Wrong-V는 사용자 정의의 spatial NoCorr를 제거하는 실험도 아니다.

## 실행 전제와 판정

먼저 Normal32가 기존 actual32 archive의 predicted/GT latent와 packed voxel,
모든 per-asset record를 정확히 재현해야 한다. 이를 통과한 실제 canonical 결과가
없으면 네 개입을 실행하지 않는다. 모든 실행은 실제 initial noise·RGB/camera·
global/negative·native projection·후보/prior/carrier/use를 기록하고, NN/source state의
변경 및 optimizer update가 없는지 확인한다. 원래 source tuple·sampler12/CFG3·
batch8·deterministic protocol·DEV32를 유지한다.

각 GPU worker는 별도 OS session을 소유하며 성공·실패 시 모든 자식을 정리한 후
GPU 예약을 반납한다. 실패와 부분 산출물은 보존하고 PASS로 바꾸지 않는다.
GPU0–3만 사용한다. 현재 Top-1 PBR1900 완료 snapshot은 자원 순서의 barrier이며,
이 SS 진단의 모델 가중치로 PBR을 사용하는 뜻이 아니다. Historical 학습·최종 평가를
우선하고 비는 GPU에서 실행한다.

주 지표는32개 자산별 평균 voxel IoU다. 네 개입−Normal의 cluster bootstrap4000회,
seed20261023 개별95% CI와 IoU 네 비교 family 보정 CI를 함께 기록한다.
Precision/recall/F1은 탐색적 보조 지표다. 실제 head 변화가 있어도 생성 결과가
나빠지지 않으면 그 한계를 그대로 보고하며, 개선을 요구하는 통과 조건은 없다.
결과·SS GT/Pred/Diff max-projection을 W&B에 남긴다.

현재 synthetic CPU 검사는 attention 산술·Top-1/비균등 prior 의미·입력 불변식·
BF16 fingerprint·요약의 잘못된 모델/GT/noise 거부를 다룬다. 실제 Spatial480 GPU
Normal 재현이나 개입 품질 결과는 준비·CPU 검사로 대체하지 않는다.
PBR 후속, historical 재현, 독립 seed, 선택 조합 DEV cascade와 최종508개10 metric은
계속 남아 있다. 이 진단만으로 Full 채택 또는 독립적인 semantic/QKV 이득을 주장하지 않는다.
