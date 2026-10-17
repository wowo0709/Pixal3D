# SS matching v2 실행 기록

작성일: 2026-10-17 KST. 아래 수치는 실행 중 snapshot이며 완료된 새 SS 학습 결과가 아니다.

## 실행 범위와 기존 학습 중단

[실험 계획](2026-10-17-ss-matching-v2-experiment-plan-ko.md)에 따라 원인 진단, H1/H2/H3 학습, 공통 SS panel 재평가를 시작했다. 사용 장치는 **physical GPU 0~3만**이다. 이전에 할당됐던 GPU 7과 GPU 6은 이번 campaign에서 사용하지 않는다.

사용자의 기존 학습 중단 지시에 따라 RGB queue controller를 종료하고 trainer의 STOP/checkpoint 경로로 다음 run을 종료했다.

| 기존 run | 최종 저장 step | 상태 |
| --- | ---: | --- |
| SS C2 | 90 | denoiser/conditioner/optimizer checkpoint 저장 후 종료 |
| PBR B3 | 979 | denoiser/conditioner/optimizer checkpoint 저장 후 종료 |

기존 trainer가 남아 있지 않음을 확인했다. 두 run의 결과와 checkpoint는 file4의 원래 `rgb_v1/runs_r9_v5/`에 보존했다. 기존 PBR C1/C2는 자동 시작되지 않는다.

## 데이터와 실험 고정

- SS population은 DeformNeXt RGB train 9,149 / validation 508을 유지한다.
- Matching teacher 인증 실패 train 6개는 head supervision에서 제외한다. 인증된 train 9,143개 중 source cluster 179개를 calibration으로 분리해 **head train 8,570 / calibration 573 / validation 508**을 고정했다.
- `panels/panel-32.json`, `panel-128.json`, `seed-map.json`으로 평가 ID와 sample별 noise를 고정했다. 아래 32/128 수치는 새 공통 panel의 값으로, 이전의 4개 quick panel 및 legacy 32개 panel과 직접 빼지 않는다.
- 입력은 M1 camera8 RGB와 M0 cameras9/10/11 RGB다. Source-only baseline만 camera8을 M0 RGB로 바꾼다. 두 조건 모두 서로 다른 카메라 4개다.
- 추가 원본 다운로드나 SS latent 재구축은 하지 않았다. 기존 RGB, world-v4 latent/projection, Omega epoch7 및 DINO feature를 재사용한다.

## 후보 recall 측정

Train 내부 calibration에서 사전 고정한 선택 규칙은 “recall 90%를 넘는 가장 작은 K, 없으면 K256”이다. Validation 수치를 보고 K를 고르지 않는다.

| 후보 수 | Calibration recall | Validation recall |
| --- | ---: | ---: |
| 64 | 57.18% | 63.86% |
| 128 | 74.46% | 79.94% |
| 256 | 88.39% | 93.16% |

전체 9,657개 proposal cache를 약 292초에 구축했다. 선택된 K는 256이다. Calibration의 목표 90%를 달성했다는 뜻은 아니며, 정해 둔 상한 규칙을 적용한 것이다. Recall 개선을 Top-1 정확도 또는 SS 성능 향상으로 표현하지 않는다.

## 새 head와 자원 배치

H1/H2/H3는 공통 population/seed42017/query sampling으로 각각 fixed-final 7 epochs를 학습한다. NULL 감독에서는 후보 hit, 실제 가시성, 선택한 correspondence의 정확도를 구분한 세 출력을 학습한다. `p_use`의 label은 후보에 정답이 있는지만으로 정하지 않고 실제 선택이 맞았는지를 사용한다.

Source raster와 입력 alpha의 IoU가 0.98 미만인 사례에서 가시성이 확인되지 않는 query는 불확실한 NULL 감독으로 간주해 제외한다. 가시 정답이 확인된 query는 유지한다. 같은 sampling 정책을 H1/H2/H3에 공통 적용한다. Sampling 규칙을 확정하기 전의 H1 partial은 `diagnostics/history/h1-before-uncertain-null-filter`에 보존하고 비교 대상에서 제외했다.

| 작업 | 실행 방식 |
| --- | --- |
| H1 | 후보 확대 + ranking loss, GPU 한 장 |
| H2 | H1 + NULL/usefulness/visibility 감독, GPU 한 장 |
| H3 | H2 + transported feature cosine loss, GPU 한 장 |
| 공통 baseline와 GT/경로 진단 | 나머지 자원에서 순차/병렬 평가 |

Head 단독 실행의 관측 메모리는 약 1.6 GiB였고 bounded SS 평가와 함께 실행해도 여유가 있어 같은 GPU를 공유할 수 있게 했다. **GPU 한 장에는 작은 작업 1개와 평가 1개까지만** 배치한다. 새로운 SS 본 학습은 exclusive 작업으로 분류해 다른 작업과 GPU를 공유하지 않는다. 장치 배치는 runtime queue에 기록하며 고정 물리 GPU 목록은 재사용 코드에 내장하지 않는다.

## 공통 panel에서 다시 확인한 기존 모델

32개 공통 panel, sampler12 / CFG3 / generation batch8 기준이다.

| 기존 모델/조건 | Voxel IoU |
| --- | ---: |
| Source-only zero-shot | 0.394997 |
| Edited-RGB zero-shot | 0.426715 |
| V1 final | 0.437100 |
| B3 final | 0.438276 |
| B3에서 추가 semantic 경로만 OFF | 0.438709 |
| C1 final | 0.452509 |
| C1에서 adapter만 OFF | 0.435895 |
| C1에서 추가 semantic/adapter 경로 OFF | 0.435649 |

128개 공통 panel의 재평가도 완료했다.

| 기존 모델/조건 | Voxel IoU | V1 대비 차이 |
| --- | ---: | ---: |
| Source-only zero-shot | 0.379171 | −4.893%p |
| Edited-RGB zero-shot | 0.416098 | −1.200%p |
| V1 final | 0.428099 | 기준 |
| B1 final | 0.430146 | +0.205%p |
| B2 final | 0.430537 | +0.244%p |
| B3 final | 0.430925 | +0.283%p |
| C1 final | 0.440281 | +1.218%p |

기존 C1과 V1의 차이 **+1.218%p**에 대해, 93개 source cluster를 단위로 한 paired bootstrap 4,000회의 95% 구간은 **[+0.460, +2.013]%p**다. 이는 하나의 학습 seed 결과이므로 여러 training seed에서의 재현성을 뜻하지 않는다. B1/B2/B3의 작은 점추정 차이에 통계적 유의성을 주장하지 않는다.

**이전 4개 quick panel에서 관측한 C1 하락을 전체 성능으로 일반화하면 안 된다.** 공통 128개 평가에서는 C1의 양의 차이가 관측됐다. Adapter OFF 진단은 같은 학습 weight에서 출력을 제거한 비교이며, adapter 없이 새로 학습한 독립 arm과 같은 의미는 아니다. 어떤 source 정보를 adapter가 이용하는지는 추가 wrong-object/GT 진단으로 구분한다.

## GT transport 진단과 본 학습의 진행 조건

동일한 V1 checkpoint에서 raw source feature를 GT correspondence로 주입한 32개 validation 결과는 native **0.437100**, GT transport **0.436965**다. 작은 gate에서 주소만 정확하게 바꾼 것으로는 개선이 확인되지 않았다. 학습된 B3에서도 normal **0.438276**, GT transport **0.436163**이다.

Train calibration 32개에서 주입 강도를 점검한 결과:

| 조건 | Voxel IoU |
| --- | ---: |
| Native | 0.426813 |
| GT / scale0.0431 | 0.426816 |
| GT / scale0.25 | 0.421874 |
| GT / scale0.5 | 0.408292 |

Calibration에서 선택된 scale은 기존 작은 scale과 같았으며, validation에서의 개선 근거는 없다. 따라서 계획의 “GT matching에서도 개선이 없으면 주입 구조를 먼저 재검토” 조건에 따라 **새 B2-v2/B2-v2-R/B3-v2 본 학습은 transport gate에서 HOLD**한다. 이 HOLD를 실행 오류 또는 사용자 승인 대기로 표현하지 않는다.

기존 C1의 추가 32개 진단에서 normal **0.452509**, GT correspondence **0.440061**, wrong-object source feature **0.448433**이었다. Wrong-object는 native RGB와 target query를 유지한 채 semantic 경로의 source value만 같은 batch의 다른 asset 값으로 교체한다. GT 주소로 바꾸는 것이 개선되지 않았고, 잘못된 object의 feature를 넣어도 adapter 효과가 상당 부분 남는다. 따라서 C1의 양의 결과를 정확한 semantic matching의 효과로 해석할 수 없다. Source value의 물체 식별 정보와 retargeter가 만드는 보정 성분을 더 구분해야 한다. 이 32개 경로 제거/교체 진단은 독립 재학습 ablation이 아니다.

H1/H2/H3 학습은 계속한다. Epoch 2 validation Top-1은 H1 **20.09%**, H2 **20.11%**, H3 **21.38%**로 각 epoch 1보다 증가했다. Fixed-final epoch 7 이전의 중간 수치를 최종 선택에 사용하지 않는다. 새 SS arm의 5-epoch 학습을 이미 시작했다고 보고하지 않는다.

## 구현과 검증

- 후보 수가 256이어도 SS 학습에서 `[B,Q,K,1024]`를 매번 만들지 않도록 frozen head의 Top-1/weighted evidence를 사전 집계하는 새 cache 경로를 구현했다. Native projection과 LoRA continuation은 유지한다.
- 새 cache는 RGB input fingerprint, sample/split, head/contract SHA로 fencing한다. C1에는 deformation 채널이 없고 C2에서 RAW2048 채널을 물리적으로 추가한다.
- 실제 GPU에서 epoch1 head를 사용한 **진단용** integration smoke를 수행했다. B2-no-reject/B3-reject/C2-reject 모두 native condition 최대 절대오차 0, fusion gate gradient 양수를 확인했다. C2 adapter output gradient도 양수였다. 이 epoch1 head를 production matcher로 채택한 것은 아니다.
- 새 SS 본 학습이 진행 조건을 통과할 경우 최종 head/cache로 native proof와 최대 microbatch의 실제 2-update memory probe를 다시 수행하도록 구성했다. Global96/micro96의 새 본 학습 capacity 검증이 이미 완료됐다는 뜻은 아니다.
- 관련 CPU 회귀 검사 113개가 통과했다. 안정적인 sample seed, cluster-disjoint calibration, variable K padding, 초기 native 보존, rejected query의 adapter 포함 전체 fallback, cache fencing과 GPU 자원 배타성을 확인했다. 실제 SciPy calibration 경로도 별도 확인했다.

## 보관·관측 위치

- File4 root: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/`
- Local runtime: `/root/dev/.cache/deformnext-retargeting/ss-matching-v2-20261017/`
- Queue: `queue-jobs.json`, `queue-state.json`, `queue.log`.
- 진행 조건: file4 root의 `oracle-scale.json`, `transport-gate.json`; SS가 실제 실행될 경우 `ss-decision.json`, `c-promotion.json`.
- 새 weights: `heads/{h1,h2,h3}/`; inference cache를 생성하는 경우 `matches/`.
- 평가: `screening/`, `evaluations/`, `calibration-oracle/`, `retargeting-oracle/`.
- 비교 근거: `diagnostics/c1-v1-common128.json`, `proposal-audit.json`, `smoke/verification/`.
- W&B [공통 평가](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-matching-v2-evaluations), [H1](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-matching-v2-certified-h1), [H2](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-matching-v2-certified-h2), [H3](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-matching-v2-certified-h3).

File4 sync worker와 평가 reporter를 별도로 실행한다. 완료된 공통 panel의 voxel metric과 GT/pred/diff 이미지를 W&B에 기록한다. 기존 checkpoint를 보존한 storage sync도 확인했다. 종료 시 생성되는 `campaign-result.json`은 완료/보류/실패 작업을 구분한다.
