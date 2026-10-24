# 필수 score 대조 실행 연결과 독립 학습 결과의 논문 반영

2026-10-24 21:45 UTC. 이전 goal turn은 실제 계획서·claim 원장 수정과 commit/push를 완료했으므로 progress다. 이번에는 실행 연결과 실제 논문 결과 반영을 진행했다. 전체 목표인 최종 모델 선정·matched 대조·독립 확인·TEST508·최종 논문은 아직 미완료다.

## 1. 유지 중인 실제 작업

- GPU0–3: seed202 PBR vanilla1900 DDP4. 실제 wrapper/child가 살아 있고 학습 progress가 증가한다. R PBR은 다음 의존 작업이다.
- GPU6: Learned의 추가 DEV cascade. R/Fixed의 추가96개씩은 완료됐고 Learned의 남은 사례를 처리한다. Official metric·전체 paired 집계는 별도 후속 작업이다.
- GPU7: 원래 FULL의 voxel-query DEV128×3. Noise0/1 완료, noise2 실행 중이다. 전체12모델 통계는 세 noise가 실제 종료된 뒤 기존 collector가 집계한다.

기존 NN 프로세스를 중단하거나 재시작하지 않았다. Voxel checkpoint 묶음의 config hash 문제는 terminal LF1byte 차이로 확인하고, 원본 학습·checkpoint를 변경하지 않은 별도 메타데이터 후속 작업으로 복구했다.

## 2. Uniform-K를 필수 대조로 실제 연결

동일 후보 주소·valid mask·identity value·carrier/source-use/R 경로에서 posterior weighting을 uniform으로 바꾸는 `A_FIXED_UNIFORM`을 대상으로 한다. 이는 사용자 정의 No-score=실제 Top-1과 다른 대조이며, 추가 learned Q/K의 효과도 별도다.

| 단계 | 실제 연결 | 통과 조건 |
|---|---|---|
| 초기 GPU 기능 검사 | 이미 완료 | Native 초기 conditioning/flow 일치, 실제 backward, Uniform weight 정확성, optimizer0 |
| Micro96 capacity | GPU7의 기존 voxel 평가 뒤 등록 | 단일 GPU·global96·micro96·accum1, 실제 임시3 updates, 초기native32, frozen base, source gradient·optimizer 상태 |
| 과학 학습 | Capacity 완료에 의존해 등록 | 원본 SS4210 LoRA에서 새 학습480 updates/5epochs, 동일 DeformNeXt9149·seed101·native objective·학습 noise, epoch별 validation |
| Checkpoint 묶음 | 과학 학습 완료에 의존 | 실제 학습 terminal·frozen 상태·config byte identity·own physical source10개·동일480 denoiser/source pair |
| SS DEV128×3 | Checkpoint 묶음 완료에 의존해3개 등록 | 같은 기존128 asset/93 source·3 paired generation noise·원래 sampler, 실제 prediction/GT archive |

Capacity를 아직 통과한 것으로 쓰지 않는다. Capacity가 실패하면 본 학습은 실행되지 않으며, 본 학습과 SS 평가 완료도 아직 주장하지 않는다. 새 과학 학습은 이 필수 Uniform-K 하나만 등록했다. Final No-R/learned-logit-prior 대조의 본 학습은 최종 후보 선택에 따라 필요한 것만 진행한다.

외부 결과 경로는 `PBR_NATIVE_SCIENCE_RUNTIME/unified-module-pipeline-v1/shared-evidence-final-controls-v284` 아래다.

- `UniformK-capacity-GPU7-v287`: 실제 proof 기반 capacity config/plan/queue와 file4 SHA/size closure5개.
- `UniformK-science-and-SS-evaluation-v289`: 조건부 과학 학습·sealer·3개 SS 평가 config/plan/queue, file4 closure12개.
- 출력은 `A_FIXED_UNIFORM/{capacity-v1,science-v1,final-evaluation-v1}`에 보존한다.

Science config를 native writer와 동일한 indent2+terminal LF로 미리 저장해 strict byte hash 비교를 유지한다. 시작 시 checkpoint 성장 예약을 포함한 실제 disk budget을 다시 검사한다. Data/원본 checkpoint/기존 final output은 삭제하지 않는다.

## 3. 새 대조의 cascade 연결 구현

`experiments/deformnext_retargeting/shared_ablation_pipeline_contract.py`를 추가했다. 기존 E_FIXED/E_LEARNED로 이름을 바꾸지 않고 각 대조의 own schema, No-R의 실제 source5/12개, Uniform10개, no-logit-prior17개를 구별한다. Config·denoiser·source checkpoint의 같은480 경로와 SHA256, 실제 평가의 source/head 구성·RGB4 입력·sampler·128개 모든 예측/GT/noise를 검사한다.

GPU case worker는 기존 native Shape/PBR/render 호출을 유지하고 own binding 검증과 prefix를 포함한128개 admission을 사용하도록 별도 준비했다. 원본 worker는 변경하지 않았다. 이 연결의 단위/계약 검사54개와 기존115개가 합쳐 **169 passed in1.43s**다. 실제 학습된 새 checkpoint의 full cascade forward·8개 metric 계산은 아직 실행하지 않았으므로 구현/검사 완료와 품질 평가 완료를 구분한다.

## 4. 완료된 독립 SS 결과를 실제 논문에 반영

실제 `NATIVE_TWO_TRAINING_SEED_SUMMARY`의 closed `statistics.json`/`audit.json`을 사용했다.

| Training seed | V1 macro IoU | Top-1 semantic+R | Paired 개선 | Source-cluster CI95 |
|---|---:|---:|---:|---|
| 101 | 0.467355 | 0.472600 | +0.005245 | [+0.001674,+0.009030] |
| 202 | 0.467177 | 0.472346 | +0.005169 | [+0.001780,+0.008677] |

동일 DEV128/93 source·generation noise3회를 사용한다. 각 seed 안의480개 input/noise/timestep trace pairing, 두 seed 간 실제 학습 stream 차이, 원본 LoRA 초기값 및 최종4개 checkpoint의 frozen701 NN tensor 일치를 확인했다. R MLP나 matching decoder 단독 효과로 해석하지 않는다.

논문 본문과 `tables/dev-independent-ss.tex`, README를 갱신했다. 두 고정 checkpoint pair의 source-cluster CI를 training-seed 모집단 CI로 서술하지 않으며, 최종 cascade/TEST 일반화로 확대하지 않는다.

실제 컴파일 결과는 `paper-draft-independent-SS-pairs-v288`에 보존했다. Compiler return0,11pages,24references,warning/undefined/overfull/underfull0이며 새 표 수치를 closed 통계와 소수점6자리까지 대조했다. 새 표/문단이 있는6·7페이지를 렌더링해 직접 확인했고26개 artifact를 실제 file4 SHA/size와 대조했다. 로컬 preview는 `/root/dev/.cache/deformnext-retargeting/paper-preview-20261024-v288.pdf`다. 이는 최종 TEST 결과가 없는 개발 초안이다.

## 5. 다음 종료 조건

21:49 UTC 후속: 원래12모델/10모듈/2 pipeline 비교 통계가 실제 완료·file4 보존·[W&B 로깅](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-unified-twelve-model128x3-s101-v166)됐다. Uniform-K capacity도 GPU7 wrapper1854741/child1854782에서 실제 실행 중이다. 아래의 original family 완료 항목은 해소됐다.

| 원래 구조의 비교 | Δ macro IoU | 보정 family CI95 |
|---|---:|---|
| FULL ray − voxel query | −0.000236 | [−0.000878,+0.000351] |
| Retargeting at Top-1 | +0.003929 | [−0.000525,+0.008987] |
| Spatial → raw semantic | +0.001117 | [−0.003255,+0.004969] |
| Raw → decoder descriptor | +0.000648 | [−0.003386,+0.005441] |
| 원래 FULL − V1 | +0.004723 | [+0.000288,+0.009394] |
| 원래 FULL − Top-1 R | −0.000522 | [−0.003339,+0.002567] |

모듈은10개 family, pipeline은2개 family로 보정했다. 모든10개 module CI는0을 포함한다. 효과 부재나 동등성을 주장하지 않으며, 원래 FULL과 새 shared identity-value 구조를 혼동하지 않는다. 이 결과를 숨기지 않고 새 `tables/dev-original-module-effects.tex`,14개 endpoint 표와 본문/claim 원장에 반영했다.

후속 compiled 초안은 `paper-draft-original-family-v292`다.11pages·24refs·warning0,10개 Δ/보정 CI를 실제 closed 통계와 소수점6자리까지 대조했고 새 표/문단의6·7페이지를 직접 검토했다.27개 artifact의 실제 file4 SHA/size가 일치한다. 최신 로컬 preview는 `/root/dev/.cache/deformnext-retargeting/paper-preview-20261024-v292.pdf`다.

1. 완료된 원래 voxel-family 통계를 반영하고 기존 DEV128 실제8개 metric 집계 완료.
2. Uniform-K의 capacity/480/128×3와 같은 panel downstream 품질 확인.
3. 최종 matching ladder와 선택 모델의 물리적 No-R·필요한 ray/learned 대조 보완.
4. 선택 모델·필수 PBR의 독립 확인, fresh-load/protocol freeze 뒤 TEST508.
5. Main10metric·최종 ablation·기전/정성 그림과 실제 CI에 맞춰 Results/Discussion/Abstract/Conclusion 확정.

최종 구조와 효과가 확정될 때까지 [필수 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)의 전체 범위를 유지한다. 새 backbone/head/grid를 추가하지 않는다.
