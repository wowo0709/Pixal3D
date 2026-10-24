# DEV128 실제 cascade 완료와 Fixed 문맥의 필수 대조·독립 확인

2026-10-24 22:10 UTC. 이전 goal turn은 원래12모델 전체 통계·Uniform-K 실제 용량 검사/본 학습·논문 갱신을 완료했으므로 progress다. 이번에는 실제 cascade128 집계로 working candidate를 결정하고 필요한 No-R 및 독립 seed202를 연결했다. 최종 TEST freeze나 전체 목표 완료가 아니다.

## 1. 실제 동일 DEV128·8개 metric 집계 완료

128 examples/93 source clusters, training seed101, generation noise1회다. 각 모델의 기존32개 결과를 재사용하고96개를 추가해384개 actual pipeline cases를 완성했다. 공식 DreamSim/CLIP의864 views를 추가 측정하고288 views를 재사용해 총1152 views를 연결했다. Frozen Shape와 실제 Top-1 R PBR는 공통이며 GT는 원래 material을 M1 geometry에 적용한 동일 reference다. TEST/FID/KID 결과가 아니다.

| SS + 공통 R PBR | Source LPIPS↓ | Source DreamSim↓ | Source CLIP↑ | Target PSNR↑ | Target SSIM↑ | Target M-IoU↑ | CD↓ | F-Score↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-1 R | 0.053227 | 0.148809 | 0.939061 | 24.07381 | 0.948434 | 0.890535 | 0.0127795 | 0.840095 |
| Fixed | 0.053438 | 0.151909 | 0.940282 | 24.01288 | 0.948494 | 0.892593 | 0.0127572 | 0.844058 |
| Learned | 0.053275 | 0.151001 | 0.940279 | 24.04810 | 0.948579 | 0.890776 | 0.0128053 | 0.842134 |

Source는 camera9/10, target image metric은 camera8이며 geometry는 공통 좌표계의 actual generated surface에서 측정했다. Generation noise3회 평균인 SS128 study와 혼동하지 않는다.

| Fixed−R | Paired 변화 | 탐색용 source-cluster CI95 |
|---|---:|---|
| F-Score | +0.003963 | [+0.001011,+0.007314] |
| CD | −0.00002230 | [−0.00015369,+0.00012960] |
| Source LPIPS | +0.000211 | [−0.000344,+0.000759] |
| Source DreamSim | +0.003100 | [−0.001178,+0.007689] |

DEV32 screening 뒤 확장한 탐색적 panel이므로 확인용 TEST 통계로 쓰지 않는다. CI0 포함은 동등성/비열등성을 뜻하지 않는다. Learned−Fixed의 geometry 점추정은 CD+0.00004809/F-Score−0.001925로 열위이고 해당 CI는0을 포함한다. SS IoU 최고점만으로 Learned를 선택하지 않는다.

실제 통계와 file4 closure는 `PBR_NATIVE_SCIENCE_RUNTIME/unified-module-pipeline-v1/paired-DEV128-stage-evaluation-v240/metrics-v257/summary-v1/{statistics.json,audit.json,file4-closure.json}`에 있다. [실제 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/paired-three-pipeline-DEV128-noise0-v246)에도 집계됐다.

## 2. Working candidate와 채택하지 않은 claim

**Fixed SS + Top-1 R PBR**을 다음 matched 대조와 독립 확인의 working candidate로 삼았다. 관측 geometry와 단순한 readout이 선택 이유다. R은 source LPIPS/DreamSim이 더 좋은 강한 기준선으로 유지한다. 현재 Fixed가 source appearance를 개선했다거나 비열등하다는 주장은 하지 않는다.

실제 decision은 `working-Fixed-pipeline-decision-v294/decision.json`에 별도 file4 closure와 함께 기록했다. 상태는 `WORKING_CANDIDATE_FROM_ACTUAL_DEV128_NOT_FROZEN`이며 final TEST freeze와 구분한다.

Working Fixed는 학습된 Q/K head가 없는 matching-score 가중 결합이다. 이를 learned QKV attention으로 표현하지 않는다. Ray 단위 공유는 구조 설명이며 원래 FULL의 ray−voxel 결과로 품질상 필수성을 주장하지 않는다. Member는 coverage 진단과 구조를 설명하되 독립적인 생성 품질 기여를 입증했다고 쓰지 않는다.

## 3. 선택 문맥의 No-R를 실제 실행

`A_FIXED_NO_R`는 같은 Fixed carrier·source-use·candidate·identity value를 유지하고 retargeter를 물리적으로 제거한다. 원래 Top-1 No-R를 이 최종 문맥의 결과로 바꾸어 쓰지 않는다.

GPU6에서 own micro96 capacity가 실제 완료됐다. Source tensor5개, optimizer states365, global96/micro96/accum1, 임시3 updates, 모든 source gradient positive finite, frozen base exact 및 initial native32 identity가 통과했다. Peak allocated88.970GiB/reserved93.762GiB다. 이후 actual science wrapper1867835가 실행돼5에폭/480 updates 학습을 시작했다. 장기 무OOM이나 품질 향상은 아직 증명하지 않았다.

`shared-evidence-final-controls-v284/Fixed-NoR-capacity-GPU6-v295`와 `Fixed-NoR-science-and-SS-evaluation-v295`에 각각5개/12개 prelaunch artifact의 실제 file4 SHA/size closure가 있다. Capacity→science480→own checkpoint 묶음→동일 DEV128×3 평가를 연결했다. Uniform-K는 GPU7에서 계속 학습한다. Learned No-R/no-logit-prior의 본 학습은 working candidate에 없으므로 추가하지 않았다.

## 4. 선택 Fixed의 seed202만 추가

`pipeline_confirmation_recipe_v2.make_confirmation`의 엄격한 계약으로 proof/capacity/science config를 만들었다. 바뀌는 것은 seed202와 선언된 run/reference artifact 경로다. 원본 모델·data·optimizer·FFN chunk·LoRA initializer를 유지한다. 기존 실제 V1seed202의 완료된480개 trace/startup을 pairing reference로 사용한다.

`selected-Fixed-seed202-v297/prelaunch-v1`의15개 artifact가 file4와 일치한다. Own proof→micro96 capacity→fresh480 science→checkpoint 묶음→DEV128×3의7개 작업을 GPU7 Uniform-K 평가 뒤에 연결했다. 이 새 seed의 실제 GPU proof/학습은 아직 시작 전이며, parent seed101 proof로 대체하지 않는다. Numerical reference202는 recipe-only이며 학습된 모델로 세지 않는다. V1/R을 다시 학습하거나 모든 ablation의 seed를 반복하지 않는다.

## 5. 논문과 남은 필수 작업

Actual24개 metric cell을 `tables/dev-joint128.tex`로 내보내고 paired 변화·탐색적 CI·working decision과 appearance 한계를 본문/claim 원장에 반영했다. 표는 축소로 글자가 작아지는 문제를 직접 확인해 짧은 grouped header와 native small text로 수정했다. 최종 컴파일은 `paper-draft-DEV128-readable-v298`, preview는 `/root/dev/.cache/deformnext-retargeting/paper-preview-20261024-v298.pdf`다.12pages·24refs·warning0,9페이지를 실제 렌더링해 검수했다.24개 값이 실제 closed 통계와 표시 정밀도까지 일치하며27개 artifact의 file4 SHA/size가 일치한다.

남은 핵심은 아래와 같다.

1. Uniform-K/선택 No-R의480·SS128×3·actual cascade 품질 평가.
2. **동일 Fixed 전달 경로의 Spatial→Raw→D1 cosine→D1 scalar** 대조를 구현·검증하고 누락된 학습/평가만 실행. 기존 Top-1 ladder는 해당 문맥의 근거로 보존한다.
3. 선택 Fixed seed202와 필수 PBR vanilla/R seed202의 실제 학습·평가 및 최종 계약에 맞는 집계.
4. 선택 checkpoint fresh-load, source/protocol freeze, TEST508·134 source cluster에서 두 zero-shot/V1/R/Ours의10개 metric·paired 통계·정성 결과.
5. 최종 Main/Ablation/Mechanism/Qualitative table·figure와 실제 근거에 맞는 Results/Discussion/Abstract/Conclusion/PDF.

모든 metric의 우위나 모든 모듈의 필수성을 전제로 작업하지 않는다. [전체 필수 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)의 최종 문맥 대조·독립 확인·TEST·논문 범위를 유지한다.
