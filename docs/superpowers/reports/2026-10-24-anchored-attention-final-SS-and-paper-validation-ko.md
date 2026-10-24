# Anchored attention의 최종 SS 평가와 다음 cascade

2026-10-24 15:42 UTC, `consistent-raw-evidence-followup-v177/four-model-summary-v1/audit.json`의 실제 canonical closure를 확인했다. 동일 DEV128 asset·93 source cluster, training seed101 하나,480 updates, 생성 noise3개의 asset별 평균 macro IoU다.

| 모델 | Macro IoU |
|---|---:|
| Vanilla V1 | 0.467355 |
| Top-1 R | 0.472600 |
| E_FIXED_RAW | 0.473471 |
| E_LEARNED_RAW | **0.474794** |

| 사전 정의한 비교 | IoU 차이 | 해당2-contrast family의 보정 구간 |
|---|---:|---|
| Fixed − R | +0.000871 | [−0.001248,+0.003135] |
| Learned − fixed | +0.001323 | [−0.000409,+0.003161] |
| Learned − V1 | +0.007439 | [+0.002585,+0.012504] |
| Learned − R | +0.002193 | [−0.000521,+0.004981] |

같은 evidence 전달·identity value를 유지하고 Q/K·utility만 추가한 새 경로에서 양의 점추정을 확인했다. 그러나 learned−fixed와 learned−R의 구간이0을 포함하므로 “attention이 필수”, “R보다 확실히 좋다”는 결론은 아직 아니다. 원래 FULL의 음의 SS 점추정을 소급해 양의 결과로 바꾸지 않는다. Frozen prior와 같은 초기 함수를 유지한 별도의 follow-up이다.

실제 W&B는 [shared-evidence SS128×3](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-consistent-raw-evidence128x3-s101-v177)에 기록됐다. 이는 TRAIN seed3개가 아니라 **generation noise3개**다. 추가 학습 seed 확인은 별도로 필요하다.

이 결과는 기존 계획의 “유력하면 learned 후보 cascade 추가” 조건에 해당한다. E_LEARNED_RAW의 실제 SS480 checkpoint·conditioner·noise0 occupancy archive를 사용해 동일 DEV32를 원본 frozen Shape512/1024와 학습된 Top-1 R PBR에 연결한다. 새로운 학습 arm을 추가하지 않는다. 기존 V1/R baseline을 재사용하고 모델 선택·TEST는 해제하지 않는다. Full-cascade 결과를 stage IoU에서 추정해 채우지 않는다.

논문 SS 표와 본문·claim 원장에 이 수치와 구간을 반영했다. Matching 진단은 별도 [기전 근거 기록](2026-10-24-matching-mechanism-paper-evidence-ko.md)과 분리한다.
