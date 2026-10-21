# Teacher loss 진단과 완료된 PBR R-only 비교

2026-10-21 UTC. [앞선 source-input 결과](2026-10-21-source-input-pilot-results-and-joint-integration-ko.md)의
후속 진단이다. 진행 중인 PBR Direct의 결과는 포함하지 않는다.

## Teacher의 작은 개선 폭: 먼저 마스크 정의를 맞췄다

TRAIN1536 전체의 M1 source9/10 teacher cache와 M0 source feature를 읽어
변화 에너지 `sum((teacher-source)^2)`를 계산했다. NN forward/backward와
optimizer 실행 없이 CPU 2개에서 수행했다. 전체 MSE 0.007035208251927094는
기존 1536개 baseline과 asset별로 일치한다.

기존 source 후보 마스크는 **패치 평균 alpha >0.05**이고 teacher 진단용 마스크는
**패치 중심 alpha >0.5**다. 둘을 섞으면 경계 패치를 “사라진 형상”으로 과대계산한다.
초기 `region-energy-CPU-v1`의 혼합 기준 해석은 v2로 대체했다. 원본 결과는 보존했다.
이는 학습 loss에 있던 마스크 오류가 아니다. 실제 teacher loss에는 foreground
마스크를 사용하지 않았다.

M0/M1 실제 RGB 6144장을 읽어 TRAIN1536의 cached source mask와 teacher mask가
각자의 원래 규칙으로 정확히 재구성됨을 확인했다. 동일 규칙끼리 비교하면:

| 양쪽에 적용한 규칙 | foreground union 패치 비율 | union 변화 에너지 비율 | 배경 변화 에너지 비율 |
|---|---:|---:|---:|
| 중심 alpha >0.5 | 9.30% | 48.61% | 51.39% |
| 패치 평균 alpha >0.05 | 12.08% | 57.90% | 42.10% |

같은 패치 평균 기준에서 M0에만 foreground인 패치는 0.731%, M1에만 foreground인
패치는 0.312%다. 혼합 규칙의 3.224%/0.055%를 물리적 소실/생성 비율로 사용하면 안 된다.
이 검사는 alpha/cache 생성 규칙을 확인하며 전체 camera/geometry 정합을 새로
증명하는 검사는 아니다.

전체 패치 평균 loss가 foreground 학습 신호를 희석할 가능성은 있다. 그러나 배경에도
feature 변화 에너지의 42–51%가 존재한다. DINO 배경 descriptor도 물체 변화의
영향을 받으므로, foreground-only loss를 바로 사용하지 않는다. Foreground union만
평균낼 때의 정규화 배율 8.28–10.75배는 실제 Q/K gradient 증가량이나 품질 개선이
아니다. Loss 변경·epsilon 변경·추가 teacher 학습은 이 진단으로 자동 실행하지 않았다.

## 완료된 PBR1905: R-only와 V1

둘 다 같은 V1 762-update prefix 이후 global24/world4/micro6/accumulation1로
1143 updates를 추가해 total1905까지 학습했다. 32 assets/32 source clusters의
RGB·GT PBR latent·GT Shape·sampling noise hash와 foreground mask 비율이
모두 짝지어진 완료 결과를 사용했다. 4000 source-cluster bootstrap draws,
seed20261020이다.

| 지표 | V1 | R-only | R−V1 | 95% CI |
|---|---:|---:|---:|---|
| Foreground PSNR ↑ | 15.60537 | 15.89121 | +0.28584 dB | [−0.20313, +0.70340] |
| Target crop LPIPS ↓ | 0.120409 | 0.116477 | −0.003932 | [−0.009224, +0.000682] |
| Target crop SSIM ↑ | 0.920899 | 0.922800 | +0.001901 | [−0.001450, +0.005490] |

세 평균은 개선 방향이나 신뢰구간이 모두 0을 포함한다. 단일 training seed와
generation noise의 개발32 결과로, PBR R-only 효과가 확정됐다고 하지 않는다.
이 R-only는 기존 target-patch PBR 구조이며, 새 SS SourceContrast19 teacher와
다르다. R-only에는 attention과 RAW deformation-query projection이 물리적으로 없다.

GT Shape를 고정하고 입력 target camera08에서 렌더링한 **texture stage 진단**이다.
이 crop LPIPS는 논문의 source preservation LPIPS가 아니다. 생성 geometry의
CD/F-Score나 novel-view/source adaptation 개선도 이 결과로 주장하지 않는다.
현재 학습 중인 PBR Direct가 끝난 뒤 동일 조건으로 V1/R-only/Direct를 비교한다.
Full cascade와 요청한 source/target/global naturalness 평가 범위는 유지한다.

## Native SS joint 평가 경로

`evaluate_source_input_joint.py`를 추가했다. 동일한 완료480 run의 config,
denoiser/source checkpoint, progress, native startup, source initialization report를
함께 요구한다. Teacher112를 joint480 결과로 바꾸어 표기하거나 다른 run의 denoiser와
섞는 입력을 거부한다. 추론에는 teacher label이나 초기 teacher checkpoint를 읽지 않고
저장된 joint source state를 strict load한다. 평가 뒤 source state가 checkpoint와
완전히 같은지, 원래 optimizer state가 비어 있는지도 확인한다.

완료 파일 binding에 대한 CPU 검사 9개와 기존 19개를 합쳐 **28개 검사 통과**했다.
새 구조의 joint480 학습과 그 checkpoint의 실제 GPU 평가는 아직 실행하지 않았다.
실제 source-disjoint72×3 확인 후 자기 native SS gradient/capacity와 warm prediction
동등성을 검증하고, 동일 teacher supervision을 받은 Original19/SourceContrast19를
같은 TRAIN9149 budget으로 비교한다. 독립 training seed 검증도 남아 있다.

## Canonical 근거

Campaign `feature-readout-v1/` 아래:

- `source-feature-teacher-pilot-v1/TRAIN1536-teacher-region-energy-CPU-v2/`
- `direct-retargeting-v1/PBR-joint-continuation-v1/retarget-only-v1/final1905-vs-V1-stage32-CPU-v1/`
- `failure-driven-diagnostics-v1/teacher-energy-and-PBR-R-stage-publication-v1/`

완료 artifact의 `file4-closure.json`에 실제 원격 SHA256/size 검증이 있다.
진단 그림은 PNG/PDF, 표는 CSV로 저장하고
[W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-teacher-energy-and-PBR-R-stage32-v1)에 기록한다.
