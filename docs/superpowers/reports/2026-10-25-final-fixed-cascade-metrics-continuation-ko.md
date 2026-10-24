# 최종 Fixed 대조의 실제 cascade·metric 후속 실행 연결

실행 관측: 2026-10-24 23:37 UTC. 이전 턴은 핵심 실험 행렬과 논문 산출물을 갱신했다. 이번 턴은 실제 기준선/GT 재사용을 검증하고, 다섯 대조군의 SS 이후 평가 작업과 논문용 집계를 연결했다. 신규 대조군의 품질 측정이나 최종 목표 완료는 아니다.

## 1. 실제 구현과 등록한 작업

`final_fixed_ablation_cascade.py`는 다음 다섯 arm의 own schema·checkpoint·source count·물리적 모듈 상태를 유지한 채 기존 native cascade를 사용한다.

- M_FIXED_SPATIAL / M_FIXED_RAW / M_FIXED_D1_COS.
- A_FIXED_UNIFORM / A_FIXED_NO_R.

각 arm의 실제480 checkpoint 묶음과 DEV128 noise0 archive, 평가 producer의 종료 및 file4 보존이 확인돼야 CPU connector가 실행된다. 이후 같은 Shape512/1024, 같은 실제 Top-1 R PBR1900, 같은 camera·material reference를 쓰는128개 사례를 만든다. 원래 spatial3을 semantic64로 가짜 padding하지 않으며 기존 Top-1 모델을 새 control로 재명명하지 않는다.

다섯 connector와 최종 집계의 **6개 CPU 작업**을 실제 main queue에 추가했다. 각 connector가 준비할 것은128개 cascade + coverage/official/summary의3개 metric 작업이다. 다섯 arm 전체는 **640개 신규 cascade 사례와15개 metric phase**이며 별도의 신규 PBR ablation 학습은 없다. GPU case는 필수 PBR seed202의 vanilla/R DDP4 완료 뒤 GPU0–3에서 실행한다. GPU6/7의 진행 중 학습과 등록된 SS 검증은 유지한다.

상태는 `pending`이다. 실제 own SS가 아직 완료되지 않았으므로, connector 등록이나 prepared case count를 실제 생성/품질 완료로 보고하지 않는다. 기존 모델·GT를 다시 렌더링하는 작업도 아니다.

## 2. 실제 baseline·GT 재사용 검증

기존 R/Fixed의 모든128개 case를 원래 metric assembly의 checkpoint/archive/generation/pixel 검사로 다시 읽었다. 두 baseline의 role별 **116개 평균값이 기존 closed DEV128 통계와 정확히 일치**했다. 최대 차이는0이다. 같은384 camera reference의 original-material RGB/alpha identity와128개의 GT reference/render root를 보존했다. CPU/GPU NN forward와 새 GT render는0회다.

`reference-and-baselines-v1`의4개 artifact/2,453,600bytes가 실제 file4 SHA256/size와 일치한다. 이 실제 raw per-view 기록을 최종7모델 집계에서 재사용한다. 평균 source metric을 가짜 individual view로 복제하거나, 다른 panel의 결과를 섞지 않는다.

## 3. 실제 출력 metric과 paired 통계

각 control은 원래 native geometry → R PBR replay → render → 공통 GT scoring → official DreamSim/CLIP의 경로를 사용한다. Official forward는 GPU에서 FP32/TF32 off를 유지한다. CPU 단계는 입력 identity·저장 결과 조립·통계만 처리한다.

집계할8개 endpoint는 source LPIPS/DreamSim/CLIP, target PSNR/SSIM/M-IoU, surface CD/F-Score다. 동일128개/93 source cluster·cascade noise1회이며 SS128×3와 구분한다. 각 모델의 실제 prediction·source cluster·GT RGB/alpha·sampler·PBR checkpoint가 같게 묶였는지 다시 검사한다.

사전 지정한6개 contrast:

1. Raw semantic − original spatial.
2. Decoder cosine − raw semantic.
3. Decoder scalar Fixed − decoder cosine.
4. Fixed − actual Top-1.
5. Fixed − Uniform-K on the same support.
6. Fixed − physical No-R.

여덟 endpoint×여섯 contrast의 **48-comparison family**에 Bonferroni percentile source-cluster bootstrap 구간을 추가했다.48,000 draws로 source를 재추출하고 해당 source의 deformation들을 함께 유지한다. 기존 unadjusted exploratory DEV 결과는 별도 남긴다. 이 family는 TEST의 selected-vs-V1/R family와 다르며, 개발 결과를 최종 held-out 확인으로 표현하지 않는다.

완료되면 canonical 결과 폴더에 실제7모델 통계,8metric W&B 표, measured cascade ablation TeX가 생성된다. 이것이 최종 SS128×3 통합이나 manuscript 편집을 자동으로 완료하는 것은 아니다.

## 4. 검사와 source 보존

새 cascade 검사25개와 관련 기존 검사142개를 합쳐 **167 passed in2.16s**다. 잘못된 모델/seed/PBR·depth 입력·TEST 혼입·일부 sample 누락·source pairing 오류를 거부하며, 원래 numerical template을 유지한다.48개 통계 비교의 tail과 짝 순서도 검사한다. 실제 신규 control의 GPU 품질이나 용량 검사를 대신하지 않는다.

실제 source 감사에서 A/M case worker의 child-session lifecycle 함수와 native GPU 호출 AST가 원래 case worker와 일치했다. Official GPU phase의 차이는 view count864→384와 completion metadata뿐이었다. Audit selector의 첫 버전은 두 개의 `official` 분기를 하나로 가정해 실패했으며, 실제 실행 분기를 명시한 새 감사 버전으로 바로잡았다. 학습/모델 코드 실패나 NN 재실행은 아니었다.

Prelaunch의15개 artifact/705,151bytes와 감사의2개 artifact/17,010bytes가 실제 file4 SHA256/size와 일치한다. 신규 reusable source commit은 `8e509ae`와 `ff1d032`다. 진행 중인 NN source와 기존에 고정한 future M source/worker는 변경하지 않았다.

실제 기존128개 결과의 geometry/PBR/shared 크기는 모델별 약41.5–42.1GiB다. 같은 크기의 다섯 control 추정은 약210GiB이며 당시 local free는 약560GiB다. Render·새 checkpoint·TEST는 이 추정에 포함하지 않았다. 각 case의 원래 disk admission을 유지하며, TEST 전에 별도의 rolling canonical 보존/중간물 정리가 필요하다.

## 5. 논문 source·PDF

Method/ablation 설명에서 다음을 명시했다.

- 같은 Fixed carrier/use/R를 유지하는 것은 matching ladder 행들이며 No-R 행은 실제 R 제거다.
- Cosine→scalar는 posterior와 retained support를 함께 바꿀 수 있다.
- Uniform↔Fixed는 carrier/use/R가 공유하는 evidence weighting의 대조이며 learned attention-logit 한 항만의 효과가 아니다.
- 새48개 DEV family와 최종 held-out8C family를 구분한다.

실제 compiled 초안은 **12pages/24references, compile warning0**이다.5·9·10페이지를 렌더링해 표·텍스트 clipping이 없음을 확인했다.28개 artifact/722,752bytes가 실제 file4와 일치한다. 실제 측정된 Fixed/Top-1 값만 유지했고 새 control과 TEST cell은 pending이다.

[현재 PDF preview](/root/dev/.cache/deformnext-retargeting/paper-preview-20261024-v309.pdf).

## 6. 현재 학습과 남은 목표

23:37 UTC 실제 progress 및 wrapper/child process:

| 실행 | 실제 자원 | 진행 |
|---|---|---:|
| PBR vanilla seed202 | GPU0–3 DDP4 | 577/1900 |
| Fixed Uniform-K | GPU7, global96/micro96 | 162/480 |
| Fixed No-R | GPU6, global96/micro96 | 137/480 |

두 dispatcher와 세 actual wrapper/child가 살아 있음을 확인했다. 작업을 중단하거나 재시작하지 않았다. 새 최종 품질 결과는 아직 없다.

남은 전체 목표는 다섯 control의 학습·SS128×3·실제 cascade8metric, Fixed/PBR seed202의 완료와 독립 집계·fresh load, 최종 선택/protocol freeze, TEST508/134 source cluster의10metric, Main/Ablation/Mechanism/Qualitative 및 최종 writing이다. 최종 SS 표 통합과 실제 근거에 맞는 claim 확정도 남아 있다.48개 대비를 모두 양수로 만들거나 모든 모듈을 필수라고 쓰는 것을 종료 조건으로 삼지 않는다.

## 실제 artifact 위치

공통 canonical result의 `PBR_NATIVE_SCIENCE_RUNTIME/unified-module-pipeline-v1` 아래:

- `final-Fixed-controls-DEV128-cascade-v306/reference-and-baselines-v1`.
- `final-Fixed-controls-DEV128-cascade-v306/connector-prelaunch-v1`.
- `final-Fixed-controls-DEV128-cascade-v306/continuation-audit-v308`.
- `paper-draft-final-Fixed-statistical-scope-v309`.

[남은 핵심 실험·논문 산출물](2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md), [claim 원장](../paper/claim-evidence-ledger.md), [native cascade 연결 source](../../../experiments/deformnext_retargeting/final_fixed_ablation_cascade.py).
