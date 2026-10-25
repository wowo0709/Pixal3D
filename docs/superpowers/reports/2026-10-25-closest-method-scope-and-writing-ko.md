# 가까운 선행연구와 실제 구현의 구분

2026-10-25 UTC. GPU 학습을 유지하며 세 논문의 Method와 한 공식 코드의
설계를 대조했다. 새로운 실험 arm·외부 모델 실행·데이터 다운로드는 없다.
서로 다른 문제 설정의 기존 성능을 우리 모듈의 효과로 사용하지 않는다.

| 근거 | 직접 확인한 내용 | 논문 반영 |
|---|---|---|
| [Pixal3D §3.2.2–3.2.3](https://arxiv.org/html/2605.10922) | Published Direct3D-S2/DINOv2, projection·multi-view averaging | Supplied native TRELLIS.2/DINOv3 구현·checkpoint lineage와 구분 |
| [MVDiffusion §4.1–4.2](https://arxiv.org/html/2307.01097) | 주어진 대응 주변 local attention·zero-init residual | Correspondence attention과 초기 함수 보존 자체를 신규 기여로 주장하지 않음 |
| [ShapeUP §3.1–3.2](https://arxiv.org/html/2602.05676) | Geometry의 source shape+edited image, texture의 source multi-view+edited image/shape 조건 | Source-view conditioning 자체의 최초 제안 claim 제외 |
| [CorrAdapter 공식 MV-Adapter 구현](https://github.com/SuhZhang/CorrAdapter/blob/a269d224f89aa7ae2bed1b4c61b2613352e8a1d2/image-conditioned/static/MVAdapter/mvadapter/models/attention_processor.py#L517-L616) | Cross-view match 선택·local window 집계·기존 attention과 blend | Global semantic Top-K/ray 전달과의 구조 차이로 설명하고 우위는 주장하지 않음 |

CorrAdapter 본문 PDF는 직접 fetch에서403이므로 상세 메커니즘은 공식
MV-Adapter 구현을 기준으로 했으며 모든 variant나 전체 원문을 검증했다고
하지 않는다. Official commit은 `a269d224f89aa7ae2bed1b4c61b2613352e8a1d2`,
해당 file Git blob은 `ba3c319fd17767370ca2dfc7b2dcfb9c2aeffa0b`다.
현재 기본 branch와 이 immutable ref의 file identity도 일치했다.

실제 진행 중 SS config에서 `DinoV3ProjFeatureExtractor`, native grid16,
SS4210 continuation을 확인했다. GT source/target geometry를 생성 이후에
읽는 기존 cascade 경로도 대조했다. 이 검토는 model forward가 아니다.

논문의 중심은 **서로 다른 geometry state를 나타내는 관측에서 semantic
evidence를 기존3D projection 경로에 전달·적응시키고, geometry와 appearance를
따로 평가하는 것**이다. 기존 요소를 최초 제안했다고 쓰거나 외부 방법을
실행하지 않은 채 성능 우위를 주장하지 않는다. 각 모듈의 최종 empirical
claim은 진행 중인 matched ablation과 TEST의 직접 근거로 결정한다.

`related-work-verified.tex`와 claim 원장에 위 범위를 반영했다.
`paper-draft-close-prior-method-scope-v339`는13쪽·24개 인용으로 compile됐고
warning·미해결 인용·overfull/underfull0이다.1·2·6쪽을 실제 PNG로 검수했다.
신규 control과 TEST의 미측정 cell은 채우지 않았다.

Canonical `close-prior-method-scope-review-v338`의 source inspection record,
실제 config snapshot·audit3개/23,274bytes를 file4 SHA256/size로 보존했다.
이는 본문 일부·공식 코드 대조이며 exhaustive literature review가 아니다.
