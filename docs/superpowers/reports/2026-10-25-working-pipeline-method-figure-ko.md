# 실제 구현과 맞춘 SS·stage 파이프라인 구조도

2026-10-25. 원래 실험을 유지하면서 Method의 구조 설명을 보완했다.
새 모델·새 학습·추론·quality metric은 추가하지 않았다. 최종 후보를
선택하거나 각 모듈의 성능 우위를 입증한 작업이 아니다.

## 실제 코드와의 대응

| 그림의 요소 | 구현 근거 | 설명 범위 |
|---|---|---|
| Native MV4 carrier | `member_posterior_source_ray.py` trainer와 `add_source_ray_correction` | 원본4view conditioning을 계산한 뒤 projected source delta만 추가; global token 유지 |
| Frozen Omega / D1 | `member_ray_posterior.py` | Raw cosine Top64, frozen decoder/score, source-valid support, temperature0.05 |
| Source-camera ray posterior | 같은 module의 distinct ray/target-patch membership | Member별 normalized posterior를 평균한 뒤 final64 support·재정규화 |
| Native values와 shared E | `consistent_ray_evidence.py` | Omega key와 별개의 DINO1024 value; Fixed는 learned Q/K·V/O 없음 |
| R / source use | `SourceContrastRetargeter`, `evidence_use`, `pool_ray_queries` | R은 `[E-S,T]`, 같은 E를 source-use와 carrier가 사용. Native target context는 ray의 grid sites에서 pool |
| Projected delta | `native_prior_source_ray.py` | `sum_j P_j[u*g*(E-S_j)+u*R]/4`; 별도 target/visibility write mask 없음 |
| Actual cascade | 기존 stage recipe·원본 checkpoint 구성 | SS/PBR LoRA continuation, frozen Shape512/1024, PBR은 별도 Top-1 R conditioner |

현재 SS config는 실제 `DinoV3ProjFeatureExtractor`, image512/grid16,
value/query1024, Top64, score temperature0.05, g_max0.25다. SS64는 decoded
occupancy이며 ray grouping의16³ conditioning grid와 구분했다. Matching의
관측 member 집합과 R/source-use의 native ray context pool도 구분했다.
Empty member 또는 source-valid patch64 미만이면 native prior로 abstain한다.

## 그림과 본문

`draw_pipeline_overview.py`는 weights/data/metrics를 읽지 않는 Matplotlib
vector schematic이다. Git은 재현 가능한 drawing source와 TeX/caption을
소유하고 생성 PDF/PNG는 canonical result에 둔다. `working-pipeline-overview-v361`
아래 첫 sketch는 보존하고, 수식·긴 label 겹침을 수정한 `figure-v2` PNG를
직접 검수했다. Fixed 후보의 SS와 별도 PBR conditioner를 같은 모듈로
그리지 않는다. Shared evidence는 구조 설명이며 독립 causal benefit claim이 아니다.

Method에 구조도를 삽입하고 key/value 차원·temperature·conditioning grid·
abstention·native target context의 정의를 구현과 맞췄다. 그림의 cold/no-match
fallback은 target RGB output의 완전 보존을 뜻하지 않는다. 최종 선택된
모델이 바뀌면 해당 실제 구조로 그림과 설명도 수정해야 한다.

최종 control480/128×3·actual cascade8metric·독립 확인·fresh load·freeze·
TEST508/10metric·최종 정성·Results/Discussion/Abstract/Conclusion은 계속
미완료다. 이 구조도나 이전 intermediate IoU를 그 근거로 대신하지 않는다.
