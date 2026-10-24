# 논문 초안과 결과 표

최신 `paper-draft-final-control-scope-v281`은 최종 구조에 따른 query/score 대조의 의미와 D1-cosine 중간 행을 반영한10쪽 revision이다.24개 인용·compiler returncode0·TeX warning/미해결 인용0이며8/9쪽의 본문과 ablation 표를 시각 검수했다. [최종 문맥 재사용·대조 구현 기록](../reports/2026-10-24-final-context-reuse-and-physical-controls-ko.md)에 실제 source/constructor 검사와 아직 필요한 GPU proof·학습/평가 entry 연결을 구분했다. 이 수정으로 새 arm의 성능이나 최종 모델 선택을 완료한 것은 아니다.

최신 후속 revision은 `paper-draft-independent-V1-output-link-v276`이다. 독립 V1의 실제 DEV128×3 집계와 같은32개에서의 활용/출력 association을 추가했다.10쪽·24개 인용, compiler returncode0·warning/미해결 인용0이며5/6쪽을 시각 검수했다. 원본 V1/R 두-seed 통합 collector도 기존 queue의 storage-successor-v185 job이 정상 pending이고 source295개가 일치함을 확인했다. R 평가 종료 뒤 이 기존 collector를 사용하며 중복 등록하지 않았다. [실제 결과·GPU1 평가 연결 기록](../reports/2026-10-24-independent-V1-readout-and-utilization-output-link-ko.md)을 참고한다. 최종 pipeline 선정·독립 R/PBR·TEST는 여전히 남아 있다.

2026-10-24 갱신: 최신 compiled revision은 `paper-draft-core-plan-utilization-v271`이다. 실제 Fixed/Learned 활용 진단, metric7개 원논문을 포함한 총24개 인용, 실제 metric 정의와 최종-context ablation 양식을 반영했다.10쪽·compiler returncode0·TeX warning/미해결 인용0이며5/9쪽의 활용 결과·개발 결과 표·ablation 표를 시각 검수했다. `audit.json`에 검수와 PDF identity를 기록했다. 아래 이전 revision 설명은 각 시점의 기록이다. 최신 남은 실험과 writing은 [핵심 계획](../reports/2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md), metric 정의는 [대조 기록](../reports/2026-10-24-metric-definitions-citations-and-independent-V1-ko.md)을 따른다. DEV128 전체 cascade와 최종 TEST는 여전히 미완료다.

`retargeting-evidence-draft.tex`는 현재 코드에 맞춘 Method, 데이터·학습·평가 protocol, DEV 결과 해석, 최종 표 양식을 포함한다. 이 디렉터리에서 TeX를 컴파일하도록 상대 input 경로를 구성했다. 전역 학습 환경과 분리한 Tectonic0.17.0으로 실제 TeX/BibTeX 빌드를 실행했다. 최신 revision은 SS13개 endpoint·matching 기전·같은 geometry의 PBR 짝비교를 포함한7쪽이며, TeX warning과 unresolved citation 없이 빌드됐다. 표와 본문이 있는4/5/6쪽을 렌더링해 읽기와 clipping 여부를 확인했다. 이는 개발 결과 초안이며 최종 TEST 논문이 아니다.

최신 snapshot은 결과 root `unified-module-pipeline-v1/paper-draft-thirteen-endpoints-v222`에 있다. `audit.json`에 실제 compiler returncode0·PDF SHA256·검수 범위를, `actual-file4-closure.json`에23개 산출물의 실제 파일서버4 SHA256/size 일치를 기록했다. 이후 수정은 기존 snapshot을 덮어쓰지 않고 새 revision으로 빌드한다.

현재 본문은 `figures/dev-tradeoffs.pdf`를 필요로 한다. 이 파일은 실제 render에서 만든 산출물이므로 Git에 넣지 않는다. 외부 build 디렉터리에 소스를 복사하고, 결과 root의 `paper-compact-DEV-tradeoffs-v230/dev-tradeoffs.pdf`를 해당 상대 경로에 배치한 뒤 그 source 디렉터리에서 `tectonic --untrusted --keep-logs retargeting-evidence-draft.tex`로 빌드한다. 원본 figure의 SHA256/size는 같은 결과 폴더의 `file4-closure.json`과 대조한다.

실제 figure를 포함한 후속 revision `paper-draft-actual-qualitative-v231`은8쪽으로 빌드·검수했다. Compiler returncode0, TeX warning/미해결 인용0이며 figure가 있는6쪽의 패널·annotation·caption을 시각적으로 확인했다.23개 source/build 산출물의 실제 file4 SHA256/size 일치를 확인했다. 위7쪽 revision도 figure 추가 전의 immutable snapshot으로 보존한다. 두 PDF 모두 개발 결과 초안이다.

이후 Learned cascade를 포함한 실제5개 조합과 동일32/noise의 SS 순위 진단을 반영한 최신 revision은 `paper-draft-five-pipelines-panel-diagnostic-v238`이다.8쪽, compiler returncode0, warning/미해결 인용0이며4/6쪽을 시각 검수했다. Source/build 및 실제 W&B8개 수치 대조 기록25개를 file4 SHA256/size로 확인했다. 이전 revision은 덮어쓰지 않았다.

후속 `paper-draft-utilization-protocol-v260`은 planned evidence-utilization protocol과 명시적 Uniform-K ablation 행을 포함한8쪽 revision이다. Compiler returncode0이며4/8쪽을 시각 검수했다. Utilization·전체 DEV128·TEST 수치를 새로 채우지 않았고 모두 해당 상태를 유지했다. 최신 구현/실행 범위는 [기전 관측 준비 기록](../reports/2026-10-24-trained-attention-utilization-preparation-and-paper-ko.md)을 참고한다.

현재 `paper-draft-related-work-v263`은 관련 문헌17개를 포함한9쪽 revision이다. 원래 PDF의24개 문헌 identity/task와 필요한 abstract 수준의 method 설명을 primary source와 대조했고, UniRecGen→현재 RecGen3D 제목 갱신도 반영했다. Compiler returncode0·TeX warning/미해결 인용0이며1/2/9쪽을 시각 검수했다. [문헌별 원문과 claim 범위](../reports/2026-10-24-paper-primary-literature-and-claim-scope-ko.md)에 근거를 남겼다. 이 작업은 외부 방법들과의 정량 실험이나 모든 원문의 세부 구현 검증을 대신하지 않는다.

이번 도구는 [공식0.17.0 release](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0)의 Linux musl asset을 배포 SHA256과 대조했으며 전역 패키지는 변경하지 않았다. PDF·log·source snapshot은 Git에 넣지 않고 file4의 실험 결과로 보관한다.

기존 `_CVPR2026_7__Image_based_3D_Editing (1).pdf`의 다섯 페이지를 읽어 문제 정의와 Table 1의10개 metric 구성을 확인했다. 기존 PDF를 덮어쓰지 않았다. 핵심6개 참고문헌을 primary source로 확인해 Related Work 초안에 통합했으며, 기존 문헌 전체의 검증·통합은 남아 있다. 기존 abstract의 최종 우위 표현과 user-study placeholder는 현재 수행한 실험의 증거가 아니다.

초안은 native **MV4 carrier** 위의 source correction을 기술한다. 과거 target-only carrier 수식으로 바꾸면 source subtraction의 의미가 달라진다. Target의 직접 경로 유지와 output RGB 보존도 구별한다. 현재 후보의 physical NoDQ 구조를 명시적으로 존중하며, 별도 deformation-query channel을 사용했다고 서술하지 않는다.

| 파일 | 상태와 목적 |
|---|---|
| `retargeting-evidence-draft.tex` | 구현과 연결한 영문 본문 초안; 최종 모델 선택 전 |
| `claim-evidence-ledger.md` | 주장별 현재 근거와 미완료 검증 |
| `related-work-verified.tex`, `references.bib` | 검증한 Related Work와 metric 원논문을 포함한24개 참고문헌 |
| `tables/dev-ss128.tex` | 완료된13개 모델의 동일 DEV128×3 SS 결과; 새 learned 결과는 exploratory follow-up |
| `tables/dev-joint32.tex` | 동일한 V1/R per-asset baseline을 확인한 실제5개 DEV32 조합; Learned cascade 포함 |
| `tables/dev-matching32.tex` | 동일 frozen matching checkpoint의 관측 가능 관계 단위 진단; 생성 품질과 별도 |
| `tables/dev-paired-pbr32.tex` | 같은 E_FIXED_RAW SS geometry에서 vanilla/R PBR의 실제 짝비교 |
| `dev-qualitative.tex` | 실제 DEV render의 trade-off 그림과 선택 규칙; 외부 figure asset staging 필요 |
| `tables/final-main-template.tex` | TEST508의10개 metric 양식; 전부 미측정 |
| `tables/final-ablation-template.tex` | 최종 구조 안의 matched ablation 양식; 전부 미측정 |

결과 표는 실제 file4 SHA256/size closure가 있는 audit에서 내보냈다. DEV32 표를 TEST 표에 복사하지 않는다. 최종 모델이 확정되면 불필요한 conditional ablation 행은 제거하고, 동일한 조건으로 확인된 기존 실험만 재사용한다. 기준 baseline이 다른 study의 수치를 한 개의 matched 비교처럼 섞지 않는다.

## 남은 writing

1. 최종 선택 구조에 맞춰 검증한 Related Work와 현재24개 인용의 표현을 정리. Metric7개 원논문과 실제 protocol 정의 대조는 완료.
2. 최종 선택 구조에 맞춰 Method의 fixed/learned 후보 설명을 정리.
3. 최종-context ablation과 독립 확인의 수치·CI·정성 사례 반영.
4. Protocol freeze 후 TEST508·10개 metric 결과로 Main Results 작성.
5. 실제 근거의 범위에 맞춰 Abstract·Discussion·Conclusion을 작성하고 최종 수치가 들어간 PDF를 다시 빌드·검수.

외부 편집법 전반의 우위나 SOTA는 현재 비교만으로 주장하지 않는다. 본 계획의 zero-shot·vanilla·retargeting·combined 비교가 직접 지지하는 범위를 먼저 완성한다.

핵심 문헌의 확인 근거는 [Pixal3D](https://arxiv.org/abs/2605.10922), [TRELLIS.2](https://arxiv.org/abs/2512.14692), [ShapeUP](https://arxiv.org/abs/2602.05676), [VGGT-Omega](https://arxiv.org/abs/2605.15195), [RoMa-Omega](https://arxiv.org/abs/2609.09507), [A Tale of Two Features](https://arxiv.org/abs/2305.15347)의 저자 제출 primary page다. 이 확인은 각 논문의 위 핵심 설명과 서지 정보 범위이며, 전체 문헌의 독립 재현이나 모든 기술적 세부의 검증은 아니다.
