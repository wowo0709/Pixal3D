# 논문 초안과 결과 표

## 현재 revision: 2026-10-25

현재 compiled revision은 `paper-draft-complete-seven-SS-geometry-and-selection-v550`의
17쪽·24개 참고문헌 개발 초안이다. 실제 compiler returncode0이며 수정된7/14/15/16쪽을
시각 검수했다. Warning·미해결참조·overfull/underfull0, actual file4 closure를 완료했다.
모든7개 SS arm·21개 evaluation·6대조 통계가 끝났으며 원래 reducer에서 전체 통계를
정확히 재현하고 원래 W&B7행·6개 interval을 대조했다. 모든 adjusted CI는0을 포함한다.
신규5개 control의128개 cascade·8metric도 끝나 기존 Fixed/Top-1과 같은7모델 근거를
반영했다. SS/Shape/PBR binding과384 GT pixel identity가 같다. SS ablation은
CD/F-Score와 진단용 voxel IoU 표로, PBR ablation은 동일 geometry의 appearance 표로
분리했다. SS 후보의 appearance 차이를 geometry 향상의 거부 기준으로 쓰지 않는다.
전체48-family도 완료했으며48개 보정 구간은 모두0을 포함한다. 전체48개 원값 CSV와
geometry12개 구간 표를 실제 closed JSON에서 내보냈다(v546). Fixed가7모델 중
CD/F-Score 점추정에서 동시에 가장 좋아 PBR 비교용 SS로 선정했다(v544).
이는 개별 모듈의 보정 후 효과·필수성이나 최종 TEST 우위를 확정한 것이 아니다.
최종 PBR 선정·freeze/TEST·최종 writing은 미완료다.

D1-cos는 CD0.0128065·F-Score0.842414다. Fixed−D1-cos는 CD−0.0000493·
F-Score+0.001644의 점추정이며 scalar score의 확정 효과로 쓰지 않는다. 실제6개
checkpoint/common downstream/384 GT pixel identity를 대조했고 기존5개 평균은
정확히 같다(`actual-six-DEV128-geometry-compositions-v536`).

PBR의 vanilla/Top-1 R/feature/raw 네 arm은 같은 V1 geometry의 기존 평가와 별도로,
선정된 Fixed geometry·native guide·texture noise·render triangles에서 비교 중이다.
기존 같은-Fixed vanilla/R64개 case를 재사용한다. 추가 feature/raw64개 중 첫 두
실제 case의 pair proof·세 child session 종료·file4 closure를 통과했고, 나머지62개와
평가3개를 두 queue/GPU0–3·6·7에 등록했다(v545). SS/Shape는 재생성하지 않는다.
새 PBR 구조가 선택되면 그 실제 구조의 독립 seed 확인도 필요하다.

이 비교의 coverage→official GPU→4모델 집계·W&B 설정도 조건부 준비를 마쳤다(v533).
Fixed 선정 시 기존 vanilla/R의192개 official view를 재사용하고 새 feature/raw192개만
GPU에서 측정한다. `selected_pbr_official_reuse.py`는 모델 가중치·registry·source·camera·
실제 scored-NPZ SHA를 확인한 뒤 arm 이름만 대응시키며 점수와 image audit는 유지한다.
15개 검사 및 실제 기존384개 view의 평균·대조·bootstrap 구간 전체 불변을 확인했다(v531).
기존 V1 geometry 점수를 Fixed pixel에 섞으면 실제로 거절하며, 아직 없는 SS 선정
기록으로 coverage/summary를 실행해도 output·NN 전에 거절하는 것을 확인했다(v534).
이 경로는 실제 등록·실행 중이며 appearance 향상과 최종 PBR 선정은 결과 집계 후 판단한다.

처음 v548 PDF는 새 표의9pt 수학 숫자 font가 캐시에 없어 실패했다. 모든 수치·단위는
유지하고 text 표기로 바꾼 v550은 성공했으며 실패 로그도 보존했다(v549/v551).

`cascade_record_json.py`는 저장된 camera JSON key만 정수8/9/10으로 복원한다.
Mixed/잘못된 camera를 거절하고 값·row 순서는 유지한다. JSON roundtrip 전후 paired
mean·interval 불변 등14개 검사와 실제4모델의 endpoint exact 일치를 확인했다.
Pending global CPU reducer 하나를 같은 output의 successor로 교체했고 실제
HELD→pending·고유 producer·file4 admission을 확인했다. 기존 case/NN/수치 source는
그대로이며 새 full7 global reducer의 실제 성공은 아직 주장하지 않는다.

이전 v505에서는 matching probability·confidence·fallback의 범위를 원문·코드로
구분했다. 16쪽·24개 인용의2/4/14/15쪽을 검수했고 수치 표·그림은 v494와 같았다.
Native fallback은 관측/proposal support가 없을 때 적용되며 support 있는 오매칭을
모두 검출하지 않는다는 한계를 유지한다.

이전 v494에서는 최종 Spatial SS128×3 IoU0.472267과 완료 PBR V1/R 두 training seed의
1900-step4행 표를 반영하고5/10/11/13/14/15쪽을 검수했다. PBR 원본16장과 원격
4행 metric을 대조했으며 fixed4 GT-shape training QA의 mixed LPIPS/PSNR 경향이다.
Full-cascade/source/TEST 품질 또는 개별 모듈의 필수성으로 쓰지 않는다.

이전 v478에서는 DEV32/128·validation·TEST 역할과 metric/statistical protocol을
반영했다. `current-training-DEV-TEST-split-metadata-v477`에서 현재 학습 config와
실제 train9149/validation508/TEST508 metadata의 sample/source UID/cluster 무교차,
DEV128의 validation 포함과 TEST 분리를 확인했다. 원본 manifest의 현재 local/remote
SHA·size도 같으며 TEST quality나 geometry 값을 열지 않았다. 최종 모델 선정·TEST·
논문 writing은 계속 남아 있다.

이전 v458에서 PBR 독립 확인 범위·Spatial480 완료를 반영하고12/13쪽을 검수했다.
현재 claim 원장은 과거 실행 이력과 현재 미완료 상태가 섞이지 않게 정리되어 있다.

이전 v447에서 VAL36 QA·통계 protocol·독립 seed 결과/표의5/6/10쪽을 검수했고,
독립 표의27개 수치·순서는 실제 closed exporter와 같음을 확인했다.
해당 수치와 matcher Method·Discussion은 v458에도 유지했다.

Fixed의 SS128×3는 seed101/202에서0.473471/0.474730이며 V1 대비 개선의
3대조 보정 CI95가 두 seed 모두 양수다. Fixed−Top-1의 보정 구간은 둘 다0을
포함하므로 추가 이득 확정이나 개별 모듈의 필수성으로 쓰지 않는다.
`Fixed-SS-two-seed-paper-and-WandB-v446`에서 원래 W&B6행 평균·모든 대비 구간을
실제 원격 값과 대조했다. Working Fixed seed101의 VAL36는 두 fresh-load·72개
archive·tail4의 동일성을 통과했다. PBR 두-seed training QA도 완료됐으나 최종
cascade·TEST·모델 선택과 writing은 계속 남아 있다.

`matcher-training-method-provenance-v439`는 실제 사용 중인 RGB matcher의
train9143/validation508·epoch7·62545 updates를 checkpoint에서 확인한9개 산출물이다.
9651개 teacher input fingerprint 순서가 checkpoint와 같고 현재 native RGB 입력
SHA도 같다. 원본 checkpoint/history와 native train/validation manifest는 실제
file4 SHA/size를 읽기 전용으로 대조했다. Teacher-rejected6개는 matcher만의 제외이며
SS train9149는 유지된다. CPU NN forward·GPU 초기화·재학습은 하지 않았다.
새 생성 품질 결과나 최종 모델·TEST·논문 완료를 의미하지 않는다.

`tables/dev-completed-fixed-contrasts.tex`는 완료7모델의 실제128×3 결과와 전체6개
대조를 담는다. Fixed−No-R의+0.004639도
보정 CI95[−0.000811,+0.010633]는0을 포함한다. Fixed−Top-1/Uniform도 보정
구간은0을 포함한다. 전체6행의 수치를 실제 closed JSON과 반올림까지 대조했다.
통계와 남은 범위는 [핵심 실행 행렬](../reports/2026-10-25-core-experiment-matrix-and-paper-deliverables-ko.md)을 따른다.

현재 `tables/final-ablation-template.tex`의7행 SS는 모두 완료됐고 Fixed/Top-1/
Uniform/no-R/Spatial은 cascade도 완료됐다. 두 원래 NN은480과 closure를 정상 종료했으며
현재 GPU0–3/6/7에서 나머지 matching cascade를 진행한다. 전체48-family·freeze/TEST·
최종 writing은 계속 남아 있다. 완료 학습은 다시 돌리지 않는다.

빌드는 아래 이력의 두 figure와 함께
`working-pipeline-overview-v361/figure-v2/working-pipeline-overview.pdf`를
source의 `figures/working-pipeline-overview.pdf`로 배치해야 한다. 각 원본의 실제
file4 SHA256/size를 확인한 외부 source snapshot에서 전용 Tectonic0.17.0의
`--only-cached --untrusted` 빌드를 사용한다. 전역 패키지는 바꾸지 않았다.
첫 v411은 새 표의9pt math font가 캐시에 없어 PDF 생성이 실패한 기록이다.
수치는 유지하고 기존 표와 같은 text 표기로 바꿔 v412에서 성공했으며 실패 로그도
보존한다. PDF/source/log는 Git 대신 file4 실험 산출물로 관리한다.

## 최종 정성 그림 실행 준비

[Exporter 계약](final-qualitative-export-contract.md)에 실제 source·closed input·freeze 연결을
정리했다. `final_qualitative_selection.py`와 `export_final_qualitative.py`를 구현했고
30개 metadata/pixel 검사가 통과했다. 전체6열의 변경된 layout/image join도 다시
검사·시각 검수했다. 실제508 metadata·134 source와 두 고정 ID는 원래 v355 선언과
정확히 같고, 원본 manifest의 현재 remote/local SHA·size도 읽기 전용으로 대조했다.

`final-paper-export-readiness-v424/freeze-requirements.json`은 v420의 선언 spec과
source pins에 Main table exporter를 추가한 packet이다. 실제 TEST 전에 evaluation
protocol에 **병합**해야 한다. 기존 numerical
source mapping을 대체하지 않는다. 실제 freeze·최종10metric 이후의 실행만 허용하며,
현재 검수한 그림은 명시적인 engineering fixture다. 최종 TEST 그림은 계속 미완료다.

`export_final_main_table.py`도 실제 frozen TEST508의 같은8/10metric 보고서에서
Main TeX·보정 구간별 evidence·writing 범위를 내보내도록 구현했다.15개 synthetic
계약 검사가 통과했고 수정한 immutability 검사도 다시 통과했다. 카메라 역할/표본 수/
raw KID/원래 CI를 바꾼 입력은 거절한다. 테스트용 TeX를 compile·시각 검수했으며
실제 TEST Main table이나 새 품질 수치는 생성하지 않았다. 기존 v312 ablation
TeX/evidence 작업은 중복 구현하지 않고 그대로 사용한다.

## 이전 revision 이력

아래의 “최신”·pending·쪽수 설명은 각 revision 당시의 기록이다. 현재 실행 상태는
위 요약과 핵심 실행 행렬을 따른다.

최신 `paper-draft-close-prior-method-scope-v339`는 가까운 선행연구의 Method·공식 코드와 실제 native 구현의 차이를 반영한13쪽/24개 인용 revision이다.1·2·6쪽을 검수했고 compiler warning·미해결 인용·overfull/underfull0이다. [대조 근거](../reports/2026-10-25-closest-method-scope-and-writing-ko.md)에 검토 범위와 lineage를 기록했다. 신규 control/TEST 결과는 계속 미측정이다.

2026-10-25 갱신: 최신 compiled revision은 `paper-draft-actual-ray-evidence-v325`의13쪽·24개 인용 초안이다. 실제 native ray 기전 그림은 `actual-ray-evidence-figure-v323`에서 생성·검수·file4 보존을 완료했다. `dev-ray-evidence.tex`는 해당 실제 관측의 입력4장·member·Top-K·확률을 연결하고, 조건부 사례 선정과 진단 Uniform의 범위를 caption에 명시한다. 새 PDF의5·6·10쪽을 직접 검수했고 compiler returncode0, TeX warning·미해결 인용·overfull/underfull0이다. 최종 control/TEST 결과는 아직 미측정이다. 이전 revision은 아래 시점별 기록으로 보존한다.

최신 `paper-draft-final-control-scope-v281`은 최종 구조에 따른 query/score 대조의 의미와 D1-cosine 중간 행을 반영한10쪽 revision이다.24개 인용·compiler returncode0·TeX warning/미해결 인용0이며8/9쪽의 본문과 ablation 표를 시각 검수했다. [최종 문맥 재사용·대조 구현 기록](../reports/2026-10-24-final-context-reuse-and-physical-controls-ko.md)에 실제 source/constructor 검사와 아직 필요한 GPU proof·학습/평가 entry 연결을 구분했다. 이 수정으로 새 arm의 성능이나 최종 모델 선택을 완료한 것은 아니다.

최신 후속 revision은 `paper-draft-independent-V1-output-link-v276`이다. 독립 V1의 실제 DEV128×3 집계와 같은32개에서의 활용/출력 association을 추가했다.10쪽·24개 인용, compiler returncode0·warning/미해결 인용0이며5/6쪽을 시각 검수했다. 원본 V1/R 두-seed 통합 collector도 기존 queue의 storage-successor-v185 job이 정상 pending이고 source295개가 일치함을 확인했다. R 평가 종료 뒤 이 기존 collector를 사용하며 중복 등록하지 않았다. [실제 결과·GPU1 평가 연결 기록](../reports/2026-10-24-independent-V1-readout-and-utilization-output-link-ko.md)을 참고한다. 최종 pipeline 선정·독립 R/PBR·TEST는 여전히 남아 있다.

2026-10-24 갱신: 최신 compiled revision은 `paper-draft-core-plan-utilization-v271`이다. 실제 Fixed/Learned 활용 진단, metric7개 원논문을 포함한 총24개 인용, 실제 metric 정의와 최종-context ablation 양식을 반영했다.10쪽·compiler returncode0·TeX warning/미해결 인용0이며5/9쪽의 활용 결과·개발 결과 표·ablation 표를 시각 검수했다. `audit.json`에 검수와 PDF identity를 기록했다. 아래 이전 revision 설명은 각 시점의 기록이다. 최신 남은 실험과 writing은 [핵심 계획](../reports/2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md), metric 정의는 [대조 기록](../reports/2026-10-24-metric-definitions-citations-and-independent-V1-ko.md)을 따른다. DEV128 전체 cascade와 최종 TEST는 여전히 미완료다.

`retargeting-evidence-draft.tex`는 현재 코드에 맞춘 Method, 데이터·학습·평가 protocol, DEV 결과 해석, 최종 표 양식을 포함한다. 이 디렉터리에서 TeX를 컴파일하도록 상대 input 경로를 구성했다. 전역 학습 환경과 분리한 Tectonic0.17.0으로 실제 TeX/BibTeX 빌드를 실행했다. 최신 revision은 SS13개 endpoint·matching 기전·같은 geometry의 PBR 짝비교를 포함한7쪽이며, TeX warning과 unresolved citation 없이 빌드됐다. 표와 본문이 있는4/5/6쪽을 렌더링해 읽기와 clipping 여부를 확인했다. 이는 개발 결과 초안이며 최종 TEST 논문이 아니다.

최신 snapshot은 결과 root `unified-module-pipeline-v1/paper-draft-thirteen-endpoints-v222`에 있다. `audit.json`에 실제 compiler returncode0·PDF SHA256·검수 범위를, `actual-file4-closure.json`에23개 산출물의 실제 파일서버4 SHA256/size 일치를 기록했다. 이후 수정은 기존 snapshot을 덮어쓰지 않고 새 revision으로 빌드한다.

현재 본문은 `figures/dev-tradeoffs.pdf`와 `figures/dev-ray-evidence.pdf`를 필요로 한다. 두 파일은 실제 관측·render에서 만든 산출물이므로 Git에 넣지 않는다. 외부 build 디렉터리에 소스를 복사하고, 결과 root의 `paper-compact-DEV-tradeoffs-v230/dev-tradeoffs.pdf`와 `actual-ray-evidence-figure-v323/figure/dev-ray-evidence.pdf`를 해당 상대 경로에 배치한 뒤 그 source 디렉터리에서 `tectonic --untrusted --keep-logs retargeting-evidence-draft.tex`로 빌드한다. 원본 figure의 SHA256/size는 각 결과 폴더의 `file4-closure.json`과 대조한다.

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
| `dev-ray-evidence.tex` | 실제 native ray/member·후보·확률의 기전 그림; 조건부 진단과 matched 학습 결과를 구분 |
| `tables/final-main-template.tex` | TEST508의10개 metric 양식; 전부 미측정 |
| `tables/final-ablation-template.tex` | Matching ladder/aggregation/R 대조; 실제 Fixed/Top-1 두 행만 측정 |

결과 표는 실제 file4 SHA256/size closure가 있는 audit에서 내보냈다. DEV32 표를 TEST 표에 복사하지 않는다. 최종 모델이 확정되면 불필요한 conditional ablation 행은 제거하고, 동일한 조건으로 확인된 기존 실험만 재사용한다. 기준 baseline이 다른 study의 수치를 한 개의 matched 비교처럼 섞지 않는다.

## 남은 writing

최종 Fixed correspondence3개는 구현·257개 검사·actual checkpoint 생성자 정합을
완료하고 own GPU/학습/SS 평가를 등록했다. 새 품질 수치는 아직 없다.
Ablation 표는 adjacent matching ladder와 Fixed 기준 대조를 분리하고,
실제 Fixed/Top-1 두 행만 채웠다. GPU proof나 생성자 검사로 성능을 주장하지 않는다.

DEV128의 R/Fixed/Learned 실제 cascade8개 metric을 `tables/dev-joint128.tex`와
본문에 반영했다. Fixed/R PBR가 working candidate이며 최종 freeze는 아니다.
F-Score 개선의 탐색용 CI는 양수지만 appearance 비열등성은 입증되지 않았다.
선택 문맥의 No-R/Uniform-K, Fixed seed202, 필수 PBR 확인과 TEST가 남아 있다.

원래12모델의 전체 family 집계도 완료했다. `tables/dev-original-module-effects.tex`에
10개 module-family CI를 실제 통계에서 내보냈으며 모든 CI가0을 포함한다.
Ray−voxel은−0.000236, 보정 CI[−0.000878,+0.000351]이다.
`tables/dev-ss128.tex`는 완료된14개 endpoint이며, 새 shared-evidence 구조와
원래 FULL 구조의 결과를 분리해 서술한다.

완료 갱신: `tables/dev-independent-ss.tex`와 본문의 독립 학습 문단에
실제 V1/Top-1 branch seed101·202의 DEV128×3 결과를 반영했다.
개선은 각각+0.005245/+0.005169이며 combined branch의 SS 재현성으로
한정한다. 선택 모델과 PBR·cascade·TEST의 독립 확인은 별도다.

1. 최종 선택 구조에 맞춰 검증한 Related Work와 현재24개 인용의 표현을 정리. Metric7개 원논문과 실제 protocol 정의 대조는 완료.
2. 최종 선택 구조에 맞춰 Method의 fixed/learned 후보 설명을 정리.
3. 최종-context ablation과 독립 확인의 수치·CI·정성 사례 반영.
4. Protocol freeze 후 TEST508·10개 metric 결과로 Main Results 작성.
5. 실제 근거의 범위에 맞춰 Abstract·Discussion·Conclusion을 작성하고 최종 수치가 들어간 PDF를 다시 빌드·검수.

외부 편집법 전반의 우위나 SOTA는 현재 비교만으로 주장하지 않는다. 본 계획의 zero-shot·vanilla·retargeting·combined 비교가 직접 지지하는 범위를 먼저 완성한다.

핵심 문헌의 확인 근거는 [Pixal3D](https://arxiv.org/abs/2605.10922), [TRELLIS.2](https://arxiv.org/abs/2512.14692), [ShapeUP](https://arxiv.org/abs/2602.05676), [VGGT-Omega](https://arxiv.org/abs/2605.15195), [RoMa-Omega](https://arxiv.org/abs/2609.09507), [A Tale of Two Features](https://arxiv.org/abs/2305.15347)의 저자 제출 primary page다. 이 확인은 각 논문의 위 핵심 설명과 서지 정보 범위이며, 전체 문헌의 독립 재현이나 모든 기술적 세부의 검증은 아니다.
