# SS geometry와 PBR appearance의 분리 평가·선정

실제 TEST 후속: SS508×5가 정상 종료됐고 archive2,540개를 검증했다(v574).
수치 source를 유지한 `v570` cascade와10metric→표/그림→원격 logging을 진행한다.
아래 SS 실행 중·v565 표현은 이전 이력이며
[최신 운영 복구와 완료/미완료 범위](2026-10-26-frozen-TEST508-execution-and-recovery-ko.md)를 따른다.

SS arm은 geometry 향상이 목적이다. 공통 frozen Shape/PBR에서 CD↓·surface
F-Score↑로 판단하며 voxel IoU는 진단값이다. Cross-SS LPIPS/DreamSim은 SS의
geometry 개선을 거부하는 기준이 아니다. Appearance는 동일 geometry에서 PBR만
바꿔 검증한다. 최종 시스템은 원래10개 metric을 모두 보고한다.

## 완료된 SS 비교

동일 DEV128, 공통 Shape/Top-1 R PBR의 실제7개 cascade 결과다.

| SS arm | CD↓ | Surface F-Score↑ |
|---|---:|---:|
| Spatial | 0.0129704 | 0.839333 |
| Raw semantic | 0.0128789 | 0.841031 |
| Decoder cosine | 0.0128065 | 0.842414 |
| Fixed score-weighted | **0.0127572** | **0.844058** |
| Actual Top-1 + R | 0.0127795 | 0.840095 |
| Uniform Top-K | 0.0128304 | 0.842505 |
| Physical no-R | 0.0129562 | 0.835038 |

Fixed가 두 geometry 점추정에서 동시에 가장 좋아 실제 SS checkpoint를 선정했다.
원래48-comparison family를 유지했고 **48개 보정 구간 모두0을 포함**한다.
모듈별 필수성·확정 우위·동등성을 주장하지 않는다. 완료된 두 training seed의
SS 확인은 occupancy 범위이며 두-seed surface 품질 검증으로 확대하지 않는다.

## 완료된 동일 Fixed geometry의 PBR 비교

실제 DEV32/32 source에서 네1900-step PBR checkpoint를 비교했다. 생성 mesh,
native guide, texture 초기 noise, sampler, render vertices/faces와 GT material
reference가 같다. 기존 vanilla/R64개와 새 feature/raw64개로128 case를 완성했고,
192개 새 official view와 정확히 같은 pixel의 기존192개 점수를 합쳤다.
SS/Shape forward는 추가하지 않았다.

| PBR arm | Source LPIPS↓ | Source DreamSim↓ | Source CLIP↑ | Target PSNR↑ | Target SSIM↑ |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0.053513 | 0.158346 | **0.942179** | 24.47775 | 0.951534 |
| Actual Top-1 + R | **0.053290** | **0.157828** | 0.941692 | **24.54433** | **0.951563** |
| Top-1 + learned feature + R | 0.053326 | 0.157985 | 0.941759 | 24.51012 | 0.951536 |
| Raw Top-K + learned feature + R | 0.053361 | 0.158261 | 0.941559 | 24.52268 | 0.951539 |

Source view9/10의 LPIPS/DreamSim을 중심으로 **Top-1 R PBR**를 선정했다. CLIP은
보조 지표이고 target view8의 PSNR/SSIM/M-IoU는 별도 보고한다. 모든 행의
CD0.0123276·F-Score0.856566은 같다. DEV32 geometry 동일성 확인값이며 PBR geometry
향상이나 DEV128 SS 수치와의 직접 비교가 아니다.

Top-1 R−vanilla의 LPIPS는 −0.000222934, exploratory CI95
[−0.000539103,+0.000035032], DreamSim은 −0.000517370,
[−0.002250414,+0.001182970]이다. 모두0을 포함하므로 확정적인 appearance 향상으로
쓰지 않는다. Feature/Raw의 추가 경로도 Top-1 R보다 두 primary 점추정이 좋지 않다.
기존 feature/raw는 Top-1 carrier/use 경로이므로 SS Fixed와 같은 shared-evidence
PBR의 효과·실패 증거로 이름을 바꾸지 않는다.

실제 W&B12행의8개 metric을 원본 JSON과 ulp까지 대조했다. 네 arm의 첫 고정 사례
GT–Pred–Diff 원격 PNG도 원본 RGBA pixel과 정확히 같았다.
[실제 비교 run](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/selected-Fixed-geometry-PBR4-native32-v533).

## 최종 조합과 남은 실행

DEV 기반 선택은 **Fixed SS + Top-1 R PBR**다. 각 stage의 별도 목적에 따른 선택이며
모든 matching/attention 설계의 효과가 증명됐다는 뜻이 아니다. 선택 PBR는 이미
vanilla/R×seed101/2021900의 fixed4 GT-shape training QA가 있어 반복 학습하지 않는다.
이 QA는 source appearance나 두-seed full-pipeline 확인을 대신하지 않는다.

실제 checkpoint/config·원래 source/metric/sampler/camera·선정 기록을
`actual-selected-pipeline-final-freeze-v558`에 동결했다. TEST 품질을 읽기 전에 원래
다섯 development gate, selected SS와 독립 seed의 실제 model bytes, VAL36 정합을
검증했다. 두 zero-shot/V1/R/선택 조합을 TEST508/134 source에서 평가하며 원래
TEST8C 보정·10개 metric·사전 qualitative 정책을 유지한다. 이후 실제 Main/Ablation/
Mechanism/Supplement와 최종 Results/Discussion/Abstract/Conclusion을 완성한다.
동결 완료는 실제 TEST 결과나 최종 논문 완료를 뜻하지 않는다.

실제 보존 근거: `conditional-Fixed-geometry-PBR4-completion-v528/eight-metric-summary-v1`,
`actual-stage-separated-pipeline-selection-v553`,
`actual-selected-pipeline-freeze-preparation-v554/selection-v1`.
v554는 추가 SS control 다섯 개의 설명용 identifier를 실제 등록 arm ID로 바로잡은
선정 기록이며 모델·수치·선정 기준·원래14-arm population은 그대로다.

현재 실제 실행: 세 학습 SS는 GPU2/3/6, 두 original zero-shot은 GPU0/1에서 TEST508을
평가 중이다. Zero 첫 실행의 file-spec metadata 오류는 모델 계산 전 확인돼 실패 로그와
terminal을 보존했고 별도 v562 successor에서 수정했다. 수치 code·model·input SHA는
유지한다. SS 완료 뒤 bounded whole-asset cascade와 실제 cases 완료 뒤 ten-metric
chain을 기존 두 dispatcher에 연결했다(v565). GPU7은 cascade 단계에 포함된다.
19쪽/24인용 개발 PDF(v560)를 compile·4쪽 시각 검수·file4 보존했다(v563).
