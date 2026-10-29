# GT scale 미제공 TEST508 실시간 진행표

갱신: 2026-10-29 08:11 KST (2026-10-28 23:11 UTC). 이 표는 진행률이며 최종 성능표가 아니다.

| 단계 | 검증된 진행률 |
|---|---:|
| Ours V1 full cascade, file4 SHA | 508/508 |
| Ours Selected full cascade, file4 SHA | 508/508 |
| Ours Selected Top-1 PBR, file4 SHA | 508/508 |
| 같은 사례의 두 arm 모두 완료 | 508/508 |
| Official RGB1 Cycles render chunks, file4 receipt | 32/32 |
| Ours raw/+M0 Cycles render chunks, file4 receipt | 32/32 |
| Official Pixal3D RGB1 source-M0 surface | 501/501 |
| Official Pixal3D RGB1 render mesh | 501/501 |
| Official render handoff | complete |
| External seven-method learned results | 7/7 |
| External learned evaluator (PID/state/I/O) | no active evaluator found |
| Unified nine-method learned evaluation | complete |
| Unified cached evaluator | complete |
| Ours four-condition learned results | 4/4 |
| Ours source-M0 surface precache (independent of final score) | 490/490 cases; workers [] |
| Local primary cascade workers | complete/closed |
| Local GPU 6/7 tail workers | complete/closed |

| Completion gate | Receipt |
|---|---|
| Ours cascade | PASS |
| Ours surface | PASS |
| Official Cycles render | PASS |
| Ours Cycles render | PASS |
| Ours 10-metric evaluation | PASS, common-501 |
| Common 10-method table | PASS, common-501 |
| Raw 10-method table | PASS, common-501 |
| Ours paired uncertainty | PASS, common-501 |
| Raw Ours paired uncertainty | PASS, common-501 |
| Official RGB1 paired uncertainty | PASS, common-501 |
| Fair TEST claim audit | PASS |
| Fair common-success teaser | PASS |
| MV-Pixal3D zero-shot 10-metric row | PASS, common-501 |
| Geometry 6-arm ablation | PASS, TEST508/common-501 |
| Appearance 3-arm ablation | running, common-501 |

Raw and source-M0 tables use the same final common-success intersection, camera-given where supported, and no target-derived object scale at generation. The older M1-derived-scale table is an internal diagnostic.

## 2026-10-28 18:57 UTC 후속 평가 체인

- 공통 main comparison은 `PASS_LOCAL_FAIR_TEST_TABLES_INTERVALS_AND_CLAIMS`로 닫혔고, 501 edits / 133 source clusters를 사용한다.
- 추가한 `MV-Pixal3D (zero-shot)`은 DeformNeXt 미세조정 없이 T1+S3 RGB를 받는 mixed385 multi-view checkpoint다. SS IoU는 0.359802이며, full 1024 cascade는 이 시각 기준 file4 checksum receipt 466/508이다.
- cascade 이후 common-501 렌더와 10-metric 평가가 자동으로 이어진다.
- 이후 TEST508 SS 6-arm, geometry 6-arm, appearance 2-arm 및 source-view metric 평가가 상태/표본수/optimizer-update 게이트를 통과한 경우에만 순차 실행되도록 연결했다.
- geometry와 appearance는 각각 1-case 실제 GPU/file4 pilot을 통과해야 full campaign이 시작된다.
- 사용 GPU는 로컬 0,1,2,3,6,7뿐이며 node16과 GPU 4,5는 사용하지 않는다.
- Overleaf 원본은 `docs/superpowers/paper/gear3d-overleaf`에 통합했고 현재 15쪽 PDF가 빌드된다. 최종 metric이 닫히면 main/geometry/appearance 표와 본문 claim을 실제 common-501 결과로 교체한다.

### Zero-shot cascade tail reconciliation

- 완료된 rank의 GPU를 재사용하는 bounded reverse helper로 tail을 단축했다. helper는 명시된 rank의 뒤쪽 일부 case만 처리하고 기존 receipt는 건너뛴다.
- rank 10의 마지막 case에서 기존 worker와 helper가 동시에 진입해 기존 worker 하나가 non-zero로 종료됐지만, helper가 해당 case의 file4 checksum archive와 local evaluation cache를 정상 완료했다. 생성 결과 손상이나 누락은 없다.
- 따라서 원래 orchestrator의 child-exit 상태를 completion 근거로 사용하지 않고, 508개 전 case의 local archive receipt, 정확한 remote path, evaluation-cache SHA256, archive/cache 교차 SHA256를 모두 검사하는 `finalize_fair_mv_zero_cascade.py` reconciliation gate를 추가했다.
- 이 gate는 508/508이 아니거나 cache/archive hash가 하나라도 다르면 completion을 발행하지 않는다. 19:17 UTC 기준 505/508이며, 남은 작업에는 317 MB prediction을 5M-face budget으로 export하는 정상 CPU tail이 포함된다.

## 2026-10-28 19:32 UTC Overleaf source provenance

- 사용자가 제공한 최신 원본은 `docs/superpowers/reports/_CVPR2027__GEAR3D.zip`이며 SHA-256은 `c402cc29bb636a396ebdac04ab864f73a751ac174cd9e5850243ce23f95b2aae`다.
- 원본의 20개 LaTeX/스타일 파일을 `docs/superpowers/paper/gear3d-overleaf` 작업본과 대조했다. 작업본의 추가 figure와 main/ablation 표, protocol 서술 변경은 원본 위에 적용된 의도된 변경이며 원본 ZIP은 보존한다.
- 최종 제출용 ZIP은 `package_gear3d_overleaf.py`로 build artifact를 제외해 재현 가능하게 생성한다.
- MV-Pixal3D zero-shot cascade는 507/508 checksum-verified cases까지 완료됐다. 마지막 대형 mesh export가 끝나면 strict reconciliation gate와 common-501 평가가 이어진다.

## 2026-10-28 19:40 UTC paired appearance correction

- 기존 appearance 계획은 Selected cascade 안의 vanilla texture를 직접 사용하면서 Top-1/Top-K만 별도 PBR replay를 수행해 texture noise stream이 완전히 paired되지 않는 문제가 있었다. 아직 full appearance campaign은 시작 전이었으므로 발표 수치나 archive가 오염되지는 않았다.
- `appearance-common501-v2`는 vanilla와 raw-TopK를 모두 동일한 Selected geometry에서 같은 `PBR_replay_seed`로 다시 생성한다. Top-1은 이미 같은 replay protocol로 생성된 common-501 결과를 사용한다.
- vanilla와 raw-TopK 1-case 실제 pilot은 모두 `PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE`, geometry byte identity, replay-bound export, file4 checksum archive를 통과했다. 두 arm의 sample, geometry SHA, replay seed, initial texture-noise SHA가 정확히 같고 최종 texture latent만 모델별로 달라지는 것을 확인했다.
- full appearance 평가는 두 pilot과 geometry 6-arm completion gate를 모두 통과한 뒤 로컬 GPU 0,1,2,3,6,7에서 실행된다.

## 2026-10-28 19:49 UTC critical-path acceleration

- MV-Pixal3D zero-shot은 508/508 archive/cache SHA reconciliation을 통과했고 common-501 render와 10-metric 평가가 시작됐다.
- SS 6-arm은 zero-shot render 결과에 의존하지 않으므로 불필요한 선행 대기를 제거하고 GPU 0,1,2,3,6,7에서 즉시 병렬 시작했다. 각 GPU에 한 arm을 고정했으며 generation batch는 8이다.
- 19:49 UTC 기준 여섯 arm 모두 48/508이고 GPU utilization은 98--100%다. SS completion 뒤 기존 geometry pilot/full watcher가 그대로 이어진다.
- zero-shot metric이 닫히면 `compose_gear3d_main_ranking.py`가 즉시 11-method best/second ranking과 LaTeX cell을 생성하도록 연결했다.

## 2026-10-28 20:10 UTC TEST508 SS screening 완료

- 여섯 arm 모두 508/508을 완료했고 `PASS_TEST508_PREREGISTERED_SS_ABLATION`을 통과했다. 학습 update는 없으며 동일 입력·checkpoint·sample set을 사용한 geometry-stage screening이다.

| SS arm | Voxel IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Raw semantic correspondence | **0.380452** | **0.513079** | 0.595438 | **0.551199** |
| Uniform Top-$K$ weighting | <u>0.380160</u> | <u>0.512481</u> | <u>0.595529</u> | <u>0.550893</u> |
| D1 cosine correspondence | 0.379073 | 0.511808 | 0.593769 | 0.549750 |
| Matching decoder | 0.378054 | 0.508848 | 0.595274 | 0.548678 |
| Spatial correspondence | 0.377992 | 0.508314 | **0.595851** | 0.548613 |
| No residual retargeting | 0.375245 | 0.503613 | 0.595496 | 0.545714 |

- 위 표의 순위는 quick voxel metric이며 논문의 최종 CD/F-Score ablation 표를 대체하지 않는다. 이 결과로 raw semantic candidate와 retargeting 경로의 필요성은 지지되지만, 현 decoder/score aggregation의 추가 이득은 SS IoU에서 확인되지 않았다.
- 완료 receipt 생성 즉시 geometry 6-arm campaign을 시작했다. 최종 common-501 CD/F-Score가 닫히면 `tables/main_ablation.tex`을 즉시 교체한다.

## 2026-10-28 20:47 UTC geometry critical-path correction

- 최초 1024 pilot은 geometry ablation에서도 불필요한 PBR stage와 8M-face CPU surface materialization을 수행해 한 case가 10분 이상 정체되었다. full run 전 pilot에서 발견했으므로 논문 수치나 archive는 오염되지 않았다.
- Geometry stage를 native Shape512→Shape1024로 한정하고 PBR은 appearance ablation으로 분리했다. 생성 mesh가 GPU에 있을 때 face-area CDF를 한 번만 계산해 inline/raw/M0-registration surface sample을 같이 추출한다. 작은 mesh에서 기존 NumPy weighted-choice와 bit-exact한 sample을 확인했다.
- 실제 후속 pilot은 전체 geometry evidence를 18.02초, peak 8.26 GiB에 완료했다. GPU 0,1,2,3,6,7에 arm 하나씩 고정하고 GPU당 3 worker, 총 18 worker로 6-arm TEST508 full geometry evaluation을 시작했다.

## 2026-10-28 22:28 UTC common501 geometry ablation 완료

- 6개 geometry arm 모두 TEST508 508/508 checksum archive를 통과했고, 논문 비교는 기존 Full과 동일한 common-501 / 133 source-cluster population에서 수행했다.
- 등록은 입력 source mesh만 사용하는 M0 Sim(3)이며 GT object scale은 generation과 registration 모두에 제공하지 않았다. 모든 arm의 training update 수는 0이다.

| Geometry variant | CD $\times 10^4$ ↓ | F-Score $\times 10^2$ ↑ |
|---|---:|---:|
| Full ray-fused score-weighted Top-$K$ | **149.28** | <u>80.47</u> |
| Spatial retrieval | 151.25 | 80.37 |
| Raw semantic retrieval | 151.60 | 80.22 |
| Decoder matching (cosine) | <u>149.89</u> | 80.24 |
| Top-1 selection | 150.40 | 80.17 |
| Uniform Top-$K$ weighting | 150.37 | 80.38 |
| Without residual retargeting | 151.36 | **80.50** |

- Full은 CD가 가장 낮고, F-Score는 no-retargeting보다 0.03 point 낮아 사실상 동률이다. paired cluster bootstrap의 95% CI와 12-endpoint Bonferroni CI는 모든 arm-vs-Full 차이에서 0을 포함한다. 따라서 논문 claim은 “모든 endpoint의 유의한 우위”가 아니라, full 조합이 최저 surface distance를 달성하면서 F-Score를 유지한다는 범위로 제한한다.
- `tables/main_ablation.tex`과 ablation 본문을 위 결과로 즉시 교체했다.

## 2026-10-29 07:33 UTC main-table closure and appearance handoff

- MV-Pixal3D zero-shot은 common-501의 10개 metric 평가를 완료했다: LPIPS 0.05375, DreamSim 0.13730, CLIP 0.93769, FID 40.08, KID$\times10^3$ 0.667, PSNR 23.24, SSIM 0.9466, M-IoU 0.8602, CD 0.01555, F-Score 0.7920.
- 11개 방법 전체를 다시 순위화해 `tables/main_quantitative.tex`의 zero-shot 행과 best/second 표시를 갱신했다. 현재 main table에는 빈 metric cell이 없다.
- 최신 main/geometry 표를 포함한 Overleaf 작업본은 `latexmk -pdf -halt-on-error`를 통과해 15-page `main.pdf`를 생성했다.
- 남은 appearance common-501 비교는 고정 Selected geometry와 paired PBR noise에서 vanilla, 기존 Top-1, raw Top-$K$+feature의 세 arm이다. Vanilla와 Top-$K$ 각각 GPU 세 장의 forward worker와 서로 겹치지 않는 두 tail slice를 사용해 GPU 0,1,2,3,6,7에서 총 18 worker로 실행한다. 학습 update는 없고 generation 이후 source-camera render와 LPIPS/DreamSim/CLIP scorer가 자동 연결된다.
- MV-Pixal3D zero-shot은 3,507/3,507 렌더를 완료했다. 평가 환경 정리 과정에서 제거된 `dreamsim==0.2.1`과 `torch-fidelity==0.3.0`을 과거 성공 summary의 고정 버전으로 복원했고, 동일 extractor identity의 실제 offline model smoke를 통과한 뒤 metric shard만 재실행 중이다.

## 2026-10-29 08:11 KST appearance 실행 및 제출물 사전 검증

- Paired common-501 appearance 생성은 vanilla 168/501, raw Top-$K$+feature 163/501 checksum receipt까지 진행됐다. 기존 Top-1 common-501 결과는 같은 replay protocol에서 재사용한다.
- 앞쪽 57개, 중간 50개, 뒤쪽 60개 rank slice가 서로 겹치지 않도록 arm당 9 worker, 총 18 worker를 GPU 0,1,2,3,6,7에 배치했다. 24-worker 시도는 96-core CPU를 과포화해 처리량이 감소했으므로 종료했고, 현재 구성에서는 OOM·runtime error가 없다.
- 생성 완료 뒤 common-501 렌더, LPIPS/DreamSim/CLIP 평가, geometry/appearance LaTeX 행 조합이 자동으로 이어지도록 watcher를 연결했다. 중간 subset metric은 논문 표에 사용하지 않는다.
- Main comparison JSON의 11개 방법×10개 metric을 LaTeX 표와 다시 대조해 best/second 표시가 모두 일치함을 확인했다.
- 결론의 appearance claim을 실제 main table보다 강하지 않도록 ``source appearance preservation and target alignment의 favorable balance''로 수정했다.
- 제출 ZIP 패키저에서 본문에 사용되지 않으면서 누락된 `egfigure.eps`를 참조하던 CVPR 예제 `rebuttal.tex`, 미사용 template section 두 개, 중복 PNG와 이전 overall figure, provenance JSON을 제외했다. 실제로 참조되는 21개 파일의 clean ZIP을 새 디렉터리에 풀어 독립 `latexmk` 빌드한 결과 경고 없이 15쪽 PDF가 정상 생성됐다. Supplementary 페이지는 `S1, S2, ...`로 번호를 분리해 본문과 PDF page anchor가 중복되지 않는다.

## 2026-10-29 09:07 KST appearance final-tail tracking

- Paired common-501 appearance 생성은 vanilla 407/501, raw Top-$K$+feature 424/501, 합계 831/1,002 checksum-verified cases까지 진행됐다. 완료되지 않은 중간 subset에서 metric을 뽑거나 표를 갱신하지 않는다.
- 생성 worker, completion reconciler, 6-GPU renderer, LPIPS/DreamSim/CLIP scorer, geometry/appearance table composer가 하나의 gate chain으로 살아 있다. 생성 1,002/1,002 직후 별도 수동 대기 없이 render와 metric 평가가 시작된다.
- 현재 critical path 밖의 신규 실험은 추가하지 않는다. 최종 `appearance-ablation.json`이 501 samples와 세 arm을 검증하면 appearance 표와 본문 claim을 즉시 교체하고, 최종 PDF·clean Overleaf ZIP·handoff report 생성으로 진행한다.

## 2026-10-29 09:14 KST appearance stalled-rank recovery

- 남은 case를 rank별로 대조한 결과, 최초 orchestrator에서 종료된 vanilla rank 1/2의 앞쪽 각 10개가 활성 worker 범위 밖에 남아 있음을 확인했다. 두 rank의 첫 case에는 완료 receipt나 원격 archive가 없는 불완전 local replay가 남아 worker의 fail-closed 검사를 유발하고 있었다.
- 불완전 출력은 삭제하지 않고 `appearance-common501-v2/quarantine-partial-20261029T0012Z`로 이동했다. 두 정확한 gap slice만 GPU 6/7에 `skip=47, max=10` bounded helper로 다시 연결했으며 기존 forward/reverse worker 범위와 겹치지 않는다.
- 복구 worker가 정상 실행된 뒤 진행률은 vanilla 420/501, raw Top-$K$+feature 437/501, 합계 857/1,002가 됐다. OOM이나 원격 checksum 오류는 없다.

## 2026-10-29 10:50 KST common-501 appearance ablation 완료

- Vanilla와 raw Top-$K$+feature 생성은 각각 501/501 checksum archive를 통과했다. 동일 Selected geometry, paired PBR noise, source cameras 9/10에서 2,004개 렌더를 완료했고, 12개 metric worker로 LPIPS/DreamSim/CLIP을 계산했다. 기존 Top-1 결과는 동일 common-501 protocol의 pre-existing selected checkpoint에서 가져왔다.

| Appearance variant | LPIPS $\times10^3$ $\downarrow$ | DreamSim $\times10^3$ $\downarrow$ | CLIP $\uparrow$ |
|---|---:|---:|---:|
| Original conditioning | <u>53.73</u> | **134.78** | **0.9388** |
| Top-1 retargeting (main model) | 53.82 | 135.17 | 0.9384 |
| Raw Top-$K$+feature retargeting | **53.71** | <u>134.81</u> | <u>0.9387</u> |

- Raw Top-$K$는 Top-1보다 LPIPS가 $0.106\times10^{-3}$ 낮고 source-cluster bootstrap 95% CI가 $[-0.210,-0.008]\times10^{-3}$로 0을 제외한다. DreamSim과 CLIP 차이의 CI는 0을 포함한다.
- Raw Top-$K$와 original conditioning의 LPIPS, DreamSim, CLIP 차이는 모두 CI가 0을 포함한다. 따라서 common-501 결과는 appearance retargeting의 일관된 우위를 뒷받침하지 않으며, 논문 claim은 geometry 개선이 주된 기여이고 appearance variant들은 유사한 source fidelity를 유지한다는 범위로 제한한다.
- Main comparison은 test 결과를 보고 checkpoint를 교체하지 않고 기존 Top-1 selected checkpoint를 유지한다. `tables/appearance_ablation.tex`, main experiment discussion, supplementary stage-specific 설명을 위 결론으로 갱신했다.
