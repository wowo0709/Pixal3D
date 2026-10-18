# SS 14개 본 비교의 최종 결과와 모듈 재현 판단

14개 구성의 final128×3 비교가 완료됐다. **현재 새 R ladder를 Full로 채택할 근거는 없다.** 기존 C1과 native projection을 유지하는 P1/P2의 점추정치가 더 높다. 과거 실험의 Q2C 및 geometry 효과를 재현하는 후속은 이 표와 별도로 진행한다.

## 동일 조건의 최종 표

개발용128개·93 source cluster·샘플별 noise3개, sampler12/CFG3다. 아래 IoU는 aligned SS latent를 decode한 GT에 대한 TP/FP/FN 합산 micro IoU다. 같은 source의 variant/noise는 bootstrap에서 같은 cluster 안에 둔다. Source-only도 원하는 M1 상태 GT와 비교하며 M0 재구성 성능을 뜻하지 않는다.

| 구성 | Voxel IoU | V1 대비, pp [95% CI] |
| --- | ---: | --- |
| Source-only zero-shot: M0 RGB 4장 | 0.380711 | −5.122 [−6.789,−3.649] |
| Edited-RGB zero-shot: M1 1장 + M0 3장 | 0.419275 | −1.265 [−2.543,−0.110] |
| V1 vanilla LoRA fine-tuning | 0.431926 | 기준 |
| 기존 C1 | 0.442214 | +1.029 [+0.586,+1.498] |
| 기존 B1 | 0.433912 | +0.199 [+0.069,+0.330] |
| 기존 B2 | 0.435202 | +0.328 [+0.191,+0.481] |
| 기존 B3 | 0.435352 | +0.343 [+0.209,+0.498] |
| P1 native-grid + adapter | 0.441459 | +0.953 [+0.484,+1.422] |
| P2 H3 native-grid + adapter, No-R | 0.440920 | +0.899 [+0.390,+1.444] |
| 새 Top1 | 0.433644 | +0.172 [+0.005,+0.371] |
| 새 Top1 + rejection | 0.432806 | +0.088 [−0.039,+0.228] |
| 새 score-aware attention + rejection | 0.432847 | +0.092 [−0.038,+0.235] |
| 새 retargeting adapter + rejection | 0.433234 | +0.131 [−0.048,+0.327] |
| 새 RAW2048 query + rejection | 0.433664 | +0.174 [−0.014,+0.361] |

본 비교는 한 training seed의 탐색적 개발 비교이며 다중비교 보정은 없다. CI는 source cluster의 불확실성을 나타내며 training seed 모집단의 변동을 포함하지 않는다. V1은 world1/micro96, 새 arm은 world2/micro48×2로 global96은 같지만 rank별 RNG trajectory는 같지 않다. P1/P2는 원래 run을 중간부터 재개했고 rank RNG 복원이 없었다. 작은 차이를 엄밀한 인과효과나 unseen test 개선으로 확대하지 않는다.

## 추가 모듈의 차이

| 比較 | IoU差, pp [95% CI] | 判断 |
| --- | --- | --- |
| Top1−V1 | +0.172 [+0.005,+0.371] | 이 seed에서 작은 양의 차이 |
| Rejection−Top1 | −0.084 [−0.190,+0.011] | 추가 개선 미확인 |
| Score-aware attention−Top1-R | +0.004 [−0.021,+0.025] | 추가 개선 미확인 |
| Retargeting adapter−attention | +0.039 [−0.136,+0.219] | 추가 개선 미확인 |
| RAW2048 query−adapter | +0.043 [−0.134,+0.231] | 추가 개선 미확인 |
| P2 learned matching−P1 | −0.054 [−0.317,+0.209] | native 대비 추가 개선 미확인 |
| RAW2048 query−기존C1 | −0.855 [−1.366,−0.372] | 이 seed에서는 기존C1보다 낮음 |

No-score는 Top1 하나를 읽는 비교이며 uniform 평균을 쓰지 않는다. No-corr는 native spatial 대응을 사용한다. RAW query가 없는 arm에는 해당 encoder와 입력 channel 자체가 없다.

## 이전에 개선된 실험과의 관계

이전 geometry query는 predicted O/D logit·occupancy·SDF의 9channel이며 RAW Omega2048과 다르다. 이전에는 frozen SS의 12-step full rollout과 regional SS64/32 BCE로 학습했지만, 이번 본 비교는 LoRA와 conditioner의 flow-matching 학습이다. 이전 Q2C는 Top64·RGB task L1·no rejection의 D0/D1 비교였다. H3 head의 supervision/threshold를 같은 조건으로 취급하지 않는다. 이 차이를 복구하는 [원본 재현](2026-10-18-faithful-module-reproduction-ko.md)을 병행했다.

- 원본 Q2C는 24run의 모든 loss·pair/query hash·최종 weight가 일치하며 원래 효과가 재현됐다.
- 원본 G2는 기존 6checkpoint의 144endpoint가 모든 metric/BCE까지 일치했다. 128update 전체를 새로 학습한 trajectory의 재현과는 구분한다.
- [현재 RGB task](2026-10-18-task-rgb-matcher-transfer-ko.md)의 D0/D1은 3seed에서 RGB error가 6.01% 줄었지만 SS 생성 개선을 입증하지 않았다.
- [현재 geometry/regional 실험](2026-10-18-current-geometry-endpoint-pilot-ko.md)은 native spatial에 adapter를 추가하고 다음 arm에 physical9channel query를 추가하는 두 arm·3seed로 실행 중이다. 본 비교의 global96 학습을 batch1로 바꾸는 실험이 아니다.

이전 G2 자체도 DEV에서 source projection ON이 OFF보다 좋지 않았다. Geometry cleanup 재현과 source-retargeting 효과를 구분한다. 현재 common32에서 native fallback을 보존한 GT per-source-camera routing은 native 대비 +1.960pp였지만, GT oracle의 상한 진단이며 learned matching 결과가 아니다.

## Source 활용과 채택 조건

완료된 common32/1noise의 동일 checkpoint 진단에서 P2의 source-off/adapter-off/wrong-object는 normal보다 각각 −1.660/−1.797/−0.883pp 낮았고 각 source-cluster CI는 0을 포함하지 않았다. R을 붙인 새 B3/C1의 source-off 등 차이는 대체로 작아 matching의 추가 효과를 지지하는 근거가 약하다. 각 진단은 동일 checkpoint에 대한 개입이며, 다른 head·학습 run·source-off 정의를 혼합한 인과 비교로 해석하지 않는다.

최종 C2의 RAW context shuffle 진단과 현재 regional 두 arm의 3seed·source-off·joint-query shuffle을 확인한다. Full을 C2로 미리 정하지 않는다. 채택 후보의 작은 차이는 training-seed로 확인하고, SS에서 근거를 확보한 뒤 PBR 확대를 판단한다. 음의 결과도 설정·결과·미확인 가설과 함께 남긴다. 좋은 수치를 얻기 위해 arm이나 GT를 결과 확인 뒤 변경하지 않는다.

## 실행·저장 근거

새 일곱 SS run은 모두 480updates/약5epochs를 완료했다. 두 rank의 연속 update, micro48·accumulation1·global96, finite loss, native 시작 IoU0.4267154667, 실제 gradient, checkpoint, 여섯 번의 epoch 검증을 strict collector가 확인했다. C2의 steady compute는 약17.27초/update, rank별 peak allocated는48.134GiB였다.

기계 판독 근거는 file4의 `rgb_matching_v2/module-study-native-v4/final-primary-report.json`, 동명 Markdown, 각 `comparisons`, JSON의 `training_audit`·`head_contract`에 있다. 기존 V1/C1 noise1은 sampler 정보가 명시된 별도 output에 재평가하고 원래 sample별 seed·TP/FP/FN과 일치를 확인했다. 원래 기록에 없던 voxel hash를 후에 만들어 넣지 않았다. W&B는 `youngwoojeon-unist/pixal3d-deformnext-retargeting`의 `ss-matching-v2-evaluations`에 metric·GT/Pred/Diff·primary artifact를 저장한다.
