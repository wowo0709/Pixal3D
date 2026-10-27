# GT scale 미제공 TEST508 최종 표·원고 조립

2026-10-28 07:10 KST 기준. 아래 경로는 현 실행의 frozen 산출물이다. **완료 영수증이 없으면 최종 수치나 PDF를 공개하지 않는다.** 새 학습은 포함하지 않는다.

## 자동 연결된 완료 영수증

| 산출물 | 영수증·정본 |
|---|---|
| V1/Selected 508쌍 file4 checksum archive | `/root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028/cascade-two-arm-execution-v2-direct-file4/complete.json` |
| Official RGB1 501 성공 사례 Cycles | `/root/dev/.cache/deformnext-benchmark-runtime/official-rgb1-render-campaign-v2/complete.json` |
| Ours 502사례 raw/+M0 Cycles | `/root/dev/.cache/deformnext-benchmark-runtime/ours-two-render-campaign-v2/complete.json` |
| 최종 성공 교집합 10방법×10지표 | `/root/dev/.cache/deformnext-benchmark-runtime/scale-ungiven-main-table-v4/complete.json`; 원고 `paper/final-results/scale-ungiven-main-table.tex` |
| V1↔Selected 8지표 paired CI | `/root/dev/.cache/deformnext-benchmark-runtime/scale-ungiven-ours-paired-v3/complete.json` |
| Official RGB1↔V1/Selected 16지표 paired CI | `/root/dev/.cache/deformnext-benchmark-runtime/scale-ungiven-official-contrasts-v2/complete.json`; 원고 `paper/final-results/scale-ungiven-official-contrasts.tex` |
| 실제 공정 teaser | `/root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028/fair-final-teaser-v2/complete.json`; 원고 `paper/final-results/figures/paper-teaser.{pdf,png}` 및 캡션·provenance |

최종 표는 external 7법, 공식 RGB1, Ours V1/Selected의 **동일 성공 ID**에 대해 카메라 제공 가능 모델의 given camera, 생성 단계의 객체별 GT scale 미제공, 생성 후 입력 source M0만 쓰는 Sim(3), 공통 Cycles·평가기 조건이다. 공식 RGB1과 Ours의 입력 사진 수가 다르므로 시스템 비교로만 해석한다. 예전 M1 유래 scale의 내부 6행 표를 주 비교 수치와 섞지 않는다.

## 실측 뒤 작성할 문장

1. `scale-ungiven-main-table.json`의 10방법×10지표 셀과 raw/+M0 target8 표를 확인한다. 최고/2위 서식은 자동 생성되며, KID 표시값은 원값×1,000이다.
2. `paired-eight-endpoints.json`에서 Selected−V1의 source/target/geometry 8지표의 8-family interval을 확인한다. `official-contrasts-sixteen-endpoints.json`에서 official RGB1 대비 V1/Selected의 16-family interval을 확인한다. 0을 포함하는 구간은 개선·동등성 어느 쪽도 확정하지 않는다. FID/KID는 점추정만 서술한다.
3. 실제 결과에 근거한 `paper/final-results/unified-abstract-findings.tex`, `unified-results.tex`, `unified-discussion.tex`, `unified-conclusion.tex`를 쓴다. 최종 빌더는 이 네 파일을 모두 요구한다. 개별 semantic matching·score·ray sharing·retargeting의 필수성을 최종 V1↔Selected 묶음 비교에서 추론하지 않는다.
4. 신규 공정 teaser와 Figure 2–4, Main/보조 표, 내부 oracle-scale 표의 캡션을 전 페이지 시각 검수한다. Teaser의 사전 선정 규칙과 공동 성공 집합 분모를 함께 표기한다.

## 검증 조판 명령

작업 위치: `/root/dev/Pixal3D/.worktrees/ray-score-prior-stage-20261019`

```bash
python docs/superpowers/paper/build_fair_manuscript.py \
  --paper-root docs/superpowers/paper \
  --support-snapshot /root/dev/pixal3d-inspection/manuscript-geostyle-data-20261027/source \
  --main-table-json /root/dev/.cache/deformnext-benchmark-runtime/scale-ungiven-main-table-v4/scale-ungiven-main-table.json \
  --paired-json /root/dev/.cache/deformnext-benchmark-runtime/scale-ungiven-ours-paired-v3/paired-eight-endpoints.json \
  --official-contrasts-json /root/dev/.cache/deformnext-benchmark-runtime/scale-ungiven-official-contrasts-v2/official-contrasts-sixteen-endpoints.json \
  --common-manifest /root/dev/.cache/deformnext-benchmark-runtime/scale-ungiven-main-table-v4/common-manifest.json \
  --output-dir /root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028/fair-manuscript-final-v1
```

빌더는 최종 공통 N, 10지표·10방법, V1↔Selected 8지표, official 16지표, teaser PNG/receipt로 검증된 PDF SHA·캡션, 원고 Abstract/Conclusion 존재, TeX 오류, 각 결과 평균과 주 표 셀 일치를 검사한다. 출력은 `source/retargeting-final-manuscript.pdf`와 전 페이지 `review-pages/`다. 눈으로 검사한 뒤에만 `docs/superpowers/reports/`에 **새 이름**으로 복사하고 SHA/사용한 평가 JSON·TeX·fig provenance를 리포트에 기록한다. 기존 2026-10-27 M1-scale PDF는 덮어쓰지 않는다.
