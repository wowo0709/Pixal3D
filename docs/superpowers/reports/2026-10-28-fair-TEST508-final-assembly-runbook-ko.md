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

## 본문 길이 사전 점검

합성 공정 표와 임시 결과 문장을 넣은 레이아웃 시험은 **26페이지**였고, 본문 결론이 12페이지, 참고문헌이 13–14페이지, 부록이 15페이지부터 시작했다. 이는 실제 최종 결과와 teaser가 들어가기 전의 배치이므로 최종 페이지 수가 아니다. 최종 실측 서술을 넣은 뒤에는 메인 표·선정된 ablation 표·핵심 그림을 우선 보존하고, 상세 내부 oracle-scale 결과와 부가 개발 진단을 부록으로 옮길지 검토한다. 정보나 부정적 결과를 삭제해 분량을 맞추지 않는다. 투고 형식의 실제 분량 요건은 투고 전에 별도로 확인한다.

2026-10-28 확인: [CVPR 2027 공식 저자 지침](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)은 **본문 8페이지 이내(그림·표 포함), 공식 템플릿 사용**을 요구하고 참고문헌과 필수 LLM Usage Statement만 별도 페이지를 허용한다. 현재 `article` 기반의 12페이지 본문은 **연구 원고 초안**이며 CVPR 2027 투고 PDF가 아니다. 해당 연도 [공식 LaTeX author kit](https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip)을 `/root/dev/.cache/paper-tools/author-kit-CVPR2027-v1-latex.zip`에 받아 SHA256 `da1ec5db9b73531ea82a87723e4aedb631083331fe655d1972279b48a7bdb2fe`를 기록했다. CVPR 2027 투고를 목표로 한다면 실측 결과를 넣은 뒤 별도 공식 템플릿 8페이지 본문과 부록을 조립·검수해야 한다. 현재 장문 원고의 결과·증거를 지우지 않고 재배치한다.

공식 kit의 `cvpr.sty`를 사용한 최소 review PDF를 로컬 Tectonic으로 컴파일했고, 필요한 LaTeX 패키지를 캐시한 뒤 `--only-cached` 재컴파일도 통과했다. 이는 **템플릿 환경 확인**이며 8페이지 본문 작성이나 최종 투고 PDF 완성을 뜻하지 않는다.

별도 [CVPR 2027 압축 원고](../paper/cvpr2027-submission.tex)를 공식 review 템플릿으로 작성하고 [preview 빌더](../paper/build_cvpr2027_preview.py)로 4쪽(참고문헌 포함)을 조판·시각 검수했다. 레이아웃 오류, 미해결 참조·인용은 0개다. 이 버전은 공통 성공 집합의 최종 표와 teaser가 없다고 본문에 표시한 **protocol preview**다. 실측 결과와 공정 teaser를 넣은 뒤 본문 8쪽 제한 및 LLM Usage Statement를 다시 검수해야 하며, 현재 PDF를 최종 논문으로 사용하지 않는다.
