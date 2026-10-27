# TEST508 cascade 종료와 metric 준비 요청 복구

2026-10-27 07:36 UTC 기준, 동결된 다섯 pipeline의 실제 TEST508 계산은
**508 assets × 5 models = 2,540 cases**를 모두 완료했다. `v636`은 이전 세
campaign에서 검증한 277개 complete group/1,385 case를 원래 producer 경로에서
재사용하고, 남은 231개 group/1,155 case를 실제 계산했다. 최종
`actual-selected-five-model-TEST508-remaining-cascade-v636/result/audit.json`의
상태는 `PASS_ACTUAL_SELECTED_FINAL_CASE_CAMPAIGN`이고, `assets=508`, 다섯
모델 이름과 각 case의 원래 audit/worker terminal을 기록한다. 해당 audit의
`result/file4-closure.json`과 자식 작업 종료를 증명하는 `release.json`의
`release-closure/file4-closure.json`은 모두 실제 file4 SHA256/size 대조
`PASS`다. 따라서 cascade 계산과 보존은 종료됐지만, 최종 10개 metric과 논문
결과는 아직 완료되지 않았다.

첫 metric 활성화 callback
`activate-selected-final508-remaining-metrics-CPU-v636`은 exit code 1로
종료했다. 로그의 직접 원인은 세 campaign 재사용 manifest 검증에서 metric
준비 context의 `campaign_root` 키를 참조한 `KeyError`다. 이 필드는 완료된
campaign/result에는 있었으나 최초 metric context에는 빠져 있었다.
실패는 metric output 생성 및 GPU 점수 작업 등록 전에 발생했다. TEST case나
동결된 checkpoint·수치 코드를 실패한 요청으로 다시 계산하지 않았다.

새 요청 `actual-TEST508-metric-context-repair-v667/prelaunch-v1/context.json`은
실패한 context에 **동일한 실제 `v636` campaign_root 필드만 추가**했다.
원래 505개 수치 source의 byte SHA256을 재확인했고, 새 요청과 원인 audit의
file4 closure가 `PASS`다. 동결된 원본 수치 source, 모델 선택, metric 정의,
scored archive 및 원래 실패 요청은 수정하지 않았다. 이 요청으로 원래
`operational_three_metric_entry --phase prepare`를 실행한 결과,
`actual-selected-five-model-TEST508-tenmetrics-v636/activation-v1/audit.json`은
`PASS_ACTUAL_SELECTED_PHYSICAL508_METRIC_JOBS_INSTALLED`를 기록하고
file4 closure도 07:44 UTC에 `PASS`다. 11개 GPU metric 작업과 1개 CPU
collector가 등록됐으며, 07:44 UTC에는 GPU 0·1·2·6·7의 첫 다섯 작업이
실제 `running`이었다.

후속 Main/정성 export(`v637`)는 `ten-metrics/file4-closure.json`을 기다린다.
W&B 검증(`v638`)과 원고 입력 생성(`v639`)은 Main/정성의 실제 closure를
기다린다. 첫 callback 상태 `FAILED`는 운영 이력으로 보존하고, 후속 작업은
완료된 metric closure를 선행 조건으로 사용한다. **최종 TEST metric·비교 표·
결과 의존 원고·최종 PDF는 이 시점 아직 없다.**
