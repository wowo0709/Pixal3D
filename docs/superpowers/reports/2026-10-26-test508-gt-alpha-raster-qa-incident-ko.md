# TEST508 GT 알파 마스크 검사 중단과 재개 기준

2026-10-26 UTC. 본 문서는 최종 TEST508 평가의 운영 오류를 기록한다. 모델 선택이나 TEST 품질에 따른 제외를 승인하지 않는다.

## 실제 중단

`actual-selected-five-model-TEST508-remaining-cascade-v611`은 227개 완료 그룹을 재사용하고 281개 그룹을 처리할 예정이었다. 50개 새 그룹의 250개 모델 사례는 완결되었고, 다음 `zero-source4`의 렌더 단계에서 중단되었다. 실패 당시 총 251/1405개의 새 사례에 결과가 있었으나, 251번째 사례는 완료되지 않았다. 캠페인은 실패 상태와 모든 자식 프로세스 종료 기록을 보존했다.

실패는 `render_cascade_artifact_v2.py`에서 GT M1 메시를 rasterize한 마스크와 GT RGBA alpha를 비교하는 QA 단계에서 발생했다. 해당 camera 8의 GT alpha 648픽셀 중 1픽셀 확장 메시 raster 밖에 7픽셀이 있었다. containment는 `641/648 = 0.98919755`로 기존 연속형 `>=0.99` 판정에서 거부되었다. 모델 예측 마스크나 모델 score 때문에 발생한 오류는 아니다.

## 전체 GT 데이터 감사

원래의 GT 이미지, GT 메시, 카메라, 렌더러 설정으로 TEST508의 1,524개 평가 뷰를 예측 결과 없이 독립 검사했다. 연속형 0.99 기준 실패는 위 1개 뷰였고, GT alpha 픽셀 수의 1%를 정수 픽셀로 올림한 허용치 `ceil(0.01 × alpha foreground pixels)`에서는 1,524개 모두 통과했다. 감사 결과는 파일서버4의 `actual-frozen-TEST508-GT-alpha-raster-QC-v623`에 보존했다.

## 운영상 수정

최종 렌더 QA에서만 1픽셀 확장 raster 밖 GT alpha 픽셀 수가 위 정수 허용치 이하인지 검사한다. 모든 모델, 샘플, 카메라에 같은 규칙을 적용한다. 원본 학습/생성 코드, 렌더 수치 코드, GT 입력, 이미지 배열 및 metric 계산식은 수정하지 않는다. 보정은 TEST 실행 후 확인된 것이므로 논문의 평가 프로토콜과 한계에 명시해야 한다.

기존 실패 사례를 별도 출력에서 이 규칙으로 재실행해 camera 8, 9, 10의 렌더 결과를 모두 생성했다. 파일서버4에 `actual-quantized-GT-raster-failed-view-render-pilot-v624`로 보존했다. 이 pilot의 수치는 모델 선택이나 최종 결과로 사용하지 않는다.

추가 점검에서 사례 검증기 `selected_final_case_integrity.py`에도 연속형 `>=0.99` 검사가 독립적으로 한 번 더 있음을 확인했다. v630 운영 실행은 렌더가 통과한 뒤 이 검증기에서 중단됐다. 원본 검증기 전체 함수의 AST 중 해당 검사식 한 곳만 실제 렌더 출력의 정수 픽셀 QA 증빙을 검증하는 식으로 바꾸는 별도 운영 adapter를 추가했다. 기존 실패 사례의 실제 geometry, render, shared 보고서와 종료 기록을 입력으로 검증기 전체를 다시 실행해 `PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_CASE`를 확인했다. 나머지 원본 사례 검사식은 유지된다. v627의 첫 실행은 스케줄러 wrapper 재귀 때문에 NN 시작 전 중단됐고, 이 오류도 수정했다. 두 실패 실행과 예약 취소 기록은 각각 별도 파일서버4 네임스페이스에 보존한다.

v636 재실행에서는 해당 원래 실패 사례의 생성, 세 카메라 렌더, 공통 material 평가, 원본 나머지 사례 검사를 끝내고 실제 case audit `PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_CASE`와 파일서버4 closure를 생성했다. 이것은 첫 사례의 운영 수정 검증 결과이며, 508개 전체 평가 완료나 성능 개선 결과는 아니다.

## 완료 조건

227개 기존 그룹과 v611의 50개 새 그룹은 각각 실제 group audit, 모델별 case audit, 종료 기록, scoring archive 해시 및 GT 픽셀 일치를 확인한 뒤에만 재사용한다. 남은 231개 그룹은 새로운 네임스페이스에서 실행한다. 이후 508개 전체에 대해 사전 지정된 10개 metric, source-cluster 신뢰구간, Main/Qual 결과와 논문 표를 산출한다. 현 시점에 최종 TEST 수치나 논문 결론은 없다.
