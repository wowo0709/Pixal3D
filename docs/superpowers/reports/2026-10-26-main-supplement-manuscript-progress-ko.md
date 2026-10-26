# 본문·Supplement 구성과 실제 TEST 실행 후속

SS는 geometry, PBR는 appearance라는 사용자 기준에 맞춰 원고를 재구성했다.
최종 TEST 결과를 개발 수치로 대신하지 않았다. 실제 평가와 최종 결과에 따른
writing은 계속 진행해야 하며, 이 기록은 논문 완료 선언이 아니다.

## 원고에서 완료한 작업

새 진입점은 `../paper/retargeting-final-manuscript.tex`다. Introduction, Related
work, Method, Experiments, Results, Discussion, Supplement를 별도 section으로
작성했다. 선택된 Fixed SS와 별도 Top-1 R PBR의 구조도를 넣었다. 기존 개발 초안과
산출물은 보존한다.

- 본문: 실제 구현의 ray membership·matching/value 분리·weighted evidence·source
  correction을 설명하고, SS geometry ablation과 동일 geometry의 PBR appearance
  ablation을 별도 표로 배치했다. Geometry 점추정과 appearance 점추정의 선정 이유를
  각각 설명하며 모든 관련 zero-containing CI를 유지한다.
- Supplement: 원래12-arm family와 두 follow-up, 독립 SS 두 seed, PBR 두-seed
  fixed-four GT-shape training QA, 전체 matched contrasts와 이전 geometry별 PBR
  비교, matching 진단·실제 ray 예제를 남겼다. 원래 study의 조건부 결과를 최종
  구조에서의 모듈 필수성으로 확대하지 않는다.
- 검증 범위: 선택된 Fixed SS의 VAL36 두 fresh-load와 V1/Top-1 대조군의 기존
  checkpoint-binding/native parity 검사를 구분했다. 모든 ablation checkpoint가
  두 fresh-load 검증을 받았다는 표현은 사용하지 않는다.
- 정성 그림: TEST 이미지를 보기 전에 원래 v355 declaration의 첫 outcome-blind
  예제를 Main에, 두 번째와 모든 conditional 예제를 Supplement에 싣도록 배치를
  고정했다. 원래 사례 선택·category·이미지 변환 규칙은 그대로다.

`stage-separated-main-and-supplement-manuscript-preview-v578`에서 실제 PDF를
컴파일했다. **13쪽 = 본문6 + 참고문헌2 + Supplement5**, 인용24개다. 모든13쪽을
렌더링해 실제로 검수했다. 표/그림 clipping·overfull·미해결 인용·missing glyph는
없다. Underfull 경고32개는 원본 log와 review 기록에 남겼다. 이는 개발 preview이며
최종 TEST Main/정성 그림·결과 종속 Abstract/Results/Discussion/Conclusion은 대기한다.

기존 numeric table14개, 참고문헌과 실제 ray 그림의 byte를 이미 file4에서 검증된
v560 snapshot과 대조했다. 두 단 조판에서는 table→table* 및 figure→figure*의 float
환경만 바꾸고 기존 cell·caption을 유지했다. 선택 구조도는 기존 도형·수식·경로를
유지하고 working/selected 제목을 구분했다. Frozen numerical source505개는 현재
byte까지 모두 일치한다. 모델·학습·sampler·평가 endpoint는 바꾸지 않았다.

첫 v576은13쪽으로 컴파일했고, v577은 참고문헌의9pt italic font가 local cached
compiler에 없어 실패했다. 실패 source/log/terminal을 보존하고 원래10pt 참고문헌을
유지한 새 v578에서 성공했다. NN 실행이나 전역 환경을 변경하지 않았다.

## 실제 평가 상태와 남은 종료 조건

2026-10-26 13:42 UTC 현재 campaign은
`actual-selected-five-model-TEST508-remaining-cascade-v611`이다. 이전 실패
실행에서 완결·검증된 227 asset/1,135 case를 불변 재사용하고, 남은 281 asset/
1,405 case 중 186 case를 새로 완료했다(전체 논리 진행 1,321/2,540).
CPU coordinator와 GPU0·1·2·6·7의 case worker/하위 geometry process를 실제
PID·명령·GPU 점유로 확인했다. 작업자 교체와 짧은 회수 구간은 campaign 종료로
해석하지 않는다. 10metric→Main/Qual→W&B readback→원고 입력 callback은
완전한 cascade closure 후 실행되도록 대기 중이다. 최신 운영 증거는
[v611 복구 기록](2026-10-26-TEST508-immutable-reuse-and-disk-recovery-ko.md)을 따른다.

1. 동일 frozen 다섯 pipeline의 TEST508 cascade와 raw-retirement closure를 완료한다.
2. 연결된 원래10metric collector에서 전체2,540 case·1,524 image/model을 평가하고
   TEST8C=16 family의 source-cluster interval과 signed raw FID/KID를 보존한다.
3. 실제 closed Main/qualitative export와 원격 W&B readback을 완료한다.
4. 그 결과로 최종 Main·Supplement의 수치와 그림을 채우고 Abstract/Results/
   Discussion/Conclusion을 확정한다. 기존 foreground/GT-crop score는 Supplement의
   sensitivity로 보고하며 primary endpoint나 가설 family를 바꾸지 않는다.
5. 실제 최종 PDF를 다시 compile·시각 검수·file4 보존하고 전체 요구를 감사한다.

현재 선택은 DEV-only **Fixed SS + Top-1 R PBR**로 고정돼 있다. PBR의 기존
feature/raw arm은 Top-1 carrier/use를 유지하므로 SS Fixed와 같은 shared Top-K
PBR의 효과를 입증하지 않는다. 이 경로의 appearance 개선이나 모든 모듈의 필수성을
논문 claim에 넣지 않는다. 완료된 학습을 반복하거나 peripheral grid를 추가하지 않는다.

실제 산출물은 file4 결과 root의 `unified-module-pipeline-v1/` 아래 v576/v577/v578에
보존한다. 최신 closure와 live 상태는 agent handoff의 후속 기록을 따른다. 본문 숫자는
개발 결과이며, 아직 실제 최종 논문·TEST 결과가 완성된 것은 아니다.
