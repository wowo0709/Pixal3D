# 실제 ray 기전 그림의 논문 반영과 최종 평가 준비

2026-10-25 UTC. 직전 턴은 전체128개 입력 검증·pending connector 수정과
실제 기전 그림 생성으로 진전이 있었다. 이번에는 manuscript 반영·compile·검수,
빠진 Fixed VAL36 연결과 실제 저장 공간 점검을 진행했다.
최종 ablation 품질·모델 선정·TEST·최종 논문은 미완료다.

## 실제 논문 PDF

`dev-ray-evidence.tex`와 본문에 실제 RGB4장·관측 ray member·Top-K 후보와
저장 posterior를 연결했다. 사전순 첫 Top-1 miss/Top-K hit·관측 member2개
이상의 관계이며 downstream 품질을 보고 고르지 않았다. 예시 GT 후보는
rank2, probability mass25.64%다. 저장된 observed member와 전체 native
ray context를 구분하고 source-use를 calibrated visibility로 해석하지 않는다.
DEV32 recoverable31,408 relation의 Uniform1.63%/Fixed6.46%/Learned6.18%는
진단이며 matched Uniform 학습 결과나 최종 품질·성공 빈도의 증거가 아니다.

새 `paper-draft-actual-ray-evidence-v325`는13쪽·24개 인용이다.
Compiler returncode0, TeX warning·미해결 인용·overfull/underfull0이며
5·6·10쪽을 실제 PNG로 검수했다.31개 artifact/1,061,771bytes는 실제
file4 SHA256/size와 일치한다. 신규 control/TEST cell은 채우지 않았다.

[실제 개발 PDF](/root/dev/.cache/deformnext-retargeting/paper-preview-20261025-v325.pdf).

## Fixed VAL36 실제 후속 작업

Queue에서 V1/R의 기존 validation36 및 Learned의 두-load 검증은 완료됐지만
Fixed의 작업은 없음을 확인했다. 동일 완료된 Fixed480 checkpoint와 noise0
DEV128의 원래 prefix36을 사용하는 두 fresh-load를 기존 public factory로
준비했다. 새 학습 arm이 아니다.

| 작업 | 실제 시작 조건·역할 |
|---|---|
| Fixed fresh-load0, GPU7 | Uniform 최종 noise2 종료 뒤36개·batch8 |
| Fixed fresh-load1, GPU7 | load0 뒤 별도 process에서 같은36개·seed |
| CPU parity reducer | 두 실제 결과·archived occupancy·prefix32·tail4 대조 |

GPU 작업2개와 CPU1개를 auxiliary queue에 실제 append했고 pending 상태를
확인했다. 우선순위15로 기존 proof/evaluation10 뒤, science20 전에 실행할 수
있다. 현재 학습은 유지했다. `fixed-evidence-candidate-VAL36-v327`의
request/prepared/execution5개 폴더는 각각 실제 file4에 보존됐다.
실제 parity 성공·최종 채택·TEST release는 아직 아니다.

## 기존 실제 GPU render proof 재사용

CPU campaign preflight192 이후 source5개가 바뀌었지만 이후의 실제
`final-render-compatibility-and-source-lock-v208`과 대조했다. 현재 campaign
required source46개 모두 GPU proof의 source59 lock에 포함되고 hash가 같다.
Render vertices/faces·GT·camera와 세 view의 shaded/base_color/alpha/mask는
모두 byte-identical이며 원래 child returncode0·session drain도 확인했다.
새 GPU 실행은 하지 않았다. 한 DEV 사례의 integration 근거이며 TEST 전체
재현성이나 최종 품질의 증명이 아니다. 재확인과 새 pending descriptor의
근거는 `current-render-lock-and-Fixed-VAL36-admission-v329`에 보존했다.

## 실제 저장 크기와 TEST 계획

완료된 Fixed DEV128/93 source의 case 평균은347.17MiB이며, 기존 raw
retirement 경로가 보존하는 부분은 평균43.95MiB다. 관측 free552.92GiB에
대해 신규5개 control의640 case와 SS6개/PBR2개 실행의 남은 checkpoint
증가량 추정은308.17GiB다.

| TEST 최종 모델 수 | 모든 raw를 유지한 뒤 예상 free | Rolling raw retirement 사용 시 예상 free |
|---|---:|---:|
| 4 | −444.17GiB | 157.53GiB |
| 5 | −616.40GiB | 135.72GiB |

DEV에서 추정했으며 TEST density의 상한이나 공간 보장이 아니다.
추가 SS archive·log·metric feature·다른 사용자의 증가·transient 복제를
포함하지 않는다. Whole-asset disk credit와 workspace admission을 유지한다.
원본·GT·checkpoint·scoring 결과를 보존하고, 모든 consumer 종료·reader
부재·실제 file4 SHA를 확인한 소유 raw duplicate만 정리하는 기존 campaign
경로가 필요하다. 이번 턴의 실제 파일 삭제는0이다.
`remaining-cascade-and-TEST-storage-readiness-v326`의 실제128개 inventory와
audit2개/117,765bytes는 file4에 보존됐다.

## 학습과 다음 작업

00:55 UTC에 원래 PID/birth·child가 살아 있음을 확인했다. PBR vanilla202
915/1900, Uniform286/480, No-R261/480이었다. 중간 수치를 최종 모듈 효과로
쓰지 않으며 학습을 중단하지 않았다.

다음은 신규5개 own480·SS128×3·cascade8metric, 선택 후보와 PBR 독립 확인,
Fixed VAL36 실제 실행, 근거 기반 모델 선택·동결, TEST508/10metric과
최종 표·성공/실패 그림·Results/Discussion/Abstract/Conclusion이다.

## 후속: 같은 step의 중간 품질과 source 학습 경로

01:08 UTC에 Uniform은304/480, No-R는282/480이었다. 이후 No-R의288-step
검증이 완료돼 이를 추가했다. 완료된 quick32 항목만 대조한다. 같은32개 source-disjoint asset·같은
generation noise·GT이며 초기 per-asset 결과는 네 모델에서 정확히 같다.
네 모델의 첫288개 실제 학습 asset·noise·timestep·global input trace도
정확히 같다. 표는 macro mean-asset IoU이며
global occupancy IoU 및 최종128×3와 구분한다.

| Updates | V1 | Fixed | Uniform | Physical no-R |
|---|---:|---:|---:|---:|
| 0 | 0.458358 | 0.458358 | 0.458358 | 0.458358 |
| 96 | 0.463801 | 0.461789 | 0.461136 | 0.463828 |
| 192 | 0.467447 | 0.463634 | 0.465803 | 0.467777 |
| 288 | 0.468973 | 0.471552 | 0.474767 | 0.469064 |

Uniform−Fixed의288-step 평균 차이는+0.003215지만,14개 개선/18개 하락,
median−0.000181이다. 가장 큰 개선3개의 합은+0.107564이며 전체 차이 합
+0.102868보다 크다. 해당3개를 제외한 평균 차이는−0.000162다.
이 제외 계산은 평균의 집중도를 설명하는 진단이며, 표본을 제외해 새 성능
지표를 보고하거나 승자를 바꾸는 용도가 아니다. 반복 관측한 중간 결과에
significance·보편적 score weighting 우위·Uniform 최종 우위를 주장하지 않는다.

실제 완료된 source checkpoint9개의 CPU tensor 비교에서 matching decoder는
모두 정확히 같았고, 저장된10-step gradient entry에서 decoder gradient는0이다.
Fixed/Uniform 각각28개 entry 전부에서 retargeter·source-use·fusion gradient는
양수다. No-R의28개 entry에서도 source-use·fusion gradient는 양수이고
checkpoint에 retargeter는 물리적으로 없다. 따라서 이 관측은 source branch의
gradient 단절이나 matcher의 비의도적 업데이트를 뒷받침하지 않는다.

Step288의 fusion gate는 Fixed0.026804/Uniform0.026898이고, 마지막 training
pack의 R residual RMS는0.047621/0.047436이다. 이는 작은 검증셋의 평균
activation이 아니다. Invalid ray의0을 제외해 유도한 RMS와 post-update
bound의 비율도 약0.562/0.559이며, per-channel saturation 비율이 아니다.
이 scalar만으로 saturation이나 최종 품질 차이의 causal 원인을 결론내리지 않는다.
CPU NN forward0회, CUDA 초기화False를 확인했다.

R의 same-step 차이는 step192에서 Fixed−no-R−0.004143이었다가 step288에서
+0.002488로 뒤집혔다.288에는32개 중19개에서 Fixed가 no-R보다 높다.
초기 함수의 mismatch는 초기 결과·학습 trace·실제 cold proof와 맞지 않으며,
중간 최적화 경로의 차이를 최종 필요성으로 단정할 수 없다. 이는 예정한
480 updates와 실제 downstream 평가를 끝내야 하는 근거다. R의 최종 효과나
지연 원인의 causal mechanism은 아직 확정하지 않는다.

현재 해석은 **대응 후보에 더 큰 정답 mass를 주는 것과 최종 occupancy 개선은
구별해야 한다**는 것이다. 현재 경로가 학습되지 않는다고 단정해 구조를
바꾸지 않으며, 계획한 같은-support Uniform·no-R 및 matching3개 arm의
480/128×3·실제 cascade8metric 결과로 최종 기여를 판정한다.

처음 Canonical `shared-Fixed-interim288-source-learning-v331`의24개 artifact/
11,416,973bytes와 실제 source checkpoint8개/52,916,625bytes,
`shared-Fixed-interim288-interpretation-v332`의5개 artifact/241,770bytes를
file4에 보존했다. 실제 curve PNG도 검수했다. 이는 진행 중 NN 전체의
closure가 아니라 명시한 완료 entry/prefix/checkpoint의 보존이다.

No-R288을 포함한 후속 `shared-Fixed-interim288-source-learning-v334`의
24개 artifact/12,441,284bytes와 source checkpoint9개/57,166,731bytes,
`shared-Fixed-interim288-interpretation-v335`의5개 artifact/243,246bytes도
실제 file4에 보존했다. 아래 같은 W&B 분석 run의 표·이미지·summary를
기존 기록을 보존하며 갱신했다.

[실제 W&B 중간 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/shared-Fixed-interim288-source-learning-v333)에
32개 사례별 표·곡선·macro/global 구분을 로깅하고 remote summary 수치를
원본과 정확히 대조했다. 기존 학습 run과 최종 ablation table은 수정하지 않았다.
GPU0–3의 실제 PBR ranks와 GPU6/7의 원래 SS child가 각 GPU를 점유함을
확인했고, 해당 GPU에 별도 compute process는 없었다.
