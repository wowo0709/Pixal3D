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
