# 실제 C1 weight의 cache·AMP·CFG 검증

**실행 결과:** 실제 trained C1의 두 사례에서 BF16 cache/online·training/inference·CFG 검사를 완료했다. 유한한 trace 오차와 mask를 명시한 최종 결과는 `conditioning-contract-v2/summary.json`이다. 독립 검사·W&B22개 numeric field와4개 artifact 파일의 실제 다운로드·actual-file4 SHA256/크기 검증도 완료했다. 아래 v1 실행 계획은 이력으로 보존한다.

최초 계획의 active-path cache parity와 dropout/AMP 검증 근거를 현재 최종 후보에 연결한다. 기존 R9 보고서는 adapter를 활성화한 BF16 SS/PBR16개 사례에서 cached/online evidence 오차0을 기록했지만, 사용한 head/모델 weight의 hash를 보고서 자체에 담지 않았다. 과거 검사를 현재 checkpoint의 직접 증명으로 확대하지 않고 별도 검증을 추가한다.

현재 matched seed101 C1의 실제 final denoiser·conditioner·frozen Q2C head를 그대로 읽는다. 고정32에서 실제 target/source validity를 갖는 첫 두 사례를 선택하고, cached/online의 RGB 네 장·camera 및 transport tensor bytes를 fingerprint한다. Head·recipe·manifest·panel·weight도 SHA256으로 고정한다. Learned retargeter와 최종 projection의 native 대비 변화가 실제 nonzero여야 하며, zero initializer 상태로 검사를 통과할 수 없다.

BF16에서 candidate index exact, evidence/logit/weight/residual·최종 condition의 cached/online 일치를 검사한다. 이어서 실제 `get_cond`의 CFG 확률0과1, seed를 고정한 확률0.5의 sample별 mask를 확인한다. Global과 projection에 같은 mask가 적용돼야 하고 확률1에서는 모든 추가 조건도 완전히0이어야 한다. 같은 입력의 training p0/inference condition도 일치해야 한다. 실제 LoRA·base·conditioner를 고정한 no-grad 검사이며 optimizer state/update는0이다. 생성 품질 평가나 전체9149개 cache의 GPU parity 검사는 아니다.

CPU 검증5개는 활성 경로 누락, online/cache 차이, CFG 뒤 semantic projection 누출, mixed mask의 branch 불일치를 거부한다. 실제 GPU proof는 GPU2의 adapter-only 세 seed가 모두 끝나 해당 GPU를 반환한 뒤 실행한다. GPU3의 query arm 세 seed는 병행할 수 있고, GPU2 sampler는 이 검사 뒤 시작한다. 최초의 전체6개 완료 대기는 아래 실제 실행 시간에 근거해 자원별 dependency로 조정했다. 현재 GPU0/1 B3 학습과 완료된 V1/C1 weight는 유지한다. Canonical 결과는 file4 `rgb_matching_v2/conditioning-contract-v1/summary.json`이다. 설정/CPU 검증을 실제 GPU PASS로 표시하지 않는다.

Seed101 실제 전체 시간은 adapter-only1,786.88초, query1,996.44초였다. Query 쪽은 control 하나를 더 평가하므로 GPU2가 먼저 끝난다. Pending QA와 GPU2 첫 sampler 두 dependency만 변경하고316개 job graph의 비순환성을 검사했다. 세 wave에서 약10.5분의 GPU2 대기를 줄일 수 있다는 첫 실측 기반 예상이며 전체 완료 시각의 확약은 아니다. 모든6개 regional run·strict 집계는 여전히 필수다. 원래 정의와 변경된 job은 `provenance/gpu2-postregional-pipeline-schedule-v1.json`에 남겼고 기존 학습·평가는 중단하지 않았다.

## 실제 GPU 결과와 v1 통계 표시 수정

첫 v1 실행의 `torch.testing.assert_close`는 실제 logit 동등성을 이미 검사해 통과했다. 다만 같은 negative infinity끼리 뺀 max-error 통계가 NaN이었다. 이를 모델 오류나 동등성 검사의 실패로 해석하지 않는다. 기존 v1 JSON을 보존하고 `provenance/current-C1-v1-trace-statistic-review.json`에 기존 assert가 통계 계산보다 앞서 있었음을 기록했다.

Verifier만 수정해 NaN/+infinity를 거부하고, logit의 negative-infinity mask 일치와 finite 값의 오차를 따로 검사했다. 최종 JSON도 `allow_nan=False`로 직렬화한다. 모델·학습 수식·weight는 바꾸지 않았다. CPU 회귀 검사는 기존5개를 포함해12개 PASS였다. 같은 weight·두 sample·입력 fingerprint·CFG 결과를 유지한 v2 GPU 실행이 정상 종료했다.

| 실제 v2 검사 | 결과 |
| --- | --- |
| Evidence / finite logits / candidate weight / retarget residual 오차 | 각각0 |
| View weight 최대 절대 오차 | 5.960464477539063e−8 |
| Finite logits / 동일 negative-infinity mask 원소 | 9,600 / 121,472 |
| Cached/online global·projection 오차 | 모두0 |
| Training p0 / inference global·projection 오차 | 모두0 |
| CFG p1 / p0.5 sample mask | 모든 branch zero / `[True, False]` 일치 |
| Native 대비 active projection RMS | 0.023112159222364426 |
| Optimizer update | 0 |

실제 frozen head24개 key와 selected head를 대조했고 denoiser·conditioner·head·recipe·manifest·panel의 실제 파일 SHA256·크기를 독립 재계산했다. v1의 기존 통과 assertion 결과와 v2의 통계 이외 결과도 같았다. 이 proof는 두 실제 활성 사례의 conditioning 계약이며 생성 품질이나 전체 cache의 GPU 검증이 아니다.

W&B run은 `ss-current-C1-conditioning-contract-v2`다. `conditioning-contract-v2/independent-audit.json`은 actual remote finished/PASS,22개 numeric field와4개 artifact 파일 bytes 일치를 기록한다. `provenance/current-C1-QA-v2-completed-file4.json`은9개 산출물36,963bytes의 actual-file4 검증이다. QA-v2를 sampler V1-50 뒤에 추가해 총317개 queue가 됐으며 기존 학습·평가는 중단하지 않았다.
