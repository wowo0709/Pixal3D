# Matching score의 실제 후보 사용량과 native voxel-query 실행 검증

2026-10-19 06:23 UTC 관측. 현재 utility scale16 모델의 raw480 최종 평가는 아직 완료되지 않았다. 아래 분석은 모델의 성능 향상 증명이 아니다.

`inspect_score_prior_support.py`로 원래 development128개·93 source의 RGB 기반 캐시를 확인했다. 원래 head SHA, 입력 RGB identity, semantic Top64 ID와 valid support, target alpha의 patch support를 대조했다. 예약121개, SS latent/GT correspondence label, 새 모델 출력은 읽지 않았다. CPU 두 core에서 분석했고 optimizer update와 GPU 사용은 없다. 준비된 soft-prior 설정은 이 분석 전에 이미 temperature0.05로 고정되어 있었다.

활성 query 9,883개의 가중 평균이다. 유효 후보 수와 유효 view 수는 각각 분포 entropy의 지수이며, 서로 다른 정답 correspondence의 개수라는 뜻은 아니다.

| 진단 | 현재 T=0.005 | 준비된 T=0.05 |
|---|---:|---:|
| 최대 후보 가중치 평균 | 93.33% | 56.62% |
| 최대 후보 가중치 중앙값 | 99.92% | 52.24% |
| 유효 후보 수 평균 | 1.231 | 4.674 |
| 유효 후보 수 중앙값 | 1.007 | 3.078 |
| 유효 source view 수 평균 | 1.112 | 1.703 |
| 최대 source view의 가중치 평균 | 96.40% | 77.41% |
| Top-4 누적 가중치 평균 | 99.96% | 89.99% |
| Top-8 누적 가중치 평균 | 100.00% | 96.51% |

두 설정의 semantic 후보 ID·valid support·hard Top-1 선택은 모두 같다. 현재 score prior는 Top64 pool을 제공하더라도 대부분 한 후보·한 source view에 집중된다. 따라서 Top-K를 사용한다는 설정만으로 다양한 후보를 실질적으로 사용하는 것은 아니다. T0.05는 같은 후보의 가중치 분포를 완화한다. 이는 uniform score나 사용자가 정의한 No-score arm이 아니며, No-score는 여전히 Top-1 feature 대조다.

이 분석은 캐시의 FP32 score prior만 측정한다. 새 D1 forward, 실제 GPU AMP, 학습된 utility의 가중치, SS 생성 품질 또는 매칭 정답률을 측정한 것은 아니다. 더 넓은 분포가 오답 후보를 섞어 성능을 낮출 가능성도 있으므로, 준비된 P 후보의 고정 최종 평가와 같은 T0.05의 기존 static C1 대조가 필요하다. 이 static C1은 raw480/128개/3noise IoU0.440938로 기존 기록이 있으며, P가 이 모델보다 향상됐는지를 attention의 직접 비교에 추가해야 한다. P 대 Top-1만으로 temperature 변경과 학습된 attention의 효과를 분리하지 않는다.

Native voxel-query 후보는 비어 있던 물리 GPU3에서 단독 검증을 완료했다. 원래 control의 fresh202 학습 및 GPU QA가 끝났고, noise2 평가는 GPU2에서 계속 진행되어 자원 충돌은 없다. proof를 먼저 실행했으며 본 학습을 시작하거나 재시작한 것은 아니다.

실제 native SS 조건에서 global/proj 차이는 모두 정확히0이었다. LoRA loss gradient L1은248.3946, adapter output gradient L1은10.0507, fusion gate gradient는0.03655였다. 임시 adapter output perturbation으로 utility까지 SS-loss gradient가 전달됨을 확인했고, 해당 output weight를 정확히 복원했다. Decoder/score gradient는 frozen 설정에 맞게0, utility gradient는0.001058이었다. Optimizer update는0이다. 실행 전후11개 소스 hash와 native-proof JSON hash가 일치한다.

이는 초기 정합과 gradient 연결의 단일 샘플 GPU 증명이다. micro48/DDP2 메모리·속도와 학습 성능은 아직 증명되지 않았고, 실제 본 학습 직전에 native proof와 두-update DDP benchmark를 다시 실행한다. Standalone 결과는 `voxel-query-followup-v1/verification/voxel-query-c1/standalone-native-v1/`에, 새 본 학습의 proof는 그 부모의 별도 파일에 기록하므로 덮어쓰지 않는다.

실제 진단 파일은 `semantic-topk-search-v1/orthogonal-utility-followups-v1/score-prior-support-v1/audit.json`이다. 모든 원본 case와 입력 파일의 SHA/크기, per-case/pooled descriptive 통계를 유지한다. 학습 프로세스와 여섯 활성 forward 소스는 변경하지 않았다.
