# PBR 독립 seed의 reference 범위와 실제 집계 수정

2026-10-25. 최종 문맥의 다섯 SS 대조·downstream 평가·독립 확인·TEST·논문 계획을 유지한다.
새 학습 arm이나 NN forward를 추가한 작업이 아니다.

## 발견한 충돌과 수정 범위

기존 PBR 초기 검사는 RGB/camera/noise/GT latent/cold prediction latent의 정확한
동일성과 사전 선언한 작은 renderer metric bound를 구분한다. 반면 최종 독립
seed 집계는 두 training seed의 GT PNG까지 정확히 같도록 요구했다. 실제 PNG에서
이 두 조건이 충돌해 아직 pending인 최종 collector가 실패할 수 있음을 확인했다.

`selected_pbr_seed_confirmation_v2.py`와
`collect_selected_pbr_seed_confirmation_v2.py`를 새로 만들었다. 기존 pinned module과
학습 코드는 수정하지 않았다.

- 같은 seed의 V1/R 등 비교 모델 및 여섯 epoch 전체는 GT pixel이 정확히 같아야 한다.
- 독립 seed는 기존의 정확한 입력·latent·cold prediction 조건을 유지한다. Render
  차이는 원래 producer의 실제 증거 및 기존 bound를 다시 검증한다. PSNR0.01,
  SSIM0.0002, LPIPS0.0003의 **기존 초기 metric bound**이며 새 GT pixel 허용 오차가 아니다.
- 최종 collector는 snapshot에 쓰인 hash를 신뢰하는 대신 닫힌 원본 GT PNG의
  첫 column을 직접 읽어 map과 대조한다. Native panel 순서는 GT | Pred | AbsDiff다.
- Reference scope와 실제 seed/mode별 pixel hash를 결과에 명시한다. Seed별
  curve의 원래 metric을 유지하며 서로 다른 rendered reference를 공통 pixel 평가로
  pooling하지 않는다. Final1900·rank·trace·frozen base·LoRA 검사는 유지한다.
- 같은 seed202의 R과 vanilla GT가 다르면 계속 실패한다. 그런 경우에는 공통
  reference를 사용한 재평가가 필요하며 추가 허용으로 통과시키지 않는다.
- Generated-geometry cascade 및 TEST의 공통 GT pixel 규칙은 그대로다.

## 실제 관측과 검증

`PBR-within-seed-reference-prefix-v343`에서 실제 validation prefix·producer proof·
원본 PNG를 읽었다. Seed101 vanilla/R의0/380/760/1140/1520/1900은 모델·epoch 간
GT가 정확히 같다. Seed202 vanilla의 현재 완료0/380/760도 epoch 간 정확히 같다.
각 초기 producer 증거를 원래 cold reference에 대해 재계산했다.

독립 seed의 네 사례에서 변경 pixel은 각각131/593/51/248개이며 각 image는
65,536pixel이다. 최대 uint8 channel 차이는3/6/48/16, 전체 RGB MSE는 각각
1.995e-8/8.995e-8/6.298e-7/1.037e-7이다. 입력·GT latent·camera·cold prediction
latent는 정확히 같다. 이 관측은 renderer 내부의 특정 원인을 입증하지 않는다.

해당 관측의6개 artifact/225,767bytes가 실제 file4와 SHA256/size 일치했다.
진행 중 seed202의 source 파일은 완료 artifact로 가장하지 않고 관측한 prefix와
원본 SHA를 별도로 기록했다. 최종 collector는 완료 후 실제 closure로 다시 읽는다.

새 회귀 검사와 기존 PBR initial/trace/freeze 관련 검사의 합계107개가 통과했다.
같은 seed의 GT 변화, epoch 누락, producer proof 누락/변경, 입력/latent 변화,
불완전 학습, trace/frozen-base 변화 및 잘못된 panel layout은 거절한다.
기존 summary는 cross-seed GT 차이를 계속 거절하며, 새 policy 선택만 이를
seed별 stage reference로 명시적으로 구분한다. 테스트 fixture는 실제 결과가 아니다.

## 남은 실행과 논문 해석

기존 pending CPU collector만 새 version과 source pin으로 연결한다. Seed202
vanilla/R의 실제1,900 학습과 네 rank terminal·archive·공통 seed 내 GT 확인이
끝나기 전에는 최종 두-seed 확인 결과를 만들지 않는다. 현재 seed202 R는 아직
관측하지 않았으며 결과를 추정하지 않는다.

논문 protocol과 claim 원장에 seed별 reference 범위를 명시했다. Fixed4는 GT shape
조건의 target8 학습 추세 확인이다. Generated geometry·source preservation·최종
pipeline 재현성을 대신하지 않는다. 최종 SS 대조5개·독립 Fixed·cascade8metric·
모델 동결·TEST508의10metric·최종 표/정성/writing은 계속 미완료다.
