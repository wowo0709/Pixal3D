# 공통 R96 SS 두 arm의 실제288 비교와 전달 경로 진단

같은32개 개발 자산·GT·cluster·생성 seed의288-step 평가에서 direct-NoDQ의
평균 voxel IoU가 retargeting보다0.5584%p 낮았다. 탐색적 cluster CI도0을 제외한다.
전체 탐색과 반복 epoch 평가를 보정한 독립 test 결과는 아니며, 두 학습은480까지 유지한다.

| 모델 | Mean-asset IoU (%) | TP/FP/FN 합산 IoU (%) |
|---|---:|---:|
| Retargeting288 | **47.6711** | **44.5995** |
| Direct-NoDQ288 | 47.1126 | 43.7217 |

Direct-NoDQ − Retargeting의 평균 IoU 차이는−0.558423%p,
source-cluster bootstrap4000회/seed20261023의95% CI는
[−1.059821, −0.055254]%p다. 192-step의 차이는−0.030397%p였고 CI가0을 포함했다.
중간 panel에서 차이가 커진 사실은 기록하되, 이를 최종128×3 또는 학습 seed 간
효과로 확대하지 않는다.

합산 count는 retargeting TP172237/FP136784/FN77165,
direct-NoDQ TP171254/FP142289/FN78148이다. 이 단계의 direct-NoDQ는
TP가983 적고 FP가5505 많았다. 어떤 전달 경로가 원인인지는 이 count만으로
확정할 수 없다.

## 초기 붕괴·정지·난수 불일치 확인

초기96의32개 metric과count는 같았다. 실제96–288의 두 rank386개 snapshot에서
CPU/CUDA·NumPy·Python RNG와 consumed sampler 상태가 모두 같았다.
데이터 설정·data_dir도 같다. Loader 전용 generator는191 이후196개 snapshot에서
달랐으나 학습 RNG와 분리돼 있다. 실제 x_t/noise tensor hash를 기록한 것은 아니다.

288 Adam checkpoint에서 두 모델의 공통360 LoRA+6 R/fusion은 모두288 update,
추가 head9개는 각각192 update다. Moment tensor는 유한하고 head LR은1e−4다.
Step288 TRAIN batch 로그의 feature gradient L1은10.27082,
feature residual RMS는0.0124567, direct correction RMS는0.0119230,
head별 평균 effective K는4.1998이다. 추가 branch가 정지하거나0 출력에
머무른 상황은 아니다. TRAIN 로그의 수치를 validation 고정 자산의 기여량으로
해석하지 않는다.

192 저장 tensor의 별도 CPU 대조에서는 두 모델 모두 기준 R96의 frozen NN701개와
q2c24개가 유지됐고 LoRA360개·R/fusion6개가 갱신됐다. 이는 저장된 rank0의
중간 검증이며 최종288/480의 모든 rank tensor identity를 주장하지 않는다.

## 확인한 구현상 전달 차이와 다음 검증

현재 `DirectRayRetargetingNoQueryConditioner`는 attention feature F를
`R(Top1+F, target)`의 입력으로 사용하고, 별도로
`C(F)=g_max*tanh(F/g_max)`를 retargeting residual에 더한다.
Nearest consumer는 evidence−residual을 원래 Top1 carrier로 쓰고,
semantic delta에는 fusion gate g를, residual에는3/4 계수를 적용한다.
따라서 현재 경로는 개념적으로 다음과 같다.

`native + g*semantic_delta(Top1) + 3/4*[R(Top1+F,target) + C(F)]`

별도의 C(F)는 semantic fusion gate 밖에 있다. 기존
`RayFeatureAttentionConditioner`의 직접 F 전달은 semantic carrier 쪽에 있어
fusion gate와 함께 조절된다. 이는 두 구현의 사실상 차이이며, 어느 쪽이 더 좋다는
증거는 아니다. FP32/BF16 연산 순서와 bounded correction도 달라 한 경로를
다른 경로의 bit-exact 재현이라고 부르지 않는다.

이 차이가 현재 하락에 기여하는지 확인하기 위해, 저장된 NoDQ288 한 모델에서
direct correction C(F) 전달만0 또는g*C(F)로 바꾸는 작은 inference 진단을 추가한다.
F의 계산·matching score·correspondence·attention 확률·R 입력과출력·target context는
유지한다. 구체적 실행과 판정은
[전달 경로 진단 프로토콜](2026-10-23-SS288-direct-feature-delivery-diagnostic-plan-ko.md)을 따른다.
현재는 원인 가설과 검증 설계이며 실제 제어 결과가 나온 것은 아니다.

이 제어가 좋아져도 현재 학습된 LoRA/R의 조건부 기여다. R-only 재학습 모델과
동일한 조건도 아니고 semantic matching이나 attention 전체를 끈 ablation도 아니다.
모듈 설계 채택에는 matched 재학습·최종128×3·독립 확인과 cascade 평가가 필요하다.
기존 세 historical arm, native PBR 후속, DEV/독립 seed/최종508개10 metric은 유지한다.

Canonical 결과는 `PBR-native-science-runtime-v1/`의
`historical-SS-matched-epochs-v1/direct-nodq/step0000288`,
`historical-SS-matched-epochs-v1/two-arm-step288-comparison-v1`,
`historical-SS-matched-RNG-through288-v3`에 있다. NoDQ288의 실제 NN/source/Adam과
평가 prefix8개 파일2,877,493,228bytes, RNG 관련774개15,998,579bytes와
비교3개137,559bytes를 각각 실제 file4 SHA256·크기로 검증했다.
