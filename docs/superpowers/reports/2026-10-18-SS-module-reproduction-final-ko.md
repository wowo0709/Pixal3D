# SS 모듈 실험·재현 최종 결과

SS의 전체14개 비교, 기존 모듈 재현, 현재 데이터 이식, matched 추가 학습과 최종 평가를 완료했다. 현재 SS에 채택할 구성은 **기존 R9 topology head + retargeting adapter인 C1**이다. Deformation query와 새 confidence/rejection 조합은 채택하지 않는다. C1이 모든 다른 후보보다 유일하게 우수하다는 주장이나 PBR 성능 주장은 하지 않는다.

동일 실행 조건의 추가 실험에서 C1은 vanilla V1보다 voxel IoU **+0.954pp**, adapter를 제외한 B3보다 **+0.769pp** 개선됐다. 두 차이는 세 sampling noise에서 모두 양수였다. 성공한 부분은 원래 C1 경로와 retargeter의 추가 효과다. 정확한 correspondence 주소 자체의 SS 이득, Q2C context만의 독립 SS 이득, deformation-query 이득은 이 결과로 입증하지 않는다.

## 같은 조건의 최종 SS 비교

Train은 DeformNeXt world-v4 9,149개, validation은508개다. 실제 fine-tuning 초기 모델은 사용자가 제공한 mixed385 multi-view SS64 **step4210 complete base+LoRA**이며, 모델 구조는 dev-jaehyeok 계열 LoRA runtime을 유지했다. Clean80k는 초기 multi-view 모델과 frozen Omega의 학습 이력이고 이번 deformation 학습에 다시 넣지 않았다. Omega는 epoch7을 고정했다.

새 V1/B3/C1은 각각 fresh optimizer로 seed101,480 updates(약5epochs),global96,DDP2,micro48/rank,accumulation1로 학습했다. Final은 checkpoint480/raw weight,같은128개·93 source cluster·서로 다른3 sampling noise,12steps/CFG3/generation batch8이다. TP/FP/FN 합산 micro IoU/F1이며 동일 source의 변형·noise를 독립 표본으로 늘리지 않았다.

| 방법 | 입력/추가 경로 | Voxel IoU | Voxel F1 |
| --- | --- | ---: | ---: |
| Zero-shot source-only | M0 RGB camera08/09/10/11 | 0.380711 | 0.551471 |
| Zero-shot edited | M1 RGB camera08 + M0 RGB09/10/11 | 0.419275 | 0.590830 |
| V1 seed101 | Edited 입력의 vanilla LoRA fine-tuning | 0.432650 | 0.603986 |
| B3 seed101 | 같은 R9 semantic score/weighted feature/fusion | 0.434499 | 0.605785 |
| C1 seed101 | 같은 B3 경로 + retargeting adapter | **0.442193** | **0.613223** |

Source-only도 같은 M1 desired-state GT로 평가했다. Source-only의 네 번째 장은 deformation 전 camera08이고, edited 입력의 첫 장은 depth가 아닌 RGB다. Native spatial correspondence 기준은 V1이다. No-score는 matched Top1 하나를 사용하는 대조이며 uniform 평균으로 바꾸지 않았다.

| 같은 최종 checkpoint 대비 | IoU 차이,pp | 95% source-cluster CI,pp |
| --- | ---: | --- |
| B3−V1 | +0.185 | [+0.010,+0.362] |
| C1−B3 | **+0.769** | **[+0.386,+1.144]** |
| C1−V1 | **+0.954** | **[+0.542,+1.350]** |

이 CI는 현재 학습 weight/seed에 조건부다. 구조가 다른 세 run의 학습 random draw 전체가 bit-exact라는 주장은 하지 않는다. 이전 seed0 C1−V1의 +1.029pp와 C1−B3의 +0.686pp도 방향은 같았지만, 당시 world/micro/accumulation 조건이 달라 새 seed101과 하나의 training-seed 모집단 CI로 합치지 않는다. 개발128을 선택에 사용했으므로 untouched-test 결과도 아니다.

## 학습 경향과 초기 성능

아래는 final128과 다른 **고정32개·한 noise**의 epoch 경향이다. 같은 epoch끼리 비교했고 이 표로 best epoch를 다시 고르지 않았다.

| Epoch/step | V1 IoU | B3 IoU | C1 IoU |
| --- | ---: | ---: | ---: |
| 0/0 | 0.426715 | 0.426715 | 0.426715 |
| 1/96 | 0.432148 | 0.434742 | 0.436335 |
| 2/192 | 0.434444 | 0.433059 | 0.440602 |
| 3/288 | 0.431872 | 0.434638 | 0.443624 |
| 4/384 | 0.434300 | 0.436509 | 0.450249 |
| 5/480 | 0.434837 | 0.438354 | 0.453849 |

새 세 arm의 초기 native global/projection 오차와 생성 count 차이는0이었다. 실제 final weight에서도 각각701개 frozen base tensor가 SS4210과 dtype/shape/value까지 정확히 같고360개 LoRA tensor는 모두 갱신됐다. Final denoiser/conditioner 전체 tensor와 optimizer moment도 finite였다. B3/C1의 frozen Q2C24개 tensor가 동일하며 C1은 실제 QA에서 검증한 같은 head/weight였다. 따라서 현재 결과의 원인을 초기 base 파손·미학습·gradient 단절로 설명할 근거는 없다.

## 전체14개 본 비교도 보존했다

아래는 이전 primary seed0 실험의 동일 final128×3 결과다. 새 seed101 표와 합쳐 순위를 추정하지 않는다. P1/P2는 같은 run의 중단 상태에서 재개했으며 rank별 RNG가 보존되지 않아 uninterrupted trajectory 재현이라고 부르지 않는다.

| Primary 구성 | Micro IoU |
| --- | ---: |
| Zero source-only / edited | 0.380711 / 0.419275 |
| V1 | 0.431926 |
| 기존 B1 / B2 / B3 | 0.433912 / 0.435202 / 0.435352 |
| 기존 R9 C1 | 0.442214 |
| P1 native-grid / P2 H3-grid No-R | 0.441459 / 0.440920 |
| H3 Top1 | 0.433644 |
| H3 Top1-R | 0.432806 |
| H3 attention-R | 0.432847 |
| H3 retargeter-R | 0.433234 |
| H3 RAW-query-R | 0.433664 |

다섯 새 arm은 모두480 updates를 학습했고 final128×3를 완료했다. Retargeter와 query의2×2를 추가하지 않았다. RAW query가 없는 C1은 해당 입력 채널 자체가 없으며 C2만2048개 extra channel을 더했다. P1/P2의 근소한 차이로 H3 head의 우열이나 rejection 하나의 인과 효과를 확정하지 않는다. C1 선택은 유일한 최고점 판정이 아니라 matched 추가 학습에서 재현된 작업 recipe의 선택이다.

## 왜 이전 양의 모듈 결과가 그대로 나오지 않았나

**재현 조건이 달랐던 부분은 실제로 수정했다.** 이전 per-source-camera GT transport와 현재 target-anchor residual은 다른 소비 경로였다. 원래 geometry-content query는 predicted O/D logit·occupancy·SDF 등의9채널인데 기존 C2는 RAW Omega delta2048이었다. 과거 frozen full-rollout regional BCE와 현재 joint LoRA/conditioner의 single-timestep velocity MSE도 다른 목표다. Scalar gate1개로 원본371-parameter gate MLP를 단순화했던 차이도 복구했다. 상세한 코드·authority·통제 결과는 [원인 분석](2026-10-18-module-reproduction-cause-analysis-ko.md)에 있다.

**성공 범위를 나눠 재현했다.** 과거 Q2C는24개 학습·28,800 update와 final parameter·OOF RGB component가 과거와 exact였다. 현재 데이터의 controlled RGB-cue task도 D0→D1에서 error가6.01% 감소했다. 과거 G2는6개 저장 checkpoint의144 endpoint와 nested BCE가 원본과 exact였다. 이 G2 결과는 fresh128-update 전체 trajectory의 재현이 아니다. 원본에서도 source projection OFF가 더 좋았다는 한계까지 재현했다.

**현재 C1의 head는 RGB-cue weight가 아니다.** 실제 `4f8771f2…`는 DeformNeXt topology loss의7epoch/62,545-update R9 head다. 새 H0~H3는 다른 train/cal split에서 각각58,653 updates를 학습했다. Controlled RGB-cue task의 성공을 현재 C1의 원인으로 직접 연결하지 않는다. 현재 확인된 성공은 같은 topology head를 쓰는 B3에서 retargeter를 추가한 전체 경로의 이득이다.

**Matching이 여전히 남은 병목이다.** H3의 visible-query candidate recall은93.16%지만 exact Top1은23.71%다. Confidence의 고정 operating point는 trustworthy foreground38,044개에서 precision33.16%/coverage11.41%였다. 후보 확대나 threshold 변경만으로 유용한 source 전달이 보장되지 않는다. Q2C가 있는 Top1과 soft를 비교한 결과로 Q2C ON/OFF를 판정했던 해석도 구분했다. 자세한 분모와 곡선은 [confidence 진단](2026-10-18-matcher-confidence-coverage-ko.md)을 따른다.

**정확한 주소의 SS 효과는 아직 미확인이다.** 원본 C1의 extra-path OFF·adapter OFF·wrong-object에서는 유용한 차이가 있었지만 same-support wrong-address 대비의 CI는0을 포함했다. Native source를 보존한 GT bridge의 +1.960pp는 headroom 진단이지 learned WHERE 성능이 아니다. 최종 C1 이득을 precise correspondence의 증명으로 확대하지 않는다.

## 원본 geometry query를 복구한 현재6-run 결과

원본371-parameter gate와9채널 O/D query,physical No-query3075/query3107 channel,같은 full-rollout regional loss로 두 arm×3seed를128 updates 학습했다. Train32/cal32,3sampling noise의 고정 비교다. 이 표는 **actual SS-input GT의 macro metric**이며 위 decoded-GT micro 표와 다르다.

| 방법 | Macro IoU | Source recall32 | Stale leak32 ↓ |
| --- | ---: | ---: | ---: |
| Frozen native | 0.445530 | 0.825740 | 0.601524 |
| Adapter-only | 0.447883 | 0.816588 | 0.549600 |
| Adapter+query | 0.438332 | 0.814673 | 0.554092 |

Query−adapter는−0.955pp,CI[−1.400,−0.528]pp였고 세 model seed 모두 음수였다. Intact−query shuffle의 추가 이득도 확인되지 않았다. 마지막 학습 pass에서 두 seed의 query loss가 No-query보다 나빠 단순 epoch 부족이나 validation 과적합만으로 설명할 수 없다. 초기 native/gradient와 실제 weight 구조는 정상이다. Consumer/domain/precision/optimization 중 어느 하나의 단독 인과 효과까지는 분리하지 않았다. 이 상태의 query를 Full에 추가하지 않는다.

## Sampling과 cache 검증

계획한4개 기존 checkpoint×12/50steps의8개 진단을 완료했다.50−12 IoU는 네 방법 모두−1.1~−1.4pp였다. 낮은 생성 성능을 단순히12steps 부족으로 설명할 근거는 얻지 못했고 primary12/CFG3를 유지했다. 이것은 새 seed101 세 모델 모두의50-step 실험은 아니다. [Sampler 결과](2026-10-18-ss-sampler-sensitivity-ko.md)를 따른다.

현재 trained C1 두 실제 사례의 BF16 cached/online·training/inference condition과 CFG p0/p1/mixed mask를 검증했다. Active projection RMS0.023112로 추가 경로도 실제 활성이다. v1의 NaN은 이미 통과한 assert_close 뒤의 같은 infinity 차감 통계 문제였고 verifier만 수정한v2가 finite 오차·동일mask를 기록했다. 모델·학습 수식은 바꾸지 않았다. 관련 CPU39개 contract 검사와12개 QA 회귀 검사, 실제 GPU proof와 remote/file4 receipt를 보존했다.

## 시간·메모리와 실행 종료

장치는 RTX PRO6000 Blackwell Server Edition이다. 이번 campaign은 물리 GPU0~3만 사용했다. 새 본 학습은 global96/DDP2/micro48/accumulation1이었다. 아래 wall time은 native/memory preflight·초기화·epoch validation·저장을 포함하고 후속 final128 생성은 제외한다.

| Matched 모델 | Median compute/update | Rank당 peak allocated | 학습 job wall time |
| --- | ---: | ---: | ---: |
| V1 | 약16.67초 | 45.62GiB | 2시간25분32초 |
| B3 | 약17.26초 | 47.09GiB | 2시간30분02초 |
| C1 | 약17.28초 | 48.00GiB | 2시간30분42초 |

새 H3 ladder의 다섯 학습도17.21~17.26초/update,46.84~48.13GiB/rank였다. 원본-network local controller는 약19분의128-update train loop,평가까지 No-query 약30분/query 약33분,train peak 약6.73GiB였다. SS 모델과 controller의 update 단위는 다르다. Final B3의 세 noise는 GPU0/1/2에서 병렬 실행했다. 모델 학습·평가는 모두 정상 종료했고 네 허용 GPU의 해당 작업 메모리는 해제됐다.

## 채택 구성·PBR 판단과 재현 artifact

SS의 작업 구성은 **R9 topology Top64/T0.005 + C1 retargeter + 기존 base/LoRA**,deformation-query channel 없음,rejection 없음이다. 단순 uniform/no-score나 GT correspondence로 바꿔 얻은 결과가 아니다. 새 H3/rejection/RAW query와 original-network geometry query를 자동 채택하지 않는다.

**PBR 전체 ablation 확대는 보류한다.** 이번 SS-focused 계획의 마지막 항목인 확대 여부 판단이다. Retargeter의 SS 이득은 확인했으므로 C1은 다음 PBR 단계의 제한된 후보이지만, 새 matching/rejection/query ladder의 추가 효과와 precise source 주소의 utility는 확인되지 않았다. 이 상태에서 모든 arm을 PBR로 늘릴 근거가 없다. PBR1024의 stage별 native/feature/cache 계약과 rendering 평가를 확인하기 전에는 PBR 향상을 주장하지 않는다. 이번 보고서는 논문의 rendering10개 metric을 수행한 결과가 아니다.

실제 file4 기준 C1 파일은 다음과 같다. Denoiser에는 frozen base와 LoRA가 모두 들어 있고 conditioner는 별도 파일이다. 모델 구조·LoRA/config·head·Omega/cache도 함께 사용해야 한다.

```text
/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/ss-seed-confirmation-v1/configs/c1.json
/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/ss-seed-confirmation-v1/runs_ss64/c1/ckpts/denoiser_step0000480.pt
/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/ss-seed-confirmation-v1/runs_ss64/c1/ckpts/deform_conditioner_step0000480.pt
```

고정128 전체의 같은 사례 비교는 `ss-seed-confirmation-v1/presentation-v2/all128-fixed-order.pdf`의16페이지와 `first8-fixed-order.png`로 정리했다. PNG는 성능으로 고르지 않은 원래 순서의 첫8개다. 그림은2D max projection이고 표기 IoU는 실제3D count이므로 silhouette만으로 전체3D 정합을 판단하지 않는다. V2는 확인되지 않은physical axis 이름을빼고max-projection으로표기했으며입력·모델·metric은같다. Legacy zero-shot JSON의3D hash 누락도 presentation manifest에 그대로 표시했다.

최종 선택·동일 입력의 재평가 recipe·정량 summary는 `final-reproduction-v1/`에 있다. 원본14개와 matched3개,기존/현재 Q2C·geometry의 결과·범위를 구분해 보존한다. 317개 queue는250 completed/59 HELD/8 historical FAILED이며,59개의 required replacement와8개 실패의 완료된 후속을 실제 SHA/결과로 확인했다. 원래 상태를 성공으로 바꾸지 않았고 unresolved failure는0이다. W&B의 실제 metric/image/table/artifact readback,raw checkpoint의 actual-file4 SHA/크기와 source archive+patch 재구성 근거는 provenance receipt를 따른다. Source는 요청한 `wowo0709/feature-retargeting` 브랜치에 보존한다.
