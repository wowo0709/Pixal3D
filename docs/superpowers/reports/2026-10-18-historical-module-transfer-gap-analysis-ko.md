# 소규모 양의 모듈 결과와 현재 DeformNeXt SS 실험의 차이

## 결론과 근거 범위

현재의 작은 증분 효과를 곧바로 “이전에 유효했던 모듈이 큰 데이터에서 무효가 됐다”로 해석할 수 없다. 코드와 실험 기록을 대조하면, 양의 결과가 났던 feature 전달 경로·ablation factor·geometry 표현·학습 loss·평가 영역이 현재와 다르다. 일부 저수준 구현은 정확히 재사용했지만 **이전 양의 실험 조건 전체를 재현한 것은 아니다.** 이전 조건까지 일치했다고 판단한 부분을 정정한다.

이번 검토는 이전 `reports_final2`의 수치 authority와 실제 실행 코드를 읽고 현재 코드·resolved config·학습된 H3 state·실제 생성 결과와 대조한 것이다. 과거의 모든 raw endpoint를 새로 실행하거나 재집계한 감사는 아니다. 현재 직접 확인한 사실과 원인 가설을 구분한다. 진행 중인 C2의 최종 효과도 아직 판정하지 않는다.

## 1. 큰 양의 GT matching 결과와 현재 주입 경로가 다르다

이전 [GT-CORR WHAT0/CONTENT1 결과](/root/dev/Pixal3D/.worktrees/anchor-hybrid-prestudy/docs/superpowers/reports_final2/2026-09-18-ss-gt-corr-proj-what0-content1-results-ko.md)의 DEV8에서는 다음 결과가 있었다.

- GT_CORR_RAW의 source-exclusive recall32: 0.8113.
- 같은 carrier의 SOURCE_OFF: 0.7043, WRONG_CORR_RAW: 0.6026.
- RAW−OFF: **+10.71%p**, CI [+6.15,+15.34]%p; RAW−WRONG: **+20.88%p**, CI [+13.87,+28.12]%p.

이는 유효한 양의 결과다. 구현은 source 카메라 세 개마다 target-state front surface의 32×32 write slot을 정하고, GT CAGE read address의 실제 old-state DINO value를 각각의 map에 기록한 뒤 **각 source 카메라의 native ProjGrid**를 통과시켰다. Global/negative branch를 고정하고 source-on 조건끼리 write support와 source value multiset도 통제했다. 코드 authority는 `anchor-hybrid-prestudy/pixal3d/experiments/ss_gt_corr_proj_what.py:build_condition`이다. GT route와 oracle camera/scale을 사용하는 mechanism 진단이며 deployable matching 결과는 아니다.

현재 [MatchingV2Conditioner](../../../pixal3d/modules/matching_v2_conditioner.py)는 source 후보를 **target anchor의 Q=1024 patch별 evidence**로 집계한다. [project_native_patchmap](../../../pixal3d/modules/transport_projection.py)은 명시적으로 `camera_angle_x[:,0]`, `distance[:,0]`, `projection[:,0]`을 사용한다. [현재 주입부](../../../pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py)는 native MV condition에 이 anchor-camera projection의 residual을 더한다. 세 source 카메라의 target-state write map을 각각 재구성하지 않는다. `target_valid`도 edited target RGB의 alpha에서 얻는다.

따라서 bilinear와 support normalization을 맞춘 것은 연산 정합의 개선이지만, **세 source 시점의 공간적 주소 교정을 재현한 것은 아니다.** Target에서 가려졌지만 source에 보이는 표면을 각 source-camera map에서 직접 주소화하던 이전 메커니즘이 현재 anchor-plane 경로에서는 보존되지 않는다. Anchor 경로가 hidden geometry에 간접 영향을 줄 가능성을 배제하는 결론은 아니다.

이전의 다른 projected-voxel splice 실험은 GT WHERE에도 실패해 `PROJECTION_SPLICE_INTERFACE_REJECTED`로 종료됐다. 같은 “GT matching” 명칭으로 그 실패와 native per-view slot transport의 성공을 합치면 안 된다. 현재의 GT 진단 실패만으로 이전 성공 경로의 headroom까지 부정할 수도 없다.

## 2. 현재 Top1 대 weighted feature는 이전 Q2C ablation과 다르다

이전 [R9 Exp4](/root/dev/Pixal3D/.worktrees/anchor-hybrid-prestudy/docs/superpowers/reports_final2/2026-09-16-exg-app-cue-stage-b-r9-four-module-ablation-results-ko.md)는 같은 Top64·descriptor·RGB carrier에서 **D0-PAIRWISE 대 D1-Q2C**만 바꿨다. Controlled APP-CUE 63-pair OOF task RGB error가 0.03676912→0.03512899로 낮아졌고 gain 0.00164013의 CI는 [0.00092160,0.00240041]이었다. 세 model seed와 네 fold에서 같은 방향이었다.

현재 [MatchingV2Head](../../../experiments/deformnext_retargeting/matching_v2_head.py)에도 Q2C context block과 scalar scorer가 실제 존재한다. 원래 `_ContextBlock`·`_ScalarHead`와 현재 복사본의 AST hash가 일치했다. 학습된 H3의 attention gate는 −0.0593004, FFN gate는 −0.1850186으로 0에 고정된 모듈도 아니다.

그러나 **Top1/Top1-R/attention-R/adapter-R/query-R 모두 같은 H3 head SHA256과 같은 cache를 사용한다.** Top1의 선택에도 이미 Q2C의 score가 들어간다. 현재 b3−b2 대비는 Q2C ON−OFF가 아니라 같은 contextual score에서 **soft weighted value−Top1 value**다. 공통128×3에서 +0.004%p, CI [−0.021,+0.025]%p인 결과는 weighted value의 추가 이득 미확인으로 기록해야 한다. Q2C 자체가 효과 없다는 판정으로 사용할 수 없다. 사용자가 요청한 Top1 정의는 그대로 유지하되, 이전 Q2C 증명의 재현 여부와 구분한다.

또한 이전 R9는 correspondence loss 없이 task RGB error를 최적화했다. D1의 task error는 좋아졌지만 R@1은 0.23706→0.22731로 낮아졌다. 현재 H3는 ranking CE·selected-match correctness confidence·DINO value cosine loss로 학습했다. **정확한 patch 일치와 유용한 evidence 읽기가 같지 않다는 이전 결과를 현재 objective/선택 기준에 충분히 연결하지 못했다.** 이 차이의 SS 인과 효과는 아직 분리 측정하지 않았다.

## 3. 고정 rejection은 실제 추가 경로를 작게 만든다

현재 R arm은 threshold 0.276316212에서 `p_use * accepted`를 raw transport와 adapter 전체에 적용한다. Threshold는 train-internal calibration의 exact-match F0.5로 선택됐다. 현재 학습·primary 평가 중 이 값을 변경하지 않았다.

실제 validation508의 foreground query38,366개 중4,348개(11.33%)만 통과했고, foreground 기준 평균 usage weight는0.04277이었다. Common32/128 통과율은10.59%/9.44%다. Confidence label은 선택된 Top1의 exact correctness지만 attention consumer는 soft evidence를 읽는다. 유용한 근접/분산 evidence까지 차단할 가능성이 있다.

실제 epoch3의 대응 rank/step192개 pack에서 update RMS/native RMS 중앙값은 Top1 0.735%, Top1-R 0.090%, attention-R 0.083%, adapter-R 0.928%, P2 adapter No-R 9.875%였다. Native condition RMS와 valid-query coverage는 대응 pack에서 같았다. Gate 크기는 유사했으므로 단순 gate=0 현상으로 설명되지 않는다.

이는 R에 의한 정보 제한과 양립하는 근거다. P2의 중단/재개 RNG 및 학습된 weight 차이가 있으므로 P2 대 C1-R 차이를 rejection만의 인과 효과로 확정하지 않는다. Source-off 등 동일 checkpoint 진단을 함께 판단한다.

## 4. 이전 geometry-content 모듈과 현재 RAW query는 같은 입력이 아니다

이전 [G2 결과](/root/dev/Pixal3D/.worktrees/anchor-hybrid-prestudy/docs/superpowers/reports_final2/2026-09-18-ss-geometry-conditioned-voxel-retarget-g2-results-ko.md)의 CONTENT 입력은 **16³ 위치별 predicted original-MV/desired-1V logit·soft occupancy·SDF와 차이, 총9채널**이었다. Target/mismatch projection·차이·xyz를 함께 사용했다. Geometry는 residual content generator에 넣었다.

현재 C1은 source evidence1024와 target query1024를 읽는 **2D patch 단위 MLP**이고, C2는 여기에 Omega FRAME+GLOBAL feature delta2048을 추가한다. C1 입력2048과 C2 입력4096이 물리적으로 분리되고 query gradient도 연결된 것은 확인했다. 그러나 이는 이전9채널 spatial occupancy/SDF geometry의 재현은 아니다. 채널이 많다는 사실이 같은 geometry 정보를 의미하지 않는다.

학습도 이전에는 frozen SS flow의 full rollout→decoder logits에 SS64/SS32 source-exclusive·stale·visible regional BCE를 걸었고, 현재는 denoiser LoRA와 conditioner를 함께 **단일 timestep flow velocity MSE**로 학습한다. 현재 frozen head/cache에는 SS loss가 역전파되지 않는다. 전체 MSE가 낮아지도록 native 조건과 LoRA가 적응하는 것과, 특정 hidden geometry를 source에서 회수하도록 adapter를 학습하는 것은 다른 목표다. 이 차이는 확인된 설정 차이이며 손실 기여의 인과 분해까지 검증한 결과는 아니다.

## 5. 이전 retargeter의 성공 범위도 나눠야 한다

이전에 모든 모듈의 source-retargeting generalization이 입증됐던 것은 아니다.

- G2 CONTENT−R2의 DEV recall32 **+4.075%p**는 유효한 geometry-content 상대 개선이었다. 하지만 동일 CONTENT의 projected-source-off가 정상보다 **+3.701%p** 높아, 보고서 자체는 source generalization FAIL로 판정했다.
- GT transport 위 CONTENT−RAW는 TRAIN +4.08%p였지만 DEV primary recall32는 **+0.07%p**, CI [−1.17,+1.57]%p였다. F1-32에는 +1.16%p 이득이 있었으므로 모든 metric이 무효였던 것도 아니다.
- 따라서 강한 기존 근거는 native per-view GT routing과 controlled Q2C task gain이다. Geometry cleanup의 양의 상대 효과와 source retargeting의 성공은 서로 다른 주장이다.

이 구분은 이전 양의 결과를 무시하기 위한 것이 아니라, 그대로 재현해야 할 성공 경로와 이미 미확인으로 남았던 주장을 분리하기 위한 것이다.

## 6. 데이터·평가 차이와 구현 정합의 한계

이전은 controlled CAGE TRAIN32/DEV8 또는 GT-surface APP-CUE63 pair이며 oracle camera/normalization도 사용했다. 현재는 DeformNeXt train9149/val508이고 동일128·93 source cluster·noise3개에서 decoded target SS occupancy의 global IoU/F1을 본다. 이전의 source-exclusive regional recall32 개선과 현재 global IoU64를 같은 효과량처럼 비교할 수 없다. 데이터 다양성·baseline의 LoRA 적응도도 달라졌으나, 이것만을 원인으로 확정하지 않는다.

현재 native 초기 global/proj 차이0, 초기32개 생성의 TP/FP/FN·seed·GT hash 일치, final checkpoint와 optimizer·양rank480-step 유한 loss 감사는 PASS다. 이는 초기 정합 붕괴·loader 실패 가설을 약하게 만들지만, **이전의 유효한 source-use 메커니즘까지 재현했다는 증거는 아니다.**

## 7. 다음 판별 순서

1. 현재 같은 checkpoint의 normal/source-off/adapter-off/support-preserved wrong-object를 먼저 완료한다. 이 중 source-off는 추가 경로 전체를 끄므로 source value만의 효과로 해석하지 않는다.
2. 이전의 양의 per-view GT slot transport를 현재의 같은32개·world-v4 camera/frame·같은 frozen checkpoint에서 재현하는 최소 oracle을 준비한다. 세 source map의 correct/wrong read address와 support/value multiset을 통제하고 native ProjGrid를 유지한다. Global 입력은 네 RGB로 고정한다. GT-only 진단을 deployable 결과로 부르지 않는다.
3. 그 경로가 유효하면 같은 후보·loss에서 D0 대 D1을 분리하는 작은 head 대조와, 실제 SS endpoint의 evidence 소비를 확인한다. 현재 Top1 대 soft 비교는 별도의 질문으로 남긴다.
4. Adapter/query를 검증할 때는 이전 geometry 입력·regional supervision의 재현과 현재 RAW-context 설계를 구분한다. 작은 oracle·matched control에서 근거를 얻은 뒤 후속 장기 학습을 정한다.

새 training arm을 이 보고서만으로 추가하거나 실행 중인 C2를 취소하지 않았다. 기존14개 구성의 결과는 보존하고, 그 결과로 주장할 수 있는 범위를 수정한다. 아래 동일-checkpoint 진단과 필요한 후속의 결과를 본 뒤 Full 채택/PBR 확대를 결정한다.

## 현재 경로 진단

동일32개·32 source cluster·한 noise·sampler12/CFG3·batch8, 각 모델 내 같은 물리GPU의 저장된480-step weight를 비교한다. Source-off/adapter-off/wrong-object의 recipient RGB 네 장·camera·GT·sampling seed는 유지한다. 이는 독립 재학습 ablation이 아니다.

| 정상 대비 control로 인한 IoU 하락, %p | P2-NoR | 새 C1-R |
| --- | ---: | ---: |
| 추가 경로 전체 OFF | 1.660 [0.351,2.915] | 0.136 [−0.102,0.387] |
| Adapter 출력 OFF | 1.797 [0.556,2.976] | 0.089 [−0.139,0.331] |
| 다른 물체의 source evidence | 0.883 [0.176,1.551] | 0.093 [0.003,0.189] |

정상 IoU는 P2 0.453171, C1-R 0.436316이다. 모든 control에서32개 target voxel hash가 일치했고 actual RGB contact sheet의 SHA256도 모델 내 네 조건에서 같았다. Normal은 각 모델의 epoch5 quick32를 샘플별 count/hash까지 정확히 재현했다. Condition checkpoint와 model checkpoint도 각 모델 내에서 같다. Paired bootstrap4000회·seed42017이며 queue가 만든 비교를 독립 재계산해 일치시켰다.

No-R의 추가 경로와 adapter에는 유용성이 있고, cross-object source evidence 교체에도 반응한다. 따라서 추가 경로가 전부 무효라는 설명은 맞지 않는다. Wrong-object는 외형/물체 정체성도 바꾸므로 정확한 correspondence 주소만의 인과 효과까지 증명하지 않는다. C1-R의 경로 민감도는 작지만 별도 학습된 두 모델의 차이를 R 하나의 인과 효과로 설명하지 않는다. 근거와 code/config/hash snapshot은 file4 `rgb_matching_v2/provenance/historical-module-transfer-gap-evidence.json`에 보존했다.
