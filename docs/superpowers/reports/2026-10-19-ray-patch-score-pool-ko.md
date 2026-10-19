# Ray 내부 token score 조합의 calibration 및 SS 진단

2026-10-19 UTC. Top-K에 있는 여러 correspondence를 조합하는 후보를 추가로 점검했다. 현재 frozen patch-score 평균 방식에서는 SS 개선을 확인하지 못했으며 새 학습 arm으로 승격하지 않는다. Semantic Top-K의 가능성 전체를 부정하는 결과는 아니다.

## 입력과 변경 범위

Current world-v4 train9149의 학습 내부 calibration32개를 사용했다. 개발128개와 asset ID 및 source cluster가 겹치지 않는다. Matching head 또는 SS가 이 train 데이터를 본 사실을 배제하지 않으므로 독립 test로 부르지 않는다.32개 모두 기존 topology label이 있고 현재 mesh·RGB·camera fingerprint와 일치했다.

기존 방식은 ray의 Omega feature를 먼저 평균내어 frozen D1/scalar score를 계산한다. Prototype은 각 ray가 포함하는 **서로 다른 target patch**마다 같은64개 source 후보에 D1/scalar score를 계산하고, softmax 분포들을 평균내어 ray의 최종 단일 분포·단일 evidence vector를 만든다. 같은 patch를 참조하는 voxel의 반복으로 가중치가 중복되지 않는다. 새 파라미터나 학습 update는 없다. Query/proposal/readout에 GT depth·mesh·correspondence·SS를 넣지 않는다.

두 candidate bank(pooled Omega64와 target-patch proposal union64)에 이 readout을 적용했다. Temperature0.05, source 후보64개를 고정하고 GT는 CPU 평가에만 사용했다.

## Calibration32 결과

Pooled Omega64 bank에서 관측 가능한 ray/target-patch 관계의 Top8 exact coverage는29.60%→34.84%, near1 coverage는58.75%→66.13%였다. 하지만 ray의 GT 대응 주소 집합에 주는 평균 probability mass는 exact17.72%→16.89%, near1 43.94%→41.13%로 감소했다. Exact mass 차이−0.830%p의 source-cluster95% CI는[−1.545,−0.131]%p, near1 차이−2.808%p의 CI는[−4.191,−1.479]%p였다.

유효 후보 수는4.72→6.53으로 늘었고 최대 probability는56.81%→39.12%로 줄었다. 분포가 넓어진 사실만으로 조합의 유용성이 개선됐다고 볼 수 없다. 반대로 이 matching proxy의 감소만으로 SS 성능 감소를 확정하지도 않는다.

Source alpha-valid patch가64개 미만인 calibration asset2개는 원래 ray 모델과 동일하게 matching branch를 비활성화했다.32개 panel에서 제거하지 않고 zero active-ray 상태로 포함했다. 첫 prototype이 이 disable 분기를 빠뜨려14번째 asset에서 strict alpha 검사가 실패한 기록도 보존했다. Source-alpha 검사를 완화하거나 GT 가시성으로 조건을 바꾸지 않았다. 해당 corner case를 포함한 CPU 검사4개가 통과했다.

## 실제 저장 모델의 identity 검증

Full raw480의 실제 두 사례 RGB-cachedDINO/Omega/camera 입력에서 원래 attention probabilities를 사용해 evidence·R·ray delta를 재구성했다. Native projection과 candidate IDs·logits·weights·view weights·R residual·evidence·query support·fusion scale이 모두 exact 일치했다. Context를 해제한 뒤 원래 출력도 exact 복원됐다. GPU0 peak allocated0.840GiB였으며 SS flow/decoder나 GT를 읽지 않았다. 이2개 사례의 identity 증명을 online DINO parity나32개 quality 결과의 재현 증명으로 확대하지 않는다.

## 같은 Full480 가중치에서 SS 생성 비교

동일한 개발32개·noise0·sampler12/CFG3·generation batch8에서 실제 SS 생성을 수행했다. 원래 learned readout, scalar-prior만 사용하는 readout, patch별 score 분포를 합치는 readout을 비교했다. Source64 후보와 query support, checkpoint·camera·RGB·GT·noise를 유지했다. R 파라미터는 유지하되 변경된 evidence에 대해 같은 R을 다시 계산했다. 이는 이미 Full로 학습된 모델의 조건부 inference 진단이며 처음부터 재학습한 ablation이 아니다.

| Readout | SS IoU | F1 |
| --- | ---: | ---: |
| 원래 learned attention | **0.436073** | 0.607313 |
| Scalar score 가중합 | 0.435018 | 0.606289 |
| Patch별 score 분포 평균 | 0.433923 | 0.605225 |

| 비교 | IoU 차이 | Source-cluster95% CI |
| --- | ---: | --- |
| Learned−score 가중합 | +0.1056%p | [−0.1645,+0.4475]%p |
| Patch score 평균−score 가중합 | −0.1094%p | [−0.2754,+0.0520]%p |
| Patch score 평균−learned | −0.2150%p | [−0.5951,+0.1087]%p |

모든 차이의 CI가0을 포함한다. Learned attention의 topology GT mass가 prior보다 낮았어도 SS 수치는 소폭 높았다. 따라서 matching 정확도·probability mass·entropy 중 하나만으로 최종 feature 유용성을 판단하지 않는다. 같은 source64에서 후보를 더 넓게 사용하는 고정 평균 방식은 현재 SS 개선을 입증하지 못했다.

두 새 SS 평가는 GPU0·2에서 각각189.8초·192.2초, peak allocated6.039GiB였다. 각 평가에서 실제 readout hook4회/32assets와 unchanged candidate addresses/query support를 확인했다. 세 결과의 GT geometry hash·source cluster·noise는 모두 같았고 실제 RGB contact-sheet PNG도 byte exact였다. 생성 noise1개·학습 seed1개의 개발 진단이다. 추가 학습 seed,128×3 최종 평가 또는 독립 test를 대신하지 않는다.

## 실행 및 후속 판단

기존 여섯 arm의 numerical source55개·학습 checkpoint·global96/DDP2/micro48/acc1·raw480 예산은 유지했다. Native-spatial과 Top1의 원래 학습 PID도 바꾸지 않았다. 새 진단은 process-local context로만 실행하고 원래 함수로 복원했다.

첫 SS launcher는 native lazy import 도중 namespace package의 확장 경로가 재계산되어 새 helper import에 실패했다. 모델 생성 전의 실패였으며 원본 로그를 보존했다. 명시적인 package path list를 사용하는 새 namespace에서 실행했고 두 SS 평가가 정상 종료했다. Identity 두 helper의 source는 바꾸지 않았다. Identity driver의 첫 metadata type 오류도 보존하고 원래 산술·검사 조건을 유지하여 전체 증명을 다시 실행했다.

현재 고정 score 평균 prototype을 긴 학습으로 승격하지 않는다. 진행 중인 여섯 arm의 최종 비교를 완료하면서, correspondence의 보완 정보를 잃는 지점과 기존 양성 R의 query/carrier를 함께 검토한다. 한 가지 후속 가설은 target-alpha로 제한한 query를 source-camera ray로 다시 투영할 때 consumer 영역이 어떻게 달라지는가이다. 코드는 source-ray delta를 native ProjGrid로 투영하지만 target-alpha consumer mask를 다시 적용하지 않는다. 이 사실만으로 실패 원인이나 bug를 확정하지 않으며, projection energy와 입력-only counterfactual로 별도 검증해야 한다.

## 실제 산출물

File4 root: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/diagnostics/ray-patch-score-pool-v1/`

- `calibration32-native-support-v1/audit.json`:32개 calibration의 네 조합과 GT fingerprint, source-cluster mass 비교.
- `identity-two-cases-v1/identity.json`:실제2개 사례의 exact reconstruction과 context restoration.
- `quality-pilot-v2/SS32-{prior-only,patch-score-pool}/`:실제 evaluation JSON·GT/Pred panel·입력 PNG·evaluator log.
- `quality-pilot-v2/summary.json`와 `.md`:입력·GT/noise·source/weight 일치를 검사한 paired SS 비교.
- 최초 실패 기록과 source snapshot을 각 versioned 폴더에 보존했다. 원래 canonical128×3 최종 결과와 섞지 않는다.
