# Direct 전달과 deformation query 효과를 분리하는 대조군

## 추가한 이유

현재 input-side attention과 Direct+DQ를 비교하면 feature 전달 방식과 추가 query가 함께 달라진다. 기존 입력 채널과 projection이 없는 **Direct NoDQ**를 추가해, 같은 direct 전달 아래 query의 효과와 query가 없는 조건에서 direct 전달의 효과를 각각 비교한다. Retargeting adapter는 이 네 모델 모두에 존재한다. Retargeting 유무×query 유무 실험이 아니다.

| 모델 | 추가 feature 전달 | 추가 RAW deformation query | 공통 R/LoRA |
|---|---|---|---|
| 기존 feature attention | Input-side | 없음 | R96/Adam366 |
| Feature attention + DQ | Input-side |2048개 채널 | 동일 |
| 새 Direct NoDQ | Direct residual | **채널·projection 없음** | 동일 |
| Direct + DQ | Direct residual |2048개 채널 | 동일 |

NoDQ에서도 target-image ray query, semantic Top64, matching score, decoder context와 원래 Top1 carrier는 유지한다. 추가 RAW Omega 차이를 retargeter에 넣는 경로만 제거한다. Matcher가 쓰는 FRAME descriptor까지 없애는 arm이 아니며, 실제 관측값을0으로 채우지 않는다. SS retargeter 입력은2048, Full DQ는4096이다. PBR width에 대한 CPU 구조 검사에서는 각각3072/5120이었다. PBR NoDQ native 학습을 구현·실행했다는 뜻은 아니다.

## 검증과 연결

- CPU4개 검사: SS/PBR 두 value width, nonzero attention output/R을 가진 상태에서 DQ projection0인 full 모델과 동일 direct 전달, physical input 부재와 RAW 입력 거부, checkpoint backward 및 empty ray/hook 복원.
- CPU2개 검사: 실제 spawn child마다 올바른 NoDQ constructor를 설치·복원하고 parent도 그 worker를 전달하는지 확인했다. 처음10초 teardown 제한은 child 등록 완료 후에도 interpreter 종료가 끝나지 않아 실패했다.60초 제한과 production SDPA 환경에서 두 child의 exit0까지 확인했다. 기존 GPU 학습은 영향받지 않았다.
- 실제 TRAIN2개 micro2 검사: 시작 SS loss0.10466600209474564 및 evidence가 원래 R과 정확히 같았다. 첫 output-projection gradient L1은31.1272583, 세 번째 update까지 attention9개 모두 비영 gradient였다. NN/R/matcher는 고정됐다. Max allocated5.5676GiB, 임시3 update 후 모델을 폐기했다.
- 실제 joint micro2 검사: LoRA360 + R6 + attention9, 총375개가 모두 변경됐다. Common366 Adam은96→99, 새9개는0→3이었다. Frozen base/matcher는 그대로였고 max allocated5.8127GiB였다. Native process exit0 및 실제 file4 SHA를 확인했다. **DDP2/micro48 capacity 증거로 사용하지 않는다.**
- Native evaluation CLI도 실제 한 사례로 실행했다.0개의 deformation 채널과 projection 부재를 관측했고, 동일 cold reference의 GT hash/seed/count/metric 및 두 PNG가 같았다. Generation batch1의 실행 검사이며 성능 실험이 아니다.

GPU0/1의 현재 DQ 학습이 끝난 뒤 NoDQ의 별도 DDP2/global96/micro48/accumulation1 capacity를 수행하도록 연결했다. 통과하면 같은 R96/Adam366에서384 update를 학습해 total480에 맞춘다. 동일128×3,32 replay, strong original R/V1 및 query/delivery 분리 비교까지 총8개 작업을 연결했다. 기존 두 학습과 다른 queue recipe는 바꾸지 않았다. 현재 NoDQ scientific 학습 및 fullbatch capacity는 아직 수행하지 않았다.

추가 검토에서 NoDQ의 초기 외부 guard가 native STOP과480 완료를 구분하지 않는 경우를 발견했다. 아직 시작하지 않은 NoDQ train job만 별도 v2 guard로 교체했다. 실제 completed progress480과 checkpoint trio가 있어야 PASS를 쓰며, native STOP은 저장 상태를 유지하고 완료로 표시하지 않는다. 동결된 v1과 모델·데이터·optimizer 코드는 그대로 보존했다. 변경 기록은 `training-stop-guard-v2`에 있다.

기존 cache를 그대로 쓴다. GT depth/deformation/correspondence를 forward에 추가하지 않으며 데이터 재구축이나 raw download는 없다. Prototype은 기존 학습 peak 약50.4GiB와 별도20% allocator 상한을 근거로 GPU0/1에 제한적으로 함께 실행했고 모두 종료했다. GPU6/7은 사용하지 않았다.

## 실제 step192의 경향

동일32개·동일 GT/생성 noise에서 **같은 학습 update192**를 비교했다. IoU는 전체 TP/FP/FN을 합친 값이다. Zero-shot은 update0의 맥락 비교다.

| 모델 | Update | IoU |
|---|---:|---:|
| Zero-shot source4M0 |0|0.39499655|
| Zero-shot edited1M1 + source3M0 |0|0.42671547|
| Vanilla V1 |192|0.43444352|
| Original positive Top1 + R |192|0.44036476|
| Matched R control |192|0.44216347|
| Input-side attention |192|0.44190640|
| Input-side attention + DQ |192|0.44050169|
| Direct + DQ |192|0.44251782|

Direct−matched R은 +0.0354pp, source-cluster95% CI [-0.3607,+0.4342]pp다. Direct−input-side DQ는 +0.2016pp, CI [-0.1393,+0.5613]pp다. Input-side DQ−NoDQ는 -0.1405pp, CI [-0.3974,+0.1267]pp다. **모든 구간이0을 포함한다.** 전달 방식의 점 추정치가 유리해 보여도 아직 개선을 입증하지 못했고, query가 해롭다고도 확정할 수 없다.

Original R/V1은 같은 update의 실제 validation row를 읽었다. 새 네 모델과 original continuous R은 post96 학습 RNG가 같지 않으므로, 원래 R 대비 차이를 새 module 하나의 효과로 귀속하지 않는다. Fixed480/128×3, 올바른 module 대조 및 독립 training-seed 확인을 유지한다. 이번32개 중간 관측으로 epoch나 full 모델을 선정하지 않는다.

## 증거

Canonical root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`이다.

- `direct-no-query-v1/SS-gradient-v1`: 실제 frozen-NN/R native identity 및9개 gradient.
- `direct-no-query-v1/joint-profile-v1`: 실제375개 update 및 common Adam 복원.
- `direct-no-query-v1/evaluation-entry-probe-v1`: native evaluation/PNG 검사.
- `direct-no-query-v1/prelaunch-v1`:129개 source binding과4개 recipe 및 초기 state.
- `direct-no-query-v1/evaluation-plan-v1`, `queue-connection-v1`: capacity부터 fixed128×3까지8개 작업과 append-only 기록.
- `physical-query-epoch-trends-v1/step0000192`: 같은 epoch의6개 학습 모델, 두 zero-shot 및 paired CI.

위 artifact들은 실제 file4 SHA/size receipt로 닫았다. 성능 개선·모델 채택·전체 실험 완료는 아직 선언하지 않는다.
