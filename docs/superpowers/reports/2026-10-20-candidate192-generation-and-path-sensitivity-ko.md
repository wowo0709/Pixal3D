# Candidate-paired 모델192 step: 실제 생성 결과와 경로 점검

작성일:2026-10-20 UTC. 현재 작업은 완료되지 않았다. 새 모델의 전체480 학습과128×3 평가, 기존 SS 비교 및 PBR 학습·평가를 유지한다.

## 동일192 step의 전체32개 결과

새 모델은 기존 R96의 NN/R·Adam366을 이어받아96 update를 추가했다. DeformNeXt-10k의 기존 정합 데이터와 native RGB 입력4개를 사용했다. Global96,DDP2,micro48,accumulation1,seed101이다. 원래80k 데이터로 deformation 학습을 다시 한 것이 아니다.

| 모델/입력 | Voxel IoU(%) |
|---|---:|
| 새 candidate-paired + frozen matching + predicted reliability |44.2246|
| Matched R control |44.2163|
| 기존 feature attention |44.1906|
| 기존 DQ |44.0502|
| 기존 FullDirect |44.2518|
| Geometry-only |44.1957|
| Physical NoDQ |44.2128|
| V1 |43.4444|
| Historical Top1/R |44.0365|
| Zero-shot:edited RGB1 + undeformed RGB3 |42.6715|
| Zero-shot:undeformed RGB4 |39.4997|

Matched R 대비 차이는 **+.00827pp**, source-cluster bootstrap95% CI는 **[−.16334,+.15591]pp**다. **추가 성능 향상을 확인하지 못했다.** 구형 DQ 대비 차이+.17445pp의 CI가 양수여도, 주 비교 대상인 matched R을 넘었다고 해석하지 않는다. Historical R은 post96 학습 stream이 다르고, zero shot은 학습 budget이 다르다. 모두 source·GT voxel hash·sampling seed를 확인했다. Training seed는1개이며 이32개는 반복 노출된 개발 패널이다.

빠른8개 평가에서는 새 모델50.5091%, 같은8개의 R50.6148%였다. 전체32개와 다른 denominator이므로8개 평균을32개 평균과 비교하지 않는다.

Actual file4 증거는 `consistent-retargeting-v1/{phase1-after-attempt-v2,runs_ss64/full-v1/segment192-file4-closure.json,evaluation-step192-native32-v1,stage192-comparison32-v1}`에 있다. 이 구간의 native/wrapper/queue는 정상 종료했고, optimizer367과 두 rank RNG·scheduler를 남겼다. 전체480이 끝났다는 뜻은 아니다.

## 실제 학습 상태와 보정 신호

Checkpoint를 CPU에서 읽어 비교했다. CPU로 모델을 학습하지 않았다.

| 항목 | 실제 확인 |
|---|---|
| NN LoRA |360 tensor 모두 변경|
| NN frozen state |701 tensor가 R96과 정확히 같음|
| 공통 R/fusion |6 tensor 모두 변경|
| Frozen conditioner state |37 state가 초기값과 정확히 같음|
| 새 geometry projection |524,288개 원소가 모두 nonzero,finite;RMS.0017595|
| Adam step |NN/R192,new projection96;367 state|
| Scheduler |last_epoch96|

최근10 update의 geometry 보정 RMS 중앙값은 .00034566, R total residual RMS 중앙값은 .0186223으로 비율은 약 **1.856%**였다. 이 값은 내부 residual의 비율이며 전체 이미지 feature 대비 비율이 아니다. 최종 fusion tanh는 .0165367이다. 새 경로는 실제로 학습됐지만 신호가 작다. 이것이 낮은 성능의 원인이라는 결론은 아직 내리지 않는다. `feature_gradient_L1=0` 로그는 고정한 matching head를 가리키며, 새 projection이 학습되지 않았다는 뜻이 아니다.

`scientific192-state-audit-v1`의 actual file4 audit에 NN/R·head 상태와 optimizer 확인 결과를 보관했다. 초기32개 평가 수치와 첫 scientific97의 두 rank 전체 loss 항목이 기존 R과 같았다는 앞선 증거도 유지한다.

## 저장 weight의 진단 실험

새 모델의192 checkpoint·NN/R weight·RGB·GT·noise를 고정하고 아래10개를 실행하도록 연결했다. 이 결과는 **다시 학습한 ablation의 효과가 아니다.** 원래 Top1/R carrier는 유지한다.

| Control | 확인할 내용 |
|---|---|
| Geometry-off |새 geometry residual의 사용 여부|
| Gate-off |예측 reliability 대신 유효 ray의 gate1;불확실한 정보 억제 효과|
| Correction×4/×16/×64 |보정이 너무 약한지, 방향 자체가 나쁜지|
| Mean-inputs |같은 α로 appearance와 RAW deformation을 모두 평균한 뒤 R 적용;후보별 R 후 합산과 비교|
| Native-Top1 |추가 경로가 native semantic Top1 feature만 사용;uniform score가 아님|
| Attention-Top1 |현재 learned α의 Top1만 사용;ranking과Top-K 결합을 구분|
| Score-prior-off |α의 원래 matching prior만 제거;사용자 정의 NoScore arm과 다름|
| Attention decoder bypass |α에 D1 context 대신 원래 Omega target key 사용;reliability의 D1 context는 유지|

Decoder bypass에 DINO query를 대신 넣지 않는다. D1이 원래 입력받는 **Omega target key**를 사용한다. Mean control도 Top1 appearance를 쓰는 기존 constructor의 `composition=mean`과 구분한다. 올바른 평균-후-변환 비교를 위해 appearance와 deformation 모두 같은 α로 평균한다.

원래 모델을 보존하는지 normal8 native probe로 먼저 확인했다. 8개의 전체 per-asset 기록이 기존192 전체32 평가의 해당8개와 **정확히 일치**했다. CPU 검사2개도 통과했다. 모든 control은 optimizer update0이며, 성공·실패 결과를 모두 보고한다. 이 화면으로 λ를 골라 최종 개선으로 확정하지 않는다. Physical NoDQ는 channel/projection 자체를 제거해야 하며 Geometry-off를 그 arm으로 표기하지 않는다. Native spatial NoCorr도 별도 정의와 학습 검증이 필요하다.

`saved-control-plan-v1`, `saved-control-queue-v1`을 actual file4에 닫았다. 구현은 `consistent_ray_controls.py`, `evaluate_consistent_ray_control.py`다. 이 process에서만 함수를 바꾸며 원래 모델·학습 entry 파일은 수정하지 않는다.

실제10개 결과도 완료했다. Normal 대비 Geometry-off−.1444pp,Gate-off−.1046pp,×4−.0269pp,×16−3.6349pp,×64−11.4154pp,Mean-inputs−.0420pp,Native-Top1+.0861pp,Attention-Top1−.0481pp,Score-prior-off−.1144pp,attention decoder bypass+.0372pp였다. Native-Top1의 paired CI는[+.0205,+.1675]pp였고,×16/×64의 CI는 각각[−6.1482,−1.6587]pp/[−14.2281,−9.0972]pp였다. 개발 패널의 여러 진단 결과이며 재학습한 module 효과로 사용하지 않는다. 작은 보정을 크게 만드는 것만으로 성능이 회복되지는 않았다. 현재 payload 선택·결합을3D 목표에 맞추는 문제가 남아 있다.

다음 후보는 기존 semantic prior7과 reliability4를 고정한 채 별도 Q/K2개만 학습하는 `TaskRayRetargetingConditioner`다. Query는 ray별 DINO1024,D1 context1024,실제 RAW deformation difference2048의4096 channel이다. Q는 zero weight로 시작하며 추가 score를±1로 제한하고 fixed semantic distribution에 대한 KL을.001 가중치로 적용한다. 기존 Top1/R,후보별 geometry retargeting,ray projection은 유지한다. CPU 검사2개에서 prior 보존,실제 deformation query 사용,Q/K gradient와 score 제한을 확인했다.

새369-state native micro2 engineering probe도 정상 종료하고 `bounded-task-attention-v1/native-probe-v1`에 actual file4 증거를 보관했다. R96에서3회 disposable update를 수행했으며 첫 update의 기존 loss 항목은 원래 R96 probe와 정확히 일치했다. NN360/R6/projection1/QK2 tensor가 모두 변경됐고, frozen NN base·semantic prior·decoder·reliability는 정확히 보존됐다. 세 번째 update의 **SS MSE만으로 계산한** gradient L1은 projection .78860,Q .00248956,K 4.18895e−7이었다. Attention이 KL만 학습하는 상태는 아니다. Peak6.3705GiB,update 구간27.07초였다. 이3회는 버린 engineering probe이며 성능 평가용 학습이 아니다. 별도 초기 생성·DDP capacity·과학적 학습 gate는 아직 통과하지 않았으며 모델 채택 대상이 아니다.

## PBR 메모리 복구와 다음 실행 순서

Frozen image feature를 이미지별로 순차 처리하는 변경은 기존 initial32 per-asset 기록·metric·34 PNG를 정확히 보존했다. 원래 world2/micro12의 NAF OOM을 해결했지만 FullDirect backward는 micro12에서 OOM이었다. 원래 실패 기록은 남아 있다.

새 **world4/micro6/global24/accumulation1**은 두 arm 모두 actual 최대31900-token 반복 batch,4회 update, 중간4개 렌더 validation, 이후 update 및 DDP 동일성 검사를 통과했다. V1 peak46.4373GiB, FullDirect49.5701GiB다. 모두 actual file4에 닫고 world4를 선택했다. Selection JSON의 예전 `original_initial32_full_render_parity_not_repeated_yet=true` 필드는 작성 시점의 문구이며, 별도 `initial32-parity-v1`의 완료 증거가 현재 상태다. 원본 selection receipt를 바꾸지 않는다.

현재 scalar SS는 GPU2/3에서 학습하고, 새192의 control과 남은 기존 SS 평가는0/1에서 진행한다. 이후 같은192 state로 C193–480을0/1에서 이어받는다. 검증만8→32로 복원하고 NN/R·Adam·rank RNG·sampler·LR은 이어받는다. Scalar final 평가4개까지 C가 기다리는 의존성을 발견해 제거했다. 그4개 평가 자체와 PBR 본학습의 선행 조건은 그대로 남아 있다. 따라서 두 SS 학습을 병렬로 진행할 수 있다.

이후 새 모델의480 평가32·128×3, 기존 비교, PBR V1/FullDirect762→1905 및3개 평가·paired 요약을 실행한다. PBR V1 대 FullDirect는 stack 비교다. Matching만의 독립 효과에는 matched R-only가 필요하며 새 candidate-paired conditioner의 native PBR 포팅은 아직 수행하지 않았다. 독립 training seed·test508·논문의10개 full-pipeline metric도 현재 단계의 결과로 대체하지 않는다.

PBR quick 평가는GT target Shape 조건의 stage 평가다. 현재 crop_LPIPS는GT target render와의 거리를 기록하며, 논문의 source-preservation LPIPS는 source render 기준으로 별도 평가한다. 이전 프로토콜의 양성 후보fresh202 검증도 유지한다.

사용자의 ACL 변경 후 Shape512 step4210·Shape1024 step16850과 두 config의 읽기·복사를 완료했다. 파일서버 원본과 로컬 복사본의 SHA256·크기가4개 모두 일치했다. 각 저장 config로 실제 `ElasticSLatFlowModel`을 구성하고 LoRA rank16/alpha16을 주입한 뒤, CPU에서1060 state의 key·shape·dtype와 strict weight loading을 확인했다. 모델별700 base state와360 adapter state가 있으며180개 LoRA B 모두 nonzero다. Config와 원본 weight는 수정하지 않았고 CUDA context도 만들지 않았다. 검증 기록은 `consistent-retargeting-v1/cascade-reference-v1`에 보관한다. 두 Shape 모델은 최종 cascade 평가에서 고정한다. 이는 full-cascade 생성·metric 평가가 이미 완료됐다는 뜻은 아니며, 해당 GPU 검증은 남아 있다. SS/PBR stage 실험은 계속 진행한다.
