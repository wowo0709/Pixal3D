# 현재 유망 조합과 다음 실험

현재 Vanilla보다 좋은 조합은 있지만, semantic Top-K와 matching-aware attention이 이미 좋은 retargeting 기준보다 독립적으로 좋아졌다는 결론은 아직 없다. 최종 full 구조는 확정하지 않는다.

## 새로 완료된 SS 최종128×3

동일128개·93 source cluster·생성 noise3회·총480 update 결과다. 생성 noise3회는 training seed3개가 아니다. 아래 학습 arm은 seed101 하나이며, 현재 표는 voxel 구조 평가다.

| 모델 | Voxel IoU(%) |
|---|---:|
| Vanilla V1 |43.2650|
| Matched Top1/R control |44.0294|
| 기존 positive Top1/R |44.2757|
| 기존 feature attention |43.9242|
| Feature attention + physical deformation query |44.2249|
| Direct:semantic TopK ray attention + physical deformation query + residual retargeting |44.3692|

Direct는 원래 Top1/R carrier를 보존하고 learned feature correction을 native retargeting residual로 전달한다. 새 reliability-gated candidate-paired 모델 및 새 bounded task attention과 다른 모델이다.

Direct−V1은 **+1.10421pp,95% CI[+.61198,+1.63009]pp**다. Direct−matched R은 **+.33986pp,CI[−.05564,+.74928]pp**, Direct−기존 positive R은 **+.09349pp,CI[−.26941,+.48361]pp**다. 따라서 Vanilla 대비 이득은 이 개발 평가에서 관찰되지만, retargeting을 넘는 추가 matching 조합의 이득은 아직 불확실하다. Direct−기존 feature attention은+.44503pp,CI[+.02515,+.88703]pp이지만 이 비교만 골라 full 구조가 검증됐다고 주장하지 않는다.

원본은 `direct-retargeting-v1/matched-final-v1/{summary-v1,strong-reference-summary-v1}`이며 두 actual file4 receipt를 유지한다. 각 arm의32개 replay는 정확히 일치했다. Historical positive R과 새 matched R은 post96 학습 stream이 달라 두 기준을 모두 보고한다.

저장 Direct480의32개 path control에서는 score-prior 제거·decoder context 우회·추가 경로 제거가 normal보다 확실히 나빠지지 않았다. 이는 저장 모델의 조건부 민감도이며 재학습한 module ablation이 아니다. NoScore=Top1 payload,NoCorr=native spatial correspondence,NoDQ=채널/프로젝션 제거라는 사용자 정의는 유지한다.

## PBR에서 보인 조합

기존 step762의32개 GT-Shape-fixed texture-stage 평가에서는 D1+learned score를 사용하는 semantic Top1 bundle(B2)이 B1보다 foreground PSNR+.07232dB,LPIPS−.0006063,SSIM+.0009518이었다. 세 지표의 source-cluster CI는 해당 개선 방향이지만, 작은 exposed 개발 패널·training seed1·native rendering exact replay 미확정이라는 한계가 있다. Decoder와 score 각각의 독립 효과가 분리된 비교도 아니다.

B3 TopK 가중합은 B2보다 추가 이득을 입증하지 못했다. 따라서 재사용할 출발점은 **retargeting과 learned matching Top1 bundle**, 추가 검증할 후보는 **Direct full 및 task-specific ray attention**이다. 최종 PBR full 후보의 새762→1905 학습은 메모리 검증 이후 대기 중이며 완료된 결과로 표기하지 않는다.

## 진행 순서

1. 현재 SS 가중합 arm과 남은 final/path 평가를 유지한다. Candidate-paired 모델192→480을 이어받아 같은 final128×3와 비교한다. 이미 있는 V1·R 기준 결과는 재사용한다.
2. 새 bounded task attention은 같은 R96·NN/R Adam366에서384 update를 추가해480으로 맞춘다. 현재 native micro2 learnability만 통과했다. 자체 초기32 생성 및369-state DDP2/micro48 capacity를 확인한 뒤 본학습·동일 평가로 연결해야 한다. Engineering probe를 과학적 학습 완료로 처리하지 않는다.
3. 전체 cascade 평가를 병행 보강한다. 확보한 Shape512/Shape1024를 고정하고 실제 SS 예측을 Shape/PBR에 전달해 raw mesh/RGB/alpha를 보관한다. LPIPS·DreamSim·CLIP,PSNR·SSIM·2D M-IoU,mesh CD·surface F-Score를 사용한다. GT Shape 조건의 texture-stage 평가와 구분한다. Source-preservation 기준은 identity/material을 유지한 paired textured M1 GT이며 raw M0 pixel reference로 치환하지 않는다.
4. PBR은 이미 capacity를 통과한 global24/world4/micro6/accumulation1을 사용한다. 현재 V1/Direct 두 arm은 stack 비교다. Matching 독립 효과를 위해 동일 시작점·budget의 R-only/Top1 대조를 추가해야 하며, 해당 native loader·초기 parity·capacity 확인은 아직 남아 있다.
5. 후보가 좋아지면 matching decoder,Top1 대 TopK payload,score 사용,ray attention,physical deformation query를 같은 initializer·budget으로 비교한다. 저장 weight toggle만으로 효과를 확정하지 않는다. 최종 양성 후보는 fresh202·별도 training seed로 확인하고 locked test 및 논문 metric을 보고한다. 작은32개에서 가장 좋은 checkpoint를 골라 test 결과처럼 사용하지 않는다.

계산은 사용자 지정 GPU0~3만 사용한다. 새 데이터셋이나80k 재학습은 추가하지 않는다. DeformNeXt의 구축된 split과 기존 cache를 유지한다. 전체 목표는 진행 중이다.
