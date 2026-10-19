# Frozen semantic Top-K: 두 후속 비교

현재 scale16 후보는5에폭 고정 평가를 계속한다. 이 보고서는 그 최종 결과가 primary 기준을 통과하지 못할 때 실행할 두 후속 실험을 정의한다. 중간 epoch를 보고 성공·실패를 결정하지 않는다.

| 후보 | 현재 scale16에서 바꾸는 모델 요소 | GPU |
| --- | --- | --- |
| voxel-query-c1 | Utility query를2D target patch에서 native4view의 위치별 projected DINO로 변경 | 0·1 |
| utility-softprior-c1 | Frozen score prior temperature를0.005→0.05로 변경 | 2·3 |

양쪽 모두 frozen Omega7/DINO/R9 D1·scalar score,semantic Top64,C1 입력2048·deformation 추가채널0,zero-output utility 초기 scale16을 유지한다. Fresh SS4210 base+LoRA continuation,seed101,5에폭/raw480/global96/DDP2/micro48/accumulation1,optimizer/LR,원래 development128·93source·3sampling noise도 유지한다. 실제 초기 native/SS-loss backward와 두 update의 최대 배치 검사를 먼저 수행한다.

Soft-prior 후보는 이전의 고정 C1/T0.05나 matcher까지 공동 학습하던 Full과 다른 비교다. Score를 uniform하게 만들거나 제거하지 않고,학습 가능한 zero-start utility가 원래 score prior를 보정한다. 사용자 정의 No-score는 여전히 Top1 correspondence feature만 쓰는 대조이며,이 후보를 No-score로 부르지 않는다.

현재 모델의 실제 saved attention은 Top-K라도 Top1에 강하게 집중돼 있었다. 위치별 query와 prior 강도를 각각 바꿔 어떤 변화가 필요한지 분리한다. 두 후속 후보를 서로 비교한 차이는 두 모델 변화가 함께 다르므로 단일 모듈의 효과로 귀속하지 않는다. 각각 현재 scale16 및 scale4 reference와 직접 비교한다.

## 실행 분기와 검증

현재 scale16의 실제 raw480/final128×3·다섯 epoch/W&B·최종 tensor·GPU QA·file4 대조가 완료돼야 분기한다. V1·같은 retargeter Top1·no-retargeter Top1 대비 source-cluster95% 구간이 양수이고 세 sampling noise 모두 방향이 양수면 이미 준비된 seed202 재현을 진행하며 이 두 본 학습은 추가하지 않는다. 실제 음성일 때만 두 개의 독립 GPU pair에 후속 학습·평가를 추가한다.

각 후보의 예약 작업은 본 학습,최종128×3 generation,saved-model GPU QA,다섯 epoch의 실제 optimizer/batch/GT/noise/W&B 수치·PNG/file4 검사,최종701개 frozen base·360개 LoRA·matching24개/utility·371개 optimizer state 검사와 최종 비교다. 같은 raw480/common32/noise0의 value-address 진단도 별도로 수행한다. Normal 경로가 최종128의 해당32개와 정확히 재현되고,score·weight·support·query별 value multiset이 유지된 상태의 주소 민감도를 측정한다. Native-spatial No-corr 학습 효과나 정확한 correspondence accuracy로 해석하지 않는다.

새 epoch/final 검사 process는 각 checkpoint가 준비된 뒤 기존 검증 함수의 그대로인 body를 실행하고 종료한다. 고정된 audit template SHA를 확인하고 마지막 watcher loop만 제거하며,실행 root·source binding도 실제 file4 receipt에 포함한다. 여러 idle watcher가 base/optimizer RAM을 계속 보유하지 않도록 한다. 이 변경은 tensor·optimizer·원격 W&B·file4 검사를 생략하지 않는다.

위치별 query의 사전 GPU proof는 본 학습이 소유하는 proof 파일과 다른 하위 폴더에 보존한다. 기존 proof를 덮어쓰는 실행은 거부한다. 사전 proof는 Top1+C1 seed202의 최종 GPU 작업이 끝난 후 physical GPU2에서만 수행하며,기존 두 건강한 학습을 중단하지 않는다.

## 새 실행의 loader RAM

04:22경 두 기존 DDP 학습에서207/218초대 동시 지연이 발생한 뒤17초대로 복귀했다. Host memory PSI 상승과8GiB swap 포화가 함께 있었으며,GPU OOM은 관측되지 않았다. 이 상관만으로 단일 원인을 확정하지 않는다. 현재 학습의 worker8·sampler·PID는 유지한다.

새 두 후보에는 worker2를 적용한다. 실제 train9149 데이터 source에 stochastic 호출이 없고,실제 두 sample을 반복해서 읽었을 때 모든 필드와 CPU torch/numpy/python RNG가 같았다. 원래 ResumableSampler와 persistent DataLoader를 사용한 CPU index 실험에서는 두 rank 각각480 update의 모든 batch index와 소비된 sampler 상태·마지막 main CPU RNG가 worker8/2에서 같았다. 실제8/2 실험은 CPU두 core,peak RSS0.87GiB였으며 모델 학습을 하지 않았다.

이 검사는 index-only loader에서 pin_memory=False를 사용했다. 전체 실데이터 GPU 공급 속도·GPU copy timing·전체 GPU 수치 재현을 증명하지 않는다. 새로운 recipe는 이 실제 source/order proof SHA를 요구하며,시작 시 실제 DDP48 benchmark와 초기 loss/GT/noise identity를 추가 확인한다. Loader와 해당 proof metadata를 제외한 모델·optimizer·데이터 설정의 차이를 별도로 대조한다.

현재110개 관련 CPU 검사가 통과했다. Synthetic gate/receipt 사례는 pytest temp에만 생성했고 실제 성능 근거로 쓰지 않는다. 두 후속 모델은 아직 본 학습이나 품질 개선을 완료하지 않았다. 실제로 기준을 통과하는 후보의 추가 training seed,decoder 물리 제거·Top1 No-score·native-spatial No-corr의 효과,원래 예약121개 source-disjoint 평가는 별도로 남아 있다.
