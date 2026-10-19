# Native voxel query 후속 후보

현재 학습 중인 scale16 후보의 고정 최종 평가가 끝난 뒤 사용할 후속 설계다. 아직 GPU 검증이나 학습·품질 개선이 완료된 모델이 아니다. 현재 두 DDP run은 유지한다.

기존 첫8개 structural 진단에서는 target foreground patch bin으로 투영된 grid center7,579개가622개 patch를 공유했다. 비율은12.185였고,같은 patch 안 native four-view feature의 pooled relative RMS는16.00%였다. 이는 GT occupied voxel 수나 같은 물리적 ray의 개수가 아니다. 이 관찰은 위치별 query의 가능성을 제시하며,기존 성능 저하의 원인 증명은 아니다.

`voxel-query-c1`은 native four-view projected DINO feature를 각 voxel의 utility query로 사용한다. 각 위치는 기존 nearest target-patch mapping에 따른 같은 semantic Top64 후보·frozen D1 context·matching-score prior를 소비한다. Source value는 원래 DINO이고,temperature0.005와 초기 utility scale16을 유지한다. C1 retargeter의 target query는 원래2D DINO이며,입력은2048채널로 유지한다. Deformation 채널을 추가하지 않는다.

새 trainer와 conditioner를 별도 파일·별도 registry 이름으로 등록했다. 현재 scale16이 사용하는 여섯 forward/runtime 파일은 변경하지 않았다. 값 gather와 utility 계산을 chunk128 checkpoint 안에서 수행해 모든 voxel의 `[K,1024]` value를 retained graph에 복제하지 않도록 했다. 실제 최대 배치 메모리는 GPU benchmark로 확인해야 한다. 공용 TaskCandidateAttention Module 호출을 유지해 score·weight를 보존하는 value-address hook도 동작한다.

관련 CPU 검사80개가 통과했다. 같은 patch query를 반복하면 기존 모델의 후보·score·attention·evidence·retarget residual이 FP32와 CPU BF16에서 재현됐다. 다른 native query는 같은 후보 pool에서도 attention을 바꿨고,checkpoint를 통과한 gradient가 연결됐다. 입력 feature가 frozen인 경우도 포함한다. 초기 fusion은 native output과 정확히 같았고,invalid voxel은 학습된 gate에서도 native를 유지했다. 이 검사에는 실제 학습 성능 주장이 없다.

실행 recipe는 기존 scale16 recipe를 복사해 trainer와 voxel chunk만 변경하며,원래 SS4210 base+LoRA continuation·optimizer·LR·seed101·train9149/validation508·5에폭/global96/DDP2/micro48/accumulation1을 유지한다. 별도 port30446과 W&B run을 쓴다. 공용 launch는 실제 SS-loss/native/backward 검사와 두 update의 DDP48 메모리 검사를 먼저 수행한다. 새 entry source contract는 추가 module/trainer/registry를 포함한 열 개 source SHA를 기록한다.

Scale16이 두 Top1 대조 및 V1 대비 고정 raw480/development128·93source·3noise 기준을 통과하면 기존 조건부 seed202 재현을 먼저 진행한다. 실패하면 이 후속 후보를 실제 GPU 검증하고 같은 고정 예산으로 비교한다. 이 보고서 작성 시점에는 후속 후보의 학습을 queue에 넣지 않았다. Decoder 제거·Top1 No-score·native-spatial No-corr의 학습 효과와 예약121개 검증은 별도로 남아 있다.
