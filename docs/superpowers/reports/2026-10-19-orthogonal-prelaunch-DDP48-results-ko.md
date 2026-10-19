# 후속 두 후보의 실제 DDP2/micro48 사전 검증

2026-10-19 06:36 UTC. Scale16의 본 학습은 GPU0/1에서 raw480까지 계속 진행 중이다. 이미 완료된 Top-1 seed202의128개·3noise 평가를 마친 뒤, 비어 있는 GPU2/3에서 후속 두 후보의 두-update benchmark를 순서대로 실행했다. 본 학습을 새로 시작하거나 재시작한 것이 아니며, 사전 검증의 가중치를 본 학습에 전달하지 않는다.

| 후보 | rank0 최대 할당 메모리 (GiB) | rank1 최대 할당 메모리 (GiB) | rank0 첫/둘째 update (초) | rank1 첫/둘째 update (초) |
|---|---:|---:|---:|---:|
| Native voxel-query | 51.041 | 51.024 | 21.119 / 18.938 | 21.136 / 18.921 |
| Soft score prior T0.05 | 50.464 | 50.336 | 18.743 / 17.342 | 18.717 / 17.311 |

두 실행 모두 global96, DDP2, GPU당 micro48, accumulation1이었다. Benchmark의 loader는 공용 memory probe와 같이 workers0이고 W&B는 비활성이다. 원래 SS4210의 base+LoRA에서 새로 시작했고, optimizer371개 상태 모두 step2 및 finite moment를 확인했다. OOM은 없고 선언한91GiB allocation 기준을 통과했다. 본 학습 config·checkpoint와 현재 scale16 프로세스는 변경하지 않았다.

두 후보의 양 rank 첫-step 전체 loss 및 모든 기록된 timestep-bin loss는 scale16 reference의 원래 micro48 benchmark와 정확히 일치했다. Zero-start가 실제 global96 첫 loss까지 유지된다는 근거다. 각 후보의 단독 native GPU proof에서도 global/proj 초기 조건 차이는 정확히0이었고, SS-loss gradient가 LoRA·fusion·retargeter 및 임시 복원 probe의 utility에 연결됐다. Frozen decoder/score gradient는0이었다. 단독 proof의 optimizer update는0이며 adapter output 임시 perturbation은 정확히 복원했다.

이것은 초기 정합과 짧은 하드웨어 검증이다. 두 timing을 장시간 평균 속도나 실제5에폭 완료 시간으로 표현하지 않는다. Main은 검증된 workers2와 별도 output을 쓰고, native proof·두-update memory probe·원래 SS4210 fresh start를 다시 수행한다. 기존5에폭과 비교하는 raw480 budget은 그대로다. Validation 성능 및 장시간 메모리 안정성은 실제 main 결과가 있어야 확인할 수 있다.

원래 shared forward6개와 후보별 source11개/8개의 hash, 원래 main config hash가 실행 전후 동일했다. 두 benchmark의 resolved config·양 rank runtime·result·실제 step2 denoiser/conditioner/misc 파일은 각각 `voxel-query-followup-v1/memory-bench/prelaunch-DDP48-v1/`, `utility-softprior-followup-v1/memory-bench/prelaunch-DDP48-v1/`에 있다. 단독 native proof는 각 후보의 `verification/<candidate>/standalone-native-v1/`에 보존해 production proof와 분리했다. Main을 실행할지는 기존 scale16의 고정 raw480/128개/3noise 결과로 결정한다.
