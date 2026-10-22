# 새 native PBR의 predicted-support 평가 구현과 NAF 초기화 검사 복구

2026-10-22 UTC. Source/PBR의 성능 향상을 입증한 보고서가 아니다.
실제 GPU 정합 검사와, 학습된 모델을 생성 geometry에서 비교하기 위한 실행 경로를 기록한다.
SS 학습과 dispatcher/mirror는 재시작하지 않았다. 물리GPU0–3만 사용한다.

## 실제 완료된 source 전달 검사

`PBR-generated-R-source-support-v1/GPU-proof-v1`은 실제GPU에서 완료하고 native/wrapper
모두 file4 SHA/size로 닫았다. TRAIN 최대31,900-token support와 개발 asset의
실제 predicted support를 구분해 세 source 모드를 검사했다.

- Cold 상태에서 원래 native pipeline conditioning과 정확히 같다.
- 임시 R-output bias 개입 시 DINO와 NAF 채널 모두에 실제 비영 보정이 전달된다.
- 실제 sparse training consumer와 새 pipeline helper의 conditioning이 같다.
- CPU/CUDA sampler RNG, global/negative stream과 임시 개입 복원이 검사를 통과했다.
- Denoiser/optimizer/sampler는 사용하지 않았다. Peak allocated는19.163GiB다.

이 결과를 texture 생성이나 학습된 R의 성능 개선이라고 해석하지 않는다.

## Native micro1의 원래 실패와 새 검사

원본 vanilla micro1은 첫 encoding 직후 combined field/global/negative/RNG equality
guard에서 실패했다. 첫 NAF lazy load가 같은 구간에 발생한 로그가 남아 있다.
원본 실패와 terminal1을 보존하고, 별도 `probe_r_mediated_pbr_native_v2.py`를 만들었다.
Native constructor가 끝난 뒤 frozen NAF를 미리 로딩하며 CPU/CUDA RNG를 fork/restore한다.
Source/LoRA 초기화, optimizer와 sampler를 다시 실행하거나 학습 recipe를 바꾸지 않는다.
CPU의 RNG·예외 복구 검사2개가 통과했다.

별도 vanilla v2 GPU 실행은 실제 maximum31,900-token TRAIN asset에서 다음을 통과했다.
Native field/global/negative와 RNG가 정확히 같고, 같은 noise의 velocity SHA와 loss도
native bypass와 같다. 실제 backward3회, optimizer state360개, frozen weights 불변,
peak20.096GiB다. Top-1+R v2도 같은 입력/noise/velocity/loss를 재현하고 source10개를
포함한 optimizer370개로3회 backward를 통과했다. Peak20.236GiB다.
이것은 첫 lazy initialization이 cold RNG 검사에 섞인 문제를 해결했다는 실제 근거다.
Micro12 capacity나 본학습, 품질 개선은 이 두 micro1 실행으로 입증하지 않는다.

이후 Feature/raw 모드도 각각 같은 검사를 통과했다. Source19개를 포함한optimizer379개가
실제로3회 update됐으며 두모드 모두peak20.275GiB다. 네모드CPU gate는 초기LoRA,
공통R10, TRAIN9139 순서·입력과3개noise/timestep/global trace가 같음을 확인해완료했다.
네 모드의cold32와 DDP2/micro12 capacity는 아직 남아 있다.
원래 FAILED/HELD state는 바꾸지 않는다. 실제 v2 micro proof의 새 CPU gate는 기존과
동일한 의미의 통합 검사를 통과해야 하며, 이후 아직 실행하지 않은16개 작업은
원래 plan/command/output을 유지하는 새alias로 연결했다. 본학습 config와 예산은 같다.
접속 기록은 `PBR-native-source-micro1-v1/NAF-recovery-{prelaunch,queue}-v1`에 있다.

## 새 PBR texture replay

`replay_r_pbr_on_generated_shape.py`는 기존 vanilla/retargeting/direct loader에
새 source weight를 끼워 넣지 않는다. 다음 네 own-schema mode를 엄격히 구분한다.
Vanilla, physical Top-1+R(source10), Top-1+feature+R(source19), raw Top-K+feature+R(source19)이다.

`original16850-engineering`은 원본complete NN과 새cold source만 허용한다.
`trained-native1900`은 실제 stage-completed 기록과 같은run의final1900 NN/source pair,
frozen base/matcher/anchor 및 DDP 검사 완료가 필요하다. 임시3-update weight는 거부한다.

입력row는 RGB/camera/cache의12개 field만 허용한다. 실제 RGB fingerprint를 확인하고,
실제 predicted shape coords에서 target patch 주소를 재계산한다. Source-camera ray별
member matching과 DINO+NAF2048 전달은 실제 sparse trainer consumer를 재사용한다.
GT shape/PBR latent, GT mesh, projection 또는 legacy matching index는 입력에 쓰지 않는다.

Native PBR sampler와 CPU texture-noise 규칙을 유지한다. Frozen DINO/NAF는 seed reset 전에
로딩하며 추가 conditioning이 sampler RNG를 소비하면 실패한다. Geometry의8개 array는
dtype/order/bytes를 유지한다. Native Shape VAE는 texture guide hierarchy에만 사용하고,
새로 decode한 mesh로 원래 geometry를 교체하지 않는다. 새 prediction을 저장한 뒤에만
기존 geometry evaluation을 복사한다. Exception 시 pipeline hook을 복구한다.

Cold proof는 네 mode의 실제 첫noise, texture latent와 decoded coord/attribute SHA가
정확히 같아야 한다. Trained 비교에서는 geometry와 초기texture noise가 같아야 하되,
예측texture는 같을 것을 요구하지 않는다. 관련 CPU 검사61개가 통과했다.

## 실제32개 CPU 입력 검증과 연결된 범위

기존 고정32개 모두에서 실제 generated archive의 canonical receipt와 sample/calibration,
native32-channel shape support를 확인했다. Image-only descriptor loader를 실행해
현실의 predicted support에서 target 주소를 계산하고, GT cache route를 쓰지 않았다.
이 CPU 입력 검증과168개 plan/row/log 파일4,237,910 bytes를 실제file4와 대조했다.

4개cold sampler proof + 4모델×32개trained sampler replay + 2개CPU coverage/parity gate를
연결했다. Trained128개는 실제1900 학습 완료와cold sampler gate를 모두 기다린다.
모두 같은V1 generated geometry를 고정한 **PBR isolation**이다. 새SS geometry와 PBR을
함께 바꾸는full-stack 비교나 렌더링/source preservation metric 완료는 아니다.
렌더링과 공식DreamSim/CLIP, source/target 역할별 집계는 후속으로 연결해야 한다.

처음 준비 중 기존background activator가16개faithful-target-ray 작업을 추가해 queue
snapshot이 달라졌다. 첫connector는 준비물을file4에 닫은 후 atomic queue guard에서
중단했다. 준비물은 유지하고 별도v2에서 최신queue에134개를append했다. 원래 scope와
동시에 추가된16개도 보존했다. 새trained 계획의dependency는NAF-recovery science alias로
연결했다. `PBR-native-generated-replay-v1/{prelaunch-v1,queue-v1,queue-v2}`에 근거가 있다.

PBR 두 번째 쌍도 정상1900 종료 및canonical archive 이후 자신의unused old epoch
로컬중복만 정리하도록2개CPU 작업을 추가했다. 원본모델·데이터·final1900과file4는 유지한다.
오래된faithful-target-ray3개 science는 이PBR 완료/정리가 끝난 뒤 실행하도록dependency만
조정했다. 삭제하거나 완료 처리하지 않았다. 이후faithful3개 자체의checkpoint 증가도
별도저장예산/정리로 관리해야 한다. 현재 queue는2804개다.

SS feature/raw의결정적epoch96 독립load2회 및불활성observer는 둘 다 실제완료했다.
원래 비결정적학습기록의exact replay 실패와는 구분한다. SS final480/128×3,
독립TRAIN seed202, 모듈기여분리, 새SS/PBR full-cascade, final508과전체10metric은 남아 있다.
이 보고서의engineering 성공을모델성능향상이나목표완료로 바꾸지 않는다.
