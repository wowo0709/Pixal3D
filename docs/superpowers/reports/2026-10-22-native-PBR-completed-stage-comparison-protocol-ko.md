# Native PBR 완료 시점의 V1 비교 자동화

현재 진행 중인 native PBR vanilla와 이후 source 세 arm이 각각1900 update를
완료하면, 최종 checkpoint와 고정4개 validation을 CPU에서 검증·보관하고 V1과
순차 비교하도록 연결했다. 새로운 GPU 학습이나 rendering은 추가하지 않았다.
기존3536개 전체 DAG의 연결을 확인했고, 기존 학습·실험 descriptor와 queue-state는
수정하지 않았다(기존3529개에 CPU7개 추가).

## 실제 완료만 수집한다

1900 update 직후 학습 PID가 종료될 수 있으므로, 이 수집기는 PID가 계속 살아
있을 것을 요구하지 않는다. 대신 원래 소유 child의 PID/birth가 같은 실제
terminal0, 원래 science job의 completed/returncode0, native·wrapper의 canonical
receipt, 네 rank의 실제 완료 기록과 final checkpoint archive를 모두 확인한다.

각 rank에서 original PBR16850·fresh optimizer·seed101·TRAIN9139·1900 updates,
world4/global24/micro6/accumulation1, native360 LoRA와 source0/10/19 optimizer
상태 수, frozen base/matcher/reference와 DDP 일치 검사가 통과해야 한다.
Capacity4-update 결과나 중간 epoch 파일만으로 완료를 처리할 수 없다.

원래 NN/source/Adam/config를 읽기 전용으로 확인한다. 기존 cache 정리가 완료된
모델의 non-final 로컬 checkpoint를 지울 수 있으므로, 이 수집기는1900 final
checkpoint와 작은 완료·validation·runtime 기록만 읽는다. 삭제된 과거 .pt를
다시 요구하거나 기존 학습을 재시작하지 않는다.

## 같은 조건의 작은 panel을 비교한다

각 모델의0/380/760/1140/1520/1900 실제 validation을 모두 수집한다. 동일4개
sample/source cluster·RGB·noise·GT PBR latent·GT Shape condition·camera가
유지됐는지 확인하고, PNG의 GT pixel 영역도 실제 byte hash로 대조한다.
모델 간 cold step0의 prediction latent hash와 metric도 같아야 한다.

각 rank의 전체1900-step 실제 asset/noise/timestep/native-global trace를
canonical JSON digest로 비교한다. Source correction에 따라 바뀌어야 하는
projection feature를 같다고 요구하지 않는다. Trace가 다르면 matched 비교로
통과시키지 않고 원래 결과와 실패를 보존해 원인을 조사한다.

V1/Top-1+R이 끝나면2모델 비교, feature+R까지 끝나면3모델 비교, raw Top-K까지
끝나면4모델 비교를 생성한다. V1뿐 아니라 강한 Top-1+R 대비 추가 효과도 보고한다.
모든 epoch의 pooled-MSE PSNR과 mean-asset PSNR(dB), LPIPS·crop LPIPS 등을
구분하고1900의 per-asset 차이와 GT/Pred/Diff 이미지를 W&B youngwoojeon-unist에
기록한다. 음성 결과도 그대로 남기며 이 작은4개 panel에 유의성 CI나 자동 모델
채택 결론을 붙이지 않는다.

이는 **GT Shape가 고정된 target camera8 stage 추세**다. 논문의 source-preservation
LPIPS/DreamSim/CLIP이나 generated geometry/full-cascade 평가가 아니다.
기존 isolation128·native full-cascade 개발32·최종508의 올바른 role별8/10 metric은
별도로 계속 필요하다.

신규 CPU 검사26개가0.07초에 통과했다. 현재 실제0~1520 validation의 집계를
새 코드로 다시 계산해1e-12 이내로 재현했고, launch/config와 GT PNG pixel의
epoch 간 일치도 확인했다. 이 검사는 아직 끝나지 않은1900 학습의 증거가 아니다.

Canonical 제어 경로는 PBR-native-science-runtime-v1 아래
`completed-stage-quick4-control-v1/{prelaunch-v1,queue-v1}`이다.
Prelaunch13개95,194 bytes와 queue2개7,751,297 bytes를 actual file4 SHA/size와
대조했다. 각 모델의 향후 snapshot은 `{mode}-stage1900-completed-evidence-v1`,
비교는 `completed1900-{2,3,4}models-fixed4-comparison-v1`에 생성된다.
