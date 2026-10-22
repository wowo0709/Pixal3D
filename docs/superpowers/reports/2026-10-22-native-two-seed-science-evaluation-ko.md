# Native SS seed202 본 학습부터 두 학습 seed의 최종 비교까지 연결

Seed101에서 관측한 Top-1+retargeting의 V1 대비 개선을 확인하기 위해,
기존 seed202 engineering4건에 이어14개 후속 작업을 추가했다. 현재
진행 중인 PBR vanilla 학습이나 기존 작업의 bound 설정은 수정하지 않았다.
새 작업은 실제 선행 결과가 생겨야 실행되며 아직 seed202 성능 결과는 없다.

## 실행 순서

1. 두 모델의 실제 proof/capacity 결과를 CPU에서 검증한다. Source0/10,
   LoRA360, cold native 동등성·backward, 초기32 평가, micro96 세 update,
   Adam360/370·step3, frozen 상태와 두 모델의 실제 asset/noise/timestep/
   global trace 일치를 모두 요구한다. Capacity 가중치는 버리고 시작한다.
2. 실제 원본 SS4210에서 fresh optimizer로 vanilla와 physical Top-1+R을
   각각480 update 학습한다. TRAIN9149/global96/micro96/accumulation1,
   내부 FFN chunk24, 기존 learning rate와 dropout, 매96 update 고정32
   validation 및 W&B를 유지한다.
3. 실제 완료·canonical checkpoint 기록을 확인하고 최종480의 NN/source
   pair를 seal한다. 각 모델에 동일128개·세 generation noise를 적용한다.
4. 실제 seed101 결과와 함께 총2 training seeds×2 models×128 assets×
   3 generation noises를 분석한다. 두 seed의 결과와 방향을 각각 보고한다.

Source cluster 단위 paired bootstrap4000회를 사용하며, 각 asset에서 세
generation noise를 먼저 평균한다. 고정한 두 training seed의 평균 효과도
보고하지만 이 CI는 asset source cluster만 resample한 값이다. 두 번의 학습으로
학습 seed 전체 모집단의 변동성을 추정했다고 해석하지 않는다.

두 모델의480개 실제 학습 trace는 같은 seed 안에서 완전히 같아야 하고,
seed101과202 사이에는 실제 stream이 달라야 한다. 원래 LoRA 초기화가
네 실행에서 같고, 최종 네 denoiser의 frozen base tensor가 원본4210과
동일한지도 검사하도록 했다. 마지막 검사는 CPU에서 checkpoint tensor를
비교하며 CPU neural forward나 training은 하지 않는다.

새 집계 검사는 잘못된 seed/config, 다른 NN 경로·GT·noise, 누락된 평가,
바뀐 learning rate, 비결정적 평가, 같은 학습 stream을 재사용한 가짜 반복을
거부한다. Seed202의 효과가 음수인 synthetic fixture도 음수로 그대로 남긴다.
관련 새 집계 및 seed recipe CPU 검사19개가0.30초에 통과했다.

## 자원과 provenance

각 과학 학습은 실제 여유 공간을 관찰하고16 GiB의 checkpoint 증가량과
8 GiB workspace를 atomic하게 예약한다. PID가 사라졌다는 이유로 예약을
자동 해제하지 않는다. 실제 native wrapper의 canonical 완료 receipt를
확인해야 해제한다. 오류가 나면 기록과 예약을 보존하고 자동 재학습하지 않는다.

네 PBR1900의 학습·cache 정리, 역사적 query 재현 세 arm의 완료·cache 정리,
새 native full-cascade DEV32의8개 지표 요약 이후 SS202 본 학습을 시작한다.
두 SS202 모델은 순차로 실행하며, 첫 모델의 오래된 local checkpoint 사본을
실제 file4 SHA와 reader 검사 후 정리하고 다음 모델로 진행한다. File4에는
다섯 checkpoint 모두 남기고 local 최종480 및 원본 모델·dataset도 유지한다.

현재14개 새 작업은 pending이다. Dependency DAG3502개 전체와 새 실행
명령의 self checksum, 존재하는 input checksum을 점검했다. 기존4건과 합쳐
seed202 관련18개 단계가 연결됐으며, queue-state를 변경하지 않았다.

계획과 queue snapshot은 `native-flow-cold3-v1/independent-seed202-v1/`
`science-evaluation-prelaunch-v1/` 및 `science-evaluation-queue-v1/`에 있다.
전자는15파일740337 bytes, 후자는2파일7676539 bytes의 실제 file4 SHA/size를
대조했다. 연결·CPU 검사 통과는 실제 seed202 GPU 검증이나 성능 향상의
완료를 의미하지 않는다. 최종 test508 adapter·모델 동결·10개 지표 평가는 남아 있다.
