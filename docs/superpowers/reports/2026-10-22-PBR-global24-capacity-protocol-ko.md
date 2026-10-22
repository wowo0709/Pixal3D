# Native PBR global24의 최대 microbatch 검증

새 source 구조를 원본 multi-view PBR16850의 complete base+LoRA에서 시작한다.
첫 배치가 메모리에 들어가는 것만으로 본 학습을 허용하지 않는다.
기존 micro1 및 cold generated32 검사가 실제 완료된 뒤, 별도 임시 실행에서
GPU2장·rank당 micro12·accumulation1을 측정한다. 입력은 선언된 실제 TRAIN의
31900-token asset을 rank당12개 반복한 최대-token 배치다. 작은 asset이나
줄어든 token 수로 대신하지 않는다.

`probe_r_mediated_pbr_capacity.py`는 native sparse flow loss와 fresh AdamW,
360 LoRA 및 source0/10/19 tensor를 사용한다. Native CFG dropout.1을 유지한다.
세 번 fullbatch update 후 rank0만 고정4개 anchor-camera PBR 렌더링 validation을
진행하고, 이후 다시 fullbatch update를 수행한다. 평가할 때 source와 denoiser의
DDP wrapper를 process-local로 풀어 다른 rank의 collective forward 대기를 피한다.
평가 종료와 예외 시 원래 wrapper·모델 mode·dropout·Python/NumPy/Torch RNG를 복원한다.

각 rank의 실제 optimizer state와 step4, source gradient, 변경된 LoRA/source 수,
frozen base·matching decoder/score·초기 attention anchor의 불변성을 검사한다.
갱신 후 DDP parameter 동등성을 실제 collective로 확인한다. 네 update의 실제
noisy latent·timestep·global condition과 원래 RGB/shape/PBR input hash도 기록한다.
각 arm의 peak allocated/reserved 메모리와 반복 배치 update 시간을 남긴다.

이는 최대-token 메모리·validation 후 재학습 가능성 검사다. 반복 배치 시간을
일반 데이터 throughput으로 해석하거나 네 임시 update를 본 학습 에폭에 포함하지 않는다.
검사용 weight/optimizer checkpoint를 export하거나 본 학습으로 이어 쓰지 않는다.
두 GPU에서 OOM이면 실제 실패 로그를 보존하고 새 실행에서 GPU4장·micro6을 검토한다.
코드는 두 배치를 명시적으로 지원하지만 실제 GPU 결과 전에는 어느 쪽도 가능하다고
확정하지 않는다. 기존 정상 SS 학습이나 그 runtime 파일은 수정하지 않는다.

초기 동등성과 capacity 통과 뒤에도 실제 데이터 순서·빠른 validation·W&B를 갖춘
fresh PBR 본 학습과 SS 전달을 분리하는 full-cascade 평가가 남는다.
