# 핵심 SS 대조군의 GPU 0–3 병렬 실행 배치

UTC 2026-10-23 기준. 브랜치 `wowo0709/feature-retargeting`.

현재 PBR feature 학습은 GPU 0–3에서 그대로 완료한다. 그다음 대기 중인 SS 핵심 대조군 네 개를 각각 한 GPU에 배치했다. GPU 6·7에서는 현재 SS 두 모델과 나머지 핵심 대조군을 이어간다. 실험 arm 수는 12개로 유지하며, 새 학습 arm이나 parameter sweep을 추가하지 않았다.

| GPU | 현재 PBR 학습 완료 후 실행할 SS arm | 검증 목적 |
|---|---|---|
| 0 | `C_RAW` | RAW semantic correspondence |
| 1 | `C_SPATIAL` | 원래 spatial correspondence |
| 2 | `C_D1_COS` | scalar score head 없이 matching decoder만 사용 |
| 3 | `C_DECODER_NO_R` | Top-1/use/fusion을 유지하고 R을 물리적으로 제거 |

각 arm은 기존과 같은 global/microbatch 96, accumulation 1, 5 epoch/480 step이다. 모델·LoRA 초기 checkpoint·LR·seed·학습 데이터·validation·sampling·출력 경로를 유지했다. 변경한 설정은 실제 실행 GPU 목록, CPU affinity, 실행 recipe의 파일 경로 세 항목이다. 원래 config와 새 config의 나머지 JSON 값과 type까지 12개 phase별로 같음을 확인했다.

모델 코드를 새 GPU에 맞춘다는 이유로 변경하거나 CUDA device 정보를 위장하지 않는다. 기존 checksum으로 고정된 runtime을 그대로 사용하고, GPU 0–3이 명시된 별도 실행 config를 만들었다. 설정 보존 검사 18개가 통과했다. 아직 새 GPU에서 초기 parity와 용량 검사를 수행한 것은 아니다.

실행 순서는 각자의 실제 초기 native 동등성 검사 → microbatch 96의 임시 3-update 용량 검사 → 원래 reference와 초기 가중치·RNG·학습 asset·noise/timestep trace 비교 → 480-step 학습이다. 새 GPU에서도 이 검사를 통과해야 본 학습을 시작한다.

기존 학습·평가 의존 관계를 유지하기 위해, 실제 GPU 0–3 작업의 parent/child 종료 및 file4 artifact를 확인하는 CPU 연결 작업을 두었다. CPU 연결 작업은 neural forward나 학습을 하지 않으며, 실제 GPU 학습을 대신해 완료 상태를 만들어내지 않는다. 최종 checkpoint 로더는 원래 scientific recipe와 실행 config의 차이가 위 세 항목에 한정되는지 다시 검사한다.

네 SS 작업이 GPU를 사용한 뒤 대기 중인 raw-Top-K PBR 학습에 GPU 0–3을 돌려준다. raw PBR의 모델·config·batch·command는 변경하지 않았다. GPU 자원 반환은 모델 품질 판정과 분리했다. 특정 SS arm이 검증에 실패하면 해당 실패를 보존하고, 그 실패 때문에 독립적인 PBR 학습까지 영구히 기다리게 하지 않는다.

진행 중인 PBR 및 GPU 6·7의 SS 학습을 중단·재시작하지 않았다. queue state를 직접 수정하지 않았고, GPU 4·5는 사용하지 않는다. 기존의 지엽·중복 작업 보류도 유지했다.

이 배치는 SS 대조군을 GPU 두 장에서만 순차 실행하던 대기를 줄이기 위한 것이다. 실제 GPU 0–3의 처리 시간과 peak memory는 용량 검사 후 확인해야 한다. 최종 생성 성능 향상이나 전체 실험 완료를 뜻하지 않는다. 최종 SS 비교, 공통 PBR cascade의 8개 metric, 독립 seed 확인, 올바른 동결 이후 TEST 508개와 FID/KID 평가가 계속 남아 있다.
