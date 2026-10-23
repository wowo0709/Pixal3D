# SS 4-epoch 비교와 선택 모델의 독립 학습 seed 집계

UTC 2026-10-23. `wowo0709/feature-retargeting`.

## 새 SS 4-epoch 결과

두 projected 대조군이 384 step validation을 마쳤다. 같은 DEV32 asset·GT·source cluster·generation seed의 raw-weight **asset 평균 voxel IoU**다. pooled occupancy IoU와 구분한다. 다섯 모델의 학습 전 asset별 기록은 모두 동일하다.

| 모델 | 1 epoch | 2 epoch | 3 epoch | 4 epoch |
|---|---:|---:|---:|---:|
| V1 | **0.46380** | **0.46745** | **0.46897** | 0.47070 |
| Decoder Top‑1+R | 0.46162 | 0.46088 | 0.46799 | 0.47258 |
| Top‑1+V/output+R | 0.46101 | 0.46472 | 0.46598 | 0.47150 |
| Fixed score Top‑K+R | 0.46082 | 0.46530 | 0.46761 | 0.47401 |
| Full learned attention+R | 0.46097 | 0.46459 | 0.46649 | **0.47481** |

4 epoch에서는 Fixed Top‑K가 Top‑1+V보다 +0.002511, Full이 Fixed Top‑K보다 +0.000801 높다. 3 epoch에서 Full이 Fixed Top‑K보다 낮았던 순위가 바뀌었다. 같은 초기 native 함수에서 시작한 뒤 학습 경로에 따라 격차가 달라졌다는 관측이다. 초기 함수 자체가 손상된 결과와 구분한다.

현재 패널의 source cluster 32개를 4,000번 paired bootstrap한 구간은 다음과 같다. 중간 epoch를 반복 관찰한 결과로, 최종 확인 검정이 아니다.

| 비교 | 평균 IoU 차이 | Source-cluster 95% 구간 |
|---|---:|---|
| Full − Fixed Top‑K | +0.000801 | [−0.001349, +0.003031] |
| Fixed Top‑K − Top‑1+V | +0.002511 | [−0.000488, +0.006379] |
| Full − V1 | +0.004109 | [−0.005657, +0.013811] |
| Full − Top‑1+R | +0.002234 | [−0.004164, +0.009039] |

평균에서는 Full이 앞서지만 모두 0을 포함한다. 계획된 module family 10/pipeline family 2만 반영한 구간도 0을 포함하며, 반복 중간 관찰까지 보정한 구간은 아니다. 따라서 attention이나 Top‑K의 필요성이 통계적으로 입증됐다고 주장하지 않는다.

이미 완료된 별도 DEV128×3의 V1/Top‑1+R/Full 값은 각각 0.467355/0.472600/0.472078이었다. 그 최종 비교에서는 Full이 Top‑1+R보다 낮다. DEV32 4-epoch 숫자와 DEV128×3 최종 숫자를 같은 곡선으로 이어 붙이지 않는다. 새 대조군의 480-step DEV128×3 및 나머지 모듈 비교를 완료해야 한다. 현재 checkpoint 예산과 평가 조건은 변경하지 않았다.

## 선택 모델의 독립 seed 집계 코드

`module_training_seed_summary.py`를 추가했다. 미리 정한 12개 physical family 중 실제 선택된 후보와 V1·Top‑1+R을 비교한다. 후보가 두 기준선 중 하나이면 중복을 제거해 두 모델만 집계하며, 나머지 경우 세 모델을 집계한다. 12개 전체를 추가 학습하는 기능은 없다.

주요 검사와 출력은 다음과 같다.

- Training seed101/202의 실제 480-step asset96·x_t·timestep·global trace가 각 seed 내 모든 모델에서 같고, 두 seed 사이에서는 실제 RNG/학습 stream이 달라야 한다.
- 초기 LoRA는 동일한 원래 checkpoint이며 constructor RNG와 TRAIN manifest 순서도 seed 내에서 같아야 한다.
- 각 모델의 실제 native480 경로·physical source count·LoRA360·raw weights·batch8/12 sampling steps/CFG3·DEV128·서로 다른 generation noise 세 개·GT/cluster pairing을 검사한다.
- 세 generation-noise 평가의 config/denoiser/source SHA가 같아야 한다. 평가 중 optimizer update, EMA 혼입, 다른 sampler, 불완전한 trace나 결과를 거부한다.
- Training seed별 평균·paired source-cluster 구간을 따로 내고, 두 seed를 고정한 평균 효과도 별도로 낸다. 두 학습 seed만으로 학습 seed 모집단의 신뢰구간을 만들지 않는다. Generation noise 세 개를 세 번의 학습으로 계산하지 않는다.
- 평균 효과가 양수여도 seed202의 효과가 음수이면 양 seed에서 재현됐다는 표시는 false로 남긴다. 자동 모델 채택은 하지 않는다.

직접 검사 **46개가 통과**했다. 실제로 완료된 seed101 V1·Top‑1+R·Full의 9개 DEV128 평가와 480-step trace/초기 RNG에 새 입력 검사를 적용했다. 세 모델의 저장된 config/denoiser/source bytes가 각 generation-noise 보고서의 SHA 및 실제 file4 closure와 같은지 확인했다. 원래 SS4210의 frozen denoiser tensor도 세 최종 checkpoint와 CPU에서 데이터 비교했다. CPU neural forward나 학습은 하지 않았다.

이것은 실제 seed202 재현 결과가 아니다. seed202 기록을 복제·합성하지 않았고, 후보나 epoch를 선택하지 않았다. 실제 두 seed를 읽는 CPU 집계 job과 최종 DEV freeze의 연결은 후보 선택 뒤에 수행해야 한다. Frozen checkpoint byte 비교와 학습 완료 증거는 통계 함수 외부의 실제 입력 검사에서도 요구된다.

산출물은 `unified-module-pipeline-v1/selected-SS-two-seed-summary-v173/preflight-v1`과 `byte-verification-and-interim-CI-v1`에 보존한다. 새 GPU job은 추가하지 않았다. 진행 중인 SS·PBR 학습, 핵심 외 구 분기 HOLD, GPU 자원 전환 controller를 유지했다. 최종 cascade·독립 seed·DEV freeze·TEST508 검증은 계속 남아 있다.
