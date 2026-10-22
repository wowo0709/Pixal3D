# R-mediated 두 후보의 실제 GPU 검증과 학습 시작

[설계·구현](2026-10-22-R-mediated-source-pilot-design-ko.md)의 두 후보를 각각
GPU0/1에서 검증하고 TRAIN1536/7에폭112step의 source-only 비교 학습을 시작했다.
본 기록은 frozen SS4210에서 source parameter19개를 학습하는 pilot이며,
NN LoRA joint 학습 또는 full 모델 성능 향상을 완료한 결과가 아니다.

## 완료된 실제 검증

CPU 검사19개를 통과했다. 두 GPU의 actual micro96 cold backward와 별도의
disposable3회 TRAIN 업데이트에서도 첫 head gradient0, 세 번째 backward의19개
finite nonzero gradient, scientific0step state 불변을 확인했다. 추가 probe를 포함한
최대 allocated 메모리는 feature36.88GiB, raw Top-K37.41GiB였다. Probe는 버렸으며
학습 초기 weight나 optimizer에 이어 붙이지 않았다.

두 후보의 actual native32 cold evaluation에서 global/projection/negative/RNG와
원래 GT/noise/voxel count, four-head matching prior가 정확히 같았다. Consumer
residual이 실제 R 출력과 같고 direct head correction이 없다는 것도 확인했다.
CPU gate는 실제 capacity0step weight의19개 값이 기존 bounded control과 같은지
대조한 뒤 학습만 허용했다.

실제 첫 TRAIN batch probe의 세 번째 backward에서 query-weight gradient L1은
feature8.23e−8, raw Top-K6.67e−6이었다. Raw Top-K에서는 두 번째 backward부터
Q/K gradient가 열렸다. 이는 raw value를 R에 제공하면 probability의 gradient 경로가
달라진다는 관측이다. GT matching 또는 최종 생성 성능이 향상됐다는 뜻은 아니다.

## 학습 로그와 초기 경향

최초 실행은 이전 bounded 모델의 W&B ID를 복사한 설정 오류 때문에 `resume=never`
검사에서 optimizer 업데이트 전에 중단됐다. 학습 log/optimizer state가 없고,
저장된0step tensor가 capacity0step과 정확히 같음을 확인해 실패를 보존했다.
기존 ID를 재사용하거나 덮어쓰지 않고 후보별 고유 W&B ID와 `train-v2` 경로로
successor를 연결했다. Source·data·LR·batch 순서·native 평가 조건은 바꾸지 않았다.

| 동일 개발32개,16step | Native voxel IoU |
|---|---:|
| R-mediated feature | 0.4263159 |
| R-mediated raw Top-K | 0.4270075 |
| 기존 bounded source-contrast16step | 0.4276772 |

Cold0step은 두 후보 모두 기존 frozen SS4210 함수와 정확히 같았다. 위 수치는
실제 업데이트 이후의 초기 경향이며 초기 loader/정합 오류를 뜻하지 않는다.
현재 두 후보가 기존 bounded control보다 낫다고 판단하지 않는다. 고정112step,
동일112개 batch·19개 초기 parameter·matching support에서 세 모델을 비교한다.

실제 학습 중 peak allocated는 feature31.52GiB, raw Top-K32.09GiB였고
reserved는 약35.75/35.92GiB였다. Global96/micro96/accumulation1이며 이 기록까지
OOM이 없다. 매 에폭 native32 수치와 voxel panel을 기록한다.

[Feature W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-R-mediated-feature-TRAIN1536-7ep-s101-v2),
[Raw Top-K W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-R-mediated-raw-topk-TRAIN1536-7ep-s101-v2)에
연결돼 있다. 학습 후 CPU cached-feature normal forward로 동일 proposal/GT에서
matching set mass와 source feature cosine/MSE를 재고 native 결과에 연결하도록 했다.

Canonical 결과는 file4의 `rgb_matching_v2/semantic-topk-search-v1/`
`ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/`
`source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/`에 보관한다.
`prelaunch-v1`의 실제 mechanics 증거와 `training-recovery-prelaunch-v2`의
고유 logging ID 수정 기록, 실패 receipt를 분리해 보존한다. 기존 source72 HOLD,
전체 PBR 렌더·metric, zero-shot 두 baseline과 joint/독립 seed/test508 scope를 유지한다.
