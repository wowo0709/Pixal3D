# Native SS+LoRA 3조건: 실제 증명 통과 및 학습 시작

2026-10-22 UTC. 기존 feature-teacher 결과 대신 실제 SS flow objective에서 비교하는 새 실험이다.
Vanilla, R_feature, R_raw_topk 세 조건 모두 actual proof, micro96 capacity, CPU admission을
통과했으며 fresh scientific 학습을 시작했다. 실패한 teacher112 가중치는 사용하지 않았다.

## 실제 확인한 사항

- 원본 multi-view SS4210의 frozen base+LoRA를 이어받았고, 초기 360 LoRA tensor의 hash가 세 조건에서 같다.
- 두 R 모델은 같은 19 tensor 초기값으로 시작한다. 추가 source constructor가 native RNG를 바꾸지 않는다.
  실제 post-constructor Torch CPU/CUDA·NumPy·Python RNG가 세 조건에서 일치했다.
- 같은 TRAIN sample에서 native global/projection/negative와 같은-noise velocity/flow loss가 일치했다.
  실제 SS loss가 원래 LoRA와 R/fusion에 도달했다. Cold R의 첫 head gradient는 0이며, 별도 fresh
  disposable 모델의 세 번째 native update에서는 source19 gradient가 모두 finite/nonzero였다.
- 세 capacity에서 실제 첫 세 update의 96개 asset ID, noisy latent, timestep, global feature hash가 같다.
  모두 정확히 같은 9,149개 TRAIN case와 순서를 확인했다. Transport 때문에 asset이 필터링되지 않았다.
- Initial native32를 capacity와 scientific 시작 전에 각각 평가했다. 세 조건 모두 원본 MV4의
  기록된 GT/noise/occupancy count/metric과 같고 IoU는 **.4267154667**이다.
  Raw voxel mask는 보관하지 않았으므로 mask byte identity라고 주장하지 않는다.
- 원래 frozen parameter/buffer가 같고, 실제 native AdamW/AMP/LoRA update가 실행됐다.
  Source head 직접 bypass, GT matching 입력, per-voxel visibility write gate는 없다.

새 구현의 CPU 회귀 검증은 **23개 통과**했다. 실제 GPU 증명은 위 항목의 별도 근거이며
CPU 통과만으로 native 학습 가능성을 주장하지 않았다.

## 측정한 메모리와 시간

| 조건 | global/microbatch | accumulation | capacity peak allocated | max reserved | 안정 구간 update 시간 |
|---|---|---:|---:|---:|---:|
| Vanilla | 96 / 96 | 1 | 78.37 GiB | 82.79 GiB | 약 34.6초 |
| R_feature | 96 / 96 | 1 | 90.46 GiB | 93.38 GiB | 약 45.9초 |
| R_raw_topk | 96 / 96 | 1 | 90.46 GiB | 93.72 GiB | 약 46.7초 |

각 조건은 GPU 한 장에서 전체 batch96으로 학습한다. FFN 내부 checkpoint chunk24를 사용하며
gradient accumulation이 아니다. 실제 initial validation 뒤 fullbatch backward/update 세 번까지
검증했으며 OOM 없이 통과했다. Chunk12 대체나 batch 축소는 사용하지 않았다.

현 속도가 유지될 때 480 updates와 epoch validation을 합쳐 Vanilla는 약 4.7시간,
두 R 모델은 약 6.3~6.5시간 규모다. 첫 실행 warmup, checkpoint 저장·file4 전송 등에 따라 달라질 수 있다.
세 모델은 동시에 학습한다. 이번 launch의 물리 GPU는 Vanilla=1, R_feature=0, R_raw_topk=3이고,
GPU2는 기존 cascade 평가에 사용한다. 정상 진행 중인 기존 작업은 재시작하지 않았다.

## 학습 예산을 해석하는 방법

고정 예산은 **480 optimizer updates**, validation/checkpoint 간격은 96 updates다.
기존 recipe의 “5에폭”은 이 96-step 단위 다섯 번을 뜻한다. Native DataLoader는
TRAIN9149/world1/batch96에서 `drop_last=True`이므로 한 loader cycle은 95 batches다.
따라서 480 updates는 loader 다섯 cycle과 다음 cycle 다섯 batch이며, 문자 그대로 데이터 다섯 순회와
동일하다고 표현하지 않는다. 세 조건은 동일한 46,080개 sample presentation 예산을 사용한다.
이 설정을 바꾸지 않고 기존 native optimizer/샘플링 계약과 공정한 고정-step 비교를 유지한다.

각 96-step 구간이 끝나면 같은 native32/12-step/seed map으로 voxel IoU·precision·recall·F1과
GT/Pred/Diff 이미지를 기록한다. 초기 기록도 이미 존재한다. 느린 rendering metric은 SS epoch 평가에
추가하지 않았다. 실제 noisy input trace는 학습 내내 기록하고, source query/key/value/output gradient도
10 updates마다 기록한다. Frozen Q2C gradient는 0이어야 한다.

## W&B와 canonical 결과

- [Vanilla](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-flow-cold3-vanilla-s101-v1)
- [R_feature](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-flow-cold3-R_feature-s101-v1)
- [R_raw_topk](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-flow-cold3-R_raw_topk-s101-v1)

File4의 기존 `rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/`
아래 `source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/native-flow-cold3-v1/`에 저장한다.
`admission-v1/gate.json`은 **새 matched task 실험의 engineering PASS**다. 과거 source72의 quality HOLD를
수정하거나 모델 성능을 입증한 PASS가 아니다. 각 `{variant}/science-v1/native/`가 실제 학습 결과 경로다.
Prelaunch/proof/capacity/admission은 실제 file4 SHA/size 검증을 마쳤으며, live 출력은 mirror로 동기화한다.
최종 scientific 완료와 canonical closure는 480 updates가 끝난 후 확인해야 한다.

현재는 학습이 올바르게 시작됐다는 증거이며 모듈 성능 향상 결과가 아니다.
기존 V1 seed101은 world2/micro48이라 새 world1/micro96에 완전히 같은 학습 noise의 control이 될 수 없어
새 vanilla를 함께 실행한다. 기존 V1/teacher/Full-R 평가 결과는 그대로 보존한다.
실제 task 모델의 비교, 필요한 retrained ablation·독립 seed·PBR/cascade·final508/10 metric은 남아 있다.
