# Ray Top-K 조합을 위한 다중 대응점 supervision

## 질문과 관측 근거

목표는 정확한 Top-1 분류가 아니라 semantic Top-K에 들어온 여러 유효 대응점을 SS 생성에 유용하게 조합하는 것이다. 동일 Full480/32개 개발 샘플에서 관측 가능한 서로 다른 (source-camera ray, target patch) 관계를 세었을 때, frozen D1/scalar Top-1의 exact coverage는 8.57%, 동일 후보 64개의 coverage는 50.77%였다. 같은 view의 한 patch 이웃까지 허용하면 28.32%와 80.48%였다. 이 수치는 SS 정확도가 아니며 후보 누락도 여전히 남는다.

기존 learned utility는 prior보다 정답 후보 집합의 평균 확률 질량을 늘리지 못했다. 고정 patch-score pooling도 saved-weight SS32에서 Full 대비 IoU -0.215 percentage points였고, source-cluster bootstrap 95% 구간은 [-0.595, +0.109] pp였다. 단순 확률 확산을 새 장기 학습 arm으로 채택하지 않는다.

후보 조합 이후 이미지 condition을 쓰는 위치도 별도 검사했다. 입력 M1 RGB alpha 밖을 native condition으로 유지하는 `mask-all`과 그곳의 projected R만 제거하는 `mask-R`은 각각 SS32 IoU 0.438307, 0.438289였다. 원래 Full 0.436073 대비 +0.223 pp와 +0.222 pp이나, paired 95% 구간은 각각 [-0.699, +1.097], [-0.674, +1.047] pp다. 이 counterfactual은 버그 수정이나 모델 개선의 증거로 채택하지 않는다. 모든 비교는 학습 seed101, 개발32개, 생성 noise0의 탐색 결과다.

## 새로 고정한 소규모 학습

- DeformNeXt world-v4 train에서 category round-robin/seed101로 source가 다른 512개를 선택한다. 기존 head calibration source 179개를 전부 제외한다. Calibration은 train의 별도 32개이며 dev128과 sample/source가 겹치지 않는다.
- 원래 Full480의 utility head 5개 tensor만 초기화해 7에폭 학습한다. SS LoRA, retargeting adapter, fusion, D1 decoder, scalar score는 동결한다. 기존 6개 SS arm은 변경하지 않는다.
- Model forward는 cached RGB DINO ray query, Omega context, 후보 DINO feature, RGB alpha validity, frozen scalar prior만 받는다. GT correspondence는 별도 loss NPZ에만 둔다.
- Ray 안의 관측 가능한 각 target patch마다 같은 view/near1인 positive 후보들의 **합산 확률**에 loss를 준다. 각 patch의 loss를 ray 안에서 평균한 뒤 ray 간 평균한다. 한 ray의 아무 정답 하나만 맞춰 모든 부분을 설명했다고 처리하지 않는다. 후보 64개에 정답이 없는 관계는 proposal miss로 계속 집계한다.
- 544개 캐시 전체의 현재 teacher fingerprint를 검증했고, 캐시 입력으로 재계산한 원래 head 확률이 전부 byte exact였다. Train 512개 중 503개에 학습 가능한 ray가 있다. 나머지 9개 및 calibration 2개는 선택 기록에 유지하고 zero-supervision으로 보고한다. 데이터 전체 재구축/추가 다운로드는 없다.
- 실제 file4에서 1,099개 파일/6,725,864,757 bytes의 SHA256과 크기를 확인했다. Canonical root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/ray-set-head-supervision-v1`이다.

## 실행 예산과 평가

실측 forward/backward/AdamW profile에서 8,192 query는 8.56 GiB, 16,384 query는 17.09 GiB였다. Physical GPU 0의 CUDA allocator를 20%로 제한하고, 더 크게 실측 통과한 **16,384 queries = 32 assets × 512 rays**, accumulation 1을 사용한다. 초기 selection의 최대 8,192 계획은 실행 전 하드웨어 amendment로 변경하며, 데이터/epoch/loss/LR은 그대로 둔다. 503개 usable assets를 epoch마다 한 번 순회하므로 16 updates/epoch, 총 112 updates다. LR 1e-4, AdamW weight decay .01, BF16, gradient clip 1, seed101이다. Profile의 임시 1 update는 버리고 과학 실험에 포함하지 않는다.

Epoch0 및 매 epoch calibration loss, target-patch probability mass, ray positive-union mass, Top-1/64 coverage를 W&B `youngwoojeon-unist/pixal3d-deformnext-retargeting`에 기록한다. Epoch 선택 없이 epoch7을 내보내고, 원래 conditioner와 비교해 utility 5 tensor 외 모든 weight가 동일함을 검사한다. Auxiliary 7 epochs는 SS 모델의 추가 7 epochs가 아니다.

최종 head는 동일 Full480 denoiser, deterministic sampler12/CFG3, 동일 개발32개/noise0로 실제 SS 평가한다. 정답 후보 질량만 좋아지고 SS가 좋아지지 않으면 성공으로 선언하지 않는다. 개선 후보만 동일 dev128×3 generation noises와 필요한 독립 train-seed 재현 대상으로 올린다. 기존 6-arm 최종 평가는 별도로 유지하며, 이 탐색32개를 독립 test로 부르지 않는다.

학습 코드/캐시 source hash를 고정하고 실행 중 변경을 검사한다. 최종 epoch7 결과가 나오기 전에는 모델 채택이나 성능 향상을 주장하지 않는다.

## 고정 epoch7 실행 결과

실제 학습은 7 epochs/112 updates를 완료했다. 904.15초, peak allocated 17.1385 GiB였으며 accumulation은1이었다. Epoch0의 utility 5 tensor는 원래 Full480과 exact였고 최종 export에서는 그5개만 바뀌었다. R/D1/scalar/fusion은 모두 exact, SS 추가 update는0이다. 새 가중치도 원래 BF16/CFG/trace conditioning QA를 완화 없이 통과했다.

| Calibration 지표 | Epoch0 | 고정 epoch7 |
| --- | ---: | ---: |
| 다중 positive-set loss | 4.23930 | 3.24328 |
| Exact target-patch probability mass | 0.083433 | 0.081654 |
| Near1 target-patch probability mass | 0.275039 | 0.280371 |
| Exact ray-union probability mass | 0.169456 | 0.167912 |
| Near1 ray-union probability mass | 0.426370 | 0.431594 |

Loss 감소와 정답 집합의 평균 확률 증가가 같은 현상은 아니다. 후보64개가 고정된 이번 학습은 proposal miss를 해결하지 못한다. 초기2에폭 이후 일부 평균 mass가 다시 내려왔지만 checkpoint를 선택하지 않고 고정 epoch7을 평가했다.

| 동일 Full480 denoiser / deterministic SS32·noise0 | IoU | F1 |
| --- | ---: | ---: |
| 원래 utility head | 0.436073 | 0.607313 |
| 다중 positive-set epoch7 head | 0.434650 | 0.605932 |

Epoch7−원래 head의 IoU 차이는 **−0.14231 percentage points**, source-cluster bootstrap95% CI **[−0.53704,+0.16628] pp**였다. RGB 입력 PNG, GT fingerprint, source cluster, 생성 noise 및 sampler가 모두 paired였다. SS 개선은 확인되지 않았고, 사전에 정한 승격 조건을 만족하지 않아 이 head를 단독으로128×3에 올리거나 추가 epoch를 돌리지 않는다. 이는 현재 fixed-bank/utility5-only/503 train-assets pilot의 결과이며 semantic Top-K나 attention 일반의 불가능성을 뜻하지 않는다.

실제 결과와 GT/Pred·입력 이미지를 같은 [W&B run](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-ray-set-head-v1-s101-512-e7)에 추가했다. 원격 numeric9개와 PNG2개의 byte equality를 직접 확인했다. Canonical `SS32-publication-v1/file4-closure.json`은 SS 평가·paired 비교·export 감사·원격 로깅 감사17개 파일/201,450 bytes의 실제 file4 SHA와 크기를 검증한다. 학습 checkpoint와 history의 앞선 별도 closure는13개 파일/43,145,027 bytes다.

## 대응점 사용과 다음 분리 검사

별도의 기존 Full480 value-address control은 candidate key·score·weight·support를 유지하고 retrieved value 주소만 섞었다. 정상 IoU0.436073, 주소 교란0.431890으로 정상−교란은 +0.41833 pp, CI[+0.00860,+0.78817] pp였다. 이는 탐색적 개발32개에서의 저장 모델 조건부 민감도이며 V1 대비 향상, matching decoder의 독립 효과, 학습 seed 재현이나 정확한 Top-1의 증거가 아니다. 이 정상32개는 새 Full128/noise0의 공통32개와 exact였고, 기존 Full의 생성 noise3개 및 value/replay proof는 실제 file4에 닫혔다.

다음 작은 비교는 **후보 검색을 평균 전에 수행하는 GT-free patch union**이다. 기존 same64 coverage 검사에서 constituent target patch별 후보를 round-robin으로 합친64개는 near1 coverage를80.4807%→83.8385%로 높였다. 이 단계의 coverage gain은 SS gain이 아니다. 새 inference-only control은 기존 pooled ray query·support·D1/scalar/utility/R 계산을 유지하고 후보 bank만 바꾼다. 실제2개 샘플에서 identity의9개 trace와 projection exact, 후보64개 중복 없음, source-disable의 native 복귀, context 종료 후 원본 exact 복원을 확인했으며 CPU6검사를 통과했다.

두 SS32 비교도 완료했다. (2)의 head는 원래 pooled 후보 bank에서 학습했으며 union 전용으로 재학습한 head가 아니다. 두 비교 모두 SS 추가 update는0이며, 기존6개 SS ablation과 결정적 최종 평가를 바꾸지 않았다.

| Saved-weight SS32 조건 | IoU | 원래 Full 대비 차이 pp | 95% CI pp |
| --- | ---: | ---: | --- |
| Union 후보 + 원래 Full head | 0.434458 | -0.161491 | [-0.391174,+0.043342] |
| Union 후보 + 고정 epoch7 head | 0.435314 | -0.075886 | [-0.489950,+0.288941] |

Union 조건에서 head7−원래 head는 +0.085604 pp, CI[-0.355475,+0.399408] pp였다. 후보 coverage 증가와 이번 조합이 SS 향상으로 연결된 증거는 없으며 두 조건을 승격하지 않는다. `ray-proposal-union-control-v1/quality-pilot-v1/summary.json`에 paired 입력·GT·noise·source 및 실제 control4회/32개 검증과 모든 contrast를 보존했다.

별도의 같은 계산 모드128개×3 생성 noise 비교에서도 기존 ray Full0.433994, 가중합0.434168로 Full−가중합은 -0.017377 pp, CI[-0.088001,+0.046097] pp였다. 두 모델의 독립32→128 exact replay 및 실제 file4 closure가 검증됐다. 원본은 `deterministic-evaluation-v1/study-v1/interim-full-vs-weighted-128x3-v1/`다. 학습 seed는 여전히1개이고 전체6-arm 판정을 대신하지 않는다.
