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
