# 현재 DeformNeXt의 원본 O/D9채널 query 준비

기존 production C2의 Omega RAW2048 delta와 과거G2의 predicted geometry는 다른 입력이다. 후자를 실제 현재 RGB에서 준비하기 위해 `cache_predicted_geometry.py`를 추가했다. 이것은 cache 구축이며 새SS adapter 학습/성능 개선 완료를 뜻하지 않는다.

## 고정 입력과 예측

| Branch | RGB 입력 | Camera |
| --- | --- | --- |
| O | 변형 전 M0 네 장 | 8/9/10/11 |
| D | 변형 후 M1 target 한 장 | 8 |

Source08과target08의 camera matrix/FOV가 같음을 검사하고, 두 branch 모두 기존 world-v4의 native target-anchor frame·mesh scale을 유지한다. 원본SS64 mixed385의 complete base+LoRA step4210을 continuation loader로 읽고 frozen evaluation으로 쓴다. Native12step/CFG3·BF16 AMP·batch8이며, 동일 asset의O/D는 같은 초기 noise tensor에서 생성한다. 과거G2의 full-FT20k/CFG7.5 reference consumer와는 별도이다. Target-only 학습 arm을 추가하는 것이 아니다.

`dataset._common`의 RGB/camera 경로만 읽는다. `dataset[index]`를 호출하지 않으며GT SS latent·mesh·M1 vertex array·topology label을 열지 않는다. 기존 native camera/scale metadata는 그대로 사용한다. O의첫RGB만 실제source08로 교체하고 D는실제target08 한 장으로 만든다. 각RGB 파일·consumer checkpoint·config·population·manifest의SHA와 camera/scale·noise seed를 기록한다. 이 단계는Omega matching 후보/score를 입력으로 쓰지 않는다.

Raw SS decoder logit64에서 과거 `geometry_from_logits` 함수를 그대로 호출한다. 각 branch의logit clamp/8·sigmoid occupancy·SDF clamp/16을4³ mean-pooling해16³에서 `[D,O,D−O]`의9채널로 만든다. 결과shape은`[4096,9]`다. GT deformation을 넣거나 RAW descriptor delta를 이름만 바꿔 저장하지 않는다. Physical No-query adapter에는 이 cache를 전달하지 않는다.

## 범위와 실행 검증

먼저 이전 RGB task pilot의고정256train/64calibration, 총320개를사용한다. Geometry query는matching query eligibility와무관하게320개 모두 구축하므로, RGB task에서 관측 query가없던8개도제거하지 않는다. Full9149 proxy 구축은pilot의 실제SS 사용 검증 이후 별도로 결정한다. 기존 common32는같은panel/noise map을사용하며pilot population에합쳐집계하지 않는다.

GPU2의실제1개 preflight는성공했다. GT 입력 없이O/D occupancy가각2139/2038 voxel, query shape`[4096,9]`, 모든값finite였다. 모델 로딩 이후약8.2초, GPU peak약3.99GiB다. 이를통과한동일consumer만full cache를실행하도록검증한다. 고정320개는GPU2/3의짝수/홀수rank shard로분할하고 common32도GPU2에서먼저실행한다. 기존0/1의C2학습은유지한다.

첫8개cache batch는약43초, GPU peak약5.39GiB였다. 이는sampling+RGB encoding+decoder+저장시간이며SS학습속도가아니다. 동일code에서GTgetitem을막은입력검사·원본9채널함수재계산·채널순서오염검출을포함해관련CPU4개검증을수행한다.

`audit_predicted_geometry`는320/32개 전체의SHA·ID·disjoint rank union·seed·protocol·9채널을raw logit에서정확히재계산한다. Common32 O의occupied count를기존source-only zero-shot의TP+FP와대조한다. 기존baseline에prediction voxel SHA가없으므로count parity를full tensor parity로확대하지 않는다. Cache완료와이감사는새adapter 학습이나SS 향상을입증하는것이아니다.

## 실행 파일과 다음 판단

작업namespace는file4의`rgb_matching_v2/current-geometry-query-v1/{preflight,pilot320,common32}`다. GPU queue는0~3만사용하며wandb는`youngwoojeon-unist/pixal3d-deformnext-retargeting`에기록한다. 기존320RGB task/6head/G2 endpoint의856개immutable파일(약3.28GB)은실제file4 SHA256대조를통과했다. 근거는`provenance/faithful-current-task-and-complete-g2-file4.json`이다.

후속SS실험에서는원본geometry query·native projected-feature content·source-use control을함께검증해야한다. 특히과거G2 자체의projected-source-off우세와현재per-camera routing/fallback 진단을반영한다. Cache가있다는이유만으로기존C2를faithful query arm으로명명하거나Full로채택하지않는다.
