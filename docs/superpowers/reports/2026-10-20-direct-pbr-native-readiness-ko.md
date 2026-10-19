# Direct correspondence 경로의 native PBR 검증

## 확인한 결과

새 direct retargeting 경로를 실제 PBR denoiser와 DINO1024 + NAF1024 condition으로 실행했다. 9,823개 sparse token을 가진 TRAIN asset 한 개를 사용했고, complete multiview PBR base+LoRA와 matching decoder는 고정했다. 이는 본 학습이나 validation 성능 실험이 아니다.

- 기존 fresh C1/R과 새 physical-DQ/direct 모델의 시작 evidence 및 loss가 정확히 같았다. 시작 loss는 둘 다 `0.027075452730059624`였다.
- Semantic fusion gate=0, 기존 R output=0인 첫 backward부터 attention output projection의 DINO 행 gradient L1은 `0.30232346`, NAF 행은 `0.25603619`였다. 기존 input-side PBR prototype은 같은 시작 단계에서 head gradient가 0이었다.
- R6 + attention9 + 추가 DQ projection1, 총16개 tensor가 세 번째 임시 update까지 모두 비영 gradient를 받았다. Complete PBR NN/base+LoRA와 matcher는 정확히 유지됐다.
- 실제 RAW Omega 입력2048개 채널을 추가해 PBR retargeter 입력은 appearance3072에서 총5120으로 늘었다. RAW context는 이미 존재하는 SS sidecar에서 읽고, PBR sidecar의 DINO/NAF value 및 sparse mapping은 그대로 유지한다. GT correspondence나 GT deformation은 입력으로 넣지 않았다.
- 최대 할당 메모리는 `19.120566 GiB`, 임시 update·진단 실행 구간은 약58초였다. GPU0에서 allocator fraction0.25로 실행했다. 이 수치를 micro12/DDP capacity 증거로 쓰지 않는다.

초기 R output이 0이므로, 첫 NAF output-weight gradient는 R의 cross-channel 간접 경로로 설명되지 않는다. 새 feature 보정이 native NAF loss 경로로 직접 연결됐다는 증거다. 이 구조가 최종 rendering 품질을 높이는지는 별도 학습으로 판단해야 한다.

## 진단 결과의 한계

임시 세 번 update 후 다섯 개 saved-weight 진단이 모두 실제 실행됐다. Appearance와 RAW Omega의 correspondence 값을 함께 바꾸거나, score prior/decoder context/DQ projection을 바꾸면 evidence가 달라졌다. 각 RMS는 다음과 같다.

| 진단 | Evidence delta RMS |
|---|---:|
| Wrong appearance + RAW value address | 0.00103035 |
| Added attention score prior off | 0.00095922 |
| Added decoder context bypass | 0.00011893 |
| Added feature + DQ path off | 0.01970114 |
| DQ projection output off | 0.00000243 |

이는 코드가 그 경로를 실제 사용하는지 확인하는 검사다. 단일 TRAIN asset을 임시로 갱신한 값이므로, 각 모듈이 품질에 유익하다는 뜻은 아니다. Score-prior-off는 사용자가 정의한 matched Top-1 NoScore arm이 아니며, projection output off는 입력 채널이 물리적으로 없는 NoDQ 재학습 arm이 아니다.

## 데이터와 후속 capacity 준비

원본 TRAIN manifest 9,149개의 PBR 좌표 NPY header를 실제로 읽었다. 기존 PBR config의 asset당32,768-token 제한을 적용하면9,139개가 남고10개가 제외된다. 이는 기존 V1도 사용하던 native PBR 정책이며, 새 모델만 다른 subset을 쓰는 것이 아니다. 전체 eligible token 수의 median은6,288, 99% quantile은22,825.74, 실제 maximum은31,900이었다.

따라서 9,823-token 단일 샘플 검사로 최대 microbatch를 단정하지 않는다. 이후 용량 검사는 TRAIN의 실제 maximum 사례와 validation 이후 추가 train step을 포함해야 한다. 기존 PBR V1 step762의 NN checkpoint 및 Adam state를 재사용할 수 있도록 기존 optimizer 파일만 가져왔다. NN2.85GB를 새로 다운로드하거나 data/cache를 다시 구축하지 않았다. 해당 checkpoint는 Adam360개, 모두 step762이고 LoRA rank16/alpha16/dropout0이다. Sampler는 epoch2/idx16, source world3이므로 새 world2/4에서 공통 global offset48을 보존하도록 각각idx24/12로 변환해야 한다.

Native PBR full continuation은 아직 시작하지 않았다. 동등한 업데이트 예산과 global24, 최대 가능한 microbatch, accumulation1을 유지하는 GPU 배치를 실제 capacity로 선택한다. Rendering 평가는 표준 GT target Shape를 사용하는 PBR stage 진단이며 full cascade나 novel-view 평가로 해석하지 않는다.

## 검증 기록

공통 root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/direct-retargeting-v1`이다.

- `PBR-gradient-v1/audit.json`: 실제 native PBR identity, 첫 DINO/NAF gradient, 모든16 tensor gradient 및 진단.
- `PBR-capacity-inputs-v1/audit.json`, `token-counts.jsonl`: 실제 TRAIN token header와 기존 filter의 eligibility.
- `pbr-reference-v1/misc_step0000762.pt`: 기존 V1 continuation state. 별도 proof로 원본 file4와 일치하는지 확인 후 사용한다.

새 데이터 loader의 CPU 검사4개가 통과했다. 잘못된 RGB hash, FRAME prefix, legacy deformation 입력을 거부하고 실제 PBR latent 및 sparse mapping 객체를 보존하는지 검사했다. 학습 개선·모델 채택·전체 실험 완료는 주장하지 않는다.
