# 최종 Fixed 문맥의 correspondence 대조 구현과 실제 실행 연결

2026-10-24 22:43 UTC. 이전 goal turn은 실제 DEV128 metric·working candidate·No-R 본 학습과 Fixed seed202 등록을 완료했으므로 progress다. 이번에는 빠져 있던 최종 문맥의3개 correspondence arm을 구현·검증하고 실행 queue에 연결했다. GPU 학습/평가 완료나 전체 목표 완료가 아니다.

## 1. 새 학습은 기존 Fixed 전달 경로의3개 누락분

| Arm | 실제 후보 | Matching network | 전달 경로 |
|---|---|---|---|
| M_FIXED_SPATIAL | 원래 projection3개 view 후보 | Decoder/scalar 물리적 제거 | Fixed weighting·동일 carrier/use/R·identity value |
| M_FIXED_RAW | Raw semantic64 | Decoder/scalar 물리적 제거 | 동일 |
| M_FIXED_D1_COS | D1 contextual cosine64 | Frozen decoder 유지, scalar head 물리적 제거 | 동일 |
| 기존 E_FIXED_RAW | D1 scalar64 | Frozen decoder/scalar 유지 | 이미 학습·SS/cascade 평가 완료한 기준선 |

원래 C_SPATIAL/C_RAW/C_D1_COS는 Top-1 appearance이므로 새 weighted Fixed 대조로 재명명하지 않았다. 세 arm 모두 실제 Fixed의 같은 초기 core10, 원본 SS4210 LoRA, DeformNeXt9149/validation508, native MV4 global/projection/negative, target RGB8+source RGB9/10/11, global96/micro96/accum1, FFN chunk24, optimizer/480 updates/5epochs를 유지한다. 새 head나 feature-teacher initializer를 추가하지 않는다.

Spatial3 대 semantic64는 correspondence module substitution이다. 이를 같은 K에서 주소만 바꾼 효과로 쓰지 않으며 fake64 padding을 하지 않는다. Cosine→scalar는 posterior와 최종 support 선택을 함께 바꿀 수 있다. 같은 후보 support의 weighting 효과는 이미 진행 중인 Uniform-K↔Fixed로 별도 검증한다.

## 2. 실제 코드 경로와 검사

- `pixal3d/modules/fixed_correspondence_evidence.py`: 원래 raw/member/D1-cosine posterior helper와 실제 camera-projected spatial key/value를 재사용한다. Spatial의 view-slot 주소와 semantic image-patch 주소를 구분하고, 같은 가중 evidence를 carrier·source-use·retargeter에 전달한다. Retargeting/use/fusion/fallback/return은 parent Fixed와 AST가 일치한다.
- `fixed_correspondence_native_recipe.py`: 실제 parent config/hash와 input-cache matcher provenance를 묶는다. Data·model·initializer·LR·batch·query channel 등 다른 변경을 거부한다. 세 source의 실제 초기 trainable10 값이 parent와 동일한지 비교한다.
- `fixed_correspondence_native_runtime.py`: 원래 flow loss/noise/data helper·실제 초기 함수/backward 검사를 유지하며 matcher가 None인 경우를 명시적으로 처리한다. Own schema/source/hash/initial state의 GPU proof와3-update micro96 capacity가 있어야 본 학습한다.
- `evaluate_fixed_correspondence_checkpoint.py`: 같은 step의 own denoiser/source/config만 strict하게 허용한다. 원래 deterministic DEV128·batch8·12steps를 사용하고 실제 SS prediction/GT archive를 남긴다.
- `fixed_correspondence_pipeline_contract.py`: 기존 Top-1나 parent E checkpoint를 새 M으로 잘못 표기하지 못하게 하고,128개 전체의 sample/source/noise/GT/count pairing을 검사한다. 실제 Shape/PBR/render numerical calls를 유지하는 own case worker도 별도 준비했다.

새88개와 관련 기존169개를 합쳐 **257 passed in2.77s**다. 최종 source-pin 목록을 확인하는9개 config roundtrip도 추가 통과했다. 이 검사는 실제 생성 품질이나 GPU capacity의 증명이 아니다. 현재 실행 중인 NN source는 변경하지 않았다. 구현 commit은 `602a936`이다.

## 3. 실제 checkpoint 생성자 검증

`shared-Fixed-correspondence-construction-v300`에서 실제 file4 D1 checkpoint의 SHA256을 로컬과 원격에서 대조했다. 세 actual constructor 모두 CPU RNG가 보존되고 parent Fixed/Top-1와 동일한 초기 core10을 갖는다. RAW/spatial의 q2c weight는 state에 없으며, D1 cosine에는 decoder weight가 있고 scalar-head weight는 없다. Source schema도 서로 구분돼 다른 mode의 checkpoint를 strict load하지 못한다.

Global NN-forward hook으로 CPU neural forward를 금지한 상태에서 수행했고 실제 호출0·GPU forward0·CUDA 초기화False였다. Constructor/초기 parameter 정합의 근거이며, native GPU 초기 함수·메모리·학습 성공의 근거로 확대하지 않는다. 실제 file4 closure는7개 artifact를 보존했다.

기존 원래 spatial control은 FFN chunk24/micro96으로 실제3-update capacity를 통과한 기록이 있다(peak allocated88.218GiB/reserved92.986GiB). 이는 새 M_FIXED_SPATIAL의 capacity를 대신하지 않는다.

## 4. 실제 등록한21개 작업

`shared-Fixed-correspondence-campaign-v301/prelaunch-v1`의38개 artifact(약1.88MB)가 실제 file4 SHA/size와 일치한다. 각 arm은 own proof→capacity→480 science→checkpoint 묶음→DEV128 noise0/1/2의7개 작업이며 총21개를 append-only로 등록했다.

| 작업 | 실제 자원/의존 |
|---|---|
| 세 arm의 proof/capacity | GPU6, 진행 중인 No-R의480/128×3 뒤. 새 proof0-update와 실제 micro96/3-update 검사가 필요 |
| Spatial/Raw 본 학습 | GPU6에서 own capacity 이후 순차 |
| D1-cosine 본 학습 | GPU7에서 own capacity와 선택 Fixed seed202의128×3 완료 이후 |
| 같은 checkpoint의 SS128×3 | GPU6/7 중 가용 자원, own checkpoint 묶음 이후 |

GPU0–3의 필수 PBR DDP4, GPU6 No-R, GPU7 Uniform-K를 중단/재시작하지 않았다. 새21개는 기록 시 모두 pending이며 실제 GPU proof·본 학습·품질 평가는 아직 시작 전이다. Capacity 실패 시 과학 학습을 자동 재시도하거나 성공으로 간주하지 않는다. 각 phase는 session drain·frozen-state·source hash·실제 file4 closure를 유지한다.

## 5. 논문 표와 실제 PDF 갱신

Matching ladder는 인접 행끼리, aggregation/retargeting은 Fixed 기준으로 비교하도록 표를 재구성했다. Spatial을 decoder+scalar Full에서 semantic 하나만 제거한 것으로 서술하지 않는다. Actual Fixed/Top-1의10개 cell만 채우고 새 control25개 cell은 pending으로 유지했다. SS128×3와 cascade128×1의 noise scope를 caption에 구분했다.

`paper-draft-Fixed-correspondence-ablation-v302`의 실제 컴파일은12pages/24refs/warning0이고9·10페이지를 렌더링해 직접 검수했다. 두 행의10개 값이 실제 closed SS/cascade 통계와 표시 정밀도까지 일치하며28개 artifact를 실제 file4 SHA/size로 보존했다. 최신 preview는 `/root/dev/.cache/deformnext-retargeting/paper-preview-20261024-v302.pdf`다.

## 6. 남은 전체 목표

1. 진행 중인 Uniform-K/No-R와 새3개 correspondence arm의 실제 proof/capacity/480·SS128×3를 완료한다.
2. 학습된 own SS archive를 같은 frozen Shape/Top-1 R PBR에 연결해 CD/F-Score·LPIPS/DreamSim을 포함한 실제 downstream 평가를 한다. Case binding 구현과 실제 cascade 품질 완료를 구분한다.
3. 선택 Fixed seed202 및 필수 PBR vanilla/R seed202를 완료하고 실제 checkpoint/trace/두 seed 통계·fresh-load를 검증한다.
4. Working candidate와 모듈별 claim을 실제 효과에 맞춰 확정한 뒤 protocol/checkpoint를 freeze하고 TEST508·134 source cluster·10metric을 실행한다.
5. 최종 Main/Ablation/Mechanism/Qualitative 표·그림 및 Results/Discussion/Abstract/Conclusion/PDF를 실제 근거로 마무리한다.

현재 학습 snapshot은 PBR vanilla347/1900, Uniform78/480, No-R52/480이다. 이 진행 수치나 생성자 검증을 최종 성능 향상으로 보고하지 않는다. [전체 필수 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md)과 [DEV128 선택 근거](2026-10-24-dev128-complete-fixed-controls-and-independent-confirmation-ko.md)의 범위를 유지한다.
