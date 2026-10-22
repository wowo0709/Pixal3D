# 새 PBR native task 학습과 빠른 validation runtime

원본 multi-view PBR16850의 complete frozen base+LoRA에서 fresh optimizer로
시작하는 실행을 구현했다. 기존 deformation V1의762-step weight/Adam이나
micro1/capacity의 임시 weight를 이어 쓰지 않는다. 대상은 같은 DeformNeXt의
eligible TRAIN9139이며 clean80k를 추가하지 않는다.

`r_mediated_pbr_science_recipe.py`는 기존 실제 micro1 config의 모델·data·
normalization·optimizer·source 설정을 유지한다. 변경하는 값은 DDP2/micro12
또는 실제 검증된 DDP4/micro6, global24/accumulation1, num_workers2,
checkpoint/validation380, budget1900이다. Native dropout.1, NN LR5e−6,
source LR1e−4, warmup25, BF16 및 adaptive clipping은 유지한다.
Native loader380개 batch를5회 소비하며, max-token32768 필터도 동일하다.

Vanilla, physical Top1+R, Top1+feature+R, raw Top-K+feature+R를 같은 recipe로
구성한다. Source는 물리적 NoDQ, actual source0/10/19 parameter이며 DINO+NAF2048을
사용한다. 초기 native denoiser360개 LoRA와 seed101 공통 R10의 실제 SHA를
기존 cold generated32 검사와 대조한다. 독립 seed202는 별도 시작값으로
동일한 고정 noise의 초기 기능 검사를 거친다.

## 실제 첫 update 이전의 검사

`train_r_mediated_pbr_native.py`는 실제 micro1, cold generated32 및 같은
world/global/microbatch의 capacity 결과가 모두 file4 검증된 PASS일 때만
시작할 수 있다. Rank0 초기4개 평가의 실제 RGB·camera·GT Shape/PBR latent·
noise·예측 PBR latent와 rendering metric이 앞선 cold32의 해당 asset과 같아야 한다.
그 다음 첫 update에서 admission flag를 모든 rank에 broadcast한다.
초기 평가 실패를 무시하고 optimizer를 돌리는 경로는 허용하지 않는다.

## 빠른 epoch 평가와 로깅

`r_mediated_pbr_science_metrics.py`는 vanilla와 FeaturePBR dataset을 각각
실제로 구성한다. 둘 다 명시된 같은4개 asset에서 native flow loss와
sampling12/CFG3·256px anchor rendering을 계산한다. PSNR/SSIM/LPIPS,
foreground 및 base-color 지표와 GT/Pred/Diff PNG를 local/file4와
youngwoojeon-unist의 W&B project `pixal3d-deformnext-retargeting`에 기록한다.
기존 quick evaluator가 FeaturePBR를 인식하지 않아 빈 metric을 반환하는
문제를 이 process-local callback에서 처리하며 공유 evaluator를 수정하지 않는다.

평가는 GT target Shape를 고정한 PBR stage 진단이다. Source preservation이나
predicted-geometry full-cascade 평가로 부르지 않는다. Rank0에서 DDP wrapper를
풀고, 종료와 예외 시 mode·dropout·Python/NumPy/Torch RNG를 복원한다.
모델 전체 state, optimizer step, sampler 및 scheduler가 평가 전후 같은지 검사한다.

## Checkpoint 보관 및 아직 남은 실행 준비

기본 checkpoint 자동 pruning을 끄고 매380 update 저장이 완료된 뒤
명시된 archive command로 NN/source/Adam의 실제 file4 SHA/size를 검증한다.
Native run과 epoch hook이 같은 step의 save를 두 번 요청해도 이미 닫힌
checkpoint를 덮어쓰지 않는다. 같은 step에 parameter/state가 달라졌다면 실패한다.
Archive 실패를 PASS로 기록하거나 로컬 checkpoint를 삭제하지 않는다.

현재 구현 및 CPU 검사는 GPU 학습 완료 증거가 아니다. 실제 engineering GPU
검사들은 기존 SS 학습 종료를 기다린다. 실행용 immutable config/plan의 file4 대조,
GPU group wrapper, 저장 공간 admission 및 완료된 cache의 보관량 조정이 더 필요하다.
이 runtime만으로 새 PBR science job을 실행했다고 보고하지 않는다.

최종 predicted geometry에 적용할 때는 cached GT-support patch address를 그대로
쓰지 않고 실제 predicted support에서 target/source camera address를 계산하는
별도 replay 검증이 필요하다. 고정 원본 Shape512/1024를 쓰는 cascade, source/target
10 metric, 독립 TRAIN seed 및 최종508 평가 범위는 유지한다.
