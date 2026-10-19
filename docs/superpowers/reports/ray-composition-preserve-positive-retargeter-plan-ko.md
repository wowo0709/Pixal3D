# 기존 양성 R 경로를 보존하는 ray Top-K 조합의 다음 구현

현재 source-camera pooled-ray 후보의 utility5-only supervision 및 patch-union counterfactual은 SS 개선을 보이지 못했다. 기존 Full−가중합128×3 차이도 -0.0174 pp로0을 포함한다. 이 결과 뒤에 같은 구조의 epoch나 head-only loss만 추가하지 않는다. 기존6개 arm과 최종 평가 큐는 그대로 완료한다.

다음 구현의 기준은 두 seed에서 Top-1 대비 향상을 확인한 `static-top1-c1`이다. 실제 config/checkpoint를 strict load하면 `projection_mode=nearest`, Top-1 aggregation, R 입력2048이며 utility attention은 없다. 기존 문서의 camera0 ProjGrid 설명은 잘못되어 정정했다. 현재 source-camera3 bilinear 전달과 원래 nearest target-anchor 전달은 같은 경로가 아니다. 어느 차이가 원인인지는 아직 단독 인과 실험으로 측정하지 않았다.

## 유지할 기준선

- 저장 old Top-1+R raw480의 full SS base+LoRA 및 conditioner를 사용한다. 초기 단계에서는 SS LoRA, R, D1/scalar를 전부 동결한다.
- Edited RGB cam08의 target patch 하나를 query 단위로 사용한다. 같은 target-anchor `patch_grid`를 참조하는 voxel들은 같은 query/evidence를 받는다. 개별 voxel query나 source-ray 안의 여러 target descriptor 평균을 새로 넣지 않는다.
- Native condition, alpha validity, camera, nearest gather, semantic fusion 및 R residual 계수까지 원래 경로를 유지한다. 새 deformation-query channel은 추가하지 않는다.
- 입력은 M1 RGB cam08 + M0 RGB cams09/10/11, DeformNeXt world-v4 train9149/val508이다. 캐시된 DINO/Omega와 기존 semantic64 주소/D1/scalar를 우선 재사용한다. GT는 필요한 경우 별도 loss 파일과 metric에만 사용한다.

## 추가할 분기

Ray의 Top-1 evidence를 `e1`, 동일 semantic64의 matching-aware readout을 `eK`라 하면, R에 전달하는 evidence는 `e1 + lambda * (eK-e1)`다. `lambda=0`으로 시작해 기존 모델을 정확히 재현한다. Matching score는 soft readout의 prior로 쓰고, native DINO target query 및 contextual Omega와 candidate feature로 utility를 학습한다.

Gate를 [0,1]에 유지할 경우 초기0의 gradient가 살아 있어야 한다. Raw clamp가 음수에 머물러 gradient가 사라진 과거 실패를 반복하지 않도록 optimizer step 뒤 parameter projection 및 경계 gradient를 검증한다. 초기에는 gate만 SS gradient를 받고 utility는 gate가 열린 뒤 gradient를 받는 정상적인 단계를 구분해 검사한다. Gate가 열리지 않으면 원인을 측정하며, 개발 성능을 보고 임의로 gate floor를 올리지 않는다. 필요한 target-patch auxiliary supervision은 기존 ray-mean supervision과 다른 새 학습 조건으로 선언하고 별도 입력/라벨 경계를 유지한다.

이 설계는 성능 향상이 보장된 full 모델이 아니다. 목적은 R의 기존 이득을 잃지 않은 상태에서 Top-K 조합의 추가 효과를 검증하는 것이다.

## 학습 전에 통과할 검사

1. 원래 저장 모델과 lambda0의 실제 condition·R·BF16·CFG 출력 exact, invalid/all-masked 회귀 exact를 확인한다.
2. 같은 결정적32개에서 원래 old Top-1+R와 GT/noise/source/TP·FP·FN·입력 PNG가 exact인지 확인한다. 이전 비결정적 epoch 기록으로 대체하지 않는다.
3. 실제 SS loss에서 gate와 활성화된 utility branch의 finite/nonzero gradient를 확인한다. 처음부터 모든 utility tensor gradient가0이면 즉시 버그라고 판정하지 않고 zero-init 단계와 분리한다.
4. 작은 실제 forward/backward/optimizer profile로 GPU memory와 microbatch를 확인한다. GPU0–3만 사용하며 기존 DDP 작업과 검증되지 않은 SS 학습 overlap을 만들지 않는다.
5. 이후에만 고정 budget의 SS-loss pilot을 준비한다. SS LoRA나 R까지 갱신하는 확장에는 같은 continuation budget의 R-only 대조를 추가해야 하며, 기존 V1 재학습으로 대신하지 않는다.

구현은 새로운 파일/실행 namespace로 분리하고, 현재55개 runtime binding 및 진행 중인6개 arm의 numerical source/config는 수정하지 않는다. Epoch0 parity, gradient, capacity 및 데이터 계약을 통과하기 전에는 새 과학 학습을 시작하지 않는다. 최종 판단은 실제 SS metric이며 대응점 loss나 Top-1 정확도만으로 승격하지 않는다.
