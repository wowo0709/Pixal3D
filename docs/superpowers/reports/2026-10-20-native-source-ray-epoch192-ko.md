# Native multi-view prior를 보존한 source-ray 실험: epoch192 중간 결과

2026-10-20 UTC, 두 번째 epoch 검증 완료 시점의 기록이다. 작업 브랜치는
`wowo0709/feature-retargeting`이다. 원본 데이터·실행 설정·체크포인트·생성물은
file4에 보관하고, 로컬에는 실행에 필요한 캐시를 유지한다. 사용 GPU는 물리 0–3이다.

현재 네 arm은 학습 전 native multi-view 출력의 일치를 통과했고 학습도 정상 진행한다.
그러나 epoch192에서 semantic Top-K 및 attention의 추가 효과는 아직 입증되지 않았다.
초기 출력 일치와 학습 경로의 동작을 성능 향상의 증거로 해석하지 않는다.

각 arm은 원본 multi-view SS4210의 완전한 base+LoRA에서 시작한다. Native 네 RGB view의
projection/global/negative condition을 보존하고, source camera의 같은 patch ray에 속한
voxel들에 같은 correction을 전달한다. 독립 voxel query를 만들지 않는다. 유효 correspondence가
없으면 source correction은 0이고 native prior를 유지한다. 별도의 geometry visibility gate는
사용하지 않는다. Retargeter와 feature 전달은 기존 source-ray 구조를 따른다.

GPU 한 장당 arm 하나, global batch96 = microbatch96, accumulation1이다. 총480 update,
5 epoch를 사전에 고정했고 매96 update마다 검증한다. 현재 검증은 같은32 asset/source cluster,
학습 seed101, 생성 noise0, 12 sampling step, CFG3, 평가 batch8이다. 학습 전 네 arm의
voxel IoU는 모두42.6715%였고 per-asset GT/noise/voxel record도 같았다. 이 초기 일치는
원본 multi-view checkpoint에 대한 것으로, 이미 deformation 학습을480 step 마친 V1과
출력이 같다는 뜻은 아니다.

| Arm | Epoch96 IoU | Epoch192 IoU | Epoch192 F1 |
|---|---:|---:|---:|
| Semantic Top-1 + retargeting | 43.2570% | 43.5106% | 60.6375% |
| Semantic Top-K 가중합 + retargeting | 43.3326% | 43.6004% | 60.7246% |
| Semantic Top-K scalar attention + retargeting | 43.3598% | 43.5449% | 60.6708% |
| Native spatial correspondence attention + retargeting | 43.4848% | 43.5561% | 60.6817% |

여기서 attention은 scalar readout이다. 별도 준비된 member-feature 4-head Q/K/V attention이나
physical deformation-aware query arm의 결과가 아니다. 현재 matcher/decoder 및 matching score
경로는 고정되어 있고 task attention은 학습한다. 실제191 update에서 task-attention gradient L1은
0.01630, attention 유효 K 평균은5.507, 최대 weight 평균은0.594였다. 고정 matcher/score의
gradient0을 학습 오류로 해석하지 않는다. 반대로 task gradient가 존재한다고 성능 이득을
주장하지도 않는다.

아래는 같은 epoch192의 paired source-cluster bootstrap4000회, seed42017 결과다.
단위는 IoU percentage point다. 중간 분석이며 다중 비교 보정 및 독립 학습 seed 검증 전이다.

| 차이 | IoU 차이 | 95% CI |
|---|---:|---:|
| 가중합 − Top-1 | +0.0898 | [−0.0102, +0.2054] |
| Attention − 가중합 | −0.0554 | [−0.1540, +0.0322] |
| Attention − Top-1 | +0.0343 | [−0.0457, +0.1263] |
| Attention − Spatial | −0.0112 | [−0.3692, +0.3591] |
| Spatial − Top-1 | +0.0456 | [−0.2849, +0.3619] |

모든 구간에0이 포함된다. Epoch96의 attention−Top-1 작은 양의 구간도 epoch192에서는
유지되지 않았다. 현재 수치로 full 모델을 선택하거나 epoch를 추가하지 않고 고정480 update를
완료한다. 과거 V1 periodic voxel quick panel은4 asset이므로 현재32 asset aggregate와 비교하지
않는다. 동일32개를 추출한 completed V1 step480은 별도 참고값으로만 사용한다. 최종 비교는
정해진128 asset × 3 generation noise 평가와 matched control로 수행한다. 생성 noise 세 번을
독립 학습 seed 세 번으로 세지 않는다.

실제 GT/Pred/절댓값 차이 PNG는 epoch0/96/192에서 GT 열이 네 arm 모두 바이트 단위로
같았다. 이는 SS occupancy의 max projection으로, RGB 렌더링이나 mesh M-IoU가 아니다.
검증 곡선, 네 PNG 및 paired CI 표는
[epoch192 W&B](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native-prior-epoch192-readouts-s101-v1)에
업로드했다. 네 arm의 epoch192 full denoiser/base+LoRA, conditioner 및 optimizer 상태는
27개 파일, 11,442,388,176 byte이며 실제 file4 readback SHA256 검증까지 완료했다.

후속 일곱 SS arm의 native GPU 구조 검사는 완료됐지만, 자기 모델의 micro96 capacity 검사와
본 학습은 아직 남아 있다. Member posterior는 source ray 구성원별 proposal을 사용하고,
member-feature arm은 실제4-head Q/K/V로 correspondence feature를 읽는다. Physical Top-1,
native spatial control 및 channel/projection 자체를 제거한 No-DQ로 효과를 구분한다.
현재 네 scalar arm을 이 후속 구조의 성능 결과로 대체하지 않는다.

새 SS 구성의 cascade 평가는 Shape512/1024 및 V1 PBR762를 고정해 SS 효과를 분리한다.
준비된14 model × 32 asset의 전체 비교 결과는 아직 모이지 않았다. 기존 세 model의96개
생성 결과는 재사용하고 새 구성들을 추가한다. Camera8의 target 결과와 camera9/10의
source adaptation 결과를 나누고 LPIPS, DreamSim, CLIP, PSNR, SSIM, M-IoU를 비교한다.
Appearance reference는 원래 source material을 유지한 M1 geometry의 공통 renderer 결과다.
변형 전 M0 RGB와 픽셀이 같다는 의미의 preservation으로 표현하지 않는다. CD/F-Score는
원본 생성 mesh와 source-defined normalization을 사용하고 출력별 ICP나 재정렬은 하지 않는다.
공통1M face 예산은 렌더링에만 적용한다. 최종 별도 test508의 FID/KID를 포함한10 metric,
후보 고정, 필요한 독립 TRAIN202 및 모듈 attribution은 아직 완료되지 않았다.

PBR은 V1과 retargeting-only가 step1905까지 완료됐고 Direct continuation은 대기 중이다.
학습된 PBR의 full-stack와 동일 generated shape를 사용하는 PBR-isolation 비교는 기존 후속
계획을 유지한다. 현재 SS 결과만으로 이 PBR 효과를 주장하지 않는다.

최종508개 평가의 큰 raw mesh가 로컬에 누적되지 않도록
[cascade remote IO](../../../experiments/deformnext_retargeting/cascade_remote_results.py)와
[PBR replay remote IO](../../../experiments/deformnext_retargeting/pbr_remote_results.py)를 추가했다.
기존 entrypoint와 대기 중 학습 소스는 수정하지 않는다. 명시한 archive 읽기/쓰기·SHA·metadata
pointer만 바꾸고, 나머지 native AST가 원본으로 정확히 역변환되는지 검사한다. PBR sampler,
RGB 입력, CPU/CUDA noise reset 및 예측 완료 뒤 GT 평가 파일을 복사하는 순서는 유지한다.

관련 CPU 검사32개가 통과했다. 실제 기존 generation 및 PBR replay 파일에 원본 archive
statement를 실행해 모든 배열의 dtype/shape/layout/값과 출력 NPZ 바이트 일치를 확인했다.
큰 NPZ는 file4에 직접 저장하고, metric reader가 사용하는 작은 camera NPZ는 유지한다.
이는 새로운 neural inference나 모델 성능 검증이 아니다. 실제 GPU renderer 픽셀/metric 재현과
PBR sampler/decoder 재현은 각각 별도 finite job으로 연결했고 다음 유휴 GPU0–3에서 수행한다.
진행 중인 네 학습을 중단하거나 함께 얹어 실행하지 않는다. 새 IO 프로세스의 NumPy allocation
설정만 사용하며 기존 학습의 환경이나 시스템 전체 설정은 바꾸지 않는다.

근거는 file4 결과 공통 root의 `ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`
아래에 있다. `native-prior-source-ray-v1/epoch192-interim-v2`, `epoch192-visualization-v1`,
`epoch192-comparison-Wandb-v1`, `epoch-checkpoint-archive-v1/step0000192`,
`final-cascade-remote-native-IO-probe-v2`, `final-PBR-remote-native-IO-probe-v1`의
`file4-closure.json`은 실제 원격 파일의 SHA256/byte 수를 검증한 기록이다. 두 GPU IO 검사는
각각 `final-cascade-remote-render-parity-v1`, `final-PBR-remote-GPU-parity-v1`에 계획만 연결된
상태다. 전체 실험과 최종 모델 채택은 계속 진행 중이다.
