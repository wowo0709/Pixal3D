# 동일 Top-K evidence를 사용하는 전달 경로: 구현·검증·실험 연결

관측 시각: 2026-10-23 20:33 UTC. 소스 커밋: `a07be7d`, `f152343`.
브랜치: `wowo0709/feature-retargeting`.

## 완료한 변경

기존 Full은 raw Top-K 조합과 학습된 feature 변환을 retargeter 입력에 사용하지만,
최종 carrier와 source-use gate에는 hard Top-1을 사용한다. 또한 학습된 matching
logit 보정은 bound=1로 제한된다. 두 사실은 실제 소스 및 학습된 파라미터 감사에서
확인했다. 다만 이 구조가 최종 성능 정체의 원인이라는 인과적 결론은 아직 없다.

이 병목 후보를 직접 확인하는 **새 학습 arm 두 개**를 구현하고 실제 큐에 연결했다.
기존 실행 중인 소스·config·worker·학습 프로세스는 수정하거나 재시작하지 않았다.
새 arm은 원래 12개 arm의 Full을 대체하거나 같은 이름으로 보고하지 않는다.

| 구성 | 선택하는 source evidence | 특징 변환 | 학습 가능한 source 텐서 수 |
|---|---|---|---:|
| 기존 Top1+R | D1 member posterior의 Top-1 특징 | 원래 retargeter | 10 |
| E_FIXED_RAW | 고정 D1 member posterior로 가중합한 raw Top-64 특징 | 같은 retargeter | 10 |
| E_LEARNED_RAW | 같은 후보·prior에 학습된 Q/K/utility의 상대 score를 반영한 raw 특징 | 같은 retargeter | 17 |

두 새 arm 모두 **선택된 동일 evidence E**를 carrier, source-use gate,
retargeter의 appearance 입력에 전달한다. 새로운 V/output projection과 별도
attention correction branch는 없다. 원래 D1 matcher는 frozen이며, 새 head의
학습 대상은 Q/K/utility 7개 텐서다. 별도의 feature-teacher 학습이나 초기화는 없다.

E_LEARNED_RAW의 확률은 `softmax(log p_D1 + s_theta - s_initial)`이다.
수치 안정화를 위해 상수항을 중심화하지만 기존 `tanh` logit bound는 적용하지 않는다.
초기에는 score 차이가 정확히 0이므로 원래 prior를 유지한다. 원래 prior에서
정확히 0인 후보 질량은 계속 0이다. 이 설계가 실제 GT 후보의 질량을 충분히
키우는지는 향후 GPU 학습 및 평가로 확인해야 한다.

## 유지한 전달식과 초기 정합

native multi-view 4장 조건에 다음 source-camera-ray 보정만 더한다.

`C = C_native_MV4 + sum_v Proj_v[u_v*g*(E_v-S_v) + u_v*R(E_v-S_v,q_v)] / 4`

S는 실제 native carrier에 포함된 원래 source feature다. target-only carrier에서
source를 빼는 수식으로 바꾸지 않았다. q는 source 카메라의 동일 ray에 속하는
voxel들의 target context다. 개별 voxel별 write 결정을 만들지 않는다.

target RGB와 native global/negative 조건을 유지한다. 대응 evidence가 없으면
native source prior를 그대로 사용한다. 별도 surface visibility gate, deformation-query
채널, 추론용 GT depth/geometry/correspondence는 추가하지 않았다.

fusion gate=0 및 retargeter의 마지막 layer=0인 초기 상태에서는 보정이 정확히 0이다.
이는 설계상의 성질이며, **실제 GPU의 native-field/velocity/loss 및 초기32 정합은
앞으로 실행될 각 arm의 필수 검사**다. 코드 검사 통과를 실제 GPU 검증 통과로
간주하지 않는다.

## 실제 연결된 실행

canonical 결과 경로:

`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/PBR-native-science-runtime-v1/unified-module-pipeline-v1/consistent-raw-evidence-followup-v177`

기존 DeformNeXt 정합 TRAIN 9,149개와 VAL 508개에서 선택한 기존 DEV panel을 재사용한다.
mixed385의 clean 80k를 deformation 학습에 사용하지 않는다. 새 다운로드나 데이터 재구축은 없다.
두 arm 모두 원래 SS4210 full-base+LoRA checkpoint에서 fresh optimizer로 시작한다.
5에폭=480 update, global/microbatch=96, world1, accumulation1, FFN chunk24,
동일 LoRA·optimizer·learning rate·TRAIN seed101을 유지한다.

GPU 6·7의 전용 큐에 **17개 작업(12 GPU, 5 CPU)**을 append했다.
각 GPU 작업은 두 카드 중 빈 카드 **한 개**만 사용한다. 현재 학습을 중단하거나
메모리에 겹쳐 넣지 않는다. GPU 0~3의 기존 학습 및 이후 PBR DDP4 전환은 유지한다.

각 arm의 실행 순서는 다음과 같다.

1. 실제 TRAIN 입력 1개: native field/global/negative/RNG 및 같은-noise velocity/loss 정합,
   cold backward, 초기 attention prior 일치 검사. optimizer update는 하지 않는다.
2. 실제 micro96: 초기 DEV32 정합 후 실제 update 3회, 모든 source gradient 및 frozen weight
   검사. 여기서 만든 optimizer 상태나 학습된 source weight를 본 학습에 사용하지 않는다.
3. CPU admission: 기존 Full과 초기 LoRA/RNG/TRAIN 순서 및 3회 학습 입력 trace를 대조한다.
4. fresh 480 update: 자체 proof/capacity와 초기 weight를 대조하고, 실제 V1과 첫 5회
   asset/noise/timestep/global input을 대조한다. 매 에폭 DEV32 평가 및 시각화를 기록한다.
5. 자체 NN/source checkpoint pair를 봉인하고 동일 DEV128 × generation noise3를 평가한다.
6. 기존 실제 V1·Top1+R 결과와 source-cluster paired bootstrap으로 비교한다.

새 arm의 W&B entity는 `youngwoojeon-unist`, project는
`pixal3d-deformnext-retargeting`이다. 본 학습이 시작되면 loss/gradient와 validation
이미지·지표를 기록한다. 학습형 head는 prior 대비 KL, entropy, Top-1 변화율도 기록한다.

새 checkpoint schema는 기존 Full/Top1/projected 모델과 구분한다. 다른 evidence mode,
변조된 schema 값·dtype·shape, 잘못된 NN/source pair를 거부한다.

관련 검사 **98개 통과**. CPU에서는 helper math, 생성자 및 NN을 stub으로 대체한
전달 경로를 검사했다. 실제 CPU neural training/forward는 수행하지 않았다.
실제 파일서버4 SHA256/size 확인: prelaunch 32개 파일, queue 2개 파일, applied 1개,
postconnection 1개 파일 PASS. 기존 aux 큐 401개 descriptor는 그대로이며 새 17개만 추가했다.
관측 당시 새 작업 17개는 모두 pending으로, GPU 결과는 아직 없다.

## 현재 결과와 해석

완료된 raw480 DEV128 × noise3, TRAIN seed101의 mean-asset IoU:

| 구성 | IoU |
|---|---:|
| V1 | 0.46735504 |
| Top1+R | 0.47260025 |
| Top1+R + V/output | 0.47221778 |
| 고정 Top-K + V/output | 0.47274517 |
| 기존 Full | 0.47207818 |

기존 Full−V1은 +0.00472314이고, DEV에서 pipeline family2 보정 구간은
[0.00028801, 0.00939391]이다. Full−Top1+R, V 변환, 고정 Top-K 및 학습형 attention의
추가 효과 구간은 0을 포함한다. 고정 Top-K의 평균이 가장 높다는 이유만으로 선택하지 않는다.
Top1+R 대 V1의 조합 이득을 retargeter 단독 기여의 증명으로 확대하지 않는다.
R의 별도 기여는 실행 중인 C_DECODER_NO_R와 비교해야 한다.

현재 6개 학습의 실제 진행:

| GPU | arm | update / 480 | 최근20회 median 초/update | 최대 allocated GiB |
|---:|---|---:|---:|---:|
| 0 | C_RAW | 161 | 36.91 | 88.67 |
| 1 | C_SPATIAL | 156 | 37.64 | 88.40 |
| 2 | C_D1_COS | 161 | 36.94 | 88.68 |
| 3 | C_DECODER_NO_R | 166 | 36.03 | 87.92 |
| 6 | FULL_QK_ONLY | 98 | 49.85 | 90.70 |
| 7 | FULL_VOXEL_Q | 28 | 148.58 | 92.75 |

GPU 7의 개별 voxel query arm은 ray-wise arm보다 느리다. GPU 7이 GPU 6보다
먼저 비게 된다는 이전 예상은 현재 속도로는 맞지 않는다. 새 작업은 6·7 중 실제로
비는 카드에 배치하며, 기존 학습을 재시작하지 않는다.

같은 32개·같은 noise·96 update의 **중간** mean-asset IoU는 V1 .46380091,
Top1+R .46161876, C_RAW .46032510, C_SPATIAL .46265811,
C_D1_COS .46009013, C_DECODER_NO_R .46373792, FULL_QK_ONLY .46155038이다.
이 시점에서 새 ablation의 명확한 개선은 없다. 완료된128×3 결과와 이32개 중간값을
섞지 않으며, native pooled IoU와 mean-asset IoU도 섞지 않는다.

## 남은 완료 조건

두 사전 선언 module contrast는 E_FIXED_RAW−Top1+R와 E_LEARNED_RAW−E_FIXED_RAW다.
pipeline contrast는 E_LEARNED_RAW−V1와 E_LEARNED_RAW−Top1+R다. 각각 family2를 유지한다.
이는 기존 모듈 결과를 본 뒤 설계한 **exploratory follow-up**이며, 원래 12-arm family를
줄이거나 기존 Full 결과를 새 arm의 결과로 바꾸지 않는다.

현재 12개 arm의 최종 평가, 이미 연결한 공통 PBR/공통 geometry 및 실제 SS+PBR joint
평가도 완료해야 한다. 선택 모델의 독립 TRAIN seed202 재현, 실제 두 번의 load 및
tail batch 정합, 모델/평가 프로토콜 freeze 후 TEST508·10개 metric 평가가 남아 있다.
새 두 모델을 선택한다면 별도 schema를 해당 seed/population/freeze 검증에 정식으로
지원해야 한다. 기존 12개 arm용 verifier를 통과시키기 위해 이름을 바꾸어 넣지 않는다.

최종 전달 구조가 바뀐 모델에서 이득을 확인하면 그 모델의 최종 파이프라인에 맞춰
필수 module ablation을 대조한다. 기존 구조의 대조 결과만으로 새 구조에서 모든 모듈의
필요성이 증명됐다고 서술하지 않는다. 현재 최종 모델 채택 및 전체 목표 완료는 아니다.
