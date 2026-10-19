# Ray 단위 query 해석과 과거 구현 재감사

사용자가 말한 ray-wise query는 **같은 camera ray에 속한 voxel들을 하나의 query 단위로 묶는 것**이다. 개별 voxel에 별도 attention을 적용하거나 near-to-far sequence encoder를 추가한다는 뜻이 아니다. 앞서 이를 individual-voxel query로 해석한 것은 잘못이었다.

현재 `voxel-query-c1`은4096개 native voxel 위치별 four-view projected DINO를 utility query로 쓴다. 같은 target patch를 참조하는 voxel도 query와 attention이 달라질 수 있다. 따라서 사용자 요청의 ray-group arm으로 부르면 안 된다. 기존 모델을 수정하거나 새 이름의 ray-group 결과로 재표시하지 않는다.

`utility-softprior-c1`과 이전 patch-query arm들은 target anchor의2D patch 하나에서 attention 결과 하나를 만들고 해당 anchor patch를 참조하는 voxel들로 전달한다. 그러나 voxel group을 query로 요약하는 연산이 없고, 세 source camera 각각의 native write slot을 보존하는 구조와도 다르다. Anchor patch 공유만으로 과거 camera별 ray-query 경로를 재현했다고 할 수 없다.

## 실제로 찾은 과거 구현과 양의 결과

| 근거 | 실제 내용 | 판정 범위 |
|---|---|---|
| `ss_target_local_ray_selection.py:target_proxy_ray_summary` | Target-only predicted16³ latent/SS64 logits를 각 source camera에 native projection하고, 같은32² ray slot에 모인 voxel의 mean/occupancy-weighted summary를 `index_add_`로 계산한다. Output은 `[B,3,32,32,21]`이며 query는 slot당 하나다. | 요청한 ray-group query 단위를 과거 구현한 것이 맞다. 자체가 성능 향상 증거는 아니다. |
| `2026-09-18-ss-gt-corr-proj-what0-content1-results-ko.md`, Wave0 | Source camera별 native image feature를 target-state write slot으로 운반한 뒤 unchanged ProjGrid를 사용한다. DEV8 source-exclusive recall32는 OFF0.7043,GT_CORR_RAW0.8113,WRONG0.6026이다. RAW−OFF+0.1071,CI[+0.0615,+0.1534],8/8양수다. | Ray/write-slot 기반 source transport의 양의 oracle 결과다. GT correspondence와 oracle frame 조건이며 learned query/Q2C의 효과는 아니다. |
| `2026-09-17-ss-static-ray-native-projection-results-ko.md` | Native ray routing은 correct/wrong을 구분했다. 그러나 learned local WHAT는 GT hard-ray보다 recall32−0.00077,CI[−0.00142,−0.00011]이었다. | Carrier와 주소의 효과를 learned attention의 추가 효과로 확대하지 않는다. |
| `2026-09-17-ss-target-local-ray-selection-results-ko.md` | Ray-pooled local query 실험은 사용자 요청으로 terminal budget 전에 중단됐다. Intermediate128의 local-Q/shuffled-Q 차이도 작았다. | Terminal 성공/실패를 새로 선언할 수 없다. |

원본은 `/root/dev/Pixal3D/.worktrees/anchor-hybrid-prestudy/`의 `pixal3d/experiments/ss_target_local_ray_selection.py` 및 `docs/superpowers/reports_final2/`다. 위 수치는 원본 리포트의 해당 scope이며 현재 DeformNeXt micro-IoU와 합산하지 않는다. Coordinate-invalidated2026-09-15 ray-guidance 결과는 이 판단의 근거로 쓰지 않았다.

## 후속 모델에서 바로잡을 계약

Native write slot을 `(camera,ray/patch)`로 명시한다. 같은 ray의 voxel group에서 query 하나를 만들고, 그 ray의 semantic Top-K/score를 읽는 attention 결과 하나를 만든 뒤 해당 camera의 native feature map/ProjGrid로 전달한다. Retargeting도 이 단일 ray feature에 적용해야 같은 ray에서 제각각 feature가 생기지 않는다. 여러 view의 동일 pixel index는 서로 다른 ray이며 섞지 않는다.

Query를 만드는 입력과 pooling 방식, target anchor와 source camera write slot의 관계, hidden/empty ray 처리, bilinear native projection과 ray bin의 관계를 각각 명시해야 한다. Native32² patch slot은 연속적인 정확한 pixel ray와 같은 용어로 과장하지 않는다. GT target geometry/depth/correspondence를 inference query나 write address에 넣지 않는다. 과거 positive oracle의 route를 그대로 새 모델 입력으로 사용하면 안 된다.

검증은 same-ray voxel의 순서를 바꾸어도 ray query/attention이 같음, view 간 group이 섞이지 않음, ray마다 feature 하나만 쓰임, native projection·zero-init·global branch·CFG가 보존됨을 포함한다. 최종 성능은 같은 retargeter의 Top-1 및 기존 anchor-patch arm과 고정 raw480/dev128×3 조건으로 비교한다. Source address/score/decoder 효과와 ray-group 효과를 분리하고 개선을 미리 가정하지 않는다.

이 문서는 source 감사와 수정할 계약이다. 새 ray-group production model의 구현·학습·성능 향상을 완료했다고 표시하지 않는다. 기존 V/P는 원래 recipe로 계속 학습하며 individual-voxel 및 anchor-patch 비교 근거로 보존한다.

## 사용자 지시 반영 및 실제 구현 시작

Retargeting adapter를 공통 블록으로 유지하고, 앞으로의 semantic matching/matching-aware attention 비교는 ray-group query를 공통 기반으로 한다. 개별 voxel 또는 단순 anchor-patch 모델의 자동 seed202 확장 factory는 실행 전 pending 상태에서 예약을 철회했다. 원래 descriptor·pending 이력·queue backup을 보존했으며 현재 학습·고정 최종 평가·게시 작업은 중지하거나 변경하지 않았다.

`wowo0709/feature-retargeting`을 별도 `/root/dev/Pixal3D/.worktrees/ray-group-query-stage-20261019` checkout으로 옮겼다. 기존 학습 checkout은 같은 commit에 detached 상태로 유지해 실제 forward source11/8개를 바꾸지 않는다. 새 `pixal3d/modules/ray_group_query.py`는 explicit camera/patch membership으로 `[B,V,R,D]`의 ray당 query 하나를 만들며 빈 ray는 valid=false로 구분한다. Native ray membership은 과거 mean-Q의 nearest32² patch-bin 계약을 따른다. Bilinear native ProjGrid를 다른 voxel splice로 대체하지 않는다.

Camera/batch 분리·same-ray voxel 순서 불변성·빈 ray·gradient·invalid index 관련 CPU 검사7개를 통과했다. 원래 dev128의 첫2개 실제 DeformNeXt camera를 써 과거 `target_proxy_ray_summary`의 unweighted query와 새 pooling을 대조했다. Group membership/populated mask가 같고, synthetic varying-latent probe의 두 query tensor 최대 차이가 모두0이었다. Source-camera별 활성 ray 수는421/414/416과840/852/849이며 ray당 최대 voxel 수는27과16이다. 이는 실제 camera에서 grouping 산술을 재현한 결과로, GT latent를 열거나 새 모델 성능을 평가한 결과가 아니다.

통합할 query의 feature 출처와 semantic key/score의 ray 단위 정의를 별도로 고정하고, ray feature 하나→retargeter→native per-view ProjGrid를 구현·검증한 뒤 같은 ray 기반 Top-1/native-spatial/decoder 대조를 구성해야 한다. Core pooling의 통과만으로 conditioner·GPU proof·본 학습이 완료됐다고 표시하지 않는다.
