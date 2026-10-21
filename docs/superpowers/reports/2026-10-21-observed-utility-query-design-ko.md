# 관측 입력으로 utility query를 구성하는 단일 변경

2026-10-21 UTC. `wowo0709/feature-retargeting`.
이 변경은 CPU 검증을 마친 모델 설계 후보다. 아직 GPU native/capacity proof나
새 학습을 시작하지 않았으며, 성능 향상으로 해석하지 않는다.

## 겨냥하는 실패 원인

Member posterior는 distinct target patch를 보존해 후보 coverage를 개선했지만,
native utility query는 source-camera ray의 전체 voxel feature를 평균한다.
실제32개 캐시 진단에서 그중 입력 target alpha 밖의 비율은 평균0.5134다.
Matching에 사용하는 target member를 보존하면서도 QKV/score utility에 배경 평균이
섞이는 것이 병목일 수 있다. 이는 frozen 최종 NoDQ의 query-only 검사로 먼저 확인한다.

## 바뀌는 연산 하나

기존 utility는 `pool(native_voxels, source_ray_ids)`다. 새 utility는 입력 target
foreground 바깥 voxel의 ray index만−1로 둔 `pool(native_voxels, observed_ray_ids)`다.
각 source-camera ray에는 여전히 query 하나가 있다. Native feature는 `[B,N,1024]`,
원래 ray ID는 `[B,3,N]`, 입력 foreground support는 boolean `[B,N]`다.

이 utility를 사용하는 score selector와 four-head Q만 바뀐다. 원래 unmasked
`pool(target_native_voxels, source_ray_ids)`는 R의 target context로 그대로 유지한다.
Member matching, candidate64, frozen D1 decoder, matching prior, Top-1 carrier,
source-use 판단, R, source-camera projection 및 full-ray voxel write도 유지한다.
유효한 query/source가 없으면 원래 native four-view prior를 보존한다.

추가 parameter나 channel은 없다. 물리적 DQ는 포함하지 않는 NoDQ 설계다.
Target input alpha를 query에 사용하는 것은 GT surface visibility로 voxel write를
막는 것과 다르다. 배경 voxel을 feature0이라는 관측값으로 평균하는 것도 아니다.

## 구현과 검증 범위

`observed_member_utility.py`는 기존 NoDQ forward의 두 ray-pool 호출 중 첫 utility만
변경한다. 호출 순서·tensor identity·원래 num_rays를 검증하고 두 번째 target pool은
원래 함수와 인자로 실행한다. 공유 NN 파일이나 state_dict 구조를 수정하지 않는다.
Context를 벗어나거나 오류가 나면 원래 forward/pool 함수를 복구한다.
통계는 scalar 누계만 보관한다.

실제 관측 utility tensor가 head checkpoint의 입력으로 전달되므로, context를 제거한
후 backward하더라도 재계산에 올바른 query가 남는다. CPU에서 다음을 확인했다.

- 기존 eval-only single-argument query intervention과 최종 출력·R residual 정확히 일치
- Cold feature output0에서 기존 native-prior 출력 보존
- Checkpoint backward와 직접 forward/backward의 모든 parameter gradient 정확히 일치
- Query/source 결손의 원래 native fallback과 DDP zero-gradient edge 보존
- Parameter/buffer 값 보존, 예외 복구, native CLI 전달 및 runtime 메타데이터

새 설계7개와 기존 query/readout23개, 총30개 검사가9.80초에 통과했다.
이는 CPU core/runtime 검증이다. 기존 NoDQ의 GPU proof를 새 query 설계의 proof로
재사용하지 않는다.

## 실행 전 의사결정

최종 NoDQ의 같은32개·같은 weight/noise에서 query support 대조를 실행하고,
matching-prior-only / readout-logprior 제거 / feature correction0 진단도 함께 본다.
Query-only 결과는 학습하지 않은 입력 분포 변경이므로, 음성 결과가 새 설계의
재학습 가능성을 완전히 배제하지는 않는다. 이 한계를 포함해 원인에 맞는 변경
하나를 선택한다. 더 많은 epoch나 넓은 조합 탐색으로 음성 결과를 덮지 않는다.

선택하면 자체 initial32 native identity·실제 fullbatch96×3 capacity proof를 수행하고,
같은 LoRA4210 시작점, seed101,480 update, global96/micro96/accum1로 원래 NoDQ와
비교해야 한다. Budget·optimizer·sampler·dataset/camera/latent는 바꾸지 않는다.
Positive 후보는 같은 Top-1/spatial 및 독립 seed 확인과 full-cascade metric이 필요하다.
Query 하나의 변경만으로 matching decoder의 독립 효과까지 입증했다고 주장하지 않는다.
