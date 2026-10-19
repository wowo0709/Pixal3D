# Matching 정보의 전달 경로와 PBR continuation 검증

## 현재 판단

Semantic Top-K를 계산하고 attention에서 여러 후보를 사용한다는 사실만으로 품질 향상이 보장되지는 않는다. 기존 input-side feature attention은 native semantic fusion gate에 의해 새 feature의 영향이 약해질 수 있었다. Physical deformation query와 bounded direct feature retargeting은 이 병목을 겨냥한 별도 후보다. 기존 ray 단위 carrier와 retargeting 경로를 유지하고, 추가 보정만 native residual 경로로 전달한다.

현재 validation 성능 개선은 확립되지 않았다. 같은 step96에서 시작한 SS R 대조군과 input-side attention의 고정32 사례 결과는 다음과 같다. 이는 per-asset IoU 평균이며 최종128×3 pooled-count 평가와 집계 방식이 다르다.

| 전체 update | R control | Input-side feature attention | 차이 | Source-cluster paired 95% CI |
|---|---:|---:|---:|---:|
| 192 | 0.44216347 | 0.44190640 | -0.00025707 | [-0.00140694, 0.00092577] |
| 288 | 0.44592041 | 0.44533058 | -0.00058983 | [-0.00222685, 0.00081984] |
| 384 | 0.44748878 | 0.44793061 | +0.00044184 | [-0.00249228, 0.00401160] |

세 구간 모두 신뢰구간이0을 포함한다. Best epoch를 고르지 않고 final480에서 비교한다. 이후의 physical-DQ/direct SS 학습은 별도 capacity 검사 뒤 진행하도록 연결돼 있다.

## 실제 PBR warm continuation 검사

기존 PBR V1 step762의 complete frozen base + LoRA checkpoint와 원본 Adam state를 재사용했다. NN 파일을 다시 다운로드하거나 데이터/cache를 재구축하지 않았다. 원본 file4의 NN, misc, config SHA256 및 크기를 실제로 확인했다.

- 공통 NN LoRA360개의 Adam moment와 step762를 복원했다. Direct arm에는 R6 + attention9 + DQ1만 fresh state로 추가했다.
- 동일 TRAIN asset9,823 tokens, micro1에서 첫 denoiser 예측 SHA256와 loss가 두 arm에서 정확히 같았다. 예측 SHA256는 `b27147a16f51cb06855de0fc7a1d92e6bb7cb6e2607e7c68e7bd89ba24289a2b`이다.
- 임시4 update 뒤 control360개, direct376개 parameter가 모두 변경됐다. NN moment는 step766, 새16개 moment는 step4였다. Frozen NN base와 matcher는 유지됐다.
- 두 arm 모두 실제4 사례의 PBR generation, decoding, rendering metric을 계산하고, validation 직후 추가 train update까지 통과했다. RGB/noise/GT PBR/GT Shape/source/seed는 arm 사이에서 같았다.
- 실제 최대 할당 메모리는 control19.3384 GiB, direct19.4571 GiB였다. 이는 micro1 correctness 검사이며 full microbatch의 capacity 증거가 아니다.

원본 source config는 `balanced_sampler=False`이며 실제 sampler는 `ResumableSampler`다. Source world3/idx16의 global offset48을 새 world2/idx24 또는 world4/idx12로 옮긴다. 실제 unbalanced sampler의 다음 global24 asset 집합이 유지되는 CPU 검사2개를 추가했다. Balanced sampler 검사만으로 원본 동작을 검증했다고 해석하지 않는다. Adam 검사와 outside-proposal 검사까지 총7개가 통과했다.

PBR의 학습 예산은 공통 source762에서 total1905까지1,143 update continuation으로 맞춘다. Global24, accumulation1을 목표로 실제 maximum31,900-token TRAIN 사례와 validation 뒤 train step을 포함한 capacity 검사로 DDP2/micro12 또는 DDP4/micro6을 선택한다. 아직 fullbatch capacity와 PBR 본 학습이 수행됐다는 뜻은 아니다. Read-only initial32 rendering 동등성 검사도 본 학습과 구분한다.

## Correspondence 후보와 attention 조합의 효과 분리

기존 wrong-value-address 검사는 Top-K 안의 값을 섞으므로, 선택된 후보 집합 자체의 유용성과 후보별 가중치의 유용성을 구분하기 어렵다. 새 outside-proposal 검사는 다음 조건을 유지한다.

- Original Top-1 carrier, candidate key, cached matching score, attention probability를 유지한다.
- 추가 attention 경로의 appearance와 RAW Omega value만 semantic Top64 밖의 같은 asset RGB-valid source patch로 함께 바꾼다.
- GT correspondence, GT deformation, 다른 asset의 feature를 넣지 않는다. 후보 밖 donor가 부족하면 donor를 재사용하고, 전혀 없으면 해당 query는 유지하고 기록한다.

실제 native SS one-asset entry에서2,624개 valid slot이 교체됐다. 이는 실행 검사다. 학습된 final480 weight의 동일32 패널 검사는 queue에 추가했으며, 기존 다섯 진단을 유지했다. 이 saved-weight intervention은 사용자 정의 native-spatial NoCorr 재학습 arm을 대체하지 않는다.

최종 후보 평가는 새 matched R 대조군과 V1 외에, 이전에 양의 효과가 관측된 original positive R480을 반드시 포함하도록 별도 분석을 연결했다. 후보 탐색 결과는 exploratory로 기록하며, 양의 결과가 나와도 독립 training seed 및 적절한 물리적 module ablation 없이 모델 채택이나 전체 실험 완료를 주장하지 않는다.

## 기록

공통 canonical root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`이다.

- `epoch288-matched-v2`, `epoch384-matched-v2`: 실제 paired validation 분석과 file4 SHA receipt.
- `direct-retargeting-v1/PBR-joint-continuation-v1/warm-correctness-v1`: native paired PBR 검사, source snapshot, 실제 file4 closure.
- `direct-retargeting-v1/outside-proposal-entry-probe-v1`: 실제 native 실행과 intervention event.
- `direct-retargeting-v1/proposal-and-strong-reference-plan-v2`: 고정 final 평가 recipe 및 추가3개 queue 연결 기록. 미사용 v1 plan의 잘못 상속된 adapter flag는 GPU 실행 전에 발견했고 v2에서 수정했다. v1은 실행하지 않았다.

PBR rendering은 표준 GT target Shape를 조건으로 쓰는 texture-stage 진단이다. Crop LPIPS는 target alignment 진단이며 source preservation LPIPS로 보고하지 않는다. Full cascade, novel-view, 최종10개 metric의 성능 향상은 이 검사로 추론하지 않는다.
