# 10개 SS 대조군 최종 DEV 비교와 진행 상황

기준: 2026-10-24 02:57 UTC. 학습 seed101 하나, 동일 DEV128개·93 source cluster·생성 noise3개 평균이다. Noise 반복은 독립 학습 seed 반복이 아니다. 전체 실험과 최종 모델 선택은 진행 중이다.

| 구조 | Voxel IoU |
|---|---:|
| Vanilla V1 | 0.46735504 |
| Spatial correspondence + retargeting | 0.47124225 |
| Raw semantic Top-1 + retargeting | 0.47235972 |
| Decoder correspondence, cosine Top-1 + retargeting | **0.47300814** |
| Decoder correspondence, learned-score Top-1 + retargeting | 0.47260025 |
| Decoder Top-1, retargeting 제거 | 0.46867128 |
| Top-1 V projection | 0.47221778 |
| Fixed Top-K | 0.47274517 |
| 기존 Full | 0.47207818 |
| Full에서 additive attention score prior 제거, QK-only | 0.47244621 |

각 evaluation과 실제 종료 terminal의 closure, sample/source/noise/GT 대응과 metric 범위를 검증했다. 원래 module family10, pipeline family2를 완료된 비교 개수로 줄이지 않았다. 전체12 arm의 결과로 표시하지 않는다.

Full−QK-only는 −0.00036803, module family 보정 CI95는 [−0.00237189, +0.00169370]이다. Additive matching-score attention prior의 추가 효과는 확인되지 않았다. QK-only도 semantic 후보 검색과 ranking score를 유지한다. 사용자가 정의한 Top-1 arm과 다르다.

Decoder-cosine−raw semantic은 +0.00064843, raw semantic−spatial은 +0.00111747이나 각각 family 보정 CI가 0을 포함한다. 현재 최고 평균 구성이라는 사실과 각 모듈의 필요성 입증을 구분한다. Spatial→semantic 비교는 후보 수까지 바뀌는 pathway 비교이며, 같은 K에서 correspondence 주소만 바꾼 효과가 아니다.

Top-1 문맥의 retargeting 추가는 +0.00392896으로 가장 큰 모듈 차이다. 일반 CI95 [0.00060345, 0.00755687]는 양수지만, 사전 지정 module family10 보정 CI [−0.00052537, 0.00898685]는 0을 포함한다. Retargeting 필수성을 확정하지 않는다. Full−V1은 +0.00472314, pipeline family2 보정 CI [0.00028801, 0.00939391]이며 이 결과의 범위는 SS conditioning이다. 전체 3D cascade 성능이나 독립 학습 seed 재현으로 확대하지 않는다.

## 실제 실행 상태

- GPU0–3: raw semantic Top-K+feature PBR, 601/1900 update. Global24, rank당 micro6, accumulation1. 최근30 update compute 중앙값15.58초, rank0 allocated peak44.31 GiB. 현재 프로세스는 실행 중이다.
- GPU6: QK-only DEV128×3 종료 후 E_FIXED_RAW proof와 3-update micro96 capacity를 통과했다. 이 arm과 후속 learned arm은 후보 선택과 전달 feature의 evidence를 맞춘 상태에서 fixed combination과 learned attention을 비교한다. 새로운 지엽적 탐색 arm을 추가하지 않았다.
- GPU7: 개별 voxel-query 대조군 복구 실행, 14/480 update. Global96, micro96, accumulation1. 최근 compute 중앙값142.64초, 현재까지 allocated peak91.55 GiB. 기존 run은146 update 후 OOM으로 종료됐다. FFN 내부 asset chunk24→12만 바꾼 새 실행이며, 동일 frozen block의 BF16 output/input gradient가 실제 검사에서 bitwise 동일했고 native identity와 3-update capacity도 통과했다. Capacity allocated peak는92.6053→91.4803 GiB로 감소했다. 아직 기존 실패 지점을 지나지 않아 장기 OOM 안정성을 확정할 수 없다.

PBR step380의 동일4개 GT-shape anchor 검증은 step0 대비 LPIPS 0.095142→0.094741, SSIM 0.949160→0.949242, PSNR 14.94645→14.90044였다. 변화가 작고 방향이 혼재한다. 이는 generated geometry, source9/10 preservation 또는 full-cascade 개선의 증거가 아니다. 새 epoch 평가 전까지 이 값이 최신 완료된 stage rendering 비교다.

원래12 arm 중 query-unit과 mean-match 대조군이 남았으며, 후속 shared-evidence fixed/learned 비교, 동일 geometry·material 조건의 SS/PBR joint 평가, 선택 모델의 독립 학습 seed 확인을 진행한다. 실제 최종 모델과 평가 설정 동결 전에는 TEST508을 사용하지 않는다. Source LPIPS/DreamSim/CLIP, target PSNR/SSIM/M-IoU/CD/F-Score 및 global FID/KID 최종 비교는 아직 완료되지 않았다.

## 증거 보관

실제10모델 수집 결과는 file4 experiment root의 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/available-primary-ten-model-DEV128x3-v191/{audit.json,file4-closure.json}`에 보관했다. 앞선8모델 W&B remote 수치·CI·이미지·artifact 확인은 `available-primary-eight-model-WandB-v189/actual-verification-v3/`에 있다. 이번10모델 결과를8모델 W&B에 업로드한 것으로 표시하지 않는다.

추가된 local raw duplicate 정리 executor의 CPU 회귀검사16개가 통과했고 이전 관련 검사와 합계436개다. 실제 파일서버 metadata fresh-hash readback도 통과했다. 실험 raw/checkpoint 삭제나 final-case GPU 실행은 수행하지 않았다. 최종 case campaign과 동적 모델 수 metric 수집의 실행 연결도 아직 남아 있다.
