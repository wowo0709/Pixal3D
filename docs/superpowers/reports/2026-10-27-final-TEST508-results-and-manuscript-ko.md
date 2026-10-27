# 동결 TEST508 평가와 최종 원고

2026-10-27 UTC. DEV 결과만으로 고른 Fixed-SS/Top-1-R-PBR 조합과 평가
프로토콜을 TEST를 보기 전에 동결했다. TEST508의 다섯 입력·모델 조건을
각각 508개, 합계 2,540개 case로 실행했다. 277개 asset group/1,385개
case는 원래 producer의 완료·검증된 archive를 불변 재사용했고, 231개
group/1,155개 case는 새로 계산했다. v636 campaign audit와 실제 file4의
SHA256·크기 검증이 통과했다. 첫 metric 활성화와 collector callback은
각각 context의 `campaign_root` 필드 누락으로 output 생성 전에 실패했다.
새 context(v667, v668)에 그 필드만 보완하고 기존 505개 수치 source,
선택 모델, scored archive와 통계 정의를 유지해 재실행했다.

## TEST508 주 결과

각 평균은 원래 deformation weighting을 따른다. Source preservation은
LPIPS·DreamSim·CLIP, global naturalness는 FID·KID, target alignment는
PSNR·SSIM·M-IoU·CD·F-Score다. CD/F-Score는 공통 geometry surface 정의를
사용한다. FID/KID는 공통 1,524 input-camera 이미지 pool의 분포 요약이다.

| 모델 | LPIPS↓ | DreamSim↓ | CLIP↑ | FID↓ | KID↓ | PSNR↑ | SSIM↑ | M-IoU↑ | CD↓ | F-Score↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot source RGB4 | 0.061884 | 0.153975 | 0.943278 | 36.0899 | 0.00149059 | 22.4062 | 0.936489 | 0.814896 | 0.0167443 | 0.766221 |
| Zero-shot edited RGB4 | 0.058324 | 0.154276 | 0.942090 | 35.0242 | 0.00161819 | 23.5416 | 0.945824 | 0.882545 | 0.0148793 | 0.808255 |
| Vanilla V1 | 0.057245 | 0.151460 | 0.942854 | 34.6588 | 0.00175376 | 23.8360 | 0.946375 | 0.889032 | 0.0141140 | 0.828473 |
| Retargeting Top-1 | 0.056329 | 0.151162 | 0.943177 | 34.9304 | 0.00182203 | 23.9966 | 0.947295 | 0.898210 | 0.0136234 | 0.834758 |
| Selected Fixed-SS/Top-1-R-PBR | 0.056201 | 0.152069 | 0.943018 | 34.3614 | 0.00160111 | 24.0283 | 0.947532 | 0.901072 | 0.0136001 | 0.834532 |

사전 선언한 두 대조×여덟 per-case endpoint의 16-endpoint family에서
source-cluster 보정 95% 구간을 썼다. Selected−V1 중 LPIPS
−0.001044 [−0.001715,−0.000446], target PSNR +0.1923 dB
[+0.0703,+0.3201], SSIM +0.001156 [+0.000538,+0.001833], M-IoU
+0.012041 [+0.005487,+0.021451]는 개선 방향으로 0을 제외한다.
DreamSim·CLIP·CD·F-Score 구간은 0을 포함한다. Selected−retargeting은
여덟 구간 모두 0을 포함하며 DreamSim과 F-Score 점추정은 약간 나쁘다.
FID는 Selected가 최저 점추정, KID는 zero-shot source RGB4가 최저
점추정이다. 두 분포 지표로 유의성 주장을 하지 않는다.

Source-only zero-shot은 변형 전 target camera를 포함한 source RGB4를
입력한다. Edited zero-shot은 편집된 target RGB1+변형 전 source RGB3다.
둘의 차이는 편집 target 정보 접근에 따른 기술 비교이지 학습 모듈의
인과 효과가 아니다.

## 논문 claim과 한계

완성된 pipeline은 vanilla fine-tuning에 비해 위 네 endpoint에서
보정 구간이 개선을 지지한다. SS geometry 개발 arm의 spatial→raw→decoder→
Fixed 대응 관계와 같은 geometry의 PBR appearance 비교를 각각 제시한다.
다만 DEV의 matched 48-comparison family 구간은 모두 0을 포함한다.
Matching retrieval coverage와 attention probability mass는 기전 진단이며
생성 품질 개선의 증명이 아니다. 최종 TEST에서도 retargeting 단독 기준과의
추가 우위는 확인되지 않았다. 논문에서 개별 semantic matching, score,
ray aggregation, retargeter의 필수성이나 모든 지표 개선을 주장하지 않는다.

TEST는 한 개 frozen full-pipeline checkpoint와 자산별 한 개 generation
noise에 조건부다. 두 SS seed와 PBR training QA는 full-cascade seed 불확실성을
대체하지 않는다. TEST source는 DeformNeXt 내부에서 분리됐으나 기존
multi-view pretraining과의 가능한 overlap은 감사되지 않았다. 사용된
input camera 평가로 novel-view 편집 품질을 주장하지 않는다.

## 보존된 근거와 원고

- Cascade: `actual-selected-five-model-TEST508-remaining-cascade-v636/result/audit.json`, status `PASS_ACTUAL_SELECTED_FINAL_CASE_CAMPAIGN`.
- 열 지표: `actual-selected-five-model-TEST508-tenmetrics-v636/ten-metrics/audit.json`, status `PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_ALL_TEN_METRICS`.
- Main 표·정성: `actual-frozen-TEST508-main-table-v637/audit.json`, `actual-frozen-TEST508-qualitative-v637/audit.json`, 둘 다 PASS와 실제 file4 closure.
- W&B readback: `actual-final-TEST508-WandB-verification-v638/audit.json`, status `PASS_ACTUAL_FROZEN_TEST508_FINAL_WANDB_TABLE_INTERVALS_AND_IMAGES_VERIFIED`와 실제 file4 closure.
- 원고 입력: `actual-final-TEST508-manuscript-inputs-v639/audit.json`과 file4 closure. 원래 생성한 표·그림의 수치/selection은 유지했고, TeX font와 float 배치만 조정했다.
- 최종 본문은 `docs/superpowers/paper/retargeting-final-manuscript.tex` 및 `final-results/`에 있다. 원고 빌드 PDF는 `actual-final-TEST508-manuscript-build-v669/build/retargeting-final-manuscript.pdf`이며, 저장소의 [`2026-10-27-retargeting-final-manuscript.pdf`](2026-10-27-retargeting-final-manuscript.pdf)에도 같은 bytes를 복사했다. 20쪽이며 오류·미해결 인용·overfull 없이 조판했고 전체 페이지를 렌더링해 표·그림 위치와 본문을 검수했다. 두 PDF의 SHA256은 `47f884e0c159d1ca307d23c0e32e6aa3d837ec8bac42b210ecb50bf2729accb9`다. 원고 source·PDF·렌더링 검수 이미지·audit 등 102개 파일, 9,812,352 bytes를 실제 file4와 SHA256·크기로 대조해 `actual-final-TEST508-manuscript-build-v669/file4-closure.json` PASS로 보존했다.

여기 적은 상대 경로는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/PBR-native-science-runtime-v1/` 아래다. 원고 repository commit/push와 전체 completion audit은 별도 완료 기록으로 덧붙인다.
