# Ours 공통 TEST501 조건별 평가 진행

갱신: 2026-10-28 08:38:27 UTC.

같은 501개 사례·133 source cluster. Source와 geometry는 조건별 공통 evaluator 완료 후 표시한다.
Target view 지표는 별도 paired 계산을 사용한다. KID는 원값×1,000이다.

| 정합 | 모델 | LPIPS ↓ | DreamSim ↓ | CLIP ↑ | FID ↓ | KID ×1,000 ↓ | PSNR ↑ | SSIM ↑ | Mask IoU ↑ | CD ↓ | F@0.02 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | Selected | — | — | — | — | — | **25.139** | **0.9555** | **0.9124** | — | — |
| Raw | V1 | — | — | — | — | — | <u>24.778</u> | <u>0.9547</u> | <u>0.9019</u> | — | — |
| Source M0 ICP | Selected | — | — | — | — | — | **23.284** | <u>0.9459</u> | **0.8619** | — | — |
| Source M0 ICP | V1 | — | — | — | — | — | <u>23.220</u> | **0.9463** | <u>0.8584</u> | — | — |

굵게와 밑줄은 같은 정합 조건에서 V1·Selected의 1위와 2위다. Source 7방법 및 전체 10방법은 [raw 표](2026-10-28-raw-common501-interim-ko.md)와 [ICP 표](2026-10-28-ours-common501-target8-interim-ko.md)를 참고한다.

완료된 공통 evaluator 조건: 0/4. 이 문서는 중간 결과이며 원고의 최종 통합표는 전체 검증 후 생성한다.
