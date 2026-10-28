# 공통 TEST501: 외부 7방법·공식 RGB1 중간 평가

2026-10-28 14:27 KST. **Ours V1/Selected 네 조건이 아직 평가되지 않았으므로 최종 논문 주 비교표가 아니다.** 같은 TEST508에서 외부 7방법과 공식 Pixal3D RGB1 생성물이 모두 존재하는 501사례만 사용했다. 원래 native 성공 수는 별도다: 공식 RGB1은 507/508 생성 성공, 공통 평가는 501/508이다.

## 완료 검증

- node16의 `unified-nine-method-final-evaluation-v4/evaluation`에는 9개 방법별 결과가 있고, 모두 **같은 501개 고유 sample ID**, source cluster 133개, source camera 9/10의 이미지 1,002장, `full_metric_computation=true`다. 공식 RGB1의 raw와 source-M0 두 조건은 별도 새 평가 결과다.
- 외부 7방법의 full502 feature cache에서 한 실패 ID만 제외해 동일 501사례로 재집계했다. replay 영수증은 `PASS_EXACT_CACHED_FEATURE_SUBSET_REPLAY`이고 기존 full 모집단 재현 최대 절대 오차는 **3.34e-11**이다. FID/KID도 동일한 501사례·1,002이미지 풀에서 다시 계산됐다.
- target camera 8의 외부 방법 및 공식 RGB1 paired JSON은 동일한 501 ID를 가진다. 아래 표는 모두 **생성 후 원본 source M0만 사용한 Sim(3)** 상태다. edited target M1은 정합에 사용되지 않았다.

## Source preservation 및 분포, 공통501·+M0

| 방법 | LPIPS ↓ | DreamSim ↓ | CLIP ↑ | FID ↓ | KID ×1,000 ↓ |
|---|---:|---:|---:|---:|---:|
| ReconViaGen | 0.07315 | 0.17719 | 0.92736 | 45.21 | 0.637 |
| Pixal3D 기존 estimated-camera·1536 | 0.08872 | 0.20355 | 0.92106 | 54.06 | 2.806 |
| ShapeUp | 0.08527 | 0.29472 | 0.88632 | 72.49 | 6.411 |
| TRELLIS.2 | 0.09093 | 0.25387 | 0.90178 | 57.73 | 1.996 |
| TRELLIS.2 + SpaceControl | 0.09134 | 0.28443 | 0.89448 | 62.19 | 2.970 |
| MVAdapter + ReconViaGen | 0.09459 | 0.27942 | 0.89333 | 64.65 | 4.169 |
| CorrAdapter + ReconViaGen | 0.09388 | 0.27669 | 0.89402 | 62.95 | 3.735 |
| Pixal3D 공식 RGB1 given-camera·1024 | 0.09790 | 0.36294 | 0.86908 | 79.64 | 8.049 |

## Target alignment, 공통501·+M0

| 방법 | PSNR ↑ | SSIM ↑ | Mask IoU ↑ | CD ↓ | F@0.02 ↑ |
|---|---:|---:|---:|---:|---:|
| ReconViaGen | 19.876 | 0.9310 | 0.6836 | 0.03607 | 0.5848 |
| Pixal3D 기존 estimated-camera·1536 | 18.767 | 0.9216 | 0.6164 | 0.04419 | 0.4846 |
| ShapeUp | 18.019 | 0.9218 | 0.6017 | 0.05045 | 0.5170 |
| TRELLIS.2 | 18.278 | 0.9210 | 0.5633 | 0.05168 | 0.4779 |
| TRELLIS.2 + SpaceControl | 17.373 | 0.9179 | 0.5639 | 0.05173 | 0.4726 |
| MVAdapter + ReconViaGen | 18.264 | 0.9180 | 0.5877 | 0.05236 | 0.4175 |
| CorrAdapter + ReconViaGen | 18.292 | 0.9202 | 0.5835 | 0.05203 | 0.4131 |
| Pixal3D 공식 RGB1 given-camera·1024 | 17.765 | 0.9212 | 0.5302 | 0.05603 | 0.3944 |

공식 RGB1 자체에서 raw→+M0는 F@0.02 **0.1659→0.3944**, mask IoU **0.3979→0.5302**, target PSNR **15.774→17.765**로 바뀐다. 이는 같은 출력물의 source 기반 후처리 전후 **점추정**이며, Ours나 다른 방법의 유의미한 우열을 뜻하지 않는다. 기존 Pixal3D 행과 새 공식 RGB1 행은 카메라 입력뿐 아니라 native runtime·cascade 해상도도 다르므로 두 행의 차이를 카메라 효과로 해석하지 않는다. ReconViaGen의 낮은 KID도 이 공통 풀에서의 분포 점추정이며 단일 사례 편집 성공률은 아니다.

## 정본과 무결성

- Node16 root: `/home/youngwoo/retargeting-unified-test508/unified-nine-method-final-evaluation-v4`
- `evaluation/summary.json` SHA256 `fa6baf92756d300eb926083c1beb7ba9c1aa4759ab9cb47b6c0e067fa0840010`
- `evaluation/cache-replay-provenance.json` SHA256 `2357af577144d62ce2add3eb1762899fb2522b965127cb677f83ae13d092eb52`
- `target8-external-paired.json` SHA256 `bd7b5e21c8afcbc51098ed16f8449fc95366bd7617fbf5b1d6d9bf53e029b8c4`
- `target8-official-paired.json` SHA256 `295d8eb76b149cd1655de826fbc3c657d64a39c409a1108f8d869cb512a41c78`
- File4 final receipt: `/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/official-rgb1-scale-ungiven-common-success-v2/final-evaluation/completion.json`, status `PASS_NINE_METHOD_COMMON_SUCCESS_TEN_METRICS`.

다음 단계는 Ours 공통502의 32청크 렌더 영수증, Ours 4조건의 동일 501 ID 평가, 10방법×10지표 최종 표와 paired interval이다. 본 중간 표를 최종 논문 표로 복사하지 않는다.
