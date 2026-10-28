# 공통 TEST501 raw 중간 통합표

모든 방법의 생성 결과에 **source M0 ICP를 적용하기 전**, 동일한 501개 성공 사례에서 계산한 점추정이다. Source view 9/10의 LPIPS·DreamSim·CLIP·FID·KID와 target view 8의 PSNR·SSIM·mask IoU, target surface의 CD·F@0.02를 표시한다. KID는 원값×1,000이다. 굵게와 밑줄은 현재 채워진 값 중 최고와 두 번째다. Ours 두 raw 행의 source·geometry 지표는 공통 evaluator 완료값이고 target view 지표는 별도 paired 평가 완료값이다.

| 방법 | LPIPS ↓ | DreamSim ↓ | CLIP ↑ | FID ↓ | KID ×1,000 ↓ | PSNR ↑ | SSIM ↑ | Mask IoU ↑ | CD ↓ | F@0.02 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours Selected, T1+S3+C | **0.04927** | <u>0.13310</u> | <u>0.93855</u> | <u>38.64</u> | 0.726 | **25.139** | **0.9555** | **0.9124** | **0.01385** | **0.8241** |
| Ours V1, T1+S3+C | <u>0.04957</u> | **0.13192** | **0.93888** | **38.41** | <u>0.656</u> | <u>24.778</u> | <u>0.9547</u> | <u>0.9019</u> | <u>0.01420</u> | <u>0.8181</u> |
| Pixal3D 공식 RGB1, T1+C·1024 | 0.12108 | 0.38396 | 0.86424 | 80.33 | 8.535 | 15.774 | 0.9026 | 0.3979 | 0.09776 | 0.1659 |
| ReconViaGen, T1+S3 | 0.08098 | 0.19097 | 0.92331 | 45.95 | **0.638** | 18.88 | 0.9274 | 0.5927 | 0.06162 | 0.3899 |
| Pixal3D 기존, T1·추정 camera·1536 | 0.10647 | 0.26709 | 0.90716 | 68.38 | 5.786 | 16.44 | 0.9111 | 0.4130 | 0.10264 | 0.1538 |
| ShapeUp, T1+source mesh | 0.08545 | 0.30228 | 0.88254 | 74.44 | 7.196 | 17.76 | 0.9242 | 0.5289 | 0.07790 | 0.4571 |
| TRELLIS.2, T1 | 0.09428 | 0.26286 | 0.90268 | 61.31 | 3.017 | 17.68 | 0.9208 | 0.4613 | 0.08841 | 0.2966 |
| TRELLIS.2 + SpaceControl, T1+anchor | 0.09694 | 0.29018 | 0.89737 | 64.86 | 3.567 | 16.70 | 0.9157 | 0.4650 | 0.08704 | 0.2952 |
| MVAdapter + ReconViaGen, T1 | 0.10361 | 0.28805 | 0.89056 | 66.54 | 4.648 | 17.04 | 0.9120 | 0.4839 | 0.08091 | 0.2359 |
| CorrAdapter + ReconViaGen, T1 | 0.10303 | 0.28693 | 0.89137 | 65.68 | 4.610 | 17.05 | 0.9140 | 0.4762 | 0.08279 | 0.2320 |

T1은 편집 target RGB, S3는 원본 source RGB 3장, C는 제공 camera다. 방법별 입력과 native 생성 조건이 다른 시스템 비교이며, 조건별 paired 차이와 각 방법의 성공 수는 최종 표에서 함께 보고한다. ICP 적용 후 10방법 중간 표는 [별도 공통501 표](2026-10-28-ours-common501-target8-interim-ko.md), 정합 전후 target paired 값은 [정합 효과 보고](2026-10-28-source-m0-target8-interim-ko.md)에 있다.

외부 7방법의 raw는 기존 full-test per-sample 지표 및 Inception 특징을 최종 501 ID와 camera 9/10 순서로 재집계했다. 기존 공통502 FID/KID의 재현 최대 오차는 **2.57e-11**이고, ICP 결과에 쓰인 동일 reference 특징의 최대 차이는 **0**이다. 재집계 정본은 node16 `/home/youngwoo/retargeting-unified-test508/raw-external-common501-replay-v2/summary.json`, SHA256 `f850794bb1de40521fc542c729987dab7fe730097768f1edc5e4f421a0a0f0ac`다. 공식 RGB1 raw는 기존 공통501 평가, target8은 paired 평가, Ours geometry는 위 ICP 보고서의 surface cache 원본이다. Selected raw는 공통 evaluator 결과로 갱신했고 나머지 조건도 완료 시 다시 검증한다.

이 summary와 7개 사례별 결과는 `/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/raw-external-common501-replay-v2/`에도 복사해 각각 SHA256 일치를 확인했다. Raw 최종 표 watcher는 Ours 네 조건과 공통 주 표가 닫힌 뒤 별도 10방법×10지표 표를 자동 생성한다.

Ours 각 조건의 공통 evaluator 결과는 전체 완료를 기다리지 않고 [조건별 실시간 표](2026-10-28-ours-common501-live.md)에 순차 반영한다.
