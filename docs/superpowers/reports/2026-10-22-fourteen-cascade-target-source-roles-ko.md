# 완료된14개 SS 모델의 target/source 분리 평가

기존14개 모델·448개 생성 case·1344개 렌더 view 전체에서 edited target camera8과
adapted source camera9/10을 분리했다. 기존 source/target 분리 분석은 V1·R·Direct
세 모델에 한정돼 있었다. 저장된 generation·render·공식 GPU DreamSim/CLIP 결과를
재사용했으며 추가 inference, metric model forward 또는 optimizer update는 없다.

## 동일 조건과 실제 검증

각 모델32개·32 source cluster·생성 noise1회다. SS만 바뀌고 Shape5124210,
Shape102416850 및 V1 PBR762를 고정한 SS-stage cascade 비교다. 진행 중인 새
PBR1900 학습이나 SS/PBR을 함께 바꾼 full 모델 비교가 아니다.

Generation에서 GT geometry를 사용하지 않았고 생성 후 source bbox 좌표계에서
surface20k CD/F-Score를 계산했다. Per-output alignment는 없다. Render3개·공식
perceptual3개의 pixel NPZ SHA 및 실제 GT RGB/alpha hash, GT-only crop 및
cross-model reference 일치를 검사했다. Reference는 **M1 deformed geometry에
원래 source material을 적용한 동일 GT**이며 undeformed M0 이미지를 복사한 것이 아니다.

기존14모델의 all-view 평균과 paired CI를 모두 재현했으며 최대 차이는0이다.
원래 결과는 유지하고 별도 source/target 집계를 추가했다. Source preservation은
camera9/10의 LPIPS·DreamSim·CLIP, target alignment는 camera8의 PSNR·SSIM·M-IoU와
asset-level CD·F-Score로 구분한다. Geometry metric을 카메라마다 새 표본으로 세지 않는다.

## 모델별 결과

| 모델 | Target PSNR ↑ | Target M-IoU ↑ | Source LPIPS ↓ | Source DreamSim ↓ | Source CLIP ↑ | CD ↓ | F-Score ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 24.4128 | .88047 | .053565 | .152908 | .941296 | .012458 | .849527 |
| Retargeting | 24.3784 | .88500 | .054451 | .156243 | .940763 | .012115 | .861474 |
| Direct | 24.4585 | .88828 | .053554 | .149921 | .943565 | .012264 | .856702 |
| Mean Top-1 | 24.2735 | .87309 | .053592 | .150776 | .942293 | .012468 | .851792 |
| Mean weighted | 24.2945 | .88512 | .052775 | .149339 | .945271 | .012450 | .851999 |
| Mean attention | 24.4681 | .88269 | .054104 | .155620 | .939432 | .012406 | .852842 |
| Mean spatial | 24.3991 | .88539 | .053948 | .155931 | .941144 | .012584 | .852896 |
| Member Top-1 | 24.2001 | .87942 | .053648 | .152455 | .944819 | .012438 | .850330 |
| Member weighted | 24.1879 | .88402 | .053062 | .156618 | .943944 | .012470 | .850171 |
| Member attention | 24.3691 | .88321 | .054265 | .150726 | .941583 | .012559 | .848634 |
| Feature NoDQ | 24.1946 | .88104 | .054379 | .155964 | .938890 | .012333 | .856130 |
| Feature DQ | 24.3251 | .87928 | .052450 | .153650 | .943086 | .012213 | .857734 |
| Feature Top-1 | 24.1680 | .88240 | .053330 | .147481 | .943648 | .012324 | .852346 |
| Feature spatial | 24.2587 | .89323 | .053844 | .156246 | .942011 | .012041 | .863882 |

SSIM을 포함한14행 전체8 metric은 canonical `target-source-metrics.csv`와 W&B table에 있다.
각 역할의45개 paired contrast와 target/source 효과 차이도 보존했다.

## 모듈 판단에 필요한 paired 결과

아래 CI는 source-cluster bootstrap4000회/seed42017의 탐색적 개별95% CI다.
여러 arm·metric에 대한 보정은 하지 않았다. 단일 학습 seed·반복 사용한 개발32개에서
얻은 관측이며 독립 confirmation이나 최종 test 결과가 아니다.

| 비교 | Metric | 차이 | 95% CI | 현재 해석 |
|---|---|---:|---|---|
| Member weighted − Mean weighted | Source DreamSim ↓ | +.007279 | [+.000464, +.014759] | 후보 탐색 개선이 이 perceptual 개선으로 연결되지 않음 |
| Member attention − Member weighted | Target PSNR ↑ | +.1812dB | [−.1303, +.4731] | Attention 추가 이득 불확실 |
| Member attention − Member weighted | Source DreamSim ↓ | −.005893 | [−.013763, +.002524] | Attention 추가 이득 불확실 |
| Member attention − Member Top-1 | Source CLIP ↑ | −.003235 | [−.007816, +.001451] | Top-K attention의 추가 보존 이득 미확인 |
| Feature NoDQ − Feature Top-1 | Source LPIPS ↓ | +.001049 | [−.000518, +.003438] | 학습형 feature 혼합의 추가 이득 미확인 |
| Feature NoDQ − Feature spatial | Target M-IoU ↑ | −.012185 | [−.020935, −.005319] | 이 조건의 semantic 경로가 spatial 기준을 넘지 못함 |
| Feature DQ − Feature NoDQ | Source LPIPS ↓ | **−.001929** | **[−.003663, −.000482]** | Query가 source 외형 적응에 도움을 줄 가능성 |
| Feature DQ − Feature NoDQ | Source DreamSim ↓ | −.002315 | [−.006347, +.001972] | 다른 perceptual metric에서는 불확실 |
| Feature DQ − Feature NoDQ | Target PSNR ↑ | +.1305dB | [−.1621, +.4360] | Target 개선은 미확인 |

Feature DQ는 저장된 실제 runtime schema상 RAW deformation query2048 channel,
R 입력4096 channel을 갖는다. NoDQ는 해당 query가 물리적으로 없다. 과거 predicted
O/D occupancy·SDF9 channel이나 앞으로 재현할 historical target-patch 구조와 같은
query로 취급하지 않는다. Source LPIPS의 한 탐색적 양성 결과로 Full에 채택하지 않는다.

Member candidate recall 개선과 final quality 개선을 계속 구분한다. 이번 역할 분리도
semantic matching+matching-aware attention의 안정적인 최종 이득을 입증하지 못했다.
반면 SS IoU만으로 query의 모든 효과를 부정해서도 안 된다. Source appearance에 대한
부분적 이득 가능성을 이후 동일 조건의 재현과 독립 확인에서 검토할 근거가 생겼다.

현재 native SS128×3의 Top-1+R 대조와 이14개 legacy arm은 구조·학습 계열이 다르다.
Native의 module attribution을 대신하지 않으며, 진행 중인 PBR과 이후 faithful SS,
독립 seed·선택 조합 DEV cascade·최종508개10 metric 범위를 유지한다. FID/KID는
이 작은 패널에서 새로 추정하지 않았다.

Canonical 경로는 `feature-readout-v1/completed-fourteen-cascade-view-roles-v1/`이다.
Prelaunch2파일1,570,253bytes와 실제 결과3파일2,725,813bytes를 file4 SHA256/size로
검증했다. 계산·logging·closure는 약24.4초였으며 CPU92/93만 사용했다.
[W&B target/source14모델 표](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/legacy-fourteen-cascade32-camera-roles-v1).
