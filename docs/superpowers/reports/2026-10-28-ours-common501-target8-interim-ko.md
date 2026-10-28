# Ours 공통 TEST501 target 이미지 중간 결과

2026-10-28. V1과 Selected의 동일한 501개 편집 사례를 target camera 8에서 비교했다. 전체 508개 중 외부 baseline과 공식 Pixal3D 및 두 Ours가 모두 성공한 교집합이며, source asset cluster는 133개다. V1과 Selected는 동일한 입력 이미지·카메라·image-only scale·고정 downstream checkpoint를 사용한다. **Source 지표와 geometry 지표는 아직 평가 중이므로 최종 통합 비교표가 아니다.**

| 생성 후 정합 | 모델 | PSNR ↑ | SSIM ↑ | Mask IoU ↑ |
|---|---|---:|---:|---:|
| Raw | V1 | <u>24.7778</u> | <u>0.954732</u> | <u>0.901854</u> |
| Raw | Selected | **25.1390** | **0.955505** | **0.912424** |
| Source M0 Sim(3) | V1 | <u>23.2197</u> | **0.946264** | <u>0.858382</u> |
| Source M0 Sim(3) | Selected | **23.2843** | <u>0.945914</u> | **0.861863** |

굵게와 밑줄은 각 정합 조건에서 두 모델의 1위와 2위다. Source M0 정합은 생성 후 source 원본 mesh만 이용하며 edited target M1은 정합에 사용하지 않는다.

- Raw에서 Selected−V1은 PSNR **+0.3612**, SSIM **+0.000773**, mask IoU **+0.01057**이다. 각각의 95% source-cluster bootstrap 구간은 [0.2237, 0.4956], [0.000085, 0.001456], [0.00752, 0.01399]다.
- Source M0 정합 후 Selected−V1은 PSNR **+0.0646**, SSIM **−0.000350**, mask IoU **+0.00348**이며, 세 95% 구간 모두 0을 포함한다. 위 구간은 다중 비교 보정 전의 탐색적 구간이다. 확정 주장은 이후 8개 endpoint의 보정 구간과 claim audit을 기다린다.
- V1과 Selected 모두 source M0 정합 후 target PSNR·SSIM·mask IoU의 점추정이 raw보다 낮다. 따라서 정합 전후를 함께 제시해야 하며 source 기준 정합이 편집된 target을 항상 개선한다고 해석하지 않는다. Source 보존과 3D geometry의 변화는 아직 확인되지 않았다.
- Selected−V1은 matching·ray 단위 집계·retargeting과 선택된 PBR 경로가 함께 바뀐 **파이프라인 비교**다. 이 차이를 개별 모듈의 독립 효과로 해석하지 않는다.

원본: node16 `/home/youngwoo/retargeting-unified-test508/ours-four-final-evaluation-v1/target8-ours-four.json`, SHA256 `4056ec0287e281609f12cf3b312761c59980a18d2fec83b1734a2f4d0f070e30`. `PASS_OURS_V1_SELECTED_RAW_AND_M0_TARGET8_PAIRED`, 501개 고유 ID가 공식 공통 subset의 순서와 정확히 일치하고 사례별 결과도 501개임을 확인했다. 별도 [baseline 중간 표](2026-10-28-common501-nine-method-interim-ko.md)의 입력 수·카메라 조건 차이를 감안해 최종 10방법 전체 표에서 함께 해석한다.

## 동일 사례의 기존 surface 점수 중간 집계

공통 evaluator가 geometry를 다시 계산하기 전, 기존 source-M0 surface 점수의 **동일 501 ID**를 집계했다. 각 arm은 501개 고유 사례를 가지며 source cluster는 133개다. CD는 대칭 unsquared L2, F-Score는 0.02 threshold다. 이 수치는 후속 공통 evaluator와 대조할 **중간 점수**다.

| 생성 후 정합 | 모델 | CD ↓ | F-Score ↑ |
|---|---|---:|---:|
| Raw | V1 | <u>0.014198</u> | <u>0.818087</u> |
| Raw | Selected | **0.013847** | **0.824140** |
| Source M0 Sim(3) | V1 | <u>0.015125</u> | <u>0.803175</u> |
| Source M0 Sim(3) | Selected | **0.014928** | **0.804705** |

- Raw Selected−V1은 CD **−0.000351**과 F-Score **+0.006053**이다. 95% source-cluster bootstrap 구간은 각각 [−0.000731, −0.000053], [0.001133, 0.011202]다.
- M0 정합 후 Selected−V1은 CD **−0.000197**, F-Score **+0.001530**이고 구간은 각각 [−0.000510, +0.000098], [−0.002698, +0.005674]다. 이 구간은 다중 비교 보정 전이며, 최종 공통 evaluator 결과와 보정 구간이 나온 뒤 주장을 확정한다.
- 기존 surface 점수는 prediction과 target M1 표면을 각각 20,000점으로 고정 seed 표본화하고, source M0에서 구한 Sim(3)을 생성 후 prediction에 적용한다. M1은 평가에만 사용한다. 동일 입력이라도 공통 evaluator의 계산 및 수치 오차를 최종 대조한다.

원본 디렉터리: node16 `/home/youngwoo/retargeting-unified-test508/ours-v1-selected-source-sim3-final-v1`. `shard0`–`shard3.jsonl` SHA256은 각각 `cff45cd840195e66f10856cbc9d78b6ab7e4731dd7e860dcea7c3045968627c9`, `7f3b8ab3cda22c24c63cbb10a49c866f604e750318e9d45f9e33127ae34aaa7f`, `12d9c96cdc51cfebba923cb79450ab9d286177d29e4193e2b061e02261281c28`, `ca2a9f5ac1372fc01cdd68d46487bb414eedbc6ce97351901156b05d1db49d53`이다. 501 ID subset SHA256은 `91c15aa37f79df13503dace479ea1baae54f9c94dd10ffb836a12bff48102189`이다. 구간은 seed 20261028, 20,000회 source-cluster bootstrap으로 계산했다.
