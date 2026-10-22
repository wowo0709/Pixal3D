# 실제 변형량에 따른 native SS attention 추가 효과

현재 native SS480 개발128개·93 source cluster·세 생성 noise 결과를 실제 M0/M1
geometry와 연결했다. “큰 변형에서만 Top-K 효과가 나타나 전체 IoU 평균에 가려진다”는
가설을 확인하기 위한 사후 분석이며, 새 모델 학습이나 test508 평가는 아니다.

## 데이터와 측정

128개에 필요한 source geometry와 final target vertices227개 파일30,123,166 bytes가
모두 기존 로컬 캐시에 있었다. 실제 file4 원본을 새로 읽어 SHA/size가 같음을 확인했다.
Geometry를 추가 다운로드하거나 다시 렌더링하지 않았다.

변형 크기의 주 공변량은 **source 표면 면적으로 가중한 대응점 변위의 RMS**다.
기존 CD/F-Score와 같은 source bbox 최대 extent로 나누며 target을 다시 중심 이동·
스케일 조정·정렬하지 않는다. Mesh 정점 수나 정점 밀도로 평균하지 않는다.
Triangle 위의 piecewise-linear displacement 제곱을 면적 적분해 RMS를 정확히 구했다.
Mean·quantile·이동 표면 비율은 source 표면20k 점을 seed20261022로 표집하고,
동일 face/barycentric 좌표의 target 대응점과 비교했다. 두 표면을 독립 표집해
임의의 점 사이 거리를 재는 방식이 아니다.

계산 검사는 정답을 아는 translation·선형 변형·triangle subdivision 불변성,
공통 scale/translation과 잘못된 topology 입력을 포함하며10개가 통과했다.
이 수치의 단위를 실제 SS64 voxel 한 칸으로 간주하지 않는다.

## 변형 크기가 커져도 Top-1+R을 넘는 안정적인 이득은 없다

개발128개의 geometry RMS로만 사분위 경계를 정했다. 경계는 source bbox extent의
1.4069% / 2.2178% / 3.3940%다. 각 그룹은32개이며 아래 차이의 단위는 IoU percentage point다.
R은 retargeting을 뜻한다. CI는 해당 그룹의 source cluster paired bootstrap95% 구간이다.

| Geometry RMS 그룹 | RMS 중앙값 | Source 표면 중 2% extent 이상 이동한 비율 평균 | Top-1+R − V1 | Raw Top-K+Feature+R − Top-1+R | 마지막 차이의 CI95 |
|---|---:|---:|---:|---:|---|
| Q1 | 1.0840% | 6.81% | +.3520 | +.0332 | [−.4740,+.5506] |
| Q2 | 1.7964% | 25.73% | +.8313 | +.0167 | [−.4140,+.4913] |
| Q3 | 2.7118% | 46.97% | +.2839 | +.1580 | [−.3463,+.6283] |
| Q4 | 5.2994% | 58.71% | +.6310 | −.4167 | [−1.0159,+.1323] |

모든 구간에서 raw Top-K의 Top-1+R 대비 CI가0을 포함한다. Q4의 평균은 낮지만
이를 확정적인 악화로 해석할 수도 없다. Feature attention+R 대비로는 Q2에서
raw Top-K가+.3394pp, CI[+.0522,+.6726]pp였으나, 같은 Q2의 Top-1+R을 넘는
이득은 불확실하다. 약한 feature 대조군만 비교하면 효과를 과대 해석할 수 있다.

변형 RMS와 raw Top-K−Top-1+R IoU 차이의 descriptive Spearman rho는−.0201,
2% extent 이상 이동한 표면 비율과의 rho는−.0255다. 연속 변형량으로도 명확한
증가 추세를 관찰하지 못했다. 상관은 인과 증거나 독립 cluster/학습 seed 검증이 아니다.

원래 전체128개 네 모델의 mean-asset 지표를 정확히 재현했다. 모든 그룹과 양수·
음수 case를 남겼으며 후보나 endpoint를 바꾸지 않았다. 그룹 경계는 TRAIN에서
미리 정한 것이 아니고, 여러 비교의 보정 전 interval이므로 탐색적 결과다.
기존 최종128 주 비교의 family 보정 CI를 이 새 bootstrap seed의 CI로 대체하지 않는다.

## 다음 판단

현재 자료는 단순한 변형 유형/크기 분포 차이만으로 attention의 추가 이득 부재를
설명하지 못한다. Score 분포·Top-K 값·feature 변환이 실제 consumer에 유용하게
반영되는지는 연결된 native480 저장-weight 개입으로 확인해야 한다. 이 분석으로
semantic matching이 불필요하다거나 PBR source-preservation에도 효과가 없다고
결론내리지 않는다. Native PBR 세 source 모델과 generated/full-cascade metric은 남아 있다.

Canonical 근거는 native-flow-cold3-v1 아래
`actual-geometry-size-strata-dev128-CPU-v1/`의 audit/statistics/per-asset/실제file4 identity다.
7개515,347 bytes가 actual file4 SHA/size 검증을 통과했다. PNG/PDF는 별도
`actual-geometry-size-visual-dev128-v1/`에 보관했다(4개223,210 bytes 실제file4 대조).

[W&B: 실제 변형량 산점도와 전체 사분위 표](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-native480-geometry-size-dev128-v1)에
동일 결과를 기록했다. 전체 실험, 독립 학습 seed와 최종 test508/10 metric은 진행 중이다.
