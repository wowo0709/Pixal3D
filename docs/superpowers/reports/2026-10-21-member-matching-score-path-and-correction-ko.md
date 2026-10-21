# Matching decoder·score·ray attention의 실제 동작과 교정 후보

기존 검증32개의 topology GT와 저장된 teacher112 weight를 사용해, correspondence
후보·frozen matching·후속 attention을 분리해 측정했다. 현재 RGB/geometry/camera의
fingerprint와 기존 GT label SHA256가 일치한다. GT는 normal forward가 끝난 뒤
평가용으로만 사용한다. 새 렌더링이나 GPU 학습은 실행하지 않았다.

이 결과는 **source-camera SS teacher112의 CPU FP32 matching 진단**이다.
Native joint SS480 또는 현재 학습 중인 target-patch PBR 모델의 생성 품질 결과가
아니다. 모든 teacher state가 전후 같고, 정상 attention 확률을 실제 중간 tensor의
logit 합으로 재구성했을 때96개 forward 모두 bit 단위로 일치한다.

## 두 종류의 score를 구분한다

Frozen matching decoder와 FusionHead가 만든 score를 member ray 단위 posterior
`π`로 모은다. 이후 네 head의 feature readout은 `log π + utility + QK`를 사용한다.
여기의 utility는 추가 task score-selector이며, frozen matching score와 별개다.
두 단계의 효과를 혼동하면 matching 전체를 제거하는 잘못된 교정으로 이어질 수 있다.

## Frozen matching decoder와 score는 GT 대응을 개선한다

1517개 GT-observable 입력 target patch에서 **동일한 coarse64 후보**를 사용한다.
GT는 그 target patch에서 보이는 M0 source correspondence의 집합이다. Near1은
같은 source view 안에서32×32 patch grid의 Chebyshev 거리1을 허용한다.

| 같은 coarse64의 선택 방식 | Exact Top-1 GT hit | Near1 Top-1 GT hit | Exact GT 확률 질량 |
|---|---:|---:|---:|
| Raw Omega cosine | 2.18% | 5.87% | 1.89% |
| Matching decoder 뒤 cosine | 19.91% | 45.35% | 12.89% |
| 실제 matching decoder+FusionHead score | 21.89% | 48.32% | 18.05% |
| Raw query를 기존 FusionHead에 전달 | 2.18% | 5.87% | 1.96% |

Source cluster32개를 resample한 paired bootstrap에서 exact Top-1 차이는
decoder-cosine−raw cosine **+17.73 pp**, CI95 [+14.17, +21.46] pp다.
실제 FusionHead score−decoder cosine은 **+1.98 pp**, CI95 [+1.07, +3.02] pp다.
따라서 이 GT matching 진단에서는 decoder와 matching score의 순위 개선을 관측했다.
Raw-query FusionHead 결과는 저장 weight의 입력 교체 진단이며, decoder 없이
새로 학습한 대조군이 아니다. Query distribution shift도 포함한다.

## Ray 단위 후보 조합에는 실제 상보성이 있다

Member를 만들 때 동일 ray의 동일 target patch는 voxel 수와 무관하게 한 번만
센다. Point64를 ray64로 합친 뒤, 반복된 member–source visibility relation53738개
중 exact correspondence 포함 수가32867→38641로 늘었다. 기존 hit1217개를
잃었지만 다른 member에서6991개를 추가로 얻었다. 이는 원래 voxel마다 독립
query를 만드는 방식과 구분되는 ray member aggregation의 후보 수준 이점이다.
이 관계들을 독립 자산 또는 독립 학습 반복으로 세지 않는다.

Source ray의 GT set은 그 ray의 distinct target-patch members에 속하는 visible
correspondence의 합집합이다. 유일한 surface point를 지정한 GT가 아니다.
이 GT set이 있는 ray18955개에서 최종64개 후보의 any-GT 포함률은 exact84.47%,
Near1 96.95%다. Point-query 수치와 ray GT-set 수치는 분모와 의미가 다르므로
서로의 accuracy 향상 수치로 비교하지 않는다.

Raw point proposal budget의 exact GT-set recall은 K64=60.38%, K128=76.01%,
K256=90.90%다. K64 Near1은88.66%, K256 Near1은99.41%다. Exact GT patch-center가
source-valid한 eligibility 상한은99.54%다. 아직 coarse 후보 단계에도 손실이 있다.
이 곡선은 **coarse cosine 검색의 관측**이며 K256 decoder·학습·생성 모델을
실행했다는 뜻이 아니다. 폭만 늘리는 것이 native 성능을 높인다는 결론도 아니다.

## 후속 learned attention의 재가중은 현재 GT 대응을 악화시킨다

아래는 Contrast19 teacher의 같은18955개 ray·같은64개 후보에서 계산했다.
Q/K·utility 항을 수식에서만 분리한 저장 weight 진단이며, 모델 output을 해당
확률로 재생성하거나 retraining한 ablation이 아니다.

| 확률 식 | Exact Top-1 GT hit | Exact GT 확률 질량 |
|---|---:|---:|
| Matching prior `π` | 26.49% | 18.06% |
| `log π + utility` | 20.24% | 16.83% |
| `log π + QK` | 20.82% | 16.79% |
| 실제 `log π + utility + QK` | 18.57% | 15.88% |

Full−prior의 exact Top-1 차이는 **−7.92 pp**, source-cluster CI95
[−11.31, −4.21] pp다. Exact GT 질량 차이는 **−2.18 pp**, CI95
[−3.49, −0.83] pp다. Near1 GT 질량 차이의 CI는0을 포함한다.
모두 개발32개의 반복 진단이며 보정되지 않은 exploratory interval이다.

이 결과는 attention의 source 전달이 꺼진 현상과 다르다. 이전 관측에서 u≈0.626,
유효 head K≈6.32로 경로가 작동했다. 현재 GT matching에서는 추가 utility와 Q/K
재가중이 기존 유용한 prior에서 확률을 빼앗는 현상을 확인했다. 다만 head별 value와
output projection이 있으므로, head 평균 GT 질량을 곧바로 native 생성 품질이나
최적 task representation으로 해석하지 않는다.

## 준비한 교정 모듈과 남은 검증

기존 bounded-prior source attention 단독의 SS128×3 IoU 개선은 확실하지 않았다.
Prior 보존만으로 충분하다는 결론은 내리지 않는다. 새로운 후보
`BoundedSourceContrastMemberFeatureConditioner`는 **이미 구현된 bounded head와
source-relative R 입력을 결합**한다. 후보·decoder·matching score·ray query·projection은
유지하고, 초기 utility/QK 함수에서 달라진 logit만 bounded residual로 더한다.
초기 head 확률은 π와 정확히 같고 R 입력에는 실제 own-source feature가 들어간다.
DQ channel은 추가하지 않으며 전체 학습 파라미터 tensor19개를 유지한다.

이 후보는 별도 checkpoint marker를 사용한다. 부모의 warm weight를 새 구조인 것처럼
자동 로드하지 않는다. CPU 검사는 초기 prior/native source delta,19개 gradient 경로,
frozen anchor, 실제 source context, strict checkpoint 구분·roundtrip·missing support를
검증한다. 현재 단계는 **모듈 구현과 CPU 확인**이다. Native RGB/GPU 정합·capacity,
명시한 새 variant의 loader/trainer 연결, matched 학습과 생성 품질 평가는 남아 있다.
기존 두 warm arm이나 실행 중인 PBR의 구조·config·대기열은 바꾸지 않았다.

No-score는 여전히 Top-1 correspondence feature를 사용하는 arm이며 uniform
확률이나 이 보고서의 π-only 수식으로 대체하지 않는다. No-corr은 원래 spatial
correspondence다. 실제 모듈 효과는 맞춘 학습 예산, 초기 정합과 native SS/cascade
metric으로 검증해야 하며, 이 matching 진단으로 최종 모델을 채택하지 않는다.

입력·GT set 구성과 view 경계·중복 member·GT eligibility의 CPU 검사4개가 통과했다.
잘못된 모델 라벨을 만든 임시 CPU v4 진단은 exit130으로 중단하고 실패 기록으로
보존했다. 수정한 v5는3모델×32개 라벨을 검증해 완료했으며, v4 결과는 사용하지 않는다.
[전체 표·source-cluster CI·진단 plot](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/source-ray-decoder-score-VAL32-CPU-v1)에 기록했다.
