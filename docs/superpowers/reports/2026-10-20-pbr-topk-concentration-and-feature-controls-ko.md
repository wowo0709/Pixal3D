# PBR Top-K가 거의 Top-1이었던 이유와 후속 검증

기존 PBR의 Top-K와 Top-1 결과가 같았던 구체적인 요인을 확인했다. 기존 PBR
temperature는 **0.005**이며, 같은 validation32에서 실제 후보 bank와 native
`categorical_weights`를 적용하면 Top-64 중 실효 후보 수가 평균 **1.229개**에
그친다. Top-1의 평균 확률은 **93.36%**이고, ray의 **80.37%**에서 Top-1 확률이
90%보다 높다. 이 결과는 Top-K의 추가 효과가 관측되지 않은 이유를 설명하는
근거다. 온도를 바꾸기만 하면 품질이 좋아진다는 증거는 아니다.

## 실제 attention concentration

동일한 32개 source cluster의 edited RGB alpha-valid target patch ray 2,511개를
사용했다. GT correspondence는 읽지 않았다. Native PBR sparse projection에서
실제로 소비하는 patch 빈도로 가중해도 같은 경향이다.

| Temperature | 평균 실효 K | Median 실효 K | 평균 Top-1 확률 | PBR site 가중 실효 K |
|---|---:|---:|---:|---:|
| 0.005, 기존 PBR | 1.2294 | 1.0064 | 0.9336 | 1.2279 |
| 0.01 | 1.5188 | 1.1603 | 0.8711 | 1.5256 |
| 0.02 | 2.2502 | 1.8219 | 0.7616 | 2.2635 |
| 0.05 | 5.0390 | 3.3611 | 0.5572 | 4.9318 |
| 0.1 | 9.9765 | 5.0162 | 0.4129 | 9.6215 |

이는 predicted score의 분포 검사이며 quality sweep이 아니다. 진행 예정인 새 SS
feature attention은 이미 고정된 **0.05**를 사용한다. 그 scientific config와 frozen
source는 변경하지 않았다. PBR은 별도 학습·평가로 검증해야 한다.

## 새 feature attention의 실제 PBR 경로

Native PBR loader는 일반 C1에서 cached Omega target/source key를 기본으로 읽지
않는다. 기존 loader를 수정하지 않고 `DeformNeXtFeaturePBR1024`를 추가해 필요한
Omega key와 native query를 명시적으로 읽는다. `spatial_indices`나 deformation
channel은 읽지 않는다. 기존 PBR sidecar의 source value는 `[3,1024,2048]`
DINO+NAF, target query는 `[1024,1024]` DINO이며, Omega FRAME key는 1024차원이다.

실제 TRAIN asset 한 개, sparse token 9,823개, GPU2에서 다음을 검증했다.

- 새 output projection이 0일 때 fresh C1/R의 PBR loss 및 evidence와 정확히 일치.
  Initial loss는 양쪽 모두 **0.027075452730059624**.
- Frozen native PBR base+LoRA와 matching weight는 검사 전후 hash가 동일.
- R/fusion 6개와 feature attention 9개에만 임시 optimizer를 적용했다. 세 update
  안에 새 feature tensor 9개 모두 PBR gradient를 받았다.
- Peak allocated **19.0932 GiB**. 이 결과는 micro1 한 asset의 correctness다.
  Global24, DDP2/micro12 capacity나 전체 PBR 학습 시간으로 일반화하지 않는다.
- PBR에서 positive trained R이 있다고 가정하지 않았다. Fresh R/fusion이 0인
  처음에는 head gradient도 0이었다. 다음 update에서 output projection, 그다음에
  나머지 attention tensor로 gradient가 전달됐다. SS는 이미 학습된 commonR96을
  사용하므로 이 cold-start 경로와 다르다.

같은 TRAIN asset으로 temperature0.05의 별도 임시 probe도 완료했다. Attention의
실효 K가 약 1.20에서 3.46으로 늘고 gradient는 유한했다. GPU가 다르고 native sparse
학습의 exact replay를 검증하지 않았으므로 두 probe의 loss 차이를 temperature의
품질 효과라고 해석하지 않는다. 두 probe 모두 폐기했고 scientific PBR update는 0이다.

## Feature 경로를 분리하는 진단

새 fixed-weight control 네 가지를 구현하고, 최종 feature SS checkpoint의 동일
32개 샘플·동일 noise·generation batch8 평가에 연결했다.

| 진단 | 유지하는 경로 | 바꾸는 경로 |
|---|---|---|
| Residual-off | 기존 Top-1 carrier, 현재 NN/R | 추가 feature residual만 제거 |
| Wrong-value-address | Query/key, score 및 attention weight, Top-1 carrier | 각 유효 후보 집합 안에서 Wv 입력 value의 주소만 순환 교환 |
| Score-prior-off | Content attention, Top-1 carrier | 추가 attention에서 cached score prior만 제거 |
| Decoder-context-bypass | Cached 후보와 score, Top-1 carrier | 추가 attention에서 D1 context 대신 raw target Omega 사용 |

사용자가 정의한 **NoScore=Top-1**, **NoCorr=원래 spatial correspondence**라는
training arm 이름과 혼용하지 않는다. Decoder-context-bypass도 cache를 만든 D1의
전체 효과를 제거한 실험이라고 주장하지 않는다. Retrained R-only control과 full의
128×3 비교는 별도로 진행한다.

CPU 검사에서 1024/2048 value 두 경우 모두 wrong-value-address가 attention weight를
정확히 유지하고 projected output만 바꾸는 것을 확인했다. 실제 PBR 임시 probe에서도
네 control이 모두 실행되고 evidence를 바꿨다. 한 asset의 SS 생성 entry도 실제로
통과했다. 이 작은 entry 검사는 generation batch1이며 quality 비교에 사용하지 않는다.
관련 CPU 검사는 총 **17개 통과**했다.

최종 학습 후 네 control 평가와 source-cluster bootstrap/actual file4 저장을 자동으로
수행하도록 연결했다. 먼저 R-only와 feature full의 128×3 비교를 만들고, scalar
composer가 끝나면 세 arm 비교를 추가한다. Generation noise 세 종류를 독립 training
seed로 간주하지 않는다. Positive 결과가 있더라도 independent training seed 확인이
남으며, 현재 full model을 채택하지 않았다.

## 근거와 실행 상태

공통 root: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1`.

- `pbr-matched32-v1/matching-concentration-v1/audit.json`: 원래 PBR predicted score 분포.
- `ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/PBR-gradient-v1/audit.json`:
  원래 temperature의 native PBR 검사.
- 같은 `feature-readout-v1/PBR-gradient-v2-softprior/audit.json`: temperature0.05 임시 검사.
- 같은 `feature-readout-v1/controls-entry-probe-v1/evaluation.json`: 실제 SS control entry.
- 같은 `feature-readout-v1/controls-plan-v1/`: 네 진단과 비교 callback의 actual file4 source/recipe.
- 같은 `feature-readout-v1/final-analysis-plan-v1/`: R-only/full 중간 비교와 세 arm 최종 비교 callback.

기존 NoDecoder/NoPrior 학습은 계속 진행한다. 새 SS scientific 학습은 reserved DDP
capacity를 기다리고 있다. 이번 PBR 검사는 새 모듈의 작동과 attention collapse의
원인을 확인한 것이며, 5 epoch PBR fine-tuning을 완료했다는 뜻이 아니다.
