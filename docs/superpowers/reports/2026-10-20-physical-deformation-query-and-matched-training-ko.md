# Semantic matching 경로 복구와 physical deformation query 실험

## 현재 판단

스토리에 맞는 성능 향상은 아직 입증되지 않았다. 기존 결과에는 후보 가중치가 사실상 Top-1로 집중되는 문제와, 과거 positive retargeter 실험과 query/carrier가 달라진 문제가 있었다. 이들은 실측으로 확인한 구현·비교 조건이다. 각각이 성능 부진의 충분한 원인이라는 뜻은 아니다.

특히 온도를 넓히는 것만으로 해결되지는 않았다. PBR의 동일 32개 입력과 고정 가중치에서 온도 0.005→0.05 변경은 foreground PSNR −0.000820 dB였고, source-cluster paired CI는 [−0.005261, +0.001762]였다. 추가 온도 sweep보다 실제 feature 전달 경로를 검증한다.

## 확인한 증거

- PBR의 유효 target patch ray 2,511개에서 원래 온도 0.005의 평균 Top-1 mass는 0.933645, effective K는 1.229362였다. Top-K 목록이 있어도 여러 후보를 의미 있게 사용하는 경로는 아니었다.
- 새 attention은 target patch의 camera ray를 query 단위로 유지한다. 기존 positive retargeter의 nearest projection과 Top-1 carrier를 보존하고, 후보 value를 학습 가능한 projection과 attention으로 조합한 residual을 추가한다. scalar mix gate는 두지 않는다.
- 실제 native PBR forward/backward에서 DINO와 고해상도 NAF 양쪽에 새 feature residual의 gradient가 확인됐다. 첫 3번의 임시 TRAIN update로 확인했으며, 이를 PBR fine-tuning 결과로 해석하지 않는다. 원래 B2/B3의 semantic delta는 DINO 부분에 주입됐고 NAF는 native 경로였다.
- 대조군과 새 attention의 실제 scientific 시작 step96 validation 32개는 IoU 0.43718718977498344로 같았고, 샘플별 TP/FP/FN, target hash, source cluster, 생성 seed가 모두 같았다. 초기 하락 없이 시작했다는 증거이며, 최종 개선의 증거는 아니다.
- 두 scientific arm 모두 실제 DDP2/global96/micro48/accumulation1이다. 새 attention의 dedicated capacity peak는 약 49.06 GiB/rank였다. 공통 LoRA360 + R/fusion6 Adam state는 step96 그대로 복구했다.

## 추가 arm: attention 다음에 실제 deformation 정보 전달

appearance feature를 attention으로 섞더라도 후보들의 appearance가 유사하면 geometry 차이가 전달되지 않을 수 있다. 이 가설을 확인하기 위해 동일 attention probability로 조합한 source RAW Omega feature와 target RAW Omega feature의 차이를 retargeter에 전달한다.

\[
d_q = \Omega_{target} - \sum_k \bar\alpha_{qk}\Omega_{source,k}.
\]

2048개 채널을 실제 추가한다. SS의 retargeter 입력은 appearance 2048에서 총 4096으로 증가한다. No-DQ 대조군은 이 추가 채널과 projection 자체가 없다. 관측값을 0으로 채워서 같은 deformation이라고 알려주는 방식은 쓰지 않는다.

추가 projection의 **가중치**만 0으로 초기화한다. 입력 Omega 차이는 실제 비영 값이다. 기존 appearance projection과 R 파라미터 형태를 그대로 유지하므로 공통 Adam moment를 padding하거나 바꾸지 않는다. 새 projection 하나는 독립된 optimizer group에서 step0부터 시작한다.

실제 2개 TRAIN asset 검증에서 시작 SS loss는 기존과 새 모델 모두 0.10466600209474564였고 evidence도 정확히 같았다. 첫 backward의 새 query projection gradient L1은 1.0232062였다. 세 번째 update까지 attention9개와 query projection1개 모두 비영 gradient를 받았다. NN/R을 고정한 이 검증은 모델을 폐기했으며 scientific update는 0이다.

별도의 실제 joint micro2 검증에서는 LoRA360, R6, attention9, query projection1이 모두 갱신됐다. 공통 Adam366은 96→99, 새10개는 0→3이었다. frozen base/matcher는 정확히 같았고 peak 5.850 GiB였다. 이 검증도 폐기했으며 DDP2/micro48 capacity 증거로 대신하지 않는다. 관련 CPU 테스트 8개가 통과했다.

## 비교와 실행

| Arm | 공통 시작 | 추가 학습 | 차이 |
|---|---|---|---|
| R control | old R step96, 공통 Adam366 | 384 updates | positive target-ray retargeter |
| Feature attention | 동일 | 동일 | semantic Top64 + matching score/content + feature attention |
| Feature attention + DQ | 동일 | 동일 | 위 구조에 RAW Omega 차이 2048개 채널 추가 |
| Scalar composer | 동일 | 동일 | 기존 scalar 가중합 가설을 별도로 비교 |

현재 R control은 GPU0/1, feature attention은 GPU2/3에서 실행 중이다. DQ는 초기 generation32 parity와 독립 DDP2 capacity를 통과한 뒤 GPU0/1에 연결한다. 기존 scalar arm은 feature attention 후 GPU2/3에 연결된 상태다. 실행 중인 모델의 numerical source와 recipe는 바꾸지 않는다.

주 판단은 고정 final480, 128개 asset/93 source cluster, 3개 생성 noise의 paired 비교다. 생성 noise 반복을 독립 학습 seed 반복으로 계산하지 않는다. 32개 replay와 saved-weight correspondence/score/decoder 경로 검사를 함께 사용한다. 새 모델이 R control보다 좋아져도 matching 경로를 꺼서 차이가 없는 경우에는 semantic matching/attention 효과라고 주장하지 않는다. 초기 개선이나 epoch 선택만으로 full model을 확정하지 않는다.

## 증거 위치

공통 experiment root는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1`이다. 아래 경로는 이 root 기준이다.

- `pbr-matched32-v1/matching-concentration-v1/audit.json`
- `pbr-matched32-v1/temperature005-to05-summary-v1/summary.json`
- `ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/PBR-channel-gradient-v1/channel-gradient-audit.json`
- `ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/scientific-initial96-parity-v1/audit.json`
- `ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/deformation-query-v1/SS-gradient-v1/audit.json`
- `ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/deformation-query-v1/joint-profile-v1/joint-profile.json`

진행 상태와 proof는 구분한다. 이 문서 시점에 새 attention의 본 학습은 진행 중이며, DQ의 본 학습과 새 PBR feature fine-tuning은 시작하지 않았다. 성능 개선·full model 채택·전체 실험 완료를 주장하지 않는다.
