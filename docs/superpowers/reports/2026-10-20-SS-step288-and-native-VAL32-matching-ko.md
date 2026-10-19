# SS update288 경향과 실제 validation 매칭 검사

## 같은 학습 update의 관측

동일32개 source cluster, 동일 GT/생성 noise, native12 sampling step와 generation batch8의 실제 validation row를 비교했다. IoU는 전체 TP/FP/FN을 합친 값이다.

| 모델 | Update288 SS IoU |
|---|---:|
| Vanilla V1 |0.43187184|
| Original positive Top1+R |0.44178756|
| Matched R continuation |0.44592041|
| Input-side feature attention |0.44533058|
| Input-side DQ |0.44121412|
| Direct DQ |0.43913141|

Direct−matched R은 **-0.6789pp**, source-cluster95% CI [-1.2237,-0.1534]pp다. 이 고정32개 중간 panel에서는 하락을 관측했다. Input-side DQ−matched R은 -0.4706pp, CI [-1.0100,+0.0309]pp다. Direct−input-side DQ는 -0.2083pp, CI [-0.8852,+0.4544]pp라 전달 방식끼리의 차이를 확정하지 않는다.

Original R과 비교한 Direct 차이는 -0.2656pp, CI [-0.9303,+0.3908]pp다. Vanilla와 비교하면 +0.7260pp지만 CI [-0.6843,+2.3199]pp라 positive module effect를 주장하지 않는다. 새 staged 모델끼리는 R96/Adam366과 postconstructor RNG를 맞췄지만 original continuous R은 post96 RNG가 다르다.

따라서 update192의 작은 positive 점 추정치를 모델 선정 근거로 삼지 않는다. 현재 DQ/Direct 학습은 미리 정한480까지 유지하고 fixed128×3, strong original R 및 saved-weight path/correspondence controls와 physical NoDQ를 연결한 상태다. 생성 noise3개는 독립 training seed3개가 아니다.

## 매칭 진단의 validation 범위

이전 확률 진단32개는 auxiliary head fitting에서는 제외됐지만 **SS TRAIN**이었다. 새 검사는 native initial96 validation row에 적힌 실제 SS validation32개를 쓴다. Auxiliary fit512와 calibration32의 sample/source cluster와 겹치지 않는지 검사한다. RGB/Omega/Top64 및 topology label은 이미 존재하며 raw download, 렌더링, latent/Omega 재생성은 하지 않는다.

Native conditioner의 실제 head 입력과 확률을 capture해 observation cache로 replay한다. Geometry teacher는 모든 native forward가 끝난 뒤 열며 loss-only label archive로 분리한다. Input에 spatial GT correspondence/deformation을 추가하지 않는다. 이 과정과 후속 평가는 optimizer update0회다. VAL32 matching 개선을 SS generation 개선 또는 독립 publication test로 부르지 않는다.

Q/K 보정의 고정 epoch7 결과가 실제 file4에 저장되면 같은 validation 입력으로 cold random Q/K, pretrained utility7을 보존한 Q-zero 기준, fixed7 warmed Q/K를 평가하도록 연결했다. Warmed state가 바꾸는 tensor는 query/key 두 개뿐이어야 하며 다른7개와 zero feature output은 그대로여야 한다. Epoch 선택 또는 모델 자동 채택은 하지 않는다.

## Native replay 검사에서 찾은 수치 문제

최초 VAL cache의 두 번째 사례는109개 ray query였다. 원래 native conditioner는109개를 한 번에 계산했지만 replay는64+45로 나눴다. 실제 GPU 진단에서 여섯 입력 tensor가 모두 정확히 같았음에도 BF16 확률의 최대 차이는0.0092818737이었다. 같은109개를 함께 재계산했을 때는 모든 확률이 정확히 일치했다. Candidate mask는 전부 유효했다. Matching/feature 주소의 불일치는 이 사례의 원인이 아니었다.

실패한 v1은1/32 partial cache와 traceback/terminal1을 보존했다. 오차 허용 범위를 넓히거나 해당 asset을 제외하지 않았다. **Native feature chunk256 및 마지막 partial chunk를 그대로 재현하는 v2**를 별도 namespace로 만들었다. 각 asset의 실제 native cold 확률 전체를 저장하고 평가 entry도 그것과 정확히 일치해야 진행한다. CPU2개 검사는 visible source relation의 view 분리, foreground/near1 label, empty ray와 주소 검사를 통과했다.

현재 frozen SS/DQ/Direct 및 Q/K training source는 바꾸지 않았다. Aux Q/K TRAIN calibration은 원래의 고정64-query partition을 계속 사용한다. 새 actual-VAL matching은 native256 partition을 사용하므로 계산 partition까지 다른 두 panel의 확률 수치를 섞지 않는다. SS reconstruction protocol은 기존 동일 generation batch8을 유지한다. Native per-asset matching replay의 일치는 whole SS generation output의 일치 증거가 아니다.

처음의 cache-entry v1은 실제 terminal1로 종료됐고, 그 failure에 연결돼 아직 실행되지 않은 fixed7 v1 job만 v2로 대체했다. 다른 scientific job/명령/가중치는 바꾸거나 재시작하지 않았다. 원본 before/after descriptor와 retired recipe도 보존했다.

## 실제 VAL32 초기 확률 검사 결과

V2 cache는32개 모두 native 확률 replay에 정확히 일치했다. 실제 native process exit0, cache 구성121.04초, max allocated0.5861GiB였다. 실제 file4에70개 artifact,427,570,350 bytes를 SHA/size 대조했다. 이는 cache/실행 증거이며 생성 성능 결과가 아니다.

이어 optimizer0회로 validation32개, 관측 가능한1,614개 관계를 실제 GPU에서 평가했다. Cold head는 저장한 native 확률과 모두 정확히 일치했고, Q-weight0 기준은 pretrained utility7과 모든 per-asset metric이 정확히 같았다. 이 entry 검사는 warmup epoch7의 평가가 아니다.

| 실제 SS VAL 분포 | Exact 정답 mass | Near1 정답 mass | Near1 Top1 coverage |
|---|---:|---:|---:|
| Cold random Q/K |16.1666%|43.3553%|47.5217%|
| Pretrained utility7 / Q-zero 기준 |16.5878%|43.8084%|47.8315%|

Exact/near1 Top64 coverage는59.9133%/88.7237%다. Random Q/K−utility7 near1 mass는 **-0.4530pp**, source-cluster95% CI [-0.8278,-0.0665]pp다. Exact mass도 -0.4211pp, CI [-0.6629,-0.1731]pp였다. 보조 TRAIN calibration과 분리된 실제 VAL에서 초기 random Q/K의 후보 확률 저하를 재확인했다. 이것을 초기 SS 출력 하락 또는 생성 품질의 인과적 효과로 확장하지 않는다. Native256 entry 평가의 max allocated는0.4113GiB, 측정 구간107.52초였다.

진행 중인 보조 Q/K fit의 실제 epoch1 checkpoint도 읽었다. 두 Q/K만 변경됐고 Adam2개의 step은32, 다른7개는 epoch0과 같으며 output projection은 여전히0이었다. SS/R update는0회다. 별도 SS TRAIN calibration의 near1 mass는45.3326%→45.5663%였으며, 이 작은 중간 변화로 epoch나 모델을 선정하지 않는다. 고정 epoch7의 실제 VAL 평가가 연결돼 있다.

## 증거

Canonical root: `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`.

- `physical-query-epoch-trends-v1/step0000288`: 같은 update의6모델, GT hash32개와 paired source bootstrap4,000회.
- `prior-anchored-QK-v1/validation-replay-diagnosis-v1`: 실제109-query partition 원인 진단.
- `prior-anchored-QK-v1/validation-cache-v1`: 실패한 partial cache 보존. 저장 SHA/size receipt가 있어도 유효한 completed cache가 아니다.
- `prior-anchored-QK-v1/true-VAL-plan-v2`, `true-VAL-followup-plan-v2`: 진단 및 native256 새 source snapshot.
- `prior-anchored-QK-v1/true-VAL-queue-correction-v2`: 실행되지 않은 invalid-partition followup만 대체한 기록.
- `prior-anchored-QK-v1/validation-cache-v2`, `validation-entry-probe-v2`: native256 cache 및 초기 확률 entry 검증.
- `prior-anchored-QK-v1/validation-fixed7-v2`: 고정7에폭 완료/실제 file4 closure 후 실행하도록 연결한 실제 validation 매칭 평가.

성능 목표는 아직 달성하지 않았다. 매칭 개선, SS 재구성 개선, 각 모듈의 효과 및 independent training seed를 각각 실제 결과로 확인해야 한다.
