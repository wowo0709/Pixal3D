# PBR R-only source-ray 모듈과 native sparse flow 연결

2026-10-22 UTC. 기존 네 SS 학습과 SHA-bound 코드는 유지했다. 새 PBR source module,
native sparse trainer와 engineering entry를 별도 파일로 구현했다. 실제 GPU 검증과 본 학습은 남아 있다.

## 모듈과 대조

`RMediatedPBRSourceConditioner`는 source-camera ray마다 같은 member semantic posterior를 구한다.
Target native DINO query와 Omega key/context는1024차원이고, appearance/value/R output은
**DINO1024+NAF1024의2048차원**이다. Head는 두 value block을 따로 정규화한다.
R은 `[appearance + learned change − own-source, target DINO context]`의3072채널을 받는다.
추가 deformation channel/projection은 없으며0으로 채우지 않는다.

| Mode | R appearance 입력 | Trainable source tensor | 물리적 feature head |
|---|---|---:|---|
| Top1-only | 실제 semantic argmax feature 하나 | 10 | 없음 |
| Top1+feature | Top1 + learned four-head feature | 19 | Q/K/V/output + utility |
| Raw-TopK+feature | 같은 learned posterior로 가중한 raw feature + learned feature | 19 | 동일 |

세 source mode의 공통 R/use/fusion10개 초기 tensor는 같다. Checkpoint marker가 서로 달라
strict load로 다른 mode를 바꿔 읽을 수 없다. Q2C decoder/score는 frozen/eval로 유지하고
feature head의 function-anchor buffer도 별도로 보존한다. Physical Top1에는 utility initializer,
Q/K/V/output 또는 dummy attention query가 없다. Top-K mode만 edited-target native projection의
DINO 부분을 ray query로 사용한다. 이 입력은 개별 voxel attention이 아니라 ray별 평균이다.

Head feature가 consumer로 직접 우회하는 경로는 없다. 전달 residual은 실제 R output과 같아야 한다.
Native four-view carrier에 더하는 correction은 다음과 같다.

```
sum_source native_projection[u*g*(Top1-own_source) + u*R] / 4
```

Target RGB/native global/negative stream과 native shape/PBR sparse support를 유지한다.
매칭이 없으면 native multiview prior를 유지한다. Ray query의 image/frustum support와
write를 구분하며, 별도 depth/GT correspondence/visibility write gate는 추가하지 않는다.

## Native trainer 연결

`RMediatedPBRNativeTrainer`는 원래 sparse flow loss·LoRA360·CFG·native optimizer/save 경로를 상속한다.
기존 `DeformNeXtFeaturePBR1024`의 정합된 cache를 사용하고, legacy target-patch의 cached
correspondence/score와 dense SS patch-grid는 새 source-ray posterior에 전달하지 않는다.
추가 공간 GT/deformation context가 들어오면 거부한다. PBR support와 texture/shape latent 정합은
기존 dataset contract를 그대로 유지한다. 새 데이터 다운로드나 latent 재구축을 요구하지 않는다.

Source constructor는 CPU/CUDA RNG를 보존한다. 초기 fusion과 R output은0이라 native condition을
유지하도록 설계했다. 이 설계와 CPU 검사는 실제 original-PBR GPU prediction 동등성의 대체 증거가 아니다.
이 구현을 기존 target-patch PBR1905 실험의 재실행이나 성능 개선으로 부르지 않는다.

## 통과한 검사와 다음 실제 검증

관련 다섯 test 파일 **37개가 CPU에서16.69초에 통과**했다. 실제 native dense/sparse projection과
gradient, FP32/BF16 cold correction0, 공통 R10 초기값, NAF/DINO 양쪽 gradient,
missing-match DDP edge, physical Top1, cross-mode checkpoint 거부, sparse trainer의 native
global/negative 보존 및 edited-target query 경로를 검사했다. 데이터 학습을 CPU에서 수행한 결과가 아니다.

`probe_r_mediated_pbr_native.py`는 다음 실제 GPU engineering 검증을 위해 구현했다.

1. 사용자 지정 원본 complete multi-view PBR16850+LoRA와 대응 config로 fresh optimizer를 만든다.
2. 동일 eligible TRAIN9139 중 선언한31900-token asset을 사용한다. Vanilla와 세 source mode를
   같은 native 모델·data·noise 조건에서 대조한다.
3. 실제 native global/projection/negative 및 encode RNG가 초기 condition과 정확히 같은지,
   same-noise velocity/loss가 native bypass와 같은지 확인한다.
4. 실제 native sparse flow loss로 micro1 세 update를 수행해 source0/10/19와 LoRA360의
   optimizer state·gradient·frozen parameter/buffer를 검사한다. 임시 가중치를 본 학습에 이어 쓰지 않는다.

이 micro1 learnability 검사는 조건부 p_uncond0을 명시한다. Config의 본래 native dropout은.1이며
향후 실제 full-microbatch capacity와 science에는.1을 유지해야 한다. Micro1 증거로 global24
학습을 허용하지 않는다. 향후5회 native loader pass의 예산은9139/global24의 floor380×5=1900 update다.
실제 GPU 수·최대 microbatch·accumulation1은 별도 측정해야 한다.

초기32개 generated rendering 동등성, global24 full-microbatch 검증, scientific validation/W&B,
독립 학습 seed·모듈 대조·full cascade·final508/10metric은 여전히 필요하다.
