# R 경유 source-ray feature 전달의 구현과 비교 학습

PBR 저장 weight 진단에서는 direct 전달을 제거하면 PSNR/SSIM이 회복됐고,
SS에서는 matching GT mass 개선이 native IoU 개선으로 이어지지 않았다.
[실측 결과](2026-10-22-matching-quality-versus-delivery-findings-ko.md)에 따라
이번 비교는 matching 정보를 source feature 변화로 전달하는 경로를 바꾼다.

## 두 후보와 유지하는 구조

Source-camera ray query, member semantic64, matching decoder/score,
bounded learned attention, Top-1 carrier, independent source-use와 target stream은
유지한다. 두 후보 모두 기존19개 source parameter의 초기값을 그대로 갖는다.
새 trainable tensor나 RAW DQ channel을 추가하지 않는다.

Own native source feature를 S, target query를 T, head output을 H라 하면
R의 입력은 [A−S,T]다. 두 후보의 A와 consumer 전달은 다음과 같다.

| 후보 | R에 전달하는 A | Consumer에 전달하는 추가 변화 |
|---|---|---|
| R-mediated feature | Top1+H | R 출력만 |
| R-mediated raw Top-K | Σ mean-head(p)·V+H | R 출력만 |

기존의 bounded H를 R 밖에서 더하는 경로는 두 후보에서 모두 없다. Raw Top-K
집계와 learned feature는 같은 실제4-head probability를 사용한다. Top-1 carrier와
source-use 판단은 그대로 두므로 두 후보의 차이는 R에 들어가는 appearance다.
Source feature의0 placeholder나 uniform score로 바꾸는 실험이 아니다.

`RMediatedSourceContrastConditioner`는 기존 실행 중인 parent 코드에 손대지 않는
새 클래스다. 모드마다 다른 state buffer key를 사용해 strict loading이 parent와
두 모드를 서로 바꿔 로드하지 못하게 한다. 학습된 parent R를 새 입력 표현에
강제로 이어 붙이지 않고 cold19+동일 frozen utility에서 시작한다.

## 초기 함수와 gradient 검증

R의 마지막 출력이0인 동안에는 H가 nonzero여도 consumer에 직접 전달되지 않는다.
Cold R와 semantic gate가0이므로 native source/target conditioning을 정확히 보존한다.
Matching support가 없으면 원래 native source가 유지되며, GT visibility나 별도
visibility 전달 mask를 추가하지 않는다. Physical R 입력은 appearance1024와
target1024의2048이며 deformation projection은 존재하지 않는다.

이 구조에서는 첫 step의 head gradient가0인 것이 의도된 결과다. 이를 dead path로
오인하거나 direct bypass를 다시 넣지 않는다. CPU 검사에서 nonreentrant checkpoint
backward와 실제3회 R 업데이트 후19개 gradient가 모두 열리는지 확인한다.

GPU capacity 단계에서는 actual micro96 cold backward를 측정한다. 별도의 disposable
module에서 TRAIN batch로3회 업데이트해 delayed gradient도 확인한다. 이 module은
버리고 scientific0step weight와 optimizer를 바꾸지 않는다. Report에 scientific
updates0과 별도 probe updates3을 구분한다. 원래 scientific state가 그대로인지
SHA를 검사한다. Source teacher나 NN의 warm start로 이 probe를 사용하지 않는다.

그다음 실제 live RGB의 frozen SS4210 native32에서 global/projection/negative/RNG,
GT/noise identity, initial four-head prior와 R-only 전달을 검증한다. CPU 결과를
GPU 정합 또는 성능 향상으로 대신하지 않는다.

## 학습과 평가

두 후보는 GPU 한 장씩 global96/micro96/accumulation1, 동일 TRAIN1536 unique
source와 동일112개 batch 순서,7에폭, seed101에서 비교한다. 기존 cached feature,
teacher, matching과 latent를 그대로 사용한다. DeformNeXt data를 쓰며 clean80k를
deformation 학습에 다시 넣지 않는다. 기존 source-only protocol의 LR1e−4,
warmup6, AdamW와 feature loss를 유지하고 매 에폭 native32 metric/image를 W&B에
기록한다. Frozen SS를 평가하는 source-module pilot이며 full NN joint 학습은 아니다.

Fixed112step에서 기존 bounded source-contrast19와 두 후보의 native IoU,
matching set-GT mass, source feature cosine/MSE, 실제 전달과 source-use를 함께
비교한다. Epoch best를 사후 선택하지 않는다. 첫 새 후보는 bypass 제거,
두 번째는 raw Top-K를 R에 제공하는 추가 효과를 확인한다. 개발32개와 한 training
seed만으로 full 모델 또는 semantic 모듈의 일반적 개선을 선언하지 않는다.
NoScore=Top-1 feature, NoCorr=native spatial, no-DQ=physical channel 제거라는
정의도 유지한다. 두 후보 자체를 이 ablation들의 결과로 이름 붙이지 않는다.

기존 plain contrast의 source-disjoint72 결과와 HOLD gate는 보존한다. 다음 독립
확인·main joint·PBR·전체 cascade·test508 scope는 이 pilot으로 대체하지 않는다.
R 전달이 PBR에서 유용했던 사실이 새 source-camera-ray SS에서도 그대로 성립하는지
이번 실제 학습과 평가로 확인해야 한다.
