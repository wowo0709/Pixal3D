# 현재 SS에서 원본 regional objective를 재현하는 실험

원본 Q2C 24개 학습과 G2 기존 checkpoint의 144개 endpoint는 재현됐다. 현재 데이터에서 geometry 모듈의 효과를 확인하기 위해 입력 표현·native projection·loss를 맞춘 frozen-controller 실험을 진행한다. 이 실험은 기존 14개 본 비교와 별도이며, 본 SS LoRA 학습의 global96/DDP micro48/5epoch 설정을 바꾸지 않는다.

## 고정 비교와 데이터

| Arm | Projection 입력 | Deformation query |
| --- | --- | --- |
| G0: adapter | target / native multiview / 차이 / xyz | encoder와 채널 자체가 없음 |
| G1: adapter+query | 동일 | frozen SS의 predicted O/D logit·occupancy·SDF 9채널 |

두 additive arm만 학습한다. Matching은 native spatial 경로를 유지하며 geometry 모듈의 효과를 분리한다. Semantic Q2C의 현재 RGB task 결과와 이 결과를 합쳐 하나의 Full 성능이라고 부르지 않는다. Query는 변형 전 source RGB 4장으로 예측한 O와 target RGB 1장으로 예측한 D에서 만든다. GT geometry는 query/condition 입력에 사용하지 않는다.

O branch에는 원래 M0 camera08도 들어가므로 query 모듈은 native의 네 RGB 외에 원래 target-view RGB의 정보도 사용한다. G1−G0를 같은 네 RGB에서 encoding만 바꾼 효과로 해석하지 않는다. Intact−joint-shuffle control은 같은 O/D 값과 입력 정보량을 유지하며 query의 공간적 정합을 검사한다.

성능을 보기 전에 고정한 320개 pilot population에서 role별 앞 32개를 사용한다. Train32/calibration32의 source cluster는 겹치지 않는다. 영역 크기·metric으로 asset을 교체하지 않는다. Validation508의 이미지/label을 이 pilot 학습에 사용하지 않는다.

## 학습과 평가

사용자가 지정한 mixed385 SS complete base+LoRA step4210을 frozen consumer로 사용한다. 원본의 12-step differentiable rollout → raw SS64 decoder → 같은 SS64/SS32 regional BCE와 correction penalty 0.001을 사용한다. Consumer의 native12/CFG3/time-rescale1을 고정하며 과거 full-FT20k/CFG7.5 결과와 구분한다.

Seed101/202/303에서 각각 128 updates를 실행한다. 원본과 같은 epoch별 pair hash order·noise hash·AdamW LR1e-4/WD0.01·clip1을 사용하고, train32를 네 번 순회한다. Controller asset batch1은 이전 frozen-controller 프로토콜이다. 본 LoRA 학습의 batch96을 바꾸거나 V1을 재학습하는 것이 아니다.

Final checkpoint/SHA freeze 뒤에만 calibration GT를 연다. 같은 cal32와 sampling noise3개에서 native / normal / projected-source-off를 비교하고, query arm에는 joint-tuple shuffle을 추가한다. Source projection OFF는 네 RGB의 native global/negative와query arm의predicted O/D query를그대로유지한다. Query에들어있는source geometry까지제거하는대조가아니며같은query 아래projected source value의추가기여를검사한다. No-query loader는 query 파일 자체를 열지 않는다. Training loader는 calibration role을 파일 개방 전에 거부한다.

Regional recall/stale/visible/BCE, TP/FP/FN과 GT/prediction voxel SHA, 고정 첫4개 asset의 Y voxel-projection GT/Pred/Diff를 W&B에 저장한다. Empty source/stale metric은 None으로 남기고 유효 population을 함께 집계한다. 이 실험의 GT는 실제 SS encoder input support, IoU는 macro asset/noise 평균이다. 본 14개 표의 decoded-latent GT/micro IoU와 직접 합쳐 비교하지 않는다.

첫 seed의 두 arm은 GPU2/3에서 실행한다. 본 14개 최종 report가 끝나면 seed202는 GPU0/1을 사용하고, seed303은 첫 seed가 끝난 GPU2/3을 사용한다. Full/PBR 판단에는 본 비교·source-use control·필요한 training-seed 확인을 함께 사용한다.

## 현재 실행 계약: V4의 GT authority

Target64는 기존에 실제 사용한 Shape encoder input support를 그대로 사용하고 SS latent에 기록된 `source_coords_sha256`과 exact 일치를 요구한다. Camera/scale은 현재 world-v4의 native target-anchor frame을 그대로 유지한다. Oracle pair-union scale로 바꾸지 않는다.

독립적인 물리적 정합 검사는 모든 GT voxel 중심과 실제 M1 triangle의 거리를 계산한다. Mesh surface를 포함하는 SS64 cell의 중심은 해당 triangle에서 cell half diagonal `sqrt(3)/128` 이내에 있어야 한다. Float32 변환 수치 오차 1e-5를 명시한다. 고정64개 전체를 검사했고 범위를 넘은 cell은 0개, SS source hash는 모두 exact였다. 관측64개는 수치 허용분을 쓰지 않아도 원래 half diagonal 안에 있었다. 실측 1.17초이며 GT cell을 제거하거나 교체하지 않았다.

Source64는 원본 native flexible64 support와 실제 source RGB에서 관측된 M0 surface point의 voxel witness를 합친다. 관측된 surface point가 cell 안에 있으면 그 cell이 실제 M0 surface를 포함한다는 직접적인 근거다. 이를 통해 coarse voxelizer가 놓친 얇은 visible 구조를 보존한다. 같은 source triangle의 barycentric을 M1에 적용해 matched source-exclusive label을 만들고, stale은 source-visible support에서 실제 target GT를 뺀다. 이 current-domain visibility 확장을 과거 CAGE label의 bitwise 재생으로 부르지 않는다.

원본 conservative cube-ray FREE/SURFACE와 원본 SS64/32 loss 함수는 그대로 사용한다. Open3D top-down raster를 raw OpenGL bottom-up 순서로 변환한다. RGBA foreground는 positive/source evidence를 제한하고 FREE는 opaque geometry z-buffer로 보수적으로 계산한다. Frame 밖 point는 boundary로 clip하지 않고 제외·기록한다. Empty region도 성능을 보고 대체하지 않는다.

## 실제 검증과 보존

원본 support/projection/cube-ray/pooling/validator 7개 함수를 AST 그대로 이식했다. CPU 검증은 AST, 실제 pooling 호출, frame 밖 point의 false boundary label 방지, target/FREE 모순 금지, 실제 SS hash와 physical mesh-cell bound를 포함한다.

Native RGB condition64 cache는 완료됐다. 실제 네 view projection의 mean과 first target projection을 고정했으며 로딩 이후 15.4초였다. GT geometry/latent 값은 condition 구축에 열지 않았다. V3의 실제 full regional gradient proof는 두 모듈의 초기 loss55.82744598이 native와 exact이고, frozen SS/decoder에는 gradient가 없었다. Query encoder의 두 번째 gradient L1은0.00546829, GPU peak는 약5.23GiB였다. V4 label에서도 별도 preflight를 수행하고 학습 gate로 사용한다.

Voxel/visibility label 계산은 CPU2core로 제한한다. 설치된 `o_voxel` package가 import 때 CUDA metadata를 조회하므로 허용된 GPU3이 보이는 환경에서 실행한다. 다른 GPU를 찾거나 조회하지 않는다. 데이터/결과는 file4의 `rgb_matching_v2/current-geometry-regions-v1`에 저장한다.

## 정합 검사에서 발견한 차이

V1 direct64와 실제 SS input 사이에5cell 차이가 발견됐다. V2의 근접도 검사와 V3 native1024 pooling 검사도 보존했다. 한 사례는 direct64 6382개와 실제6433개가 달랐지만 native1024 pooling에서는6433개가 exact였다. 다른 사례에는 equivalent transform의 연산 순서 등에 따른 missing1/extra1 차이가 남아 원인을 확정하지 않았다.

재voxelization 결과를 기존 GT와 같게 만드는 것을 최종 정합 정의로 삼지 않았다. 실제 encoder input hash와 모든 cell의 실제 M1 triangle 정합을 검증한 V4를 사용한다. 중복되고 비싼 exact-dump high-resolution 재구축 probe는 명시적으로 종료했고 그 관측·종료 이유를 남겼다. 기존 SS/SLAT GT, RGB, predicted query, 진행 중인 본 SS 학습은 수정하지 않았다. 근거는 `current-region-physical-encoder-input-registration64.json` 및 각 version의 실패/검증 기록이다.

## 실제 학습 시작

V4의64개 label은68.66초에 모두 완료됐다. 실제 SS hash/physical registration을 모두 통과했고 source witness·관측 visibility·보수적 ray label의 각 count를 보존했다. V4 full-rollout GPU gate도 두 모듈 모두 통과했다.

`runs-v2`의seed101 두 run이GPU2/3에서 실제 update를 시작했다. 첫 pair/noise가 같고 initial endpoint loss는둘다58.80086517, correction0이다. 첫 update peak는adapter6.726GiB /query6.732GiB였다. 이것은 학습 시작과 초기 정합 증명이며128-update 완료/일반화 개선을 뜻하지 않는다. 이전`runs-v1` 큐는GT검사 실패에 따른HELD 이력으로 남기고 실제학습결과로 집계하지 않는다.

본14개 최종 집계가 완료된 뒤 seed202가GPU0/1에서 시작했다. 네 controller가 동일한CPU94/95에 몰리면서 update 공급이 느려지는 것을 확인해 각 process의모든 thread를두CPU씩다르게배치했다(GPU0:88/89,1:90/91,2:92/93,3:94/95). 총8core이며process별2core 제한은유지했다. 실행중optimizer·data·noise·step을변경하거나재시작하지않았다. Pending seed303에는명시적`--cpu-affinity`를넣고GPU선택과독립적으로기록한다. 근거는`provenance/current-regional-cpu-affinity-distribution.json`이다.

최종6run은schedule·optimizer128step·GT/native voxel hash·같은32cluster/3noise를엄격히검사한뒤`all-six-run-summary-v2.json`과동명Markdown으로집계한다. Noise/model repeat를독립표본으로늘리지않고source안에서평균한다. Empty region의None과유효cluster수도그대로표시한다. 완료된14개본비교표와이regional표는각기다른GT/macro정의로구분해W&B에자동저장한다.

## Seed101의 fixed-final 정상 평가: 중간 결과

두arm 모두128update를완료하고checkpoint/SHA를고정한뒤cal32×3noise의native/normal 평가를마쳤다. GPU2/3의native96개전체metric·pred/GT voxel hash가exact였다. 아래는한training seed의중간관측이며source-off/shuffle과seed202/303은아직진행중이다. 실제SS encoder input GT의macro 집계로본14개표와직접비교하지않는다.

| Method | Macro IoU | Source recall32 | Stale leak32 ↓ | Regional BCE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Frozen native | 0.445530 | 0.825740 | 0.601524 | 95.616471 |
| Adapter-only | 0.441108 | 0.812574 | 0.571341 | 95.006540 |
| Adapter+geometry query | 0.445109 | 0.809378 | 0.533634 | 91.544493 |

Query−adapter는stale leak32 **−3.771pp,source-cluster95% CI[−5.986,−1.812]pp**,regional BCE −3.462[−6.259,−0.610]이었다. 반면IoU +0.400pp의CI[−0.409,+1.143]pp와source recall32 −0.320pp의CI[−0.935,+0.222]pp는0을포함한다. Geometry cleanup은이seed에서관측되지만source 회수나global SS 향상을함께입증한것은아니다.

Query−native는IoU −0.042pp,CI[−0.939,+0.806]pp이고source recall32는−1.636pp,CI[−2.634,−0.787]pp다. Stale 제거와source positive 손실의tradeoff가있어원래regional loss가낮아졌다는이유만으로Full에채택하지않는다. Source-only 회수의원래양의결과를재현했다고보고하지않는다. 세noise를source안에서평균한32cluster bootstrap10,000회이며세training seed나unseen test의CI가아니다.

원본목적함수는해상도64/32 각각의`0.5 × (source-positive BCE + stale BCE + balanced-visible BCE)`를합하고correction penalty0.001을더한다. 각BCE는해당영역의mean이며voxel수만으로loss지배항을단정하지않는다. IoU나source recall을직접최적화하는수식은아니므로aggregate BCE 감소가threshold0의binary recall/IoU 향상을보장하지않는다. 현재숫자는이tradeoff와양립하지만loss항별인과기여를분해한실험은아니다.

기계판독근거는`current-geometry-regions-v1/interim-s101-normal-v1.json`이다. 학습설정과나머지seed/arm을변경하거나중간결과로중단하지않는다. 3seed 최종집계와source-off·joint-query shuffle을모두확인한다.
