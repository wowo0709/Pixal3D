# Native SS480·PBR1900의 정합된 full-cascade 평가

## 현재 실제 실행

PBR 네 arm 모두 실제 DDP4/global24/micro6/accumulation1 최대 TRAIN31900-token
검사를 통과했다. 세 update 뒤 고정4 렌더 validation, 다음 네 번째 update,
전체 LoRA360개와 source0/10/19개의 학습·Adam4·DDP 동등성·frozen parameter
불변을 검사했다. 네 모델의 실제 noise/timestep/global input trace도 rank별로 같다.

| 모델 | GPU당 최대 allocated (GiB) |
|---|---:|
| Vanilla | 46.4373 |
| Top-1+R | 50.0246 |
| Top-1+feature+R | 50.2816 |
| Raw Top-K+feature+R | 50.2816 |

이 값은 같은 최대 asset을 반복한 capacity 값이며 실제 데이터 처리량이 아니다.
기존 DDP2/micro12의 source 세 arm OOM은 별도 실패 이력으로 남아 있다.

Vanilla의 실제 fresh1900 본 학습이 시작됐다. 최초 고정4 평가가 원본 PBR16850
cold reference와 정확히 일치한 뒤 optimizer update를 시작했다. 초기14 update
시점의 최근10 update median compute는13.31초, 최대 allocated는43.65 GiB였다.
초기 짧은 관측이며1900 update 전체 소요시간의 확정 예측이 아니다.
W&B는 `youngwoojeon-unist/pixal3d-deformnext-retargeting`에 연결됐다.

매380 update의 고정4 평가는 GT shape 조건에서 PSNR/SSIM/LPIPS와
GT/Pred/Diff를 기록한다. 초기26.0573 dB 및 SSIM .991635는 이 bounded stage
평가의 값이며, predicted geometry의 full-cascade 성능으로 해석하지 않는다.
특히 GT-shape alpha IoU1은 geometry 개선의 증거가 아니다.

## 연결한 여덟 모델 비교

네 SS 모델은 모두 실제480 update를 마쳤고, 동일32개·noise의 predicted voxel
archive128개를 실제 file4 SHA와 대조했다. 각각을 원본 Shape5124210 및
Shape102416850에 연결하고 다음 두 texture 조건을 평가한다.

| SS480 | SS 효과 분리 | 해당 PBR까지 학습한 조합 |
|---|---|---|
| Vanilla | 원본 PBR16850 | Vanilla PBR1900 |
| Top-1+R | 원본 PBR16850 | Top-1+R PBR1900 |
| Feature+R | 원본 PBR16850 | Feature+R PBR1900 |
| Raw Top-K+feature+R | 원본 PBR16850 | Raw Top-K+feature+R PBR1900 |

같은 SS에서 두 PBR은 실제 geometry, native predicted LOD와 초기 texture
noise를 공유한다. 둘 다 CPU seed20261020으로 PBR sampler를 새로 시작한다.
최초 geometry cascade 내부에서 소비한 PBR noise stream과 새 replay를 직접
비교하지 않는다. 서로 다른 SS가 만든 sparse support 사이에는 같은 noise
array라는 주장을 하지 않는다. Geometry GT는 생성 결과를 저장한 뒤 평가에만 읽는다.

참조 RGB/alpha는 edited M1 geometry에 원래 source material을 적용한 동일
reference다. Target camera8과 source9/10을 분리해 평가한다.
Source preservation은 LPIPS/DreamSim/CLIP, target alignment는
PSNR/SSIM/M-IoU/CD/F-Score를 사용한다. CD/F-Score는 정렬 없이 원래 source
bbox 좌표계에서 surface20k/20k, threshold .02로 계산한다.

8모델×32개×3카메라=768 view의 DreamSim과 별도 OpenAI CLIP은 기존 pinned
공식 weight registry를 재사용한다. GPU FP32/TF32off/noAMP이며 CPU로 neural
forward를 수행하지 않는다. 전체 frame이 주 지표이고 GT foreground crop은
보조 지표다. Source cluster paired bootstrap4000회로16개 비교를 보고하며,
이 개발 비교의 CI는 다중 비교 보정 전이라는 표시를 유지한다.

## 실행 상태와 저장 제한

지금 연결한 것은 **실제 가중치 완료를 기다리는 CPU activator**다.
네 PBR1900의 실제 완료·canonical SHA·중간 checkpoint cache 정리가 모두
끝나야128개 cascade case,128개 local raw retirement, official768 평가와
8-metric summary의258개 작업을 추가한다. 연결됐다는 사실은 실제 full-cascade
생성이나 품질 개선이 완료됐다는 뜻이 아니다.

Case 하나는 geometry/prior texture/matched texture의 세 raw root를 가진다.
SS 모델별로 한 case만 열고, 실제 생성·렌더·원격 SHA 검증과 모든 reader 종료
후 local duplicate 여섯 파일만 정리한 뒤 다음 case로 진행한다. File4 raw,
pixel archives, metadata, 원본 데이터와 checkpoint는 유지한다.

Lineage·noise·geometry·GT·pixel·official registry가 다른 결과를 합치지 않는
CPU 검사와 소유권/reader/canonical 경로 검사가 통과했다. 관련 새 cascade
contract/summary/retirement 및 기존 raw 보호 검사26개가1.02초에 통과했다.
별도 summary 회귀 검사와 기존 PBR summary 검사20개도 통과했다.

독립 학습 seed202, native spatial/decoder/score의 추가 효과 분리, final508 및
FID/KID를 포함한10개 최종 지표는 남아 있다. 현재 어느 새 full 모델도 채택하지 않았다.
