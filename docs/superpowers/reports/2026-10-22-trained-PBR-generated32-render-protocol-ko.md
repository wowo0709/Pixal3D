# 학습된 PBR1905의 generated-shape 평가 연결

기존160개 prediction 작업은 생성과 무결성 검사까지 연결돼 있었다. 이 뒤에
실제 rendering, official DreamSim/CLIP, target/source 역할별 paired 비교를 연결한다.
기존 prediction을 다시 생성하거나 학습 결과에 따른 quality gate를 추가하지 않는다.

## 두 비교와 공통 baseline

- Full stack96개: SS V1/R/Direct에서 생성한 각 geometry에 대응하는 PBR1905를 사용한다.
  SS와 PBR가 함께 달라지므로 개별 모듈 효과로 해석하지 않는다.
- PBR isolation64개 추가: SS V1의 동일 geometry에 R/Direct PBR1905를 적용한다.
  Vanilla baseline32개는 full-stack V1을 재사용한다. 동일 raw geometry, PBR 초기
  noise와 sampler 설정을 기존 생성 audit에서 검사한다.

전체는160개의 서로 다른 prediction,5개 관측 모델이다. 이32개 개발 자산 결과는
현재 새 source-camera-ray SS 후보의 joint 학습이나 독립 test를 대신하지 않는다.

## Geometry를 고정한 texture 평가

기존 renderer는 raw face가100만을 넘으면 native simplification을 수행한다.
이를 texture arm마다 새로 실행하면 원래 같은 geometry의 mask가 달라질 수 있다.
각 SS prediction에 대해 **이미 만들어진 native100만-face LOD**를 재사용한다.
Full-stack의 각 geometry도 해당 prediction의 동일 규칙을 따른다. PBR isolation의
세 texture arm은 같은 raw geometry와 같은 LOD를 공유한다. GT geometry로
LOD를 만들거나 output별 alignment를 하지 않는다.

새 entry는 기존 `render_cascade_artifact_v2`를 호출하면서 simplification 단계만
이 LOD로 대체한다. Raw predicted vertices/faces hash와 LOD의 SHA256을 확인한다.
실제 출력된 render geometry가 원래 LOD와 정확히 같은지도 검사한다.
나머지 camera8/9/10, native rasterization·material·AO·metric 코드는 유지한다.
CD/F-Score는 간소화 전 raw generated surface의 기존 결과를 사용한다.

먼저 이미 native 원본과 모든 array가 같은 zero-export replay를 같은 LOD에서
실제로 렌더한다. 세 camera의 모든 array와 metric이 원래 render와 정확히 같은지
확인한 뒤160개 후속 렌더를 허용한다. CPU hook/geometry 검사는 이 GPU proof를
대체하지 않는다. 실패 결과를 보존하고 자동 retry하지 않는다.

## Metric와 저장

모든 모델에 같은 original-material M1 shared-render GT pixel을 사용한다.
Camera8의 PSNR/SSIM/M-IoU 및 raw CD/F-Score는 target alignment,
camera9/10의 LPIPS/DreamSim/CLIP은 source preservation과 adaptation을 평가한다.
Source reference는 원래 M0 입력 이미지가 아니라 M1 geometry의 원래 material이다.
각 모델·camera의 fullframe score를 주 결과로, GT-only crop을 보조 결과로 둔다.
Camera들을 독립 자산으로 재추출하지 않고 자산 내부에서 평균한 뒤 source cluster를
bootstrap한다. 개발32개의 보정되지 않은 CI이며 독립 학습 반복은 아니다.
FID/KID는 최종 test508의1524-view pool에서 측정할 항목으로 유지한다.

결과 array·render metric·official metric·paired 비교와 이미지·W&B를 file4에
보관한다. 이미 file4에 SHA/size 대조 완료된 **새 replay의 local NPZ 복사본만**
렌더 후 제거할 수 있으며, 삭제 직전 remote SHA를 다시 확인하고 복구 주소를
기록한다. 원본 SS generation, LOD, weight, reference, canonical file4 결과는
삭제하지 않는다. 이 절차는 로컬 공간 때문에 후속 joint 학습이 막히지 않도록 한다.
