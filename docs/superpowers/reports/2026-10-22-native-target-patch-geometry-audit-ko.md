# Native target patch 주소와 투영 좌표 검사

진행 중인 PBR vanilla 학습을 유지하면서 기존 DEV32의 target camera0 주소를
검사했다. 원본 `ProjGrid.__init__`, 카메라 행렬 계산 및 projection 함수의 AST를
그대로 실행하여 고정 geometry buffer와 투영 좌표만 계산했다. Extractor나
pretrained 신경망을 구성하지 않았고, 신경망 forward/backward 및 학습은 없다.

SS는 실제 native16³ lattice, PBR은 각 자산의 실제 sparse64 support를 사용했다.
Cache와 sparse support 및 bound manifest/panel의 입력66개·23,674,658 bytes는
file4의 실제 SHA256/size와 대조했다. Cache는 모두 원래 생성 규칙과 정확히 같다.

| 항목 | SS | PBR |
|---|---:|---:|
| 전체 좌표 | 131,072 | 249,402 |
| Target frustum 안의 좌표 | 131,012 | 249,402 |
| 연속 투영 좌표 최대 오차, patch 단위 | 0.000002721 | 0.000002432 |
| Cache 주소와 native nearest bin이 다른 유효 좌표 | 6,491 | 6,852 |
| 다른 경우의 최대 축별 bin 거리 | 1 | 1 |
| Frustum 밖에서 border로 clamp된 cache 주소 | 60 | 0 |

연속 좌표 비교에는 native sampler의 `+0.5` image-pixel 규칙을 반영했다.
Cache는 normalized raw UV를 floor하여32×32 patch를 고른다. Native nearest
ray bin은 sampler 좌표를 round한다. Half-pixel 이동은 SS에서0.03125 patch,
PBR에서0.015625 patch이므로 경계 근처의 인접 bin 차이는 존재한다.
최대 cache-center 거리는 각각0.531143/0.515591 patch였다. 따라서 두 주소가
모두 bit 단위로 같다고 주장하지 않는다. 다만 이 패널에서는 큰 축·scale·camera
frame 불일치를 관측하지 않았다. 인접 bin 차이의 생성 품질 영향은 별도 문제다.

초기 검사 스크립트 v1/v2는 저장된 uint8 sparse 좌표를 int64 변환 전에
flatten하여 검사 자체에서 정수 overflow를 일으켰다. 실제 native runtime은
이미 long 좌표로 계산한다. 검사 v3에서 이를 바로잡았으며, 앞선 실패 스크립트와
로그도 함께 보존했다. 모델·학습·cache·queue 설정은 변경하지 않았다.

이 결과는 source camera의 투영, 전체 TRAIN/TEST508, target-alpha gating,
matching 확률 및 최종 생성 품질의 검증을 대신하지 않는다. 현재 attention의
추가 이득 부재를 큰 target 주소 정합 오류로 설명할 근거는 찾지 못했다.
후속 native480 readout 개입 검사가 후보 V·matching prior·QK 및 retargeting
출력의 실제 소비를 분리해서 확인한다.

실제 결과는 `native-flow-cold3-v1/target-patch-native-geometry-DEV32-CPU-v3/`의
`audit.json`, `actual-input-file4-SHA256.json`, `file4-closure.json`에 있다.
