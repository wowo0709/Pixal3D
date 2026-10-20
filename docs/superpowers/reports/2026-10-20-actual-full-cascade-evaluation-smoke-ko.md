# 실제 전체 cascade 생성 및6개 metric 경로 검증

기존 SS의32개 예측을 native evaluator로 다시 생성하여 latent와64³ occupancy를 저장했다. 기존 Direct480의32개 per-asset 기록과 전체 지표가 정확히 일치했다. `cascade-ss-archive32-v2`의37개 artifact/6,142,594bytes를 actual file4 SHA256·크기로 검증했다. 최초v1은 잘못 지정한 receipt 경로 때문에 native 실행 전 실패했고 기록을 보존했다. v2는 실제 final-summary/core-proof receipt를 사용한다.

이 예측 중 첫1개를 사용해 **GT Shape 없이** Shape512→Shape1024→PBR1024를 실제로 생성했다. 학습을 추가한 실행은 아니다. 아래 수치는 실행 경로를 검증한1개 샘플 결과이며 validation 전체 평균이나 모델 우열의 증거가 아니다.

## 모델과 입력

- SS는 native Direct480의 실제 예측이다. 예측64³ occupancy를 native max-pool로32³ support에 전달했다. GT occupancy·Shape latent·좌표는 생성에 넣지 않았다.
- Shape512 step4210,Shape1024 step16850의 사용자 지정 multi-view LoRA checkpoint와 원본 config를 고정했다. PBR은 모든 SS 후보에 공통으로 적용할 V1 step762를 사용했다. 새 Direct PBR 본학습 결과로 표기하지 않는다.
- 입력은 M1 RGB cam08 + M0 RGB cams09/10/11의4장이다. 동일 실제 파일에서512/1024 RGBA resize 및 black compositing 결과가 native training loader와 bitwise 일치했다. 원본 config·pipeline normalization 차이도0이었다. CPU 검사2개/.11초와 실제 파일 기반 parity audit를 통과했다.
- 기존 pipeline의 inline LR→HR quantization·native sampler·decoder를 사용한다. BF16은 flow/extractor에 한정하고 decoder는 native dtype으로 실행한다. Frozen image projection은 기존 검증된 이미지별 streaming을 사용했다.

## 실제 첫 생성

Sample ID는 `shard003_e7c02681bc2d4edf9400fbcdde95eaef_d1_fast300fpssupplyv1_balanced_JN_L_target08`이다.

| 항목 | 실제 값 |
|---|---:|
| SS32 support |750 tokens|
| Shape1024 support |3,161 tokens|
| 생성 구간 |114.48초|
| GPU peak allocation |16.2618GiB|
| Symmetric Chamfer-L1 |.01000276|
| Surface F-Score@.02 |.91348975|

Geometry는 생성이 끝난 후 처음 로드했다. 예측과 GT에 같은 source-defined 정규화를 적용했고 출력별 ICP·recenter·rescale을 수행하지 않았다. Surface point20,000개씩,seed101,양방향 nearest-distance 평균의 평균을 CD-L1으로 사용했다. Source frame에서의 실제 anchor rotation을 확인했고 translation은 numerical zero였다. Native/wrapper/queue가0으로 종료했으며 `generated-cascade-smoke-v1`의4개 artifact/57,016,501bytes가 actual file4와 일치했다.

## 카메라와 동일 렌더링 조건

cam08/09/10의 GT geometry raster 대 실제 RGBA mask IoU는 각각 .996667/.997502/.997805,한 pixel dilation containment는 모두1이었다. RGB/alpha raw array를 보관했다. cam09/10의 GT target 이미지는 입력에 포함되지 않은 M1 상태지만 카메라 자체는 M0 입력 camera와 같다. Novel-camera 성능으로 주장하지 않는다.

원본 Cycles GT와 native PBR renderer의 조명을 혼합한 첫 영상 평가는 별도 진단 기록으로 남긴다. OpenEXR codec의 기본 비활성화로 최초 render process가 실패했으며, 원래 실행을 보존하고 새 process에 `OPENCV_IO_ENABLE_OPENEXR=1`을 적용했다. 모델·metric 함수나 mesh를 바꾸거나 다시 생성하지 않았다.

최종 진단에서는 원본 target PBR dump의 실제 재질·UV를 native loader로 읽고, world-frame vertex를 복원해 **예측과 같은 renderer·forest environment·camera·AO seed0**로 GT를 렌더링했다. VAE reconstruction을 GT 재질로 사용하지 않았다. 복원된 textured mesh와 독립 GT vertex 사이 최대 최근접 차이는9.1e−8이었다. 이미 생성한 prediction render는 그대로 재사용했다.

| Camera | PSNR↑ | SSIM↑ | LPIPS↓ | 2D M-IoU↑ |
|---|---:|---:|---:|---:|
|08|20.5203|.968424|.031483|.909145|
|09|27.2882|.969562|.025653|.895637|
|10|22.4488|.970269|.026599|.753870|
|시점 평균|23.4191|.969419|.027912|.852884|

원본 Cycles 대 native GT 자체의 LPIPS도 시점별 .02364/.02629/.01755였다. 조명·renderer 차이를 모델 오차에 섞어 평가하지 않아야 한다. 동일 조건을 맞춰도 예측의 재질·형상 오류는 남는다. Foreground PSNR 평균은8.8518dB로 낮으므로 full-image PSNR만으로 appearance가 좋다고 해석하지 않는다. 이 수치를 특정 모듈의 실패/성공 원인으로 아직 분리하지 않았다.

`generated-cascade-render-v2`와 `generated-cascade-shared-material-v1`은 native/wrapper/queue0 및 actual file4 closure를 통과했다. 후자의5개 artifact는731,081bytes다. `generated-cascade-visualization-v1`의 초기 비교 그림은 원본 Cycles를 보여 주므로 동일 조명 GT의 비교 그림으로 표기하지 않는다.

## 남은 범위

현재6개 지표가 실제 생성 결과에서 작동하지만 **1개 샘플의 구현 검증**이다. 동일 설정의 V1·matched R·Direct·새 후보에 확대해야 한다. SS IoU 표를 이6개 지표의 결과로 대체했다고 주장하지 않는다. DreamSim·CLIP·FID/KID,개발/최종 패널의 paired source-cluster 비교,별도 training seed 및 최종 test는 남아 있다. 새 bounded task attention의 자체 초기 생성·DDP capacity·과학적 학습도 아직 연결해야 한다.

진행 중인 학습과 다른 사용자의 GPU 작업은 중단하지 않았다. 학습/검증에는 사용자 지정 GPU0~3만 사용했다. 첫 샘플의114초를 전체 데이터셋의 평균 시간이나 worst-case capacity로 사용하지 않는다. 전체 목표는 진행 중이다.
