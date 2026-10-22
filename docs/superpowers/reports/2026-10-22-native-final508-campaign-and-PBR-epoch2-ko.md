# 최종508 cascade·10 metric 실행 연결과 PBR 2에폭

기록은 실제 머신 UTC2026-10-22 기준이다. PBR 학습·개발 평가를 유지하며,
최종 test는 agent가 실제 개발 결과를 검토해 모델 조합과 주장 범위를 결정한 뒤 연다.
사용자 승인이나 추가 입력을 기다리는 조건이 아니다.

## 실제 PBR 760 update 검증

Vanilla가 두 번째 에폭760을 저장했고, NN·Adam·config·descriptor의 실제 file4
SHA/size 검증과 고정4 validation을 마친 뒤 학습을 재개했다. Global24/world4/
micro6/accumulation1이며 총1900 update다. 다른 source 세 모델은 순차 학습한다.

| 고정4·GT Shape 조건 | Step0 | Step380 | Step760 |
|---|---:|---:|---:|
| Render PSNR ↑ | 26.05731 | 26.34439 | 26.60228 |
| Render LPIPS ↓ | .010105 | .009688 | .009273 |
| Foreground crop LPIPS ↓ | .095126 | .092427 | .090078 |

동일한 실제 RGB·noise·GT latent/Shape·camera hash를 대조했다. 작은 GT Shape stage
패널의 연속 개선이며, source preservation이나 실제 SS→Shape→PBR 성능 개선으로
해석하지 않는다. 주요 개선이 일부 사례에 집중되는지도 보존한 per-asset delta로
확인해야 한다. Alpha IoU1은 GT Shape가 같아서 생기는 값이다.

첫760 snapshot 준비는 PID 문자열의 잘못된 치환 때문에 데이터 복사 이후 중단됐다.
훈련 프로세스나 checkpoint를 수정하지 않았으며 실패한 준비 폴더·코드는 보존했다.
실제 PID238065/birth1792670094.76을 재확인한 새 v2 snapshot은22개 파일2,316,884 bytes의
실제 file4 SHA/size 검증을 통과했다. 상속된 FIRST_EPOCH 상태 이름을 SECOND_EPOCH로
정정한 v3 sidecar도 별도로 보존했다. 수치·이미지·runtime prefix는 바꾸지 않았다.

Canonical `PBR-native-science-runtime-v1/epoch760-actual-evidence-v2/`에 실제 로그 prefix와
0/380/760 PNG가 있고, `epoch760-actual-evidence-v3/audit.json`은 정정된 상태명과
원본 snapshot SHA/receipt를 가리킨다. V3는2파일11,883 bytes를 실제 file4에서 검증했다.

## 실제 연결된 최종 평가

`activate-native-six-model-final508-cascade-and-ten-metrics-v1` CPU 작업을 대기열에
추가했다. 두 기존 SS508 activator와 여섯 모델의 실제 완료 wrapper/terminal을 요구한다.
단순히 파일이 보이는 시점을 완료로 판단하지 않고 dispatcher의 실제 terminal0도 확인한다.
학습한 네 모델과 original zero-source4/zero-edited4를 모두 보존한다.

조건이 충족되면 실제 checkpoint·config·source pair와 SS508 archive SHA를 고정하고
네 GPU를 소유하는 최종 campaign 작업과 별도 CPU 집계 작업을 추가한다.
최종 campaign의 구성은 다음과 같다.

- 원본2+native4 모델×508 asset=3048개 실제 case.
- Source와 target의 공식 DreamSim/CLIP 총9144 view.
- 모델별1524-image Inception pool6개와 공통 GT1524-image pool1개, 총10668 image.
- Source LPIPS/DreamSim/CLIP, global FID/KID, target PSNR/SSIM/M-IoU/CD/F-Score의10개 주 지표.

GPU를 모델에 고정하지 않고 네 GPU의 빈 slot에 case를 배분한다. 같은 모델의 서로 다른
asset도 독립 실행할 수 있다. Source4가 만든 각 asset의 공통 original-material M1 GT가
완료돼야 다른 모델의 해당 asset 평가를 진행한다. 동시 미정리 raw case는 최대6개다.

실제 child PID/birth/exit code, phase 출력, raw·render 픽셀과 GT|Pred|Diff를 보존한다.
실패가 생기면 새 case 투입을 중단하고 이미 시작한 child가 끝날 때까지 GPU 예약을 유지한다.
살아 있는 child를 재시작하거나 GPU가 해제됐다고 잘못 알리지 않는다.

각 case의 raw 복사본은 실제 모든 생성·렌더 child의 terminal0, 공통 GT surface join,
소유 범위, 열린 reader와 미래 consumer, 깨끗한 commit ancestry, 새로 읽은 실제 remote
SHA/size를 확인한 뒤에만 로컬에서 정리한다. Canonical file4 raw, NN, 데이터와 픽셀은 남긴다.

## 생성과 지표의 정합

원래 SS480/nativePBR1900와 original4210/16850 stage의 수치 코드를 사용한다.
GT surface reference는 실제 generation 성공 이후에 읽으며 생성기가 사용한
target vertices/faces·source center/extent가 공통20k/seed101/F@.02 reference와 같은지
검증한다. Per-output alignment나 GT geometry conditioning을 허용하지 않는다.

Native PBR replay는 예측 sparse shape support에서 자기 NN/source0·10·19를 적용한다.
원래 predicted raw geometry와 surface 점수가 바뀌지 않아야 한다. Zero-shot은 기존
원본 네 stage inline texture sampling을 유지한다. Native replay와 zero inline의 초기
texture noise가 동일하다고 주장하지 않으며 서로 다른 predicted support 간에도
동일한 noise array를 주장하지 않는다. 이 구분을 frozen protocol에 기록한다.

모든 모델은 같은 M1 geometry·원래 material의 GT RGB/alpha를 같은 camera8/9/10에서
참조한다. Target8과 source9/10 평균을 분리한다. 실제 scored FP32 RGB, alpha,
official uint8 conversion, model/sample/camera/cluster/archive SHA가 perceptual metric과
FID/KID pool 사이에서도 일치해야 한다. FID/KID는 공식 torch-fidelity0.3.0의2048 feature,
동일1524 image count, KID100 subsets×1000으로 계산한다. 음수 unbiased KID를 자르지 않으며
subset 표준편차를 source-cluster CI로 부르지 않는다. Neural feature 추출은 GPU에서,
마지막 수치 집계만 별도 CPU2 core에서 수행해 GPU를 먼저 해제한다.

역사적 handoff의 Inception GPU sanity flag는 오래된 false였다. 실제 완료된
`official-Inception2048-GPU-sanity-v1`의 terminal0와 canonical
`PASS_ACTUAL_OFFICIAL_INCEPTION_GPU_SANITY` audit을 다시 확인했다. 기존 가중치를
그대로 재사용하며 추가 다운로드나 global environment 변경은 없다.

## 모델 결정과 아직 남은 실험

Test 전에 실제 SS128·독립 학습 seed202·PBR isolation·full-cascade 개발 결과를
agent가 검토하는 조건을 **pending test activator 한 개의 requires에만 추가**했다.
학습·개발 작업의 config, source, descriptor와 queue-state는 바꾸지 않았다.
선택한 SS/PBR 조합과 개별 모듈의 효과·미확인 사항은
`development-model-decision-actual-v1/`에 실제 근거와 함께 기록해야 한다.
SS와 PBR에서 다른 mode가 유리하면 한 개의 타당한 hybrid를 먼저 개발 패널로 검증한다.
검토 없이 이 경로에 완료 receipt를 만들거나 test 점수를 보고 구조를 고르지 않는다.

새 full campaign 준비2파일69,286 bytes, append-only queue2파일7,723,495 bytes,
test gate의 이전/이후 descriptor와 audit3파일6,611 bytes를 실제 file4 SHA/size로 검증했다.
전체3523개 dependency DAG가 통과했고, 관련 CPU 검사는 **202 passed in4.24s**였다.

최종 GPU 실행은 아직 시작 전이다. 실제 PBR 네 모델·기존 PBR isolation128·역사적
target-patch 재현·독립202·native 개발8조합 평가와 개발 모델 판단이 남아 있다.
위 실행 연결이나 CPU 검사를 최종10 metric 측정 완료로 해석하지 않는다.
Retargeting의 개선 근거와 별개로 semantic Top-K/QKV의 추가 생성 품질 이득은 아직
확정하지 않으며, 전체 목표는 계속 진행 중이다.
