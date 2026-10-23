PBR 네 모델의 공통 native guide 비교 연결 — 2026-10-23 18:19 UTC

기존 상태 기록에서 PBR 네 모델의 generated-geometry32 평가가 연결돼 있다고 했지만, 실제 큐에서는 해당 평가가 보류된 이전 full-cascade factory에 묶여 있었다. 이번 변경은 이 누락을 수정한다. 이전 factory를 재개하거나 이전 reconstructed-support 결과를 새 평가로 해석하지 않는다.

현재 실행 중인 SS/PBR 학습, 기존 큐 descriptor, 보류한 주변 실험은 변경하지 않았다. 새로운 학습 arm은 없다.

비교 구성은 vanilla, top1-only, top1-plus-feature, raw-topk-plus-feature 네 개의 실제 native1900 PBR 모델이다. SS V1의 실제 native480 예측을 기존 Shape512 step4210과 Shape1024 step16850으로 전달하는 v167 공통 cascade에서 생성 geometry 32개와 native guide를 얻는다. 이 geometry와 sparse guide의 순서·scale·logit·branch를 모델별로 그대로 재사용한다. PBR 비교에서는 SS와 Shape의 신경망 forward를 다시 실행하지 않는다.

공통 cascade에서 처음 생성한 V1 텍스처는 이 비교의 baseline으로 재사용하지 않는다. 원래 cascade에서는 Shape sampling 이후의 RNG 상태가 텍스처 noise를 결정하기 때문이다. 네 모델 모두 동일한 seed 20261020으로 PBR sampling을 다시 시작하며, 실제 첫 texture noise SHA256와 sampler 값이 모델 사이에 같은지 최종 집계에서 검사한다. Vanilla도 이 경로로 별도 생성한다.

각 모델은 원래 PBR16850에서 fresh optimizer로 학습한 1900 step, DDP 4장, global batch24, micro6, accumulation1을 완료해야 한다. 실행 시 실제 네 rank의 완료 기록, frozen base·matcher·anchor의 보존, 최종 NN·source checkpoint의 파일서버4 SHA256를 확인한다. 오래된 engineering checkpoint나 다른 모델의 conditioner를 대신 쓰지 않는다.

각 사례는 native guide replay, 동일 renderer, 동일 GT reference에 대한 shared score의 세 GPU child session을 순서대로 소유한다. 각 child의 종료 코드와 실제 session drain을 확인하고 결과를 파일서버4에 보존한다. 모델별 첫 사례가 성공해야 나머지 31개가 실행된다. 실패 사례를 자동 재시도하거나 성공으로 바꾸지 않는다.

평가는 target camera8과 source camera9·10을 구분한다. Source reference는 edited M1 geometry에 원래 material을 적용한 영상이다. 변형 전 M0 영상과 픽셀을 맞추도록 강제하지 않는다. LPIPS·DreamSim·CLIP, PSNR·SSIM·M-IoU, CD·F-Score를 집계한다. PBR isolation에서는 geometry를 고정하므로 CD/F-Score가 모델별로 같은 것은 정상이며 geometry 개선의 근거로 쓰지 않는다. DreamSim/CLIP은 GPU에서 FP32, TF32 off, no AMP의 기존 공식 평가를 사용한다. FID/KID는 이 32개 탐색 평가에서 판단하지 않고 최종 고정 TEST508 평가에 남긴다.

실제 연결된 작업은 replay/render/shared 사례 128건과 coverage CPU, official384 GPU, summary CPU의 총 131건이다. 새 GPU 작업 129개 모두 기존 v172 resource permit을 요구한다. SS 네 모델 학습 후 raw PBR가 GPU0–3 네 장을 확보하는 동안 새 평가가 끼어들지 않는다. GPU6·7의 SS 비교 학습은 변경하지 않았다.

검증 결과:

- 새 contract와 기존 PBR summary 관련 CPU 테스트 36개 통과.
- Native guide capture/archive, replay lineage, GPU assignment의 기존 테스트 43개 통과.
- 실제 완료된 vanilla/top1-only PBR의 네 rank와 1900 step lineage 확인.
- 128개 계획의 공통 geometry·guide·RGB·seed·sampler 일치 확인. 추론용 계획에서 GT transport/latent cache와 이전 support audit 의존성 제거.
- 기존 수치 코드의 source hash 불일치 0개. 기존 큐 descriptor는 모두 그대로 두고 새 작업만 append.
- 실제 파일서버4 SHA256/size 검증: prelaunch 518개 파일, queue 계획 2개, 연결 기록 1개, 연결 후 감사 1개.
- 실제 dispatcher가 새 131개 작업을 모두 pending으로 인식함을 확인. 이 시점의 실제 렌더링 결과와 품질 향상은 아직 없다.

결과 경로는 파일서버4의 기존 retargeting-v1/rgb_matching_v2 아래 `semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/PBR-native-science-runtime-v1/unified-module-pipeline-v1/common-geometry-PBR4-native32-v174`이다. 실제 연결은 이 폴더의 `applied-v1/audit.json`, 계획 일치는 `postconnection-audit-v1/audit.json`에 기록했다.

18:18 UTC 확인한 진행 상태는 PBR feature 1821/1900, SS Top1-V 449/480, SS fixed Top-K 440/480이다. 해당 학습과 dispatcher, resource controller의 PID/birth가 그대로 살아 있다. Raw PBR의 유일한 미충족 release gate는 앞서 정한 SS 네 모델의 완료 gate다.

이 변경으로 최종 파이프라인이 확정되지는 않는다. 남은 요구는 12개 SS arm의 최종 DEV128×3 비교, SS 공통-PBR cascade32, 이번 PBR4 공통-geometry32, 선택된 SS/PBR 조합의 실제 joint cascade, 독립 training seed202 확인, 정확한 population/metric/model freeze, TEST508의 10개 metric과 두 zero-shot 입력 대조군이다. 별도 attention correction·teacher warm-start·새 주변 sweep을 추가하지 않는다.
