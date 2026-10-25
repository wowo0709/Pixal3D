# Frozen TEST508 실제 실행과 운영 복구

DEV 선택은 **Fixed SS + Top-1 R PBR**이며 `v558`에서 모델·입력·sampler·camera·
수치 source와10metric protocol을 TEST 전에 동결했다. SS는 geometry CD/F-Score,
PBR는 동일 geometry의 source appearance를 기준으로 선정했다. 개별 모듈의 DEV48개
보정 구간과 PBR vanilla 대비 두 primary 구간은 모두0을 포함한다.

## 실제 완료된 TEST SS

두 original RGB zero-shot, V1, Top-1 R, 선택 SS의 **508개씩 모두 정상 종료**했다.
총2,540개 archive의 실제 byte hash, float32 predicted/GT latent, packed64³
occupancy와 TP/FP/FN을 검증했다(v574). 모든 모델의 source·GT·noise pairing이 같다.

| 고정 모델 | TEST508 macro voxel IoU |
|---|---:|
| Zero-shot, source RGB4 | 0.400929 |
| Zero-shot, edited target RGB1 + source RGB3 | 0.443873 |
| Vanilla V1 | 0.469682 |
| Top-1 R | 0.477200 |
| Fixed SS | 0.477227 |

**이는 SS occupancy 진단이며 최종 geometry/appearance 결론이 아니다.** CD/F-Score는
실제 Shape cascade의 생성 surface에서, source LPIPS/DreamSim/CLIP과 target
PSNR/SSIM/M-IoU는 같은 실제 rendering에서 평가한다. Global FID/KID도 남아 있다.
위 진단값으로 TEST 이후 모델을 다시 선정하지 않는다. Native `quick_metrics`는
global micro이고 이 표는 asset macro다. v574는 두 aggregation을 별도로 검증·보존했다.

## 실제 cascade와 중간 파일 정리 검증

다섯 고정 pipeline ×508 asset =2,540 case를 GPU0/1/2/3/6/7에서 처리한다.
Shape512/1024는 원래 multi-view checkpoint를 유지한다. 두 zero-shot은 원래 PBR를,
V1은 vanilla1900 PBR를, R/선택 모델은 Top-1 R1900 PBR를 사용한다. 모든 모델은
공통 M1 geometry의 원래 material reference와 target8/source9/10으로 평가한다.

초기 `v565` 실행은 한 asset의 다섯 model에서 geometry·PBR·render·score를 완료한
뒤 raw-cache 정리 설정의 `ssh_argv` 누락으로 종료됐다. 모든 실제 child를 drain하고
keeper 종료·실패 결과·완료 및 partial artifact를 보존했다. NN/OOM 실패가 아니다.
동결된 수치 source를 바꾸지 않고 실제 SSH/remote-root 설정을 보완했다.

그 완료 asset에서 **PBR3 + geometry5의8개 retirement**를 원래 public 함수로 실행해
16개 local raw duplicate의 NAS SHA/size를 즉시 재검증하고 정리했다(v569).
Canonical raw, guide, scored image, checkpoint는 보존했다. 종료가 증명된 기존
campaign의 남은 미래 용량 예약만 canonical cancellation 근거로 해제했다.

원래 coordinator는 실패 namespace를 자동 resume하지 않으므로, `v570` 새 namespace에서
같은 모델·데이터·seed·수치 protocol로 전체 cascade를 실행한다. SS508 결과는 재사용한다.
완료된 초기 cascade 소수도 같은 고정 조건에서 replay하며 이전 결과를 보존한다.
새 실행은 첫 asset의5개 pipeline과8개 retirement를 실제 통과하고 다음 asset으로
넘어갔다. 최대6개의 whole-asset group, case당4GiB credit,16GiB workspace를 유지한다.

Reader-ledger 검증 중 scheduler가 새 case를 append하지 않도록 별도 CPU transaction
guard(v573)를 둔다. 짧은 정리 transaction 동안 CPU coordinator만 멈췄다가 반드시
재개하며, GPU/NN worker에는 signal을 보내지 않는다. 모델·generation·render·metric·
retirement의 기존 source byte와 numerical 조건은 그대로다. 이는 source/reader 검증을
완화하는 조치가 아니다. Guard의 실제 PID·journal과 coordinator identity를 추적한다.

## 연결된 후속 산출물

1. `v570` cascade/retirement 전체 성공과 실제 child release 이후, official perceptual5개와
   Inception6개의 GPU 작업 및10metric collector를 실행한다. Source-cluster134,
   TEST8C=16 primary endpoint family,1524 image/model의 원래 protocol을 유지한다.
2. `v571`은 실제 closed10metric 결과를 확인한 뒤 원래 frozen Main/qualitative exporter를
   실행한다. 사전 v355 metadata example·conditional lower-median 정책과 실제 scored
   pixel을 그대로 사용한다. 현재 최종 표/그림은 아직 생성되지 않았다.
3. `v572`는 실제 export 뒤 Main50개 numeric cell,16개 adjusted interval 및 모든
   qualitative panel을 W&B에 게시하고 원격 table/pixel을 다시 검증한다.
4. 실제 결과에 맞춰 Main/Ablation/Mechanism/Supplement와 최종 Results/Discussion/
   Abstract/Conclusion을 완성하고 최종 PDF를 compile·시각 검수한다. 현재 검수된
   v56019쪽/24인용 PDF는 개발 원고이며 최종 논문이 아니다.

현재 실제 case의 outer wall time은 대략 zero-shot170–178초, 학습 pipeline226–235초다.
초기 표본 기준 GPU6개의 순수 case budget은 약24시간이며 정리·변동 시간을 포함한
cascade 예상은 약24–30시간이다. Official metric/표·그림 생성과 최종 writing·검수는
그 후 추가3–6시간을 예상한다. 초기 소수 asset의 추정이며 누적 처리량으로 갱신한다.

실제 근거: `actual-complete-five-TEST508-SS-byte-and-occupancy-audit-v574`,
`actual-TEST508-retirement-proof-and-dead-credit-cancellation-v569`,
`actual-selected-five-model-TEST508-cascade-v570`,
`actual-selected508-cascade-and-metrics-activation-v570`,
`actual-TEST508-main-and-qualitative-export-activation-v571`,
`actual-final-TEST508-WandB-activation-v572`.
