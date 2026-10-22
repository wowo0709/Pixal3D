# PBR vanilla 세 번째 epoch 실제 검증

원본 multi-view PBR16850에서 fresh optimizer/LoRA continuation으로 시작한
vanilla가1140/1900 update와 세 번째 epoch 저장·고정4 렌더 검증을 완료했다.
물리GPU0~3, world4/global24/micro6/accumulation1을 유지했고 검증 후 학습이
재개됐다. Native parent238065/birth1792670094.76의 실제 생존을 확인했다.

| Update | Full-frame PSNR ↑ | Full-frame SSIM ↑ | Full-frame LPIPS ↓ | Foreground crop LPIPS ↓ |
|---|---:|---:|---:|---:|
| 0 | 26.0573 | .991635 | .010105 | .095126 |
| 380 | 26.3444 | .992008 | .009688 | .092427 |
| 760 | 26.6023 | .992322 | .009273 | .090078 |
| 1140 | **26.7225** | **.992467** | **.008977** | **.088230** |

0→1140의 PSNR 차이는+.6652dB다. Per-asset PSNR 차이는
+.3084/ +1.7075/ +.0005/ −.0075dB로, 실제 개선은 주로 두 자산에 있다.
Crop LPIPS도 첫 두 자산에서−.018885/−.008751이며 나머지 둘은 거의 같다.
따라서 전체 데이터의 일반적인 개선이나 새로운 모듈의 효과를 이 네 자산으로
확정하지 않는다. 실제 GT/Pred/Diff case001 이미지를 확인했으며, 형상은 GT
Shape 조건으로 같고 material/shading 차이가 남아 있다.

네 epoch 시점의 모든4개 case에서 RGB·conditioning hash·noise·GT PBR latent·
GT Shape condition·camera·sample/source cluster가 같음을 대조했다.
GT-shape-conditioned PBR stage 평가이며 source-preservation/full-cascade 평가는 아니다.
Alpha IoU는 이 설정에서 형상 생성 성능의 근거가 될 수 없다.

실제1140 NN/Adam/config/descriptor의 file4 archive receipt를 확인했다.
새 snapshot은 실제 PID/birth와 그 process의 실제 plan/output/mode를 검증하는
parameterised helper로 만들었다. 기존760 snapshot을 문자열 치환하거나 덮어쓰지 않았다.
새 evidence27개3,254,198 bytes를 실제 file4 SHA/size와 대조했다.

검증 직후 네 rank 모두1141 update를 완료했고, 최근10 update의 compute 중앙값은
약13.56초다. Rank0 peak allocated43.6595GiB/reserved46.6152GiB,
나머지 rank peak allocated43.42GiB 이하/reserved48.4609GiB다.
현재 프로세스는 정상 학습을 이어가며 새로운 OOM은 관찰되지 않았다.

Canonical 근거는 PBR-native-science-runtime-v1 아래
`vanilla-epoch1140-actual-evidence-v1/{audit.json,validation-prefix.jsonl,file4-closure.json}`과
`science-v1/vanilla/native/checkpoint-archives/step0001140/file4-closure.json`이다.

다음1520/1900 실제 저장·검증, vanilla 완료 후 native SS480 소비 경로 진단,
나머지 PBR source 세 모델, isolation/full-cascade 개발 평가, 독립 학습 seed,
개발 결과에 근거한 구조 선택과 최종508/10 metric은 아직 남아 있다.
이번 결과로 전체 목표 완료나 full 모델 채택을 선언하지 않는다.
