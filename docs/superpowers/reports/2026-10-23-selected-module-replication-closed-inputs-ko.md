독립 training seed 결과 집계와 오래된 실행 조건 수정 — 2026-10-23

12개 핵심 arm 중 실제 측정한 후보와 강한 두 baseline(V1, Top1+R)을 비교하기 위한 closed-input collector를 구현했다. 후보가 baseline 자체이면 모델을 중복하지 않는다. 새 학습 arm이나 GPU 작업은 추가하지 않았다. 기존 SS/PBR 학습 수식과 config는 수정하지 않았다.

`collect_module_seed_confirmation --plan`은 두 training seed 101/202의 실제 완료된 입력을 요구한다. 각 모델의 native480 checkpoint bundle, NN/source/config 파일의 파일서버4 hash, cold startup, 480개 TRAIN96/noise/timestep/global trace, 실제 종료 기록, DEV128의 서로 다른 generation noise 세 개를 읽는다. 평가가 주장하는 NN/source/config hash가 현재의 실제 파일과 같은지도 검사한다. 독립 seed에서 값이나 JSON 타입이 바뀐 학습 조건은 기존 recipe validator와 reducer가 거부한다.

Frozen base 검사는 CPU에서 mmap으로 checkpoint tensor data만 비교한다. 모델 생성·신경망 forward·CPU 학습은 하지 않는다. 원래 multi-view SS4210과 모든 최종 NN의 parameter key·shape·dtype가 같고, 360개 LoRA tensor를 제외한 base tensor가 정확히 같아야 한다. 학습한 LoRA 값은 바뀔 수 있다.

이전 terminal 기록에는 별도 session ID가 없는 경우가 있다. Child PID가 사라졌다는 사실만으로 그 PID를 session ID라고 추정하지 않는다. Collector는 실제 PID/birth가 종료됐는지 확인하고, 명시적인 session ID가 기록됐을 때만 해당 session의 부재를 검사한다. Producer의 drain 기록과 현재 OS가 독립적으로 확인한 범위도 구분해서 저장한다. 이 구분을 반영한 실제 검증은 `preflight-v2`이며, 이전 `preflight-v1`은 역사 기록으로만 남겼다.

새 collector·기존 통계/recipe/population 검증을 합친 CPU 테스트 172개가 통과했다. 실제 완료된 seed101 V1, Top1+R, Full 후보의 세 NN과 아홉 DEV128 평가도 새 collector로 읽어 검증했다. 실제 frozen tensor, 480-step trace, startup RNG와 평가의 asset/GT/noise 정합이 맞았다. 재확인한 mean-asset IoU128×3는 V1 0.4673550, Top1+R 0.4726002, Full 0.4720782다. 이 값은 새 품질 향상이 아니라 기존 결과의 파일·실행 provenance를 검증한 값이다. Seed202의 결과를 읽거나 합성하지 않았고, 독립 seed 확인은 아직 완료되지 않았다.

의존 관계 감사에서 또 하나의 실제 실행 문제를 발견했다. 현재 pending인 V1/Top1+R seed202 science 작업 두 개가 보류된 이전 `matched-fullcascade-v1/eight-metric-summary-v1`의 완료 receipt를 여전히 기다리고 있었다. 이 receipt는 현재 핵심 실험의 producer가 아니므로, 두 작업의 `requires`에서 이 조건을 현재 raw PBR1900 작업의 완료 receipt로 교체했다. 현재 raw PBR 작업과 GPU 자원 순서, 실제 seed202 proof·capacity·두 모델 engineering admission은 그대로 요구한다.

두 baseline은 사전에 정한 독립 재현 실험이다. 이 둘의 학습을 실행하기 위해 보류된 이전 cascade를 재개하거나 adaptive candidate 선정 결과를 기다릴 필요는 없다. 추가 후보의 seed202 학습은 실제 현재 DEV 결과와 그 후보의 선택 근거가 있어야 한다. 이를 최종 TEST508 허용이나 full model 선정으로 해석하지 않는다.

실제 수정 범위는 두 pending descriptor의 `requires`뿐이다. Command, source worker plan, NN/config 파일, optimizer, 초기 checkpoint, 학습 데이터, batch96/micro96/accumulation1, evaluation, priority, output 경로, 다른 모든 descriptor와 queue-state는 변경하지 않았다. 실제 config의 독립 seed recipe 검증을 다시 수행했고, 변경 전후 descriptor와 적용 기록을 파일서버4에 보존했다.

실제 증거 경로는 기존 canonical experiment 아래 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/selected-SS-closed-replication-v175/preflight-v2`와 `seed202-obsolete-gate-repair-v175`다. Collector는 최종 결과 집계 인터페이스를 완성한 것이며, 선택된 새 후보의 실제 seed202 launch와 집계 plan을 만드는 activation/factory, DEV 선택과 최종 freeze는 여전히 남아 있다.

18:42 UTC에 PBR feature가 1900 optimizer update와 마지막 validation을 완료했다. GT target Shape를 고정한 4개 빠른 진단에서 PSNR은 26.82601, crop LPIPS는 0.086537이었다. V1은 26.87652/0.086454, Top1+R은 26.86011/0.085186이다. Feature 추가의 최종 개선을 이 진단에서 확인하지 못했다. Target crop LPIPS를 source preservation으로 해석하지 않으며, 최종 생성 geometry32에서의 source/target 지표가 필요하다.

같은 시각 SS Top1-V가 480 update와 마지막 validation을 완료했다. 동일 DEV32 한 noise의 mean-asset IoU는 V1 0.4732792, Top1+R 0.4769671, Full 0.4782431, Top1-V 0.4790365다. Top1-V의 이 패널에서는 긍정적인 경향이 있지만, 다른 모집단인 최종 DEV128×3 결과와 섞어서 순위를 매기지 않는다. Fixed Top-K와 나머지 controls, 독립 seed, 실제 cascade 비교는 진행 중이다.

18:44 UTC에는 PBR feature의 파일서버4 최종 보존이 끝났고, GPU0–3에서 C_RAW/C_SPATIAL/C_D1_COS/C_DECODER_NO_R의 실제 proof가 완료돼 capacity 단계가 병렬 실행되는 것을 확인했다. GPU6·7에서는 Top1-V의 seal 및 예정된 최종 평가와 Fixed Top-K, 후속 design control이 기존 큐 순서로 진행된다. 검증용 collector 구현을 위해 진행 중인 NN을 재시작하거나 배치를 줄이지 않았다.
