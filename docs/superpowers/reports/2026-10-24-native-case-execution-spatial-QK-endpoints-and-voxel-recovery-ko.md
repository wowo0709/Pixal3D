# Native case 실행 연결, spatial·QK endpoint 결과와 voxel OOM 복구

기준: 2026-10-24 02:02 UTC. 구현 커밋 `985b352`, `590e074`. 전체 실험은 진행 중이며 최종 모델 선택과 TEST508은 수행하지 않았다.

## 새로 완료된 실제 결과

Spatial correspondence 대조군이 480 update와 DEV128 × 생성 noise3 평가를 완료했다. 아래는 같은128개 asset·93 source cluster·동일 GT/생성 noise를 사용한 asset별 평균 IoU다. 생성 noise3은 독립 학습 seed3이 아니다.

| 구조 | DEV128×3 IoU |
|---|---:|
| Vanilla V1 | 0.46735504 |
| Spatial correspondence + retargeting | 0.47124225 |
| Decoder Top-1 + retargeting | 0.47260025 |
| Top-1 V projection 대조군 | 0.47221778 |
| Fixed Top-K | 0.47274517 |
| 기존 Full | 0.47207818 |

Decoder Top-1+R − spatial+R는 +0.00135800이고, descriptive source-cluster CI95는 [−0.00150307, +0.00410354]다. 추가 이득은 이 평가에서 확정되지 않았다. 이 비교는 원래 spatial pathway를 proposal retrieval와 decoded correspondence로 바꾸는 결합 비교다. Raw semantic retrieval, decoder, scalar score의 각 효과를 분리한 primary contrast로 대신하지 않는다. 분리된 C_RAW·C_D1_COS 등의 실제 최종 결과가 필요하다.

각 모델의 실제 종료 terminal·file4 receipt·128개 sample/source/noise/GT 일치와 metric 범위를 확인했다. 새 control의 inline SS capture도 검증했다. 과거 V1·Top-1+R·Full metric evaluation에는 inline SS archive 항목이 없으므로, 해당 항목을 만들어 넣지 않고 실제 metric evaluation/terminal lineage를 그대로 보존했다. 최초 CPU 수집은 이 legacy field 차이로 중단됐고 별도 `actual-v2`에서 올바른 입력 계약으로 수집했다. NN이나 원본 evaluation은 수정하지 않았다.

[W&B: 실제 완료된 6모델 DEV128×3](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-unified-spatial-completed128x3-s101-v188)

QK-only, 즉 semantic 후보와 Top-K ranking을 유지하고 additive matching-score attention prior만 제거한 arm도 480 update를 완료했다. 실제 NN returncode0/session drain, frozen base·matcher 유지 및 optimizer state379개, checkpoint의 file4 closure를 확인했다. 같은 DEV32·생성 noise1의 endpoint는 다음과 같다.

| 구조 | 384 update | 480 update |
|---|---:|---:|
| 기존 Full | 0.47481289 | 0.47824306 |
| Additive prior 제거 | 0.47717633 | 0.47817716 |
| Prior 제거 − Full | +0.00236343 | **−0.00006590** |

4에폭에서 관측한 우세는 최종 epoch에서 유지되지 않았다. Attention prior가 최종 성능을 해친다는 결론을 내릴 수 없다. 이 arm의 DEV128×3와 독립 학습 seed 확인도 아직 완료되지 않았다. 이 DEV32 표와 위 DEV128×3 표의 절대 수치를 섞어 비교하지 않는다.

[W&B: 실제 480-update QK-only와 Full의 DEV32 비교](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-unified-QK-interim32-epoch5-s101-v188)

두 W&B run 모두 remote API로 실제 수치와 artifact 등록을 재확인했다. 최종 결과에 맞게 4에폭의 초기 해석을 제한하며, 모든 모듈이 필수이거나 learned attention의 추가 이득이 증명됐다고 주장하지 않는다.

## PBR 첫 epoch

Raw semantic Top-K+feature PBR는 380 update 검증을 마치고 394/1900으로 진행 중이다. 동일4개 anchor sample의 실제 RGB·GT Shape/PBR latent·camera·noise hash가 step0/380에서 일치했다.

| Metric | Step0 | Step380 |
|---|---:|---:|
| Foreground PSNR | 14.94645 | 14.90044 |
| Crop LPIPS | 0.095142 | 0.094741 |
| Crop SSIM | 0.949160 | 0.949242 |

변화가 작고 방향도 혼재하므로, 첫 epoch만으로 품질 개선을 판단하지 않는다. 이 검증은 GT Shape로 조건을 준 fixed4 anchor-stage PBR 렌더링이다. 생성된 SS→Shape→PBR 전체 pipeline이나 source9/10 preservation 평가가 아니다. Alpha IoU=1도 이 GT-shape 조건을 반영하므로 생성 geometry가 정확하다는 증거로 쓰지 않는다.

## 개별 voxel-query의 실제 OOM과 새 복구 체인

GPU7의 FULL_VOXEL_Q가 146 update를 완료한 뒤 다음 backward에서 OOM으로 종료됐다. 실제 NN PID831038은 returncode1로 종료했고 session drain 및 file4 failure closure를 확인했다. 로그는 총94.97 GiB 중 여유1.32 GiB에서 추가1.50 GiB 요청이 실패했다고 기록한다. 기존 3-update capacity가 이 후속 batch까지의 최대 사용량을 보장하지 못했다. 마지막 저장 checkpoint96과 실패 기록은 모두 보존했다.

기존 job/state/source를 바꾸지 않고 복구 작업9개를 append했다. 새 architecture arm은 추가하지 않았다. 새 proof/capacity/science 및 final128×3의 GPU 작업6개, admission/seal/전체12모델 요약의 CPU 작업3개다. 전부 기존 GPU6/7 dispatcher 아래 있으며 학습 복구는 GPU7에 배정했다.

복구 설정은 FFN 내부의 독립 asset chunk만24→12로 줄인다. Global batch96, GPU micro batch96, accumulation1, seed101, 480 update, optimizer, 학습 데이터와 matching/attention/retargeting 구조를 유지한다. 실제 통계와 lineage를 분리하기 위해 새 output과 W&B ID를 쓰고 seed101의 초기 상태에서 다시 시작한다. 96-step optimizer resume나 이미 실행한146 update를 새 run에 포함한 것으로 표시하지 않는다.

새 `voxel_ffn_recovery_proof.py`는 실제 frozen SS FFN 가중치를 읽기 전용으로 공유한 두 복사본에서 B25×4096, BF16, chunk24/12의 output과 input gradient를 비교한다. Relative-L2≤0.005, 원본 가중치·gradient 불변 및 RNG fork를 확인한 뒤 기존 native proof를 그대로 실행한다. 이후 micro96의 새3-update capacity와 원래 초기 native32/LoRA/source/RNG/noise gate를 모두 통과해야 science를 시작한다. 이 새 GPU 수치 검증과 capacity는 기준 시각 현재 대기 중이며 통과했다고 주장하지 않는다.

구식 실패 job은 FAILED로 남아 있다. 사용하지 않게 된 미래 checkpoint 증가분 reservation만 실제 종료·closure 확인 후 해제했다. 기존 partial checkpoint나 data는 삭제하지 않았다. 기존 정상 PBR, QK 학습이나 dispatcher를 재시작하지 않았다. 최종12모델 요약도 새 실제 voxel 결과를 읽는 후속 작업으로 연결했으며, 실패한 원래 job의 output을 완성된 것으로 대체하지 않았다.

## 최종 cascade 실행 코드

`run_selected_final_case.py`는 실제 closed DEV freeze·control freeze·manifest·cache QA·asset packet·모든508 SS evaluation을 읽어 자기 모델과 sample의 native plan을 만든다. 추가로 original render/shared/PBR template도 freeze의 `evaluation_protocol.final_case_templates`에 실제 hash로 묶어야 한다. Inline alias나 임의 model row로 대체하지 않는다.

각 phase는 새 closed runtime plan으로 `selected_final_phase.py`를 실행한다. 기존 native guide capture/replay 및 새 original-zero metadata adapter를 사용한다. Renderer와 shared scorer는 GPU membership literal만 명시적으로 바꾸고 기존 numerical body를 실행한다. 실제 child session이 완전히 종료된 뒤에만 terminal, phase 결과와 file4 closure를 발행한다. 실패 output은 보존하고 자동 재시도하지 않는다.

공유 SS를 쓰는 조합은 실제 완료된 geometry owner의 SS·Shape·guide·sample·freeze와 closed files 및 종료 child를 확인한다. GT surface arrays는 실제 geometry 생성이 끝난 뒤 공통20k/F@.02 reference에 join한다. 최종 audit은 기존 native PBR replay의 raw geometry 보존, guide 재사용, 같은 M1 original-material pixel과 camera 및 관측 image hash를 기록한다. 디스크 reservation은 case가 끝났다는 이유로 해제하지 않는다.

`selected_final_raw_retirement.py`는 실제 frozen composition에서 geometry의 모든 consumer를 도출한다. 예를 들어 V1 SS를 두 PBR 조합이 공유하면, 둘 다 실제 렌더링·scoring을 끝내기 전에는 owner raw geometry의 local duplicate도 정리 대상으로 허용하지 않는다. Consumer는 자신의 PBR raw만 소유하며 producer의 shared geometry를 소유하지 않는다. 현재 이 모듈은 admission contract다. 실제 fresh remote hash·현재 reader·worker 종료·inode 재확인과 unlink를 수행할 retirement executor 및 bounded campaign 연결은 아직 필요하다. 이번 작업에서 raw artifact를 삭제하지 않았다.

관련 CPU 검사 총420개가 통과했다. 새 execution/recovery 검사41개, raw consumer 검사74개와 앞선 selected 계약305개다. 실제 CPU child/grandchild의 session drain, 실패 returncode 보존, argv/callback 복구와 closed synthetic metadata/SS bytes를 검사했다. Neural weight를 대신한 fixture를 실제 GPU 검증으로 해석하지 않는다. 새 final case worker의 실제 GPU 실행은 아직 수행하지 않았다.

## 현재 자원과 남은 조건

02:02 UTC에 GPU0–3은 raw PBR DDP4, GPU6은 C_RAW final128 noise0, GPU7은 no-retargeting final128 noise1을 실행하고 있었다. QK-only의 실제480/closure는 완료됐고 이후 최종 평가3회는 기존 queue에서 진행한다. 파일서버의 canonical source325개와 기존 science config3개의 hash가 원래 pin과 일치한다.

전체12 arm 및 shared-evidence 두 후보의 최종 DEV, 실제 joint pipeline 비교, 선택 모델의 독립 SS/PBR seed 확인과 validation36 두 fresh-load가 남아 있다. 최종 모델과 평가 설정을 동결한 뒤 TEST508 full cascade, source LPIPS/DreamSim/CLIP, target PSNR/SSIM/M-IoU/CD/F-Score 및 global FID/KID까지 실행해야 한다. 새 case worker가 생겼다는 이유로 이 조건들이 완료된 것으로 처리하지 않는다.

실제 증거는 file4의 `PBR-native-science-runtime-v1/unified-module-pipeline-v1/` 아래 `voxel-query-FFN12-recovery-v188/`, `completed-spatial-and-five-controls-DEV128x3-v188/actual-v2/`, `completed-spatial-WandB-v188/`, `focused-QK-epoch5-SS32-v188/`, `paired-QK-epoch5-WandB-v188/`, `selected-final-execution-and-voxel-recovery-status-v188/`에 closure와 함께 보관했다.
