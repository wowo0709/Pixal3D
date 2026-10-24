# 실제 정성 결과와 독립 seed 확인 준비

2026-10-24 16:40 UTC 전후, 필수 learned-candidate 평가가 진행되는 동안 완료한 작업이다. 새 학습 arm이나 TEST forward를 추가하지 않았다.

## 실제 render를 이용한 성공·실패 사례

완료된 Fixed+R DEV32 집계에서 다음 규칙으로 서로 다른4개 사례를 선택했다. V1/R/Fixed의 동일 GT reference와 camera8/9 full-frame 배열을 읽었으며 모델 간 GT pixel이 정확히 같음을 확인했다. 이미지를 합성하거나 crop·alignment·retouch하지 않았다.

| 선택 규칙 | Fixed − R surface F-Score | Fixed − R source LPIPS |
|---|---:|---:|
| F-Score 증가 최대 | +0.056790 | +0.000192 |
| F-Score 감소 최대 | −0.018212 | −0.000546 |
| Source LPIPS 증가 최대 | +0.008537 | +0.009736 |
| Source LPIPS 변화의 중앙값에 가까운 사례 | +0.024811 | +0.000076 |

Source LPIPS는 camera9/10 평균이며 F-Score는3D surface metric이다. 이들은 결과 조건으로 선택한 설명용 사례이고, 개선·악화의 발생 빈도나 새로운 population effect를 추정하지 않는다. 논문 전체의 평균과 CI는 기존32개 전체 결과를 사용한다.

첫 번째와 세 번째 사례의 source camera9를 GT/R/Fixed로 나란히 놓은 compact figure도 만들었다. Geometry가 좋아져도 appearance가 함께 좋아진다고 볼 수 없다는 기존 정량 결과를 실제 출력으로 보여준다. Appearance 차이의 원인을 이 그림만으로 특정하지 않는다. Final Ours나 zero-shot 비교와 혼동하지 않는다.

결과 root `unified-module-pipeline-v1` 아래:

- `actual-DEV32-qualitative-tradeoffs-v227`:4개 사례의 target/source·V1/R/Fixed PNG/PDF, 선택 규칙, 실제 입력 SHA256, 재현 스크립트.
- `paper-compact-DEV-tradeoffs-v230`: 논문용6-panel source-camera figure와 실제 배열/GT 일치 근거.
- `paper-draft-actual-qualitative-v231`: 위 figure를 실제 포함한8쪽 PDF, source snapshot, compiler log와 시각 검수 audit.

세 결과 묶음은 실제 파일서버4 SHA256/size 일치를 확인했다. PDF는 compiler returncode0, TeX warning 및 미해결 인용0이며, figure가 포함된6쪽의 패널·annotation·caption을 시각적으로 확인했다. 기존7쪽 revision은 보존했다. 최종 TEST와 최종 ablation 표는 여전히 미측정이다.

[W&B 정성 결과](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/dev32-qualitative-tradeoffs-v227)에4개 full-panel 이미지·compact figure·선택 규칙 table을 업로드했다. Sync 완료 후 remote run summary의5개 image key를 확인했다. 근거는 `actual-DEV32-qualitative-WB-v232/audit.json`이다.

## 독립 training seed202 준비

현재 실제 실행 중인 V1 seed202의 NN PID·birth·command를 확인하고, 시작 config/startup과 최초5개의 완결된 training-noise trace를 별도 snapshot으로 보존했다. Append 중인 trace 전체를 immutable 입력으로 취급하지 않았다. 이 snapshot은 실제 시작값과 최초5개 update의 근거이며480-update 학습 완료를 뜻하지 않는다.

E_FIXED_RAW/E_LEARNED_RAW의 proof/capacity/science config6개를 기존 public factory로 준비했다. Training seed와 명시적인 run/artifact/device metadata 외의 모델·데이터·optimizer·초기 checkpoint·budget은 원본101 config와 같음을 검증했다. 참조하는 seed202 R_raw_topk 파일은 recipe-only이며 그 모델이 실제 학습됐다고 기록하지 않았다.

**이6개 config는 queue에 넣지 않았다.** DEV 결과로 최종 후보를 정한 뒤 필요한 후보 하나만 실제 GPU proof·micro96 capacity·입력 정합 검사를 수행하고, 그 근거를 닫은 후 science를 시작한다. Seed101 proof를 seed202 proof로 재사용하지 않는다.

최초 metadata 준비 v228에서는 legacy wrapper의 `active.json`에 command 필드가 없어 중단됐다. NN 실패나 재시작은0이다. 실패 기록은 보존하고, v229에서는 원래 wrapper를 변경하지 않은 채 실제 psutil PID/birth/command 관측을 별도 snapshot으로 남겨 기존 public 검증기에 전달했다. 실제 live process와 command를 다시 대조해 prefix capture를 완료했다.

Canonical 근거는 `candidate-seed202-preparation-v229`의 owner observation, actual-V1-prefix와 `recipes-NOT-QUEUED/audit.json` 및 각 file4 closure다.

## 다음 판단

필수 learned cascade의 전체8개 metric과 voxel control의 최종 평가가 필요하다. 이 결과 이후 최종 후보·필요한 최종-context 대조·독립 확인을 결정한다. 현재 준비한 config나 정성 그림을 모델 채택·일반화·전체 목표 완료의 근거로 사용하지 않는다.
