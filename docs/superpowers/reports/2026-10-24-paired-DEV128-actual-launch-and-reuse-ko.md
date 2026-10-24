# 동일 DEV128 파이프라인 평가의 실제 실행

18:16 UTC 후속: [실제 출력 metric 통합 검증](2026-10-24-DEV128-metric-real-output-integration-ko.md)에서 기존 R32의 역사적 binding schema 차이를 실제 provenance로 보완했다. 기존32개+새1개,297 view의 조립과 기존 metric2,784개 exact 재현을 통과했고 대기 중인 집계 체인을 v257로 교체했다. 실행 중인 NN·생성물·numerical metric code는 유지했다. 아래 v250 경로는 최초 연결 기록이며 현재 최종 예정 집계는 `metrics-v257/summary-v1/audit.json`이다.

2026-10-24 17:40 UTC 전후 실제 상태다. R·Fixed·Learned의 같은 panel 비교를 넓히기 위해, 이미 학습된 세 SS480 모델과 공통 Top-1 R PBR1900을 사용한다. 새 training arm, 새 deformation 데이터, TEST forward를 추가하지 않았다. 첫32개의 실제 cascade와 perceptual score는 재사용한다.

## 준비와 실제 검증

- R의 기존 final128은 수치만 저장돼 있었다. 같은 checkpoint·128 asset·seed·batch8·native sampler로 prediction을 저장했고 **128개 per-asset 기록 전체가 원래 평가와 일치**했다. 기존32개는 prediction/target latent와 packed occupancy 등 저장 배열까지 정확히 같았다. SS 재학습과 optimizer update는0이다.
- Fixed와 Learned는 기존128개의 저장 prediction을 그대로 사용한다. 세 모델 모두128개 전체의 asset/source/seed/GT/count를 검사했다. 처음32개만 검사하던 기존 validator로 나머지96개를 통과시키지 않는다.
- 추가96개에 필요한1,040개 실제 파일을 파일서버4의 SHA256/size와 대조했다. 로컬에 없던192개 GT RGBA 이미지만 file4에서 mirror로 복사했다. 새로운 외부 데이터 다운로드나 depth/matching cache 재구축은 없다. 입력/GT 이미지의512×512 RGBA header도 확인했다.
- Public contract와 통계 코드의 테스트17개가 통과했다. Suffix의 GT/seed/count 변경, 누락/중복, TEST/depth 입력, source-cluster/camera pairing 오류를 거부한다. 실제 저장된 세 모델의 native generation/replay 리포트도 별도로 검증했다.

## 세 모델의 실제 초기 사례

추가96개 중 predicted SS support가 큰 사례를 초기 확인용으로 선택했다. 이 순서는 capacity proxy이며 모든 사례의 최대 HR memory를 증명하는 것은 아니다. 세 모델의 원래 native Shape/PBR 생성과 rendering·shared material reference·scoring이 모두 완료됐다. 같은 target/source camera의 GT RGB와 alpha 배열이 세 모델에서 정확히 같음을 확인했다.

신규 검증 코드가 native PBR 리포트의 `PBR_arm`을 `mode`로 잘못 참조해 생성 후 metadata 검증이 실패했다. 원래 native NN 두 단계는 세 모델 모두 returncode0/session drain으로 성공했다. 실패 기록을 보존하고 실제 schema로 고친 뒤 **성공한 geometry/PBR 결과를 재사용해 rendering부터 이어갔다.** 생성 재실행은0이다. Source/frame/sampler/weights는 바꾸지 않았다.

또한 failure closure가 후속 작업의 성공 조건이 되지 않도록 수정했다. 새 worker는 실패 시 `failure-file4-closure.json`만 만들며, 공통 reference를 기다리는 작업은 owner의 실제 성공 `audit.json`과 closure를 함께 요구한다. 본 실행의 초기 metadata 준비에서 발견한 artifact receipt 위치와 render-template hash field 차이도 별도 실패 기록으로 보존했다. 기존 SS/voxel 학습의 실패나 재시작은0이다.

## 연결한 전체 실행 체인

1. 세 모델 초기 사례의 actual terminal·model binding·GT pixel identity 검사와 canonical pilot gate.
2. 기존32개씩을 재사용하고 추가96개씩 처리: 총384개 모델별 case 중288개가 추가 case다. 초기3개가 완료돼285개가 남은 상태에서 gate를 열었다.
3. 모든384개 성공 결과의 coverage와 source/target/camera/pixel 검증.
4. 새864개 view의 공식 GPU DreamSim/CLIP 평가. 기존288개 view는 원래 score·alias·모델 binding·실제 pixel SHA256을 확인해 재사용한다.
5. 총1,152개 view에서 같은128 asset·93 source cluster의8개 metric과 paired source-cluster 통계를 집계하고 W&B에 기록한다.

추가 case는 Fixed GPU2, R GPU3, Learned GPU6에 배정했다. 각 asset의 공통 original-material GT reference는 Fixed case가 생성하고 나머지 두 모델이 동일한 pixel을 사용한다. 이 reference 생성은 scoring 단계이며 prediction에는 GT geometry나 material을 입력하지 않는다. Original Shape5124210/Shape102416850와 native sampler·frame·normalization·20k surface points·F-Score threshold0.02를 유지한다. ICP나 output별 scale fitting은 없다.

GPU0/1의 독립 V1/R 학습과 GPU7의 voxel 대조는 유지했다. Metadata/통계는 CPU92/93·nice10을 사용하고 NN forward와 perceptual metric은 GPU에서만 실행한다.

## 실제 입력·실행 산출물

결과 root `unified-module-pipeline-v1/paired-DEV128-stage-evaluation-v240` 아래:

| 위치 | 실제 상태/용도 |
|---|---|
| `R128-capture-v1` | 실제128 prediction 저장, 원래128 기록 및 기존32 배열 exact replay 완료 |
| `input-data-v241` | 추가96 asset의 실제 file4/local byte identity와 calibrated RGBA 확인 완료 |
| `generation-reuse-proof-v248` | Metadata rejection 뒤 성공한 세 native generation을 재사용한 근거 |
| `pipeline-prelaunch-v249` | 세 모델의288개 추가 case plan과 기존32개씩의 재사용 index |
| `cases-v249` | 초기 사례3개 완료, 후속 추가 case 실행 중 |
| `pilot-pass-v249` | 실제 세 모델 성공·정확한 공통 GT 검사 통과; file4 closure 완료 |
| `execution-prelaunch-v250` | 잔여285개 case와 coverage/official/summary chain 연결 완료 |
| `metrics-v250` | 전체 case 완료 후 실제 평가/통계가 기록될 위치; 아직 결과 미측정 |

원래 metadata 실패와 native generation은 `pipeline-prelaunch-v242/v243`, `cases-v245`에 보존했다. 이들을 최종 성공 case로 세지 않는다. 성공한 source·generated output·원래 실패 이유와 후속 재사용 근거를 구분한다.

Local disk는 확인 시 약713GB 여유가 있었고, 큰 초기 사례의 저장 크기에 따른 잔여285개 proxy는 약280GB였다. 현재 checkpoint reservation과64GiB margin을 포함한 준비 검사를 통과했다. 이는 엄밀한 worst-case 크기 보장은 아니며 각 case도 원래 disk admission을 다시 확인한다.

## 해석과 아직 남은 작업

이 실행은 **training seed101·generation noise1회**의 validation128 확장이다. SS128×3와 같은 noise 평균이 아니며, DEV32 screening 후 확장한 탐색적 비교다. 아직128개 cascade 결과나 새로운 module gain을 확보했다고 쓰지 않는다. 기존5개 DEV32 조합과13개 SS endpoint 수치를 TEST/main table로 옮기지 않는다.

새로운 pipeline 조합이나 PBR arm 전체 조합은 늘리지 않는다. 기존 세 후보를 같은 panel에서 비교한 뒤 최종 구조와 부족한 matched ablation을 정한다. Original voxel 대조/family 통계, 필요한 독립 SS/PBR seed 확인, 최종 선택·protocol freeze, TEST508의10개 metric/FID/KID·정성 그림과 최종 writing은 계속 남아 있다.
