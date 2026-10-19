# Ray query로 변경한 뒤 기존 R 이득이 유지되는가

2026-10-19 UTC. 현재 ray 구조가 기존 Top-1 + retargeting 기준선을 넘는 성능은 아직 확인되지 않았다. 기존 R의 두 seed 양성 결과는 유효하며, 이를 새 query/carrier로 옮긴 효과는 별도 검증해야 한다. Native-spatial은 raw480 학습과 최종 가중치/conditioning 검사를 마쳤고 Top-1 + R은 학습 중이다. 최종 결정적128×3 비교는 아직 진행 중이다.

## 이전 R 결과와 현재 관측

고정 raw480, 128개/93 source cluster, 생성 noise3개에서 기존 Top-1 대비 R 추가 효과는 학습 seed101에서 +0.8228%p, seed202에서 +0.8806%p였다. 두 조건 모두 CI가 양수였다. [두 seed 감사](2026-10-19-retargeter-two-seed-effect-ko.md)를 현재 ray 모델의 R 효과로 확대하지 않는다.

현재 동일32개/noise0, 4에폭 비교는 다음과 같다.

| 조건 | IoU |
| --- | ---: |
| Zero-shot, source RGB4장 | 0.394997 |
| Zero-shot, edited RGB1장 + source RGB3장 | 0.426715 |
| V1, 4에폭 | 0.434300 |
| 기존 Top-1 + R, 4에폭 | 0.449885 |
| Ray 가중합, 4에폭 | 0.438088 |
| Ray Full, 4에폭 | 0.438685 |

Full−가중합은 +0.05974%p, source-cluster95% CI[−0.02869,+0.16083]%p다. 기존 R과 현재 ray 구조의 query/carrier가 달라 이 차이를 semantic matching 하나의 효과로 읽지 않는다. 이 표는 개발32개·학습 seed1개·생성 noise1개의 중간 관측이다.

이후 5에폭 raw480의 같은32개 epoch 기록에서는 Full0.436341, 가중합0.437597, V1 0.434837, 기존 Top-1+R0.454022였다. Full−가중합은 −0.12558%p, CI[−0.40892,+0.13393]%p로 추가 attention의 이득을 확인하지 못했다. Full−V1은 +0.15037%p, CI[−0.63357,+0.85369]%p이며, 기존 R보다 −1.76815%p 낮았다. 이 수치는 아래 결정적 평가로 새로 생성한 결과와 섞지 않는다. 원본은 `epoch-pair-comparisons-v1/ray-full-vs-ray-weighted/epoch05.json`이다.

Native-spatial의 같은 raw480/32개 epoch IoU는0.443367이었다. Full−native-spatial은 −0.70266%p, source-cluster95% CI[−1.70032,+0.17403]%p다. 평균은 native-spatial 쪽이 높지만 이 한 학습 seed/생성 noise의 CI는0을 포함하므로 semantic retrieval이 성능을 떨어뜨린다는 확정 판정도 하지 않는다. Epoch3/4/5의 대응 비교는 모두 실제 W&B·file4 epoch audit에서 읽었으며 `epoch-pair-comparisons-v1/ray-full-vs-ray-native-spatial/completed-epoch03to05-v1/file4-closure.json`으로 실제 file4 SHA/크기를 확인했다. Native-spatial은 후보가3개, semantic은64개인 원래 설계 차이를 유지한다.

## 코드에서 확인한 변화

| 항목 | 두 seed 이득을 확인한 기존 R | 현재 ray 모델 |
| --- | --- | --- |
| Query 위치 | Edited-target 이미지의 patch | 각 source camera의32² ray group |
| Matching 입력 | 해당 target patch의 Omega FRAME | 같은 source-ray에 모인 voxel이 참조하는 target FRAME의 평균 |
| R의 target 입력 | 해당 target patch의 native DINO | 같은 source-ray에 모인 target native DINO의 평균 |
| R 출력의 전달 | Target-anchor patch map을 camera0로 ProjGrid 투영 | Source camera별 ray map3개를 각 원래 camera로 ProjGrid 투영한 뒤1/4씩 더함 |

R 네트워크의 입력 차원2048·hidden256·bounded residual·zero output 초기화는 유지하지만 R이 받는 정보와 보정 위치는 달라졌다. Ray grouping 산술과 native projection의 재현만으로 기존 양성 모듈 전체를 동일 조건으로 재현했다고 할 수 없다.

과거 `target_proxy_ray_summary`의 query는 target-only **predicted** SS latent와 occupancy 등을 source ray로 모은21채널 요약이었다. 현재 query는 RGB projected DINO/FRAME 평균이며 그 shape-state 표현을 복제한 것이 아니다. 해당 과거 local-ray query 학습은 terminal budget 전에 중단되어, 최종 learned-query 성공 기록은 없다. [과거 재감사](2026-10-19-ray-group-query-contract-audit-ko.md)의 원본 근거와 주장의 범위를 유지한다.

9월의 큰 ray transport 양성 결과에는 GT read/write correspondence를 사용한 oracle 조건이 있다. 이것은 정확한 주소의 source observation이 유용할 수 있다는 증거이며, inference-safe semantic Top-K + learned attention의 이득을 이미 입증한 결과가 아니다. [원본 Wave0/1 보고서](/root/dev/Pixal3D/.worktrees/anchor-hybrid-prestudy/docs/superpowers/reports_final2/2026-09-18-ss-gt-corr-proj-what0-content1-results-ko.md)의 recall32와 현재 micro-IoU를 같은 metric으로 비교하지 않는다.

## 확인된 작동과 남은 원인 가설

- Full/가중합의 초기32개 결과는 원래 SS4210의 같은 입력·noise·GT 결과와 일치했다. 실제 학습에서 R output weight와 LoRA는 갱신됐고 실제 SS loss gradient가 연결됐다. 초기 모델 손상이나 완전히 끊긴 R 경로라는 설명은 현재 증거와 맞지 않는다.
- Actual common32의 input-only ray support 진단에서 native bilinear 전달 가중치는 평균97.687%였다. 평균적으로 feature 대부분이 사라진다는 증거는 없지만 learned matching의 정확도나 인과적 성능 원인을 측정한 것은 아니다.
- 서로 다른 target patch의 평균이 ray의 state 정보를 약하게 만들 수 있다. 또한 patch 단위로 학습한 frozen D1/scalar head를 새 pooled-ray query에 적용한 입력 분포 변화가 retrieval/score를 약화시킬 수 있다. R의 보정 위치가 target-anchor에서 세 source-camera map으로 바뀐 영향도 가능하다. 세 설명 모두 원인 가설이며 아직 단독 효과를 측정하지 않았다.

## 실제 저장 모델의 R-off 진단

원래 evaluator의 `--r9-path-control adapter-off`를 사용해 raw480 가중합 checkpoint에서 R 출력만 제거했다. CPU 사전 검사에서 candidate ID·score·weight·support·fusion gate는 exact했고 R residual만0이 됐다. Hook 해제 후 원래 출력이 exact 복원됐으며 저장 가중치와 실제 실행13개 numerical source는 바뀌지 않았다. 새 학습은 없다.

| raw480, common32/noise0 | IoU |
| --- | ---: |
| R 사용 | 0.437559 |
| 같은 가중치에서 R 출력만 제거 | 0.432019 |

Normal−R-off는 +0.5540%p, source-cluster95% CI[−0.4325,+1.3333]%p다. Full의 같은 진단도 완료됐으며 normal0.436426 대 R-off0.433124, +0.33023%p, CI[−0.72043,+1.21585]%p였다. 두 진단 모두 실제 hook4회가 확인됐다. 현재 모델이 R을 전혀 사용하지 않는 관측은 아니지만, CI는0을 포함한다. 이는 이미 R과 함께 학습한 모델의 조건부 의존성이지 R 없이 처음부터 재학습한 ablation이나 일반화된 효과 증명은 아니다. 생성값·입력 contact sheet·비교·실제 file4 SHA 검증은 `ray-group-study-v1/diagnostics/ray-retarget-off-v1/{ray-weighted,ray-full}/`에 있다. 두 결과는 기존 계산 모드의 진단이며 새 결정적 최종 결과와 합산하지 않는다.

## 수치 재현성 검사에서 발견한 별도 문제

가중합의 첫 QA는 반복 계산한 `score/0.05`의2개 값이 절대1e-5를 넘는 차이로 실패했다. 실제 ray arithmetic을 GPU0에서12회 반복하면 후보 ID와 R 출력은 모두 같았고, score 최대 차이2.48e-5, weight5.96e-7, projected condition1.91e-6이었다. 별도 static-ray QA는 원래 scalar head score 단위에서 기존1e-5를 적용한다. 후보 ID exact와 weight/evidence/R/projection/RGB/BF16/CFG의 모든 기존 검사는 유지했고26개 CPU 검사와 실제 저장 가중합 GPU QA를 통과했다. 이 변경은 model forward를 바꾸지 않는다.

Full의 원래 QA에서는6,291,456개 evidence 원소 중36개가1e-5를 넘고 최대0.00097775 차이가 관측됐다. 또 가중합의 epoch32와 final128/noise0에는 GT/seed가 같은32개 중1개에서 TP+5/FP+44/FN−5의 차이가 있었다. 최종 aggregate32 IoU 차이는 약−0.00375%p다. 원래 strict replay gate가 이를 거부한 기록은 보존한다. 이 반복 차이를 큰 성능 차이의 원인으로 단정하지 않는다.

Full의 실제 저장 raw480 가중치에서 deterministic algorithms와 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 사용하자 원래 trace/condition/BF16/CFG 검사를 하나도 완화하지 않고 QA를 통과했다. 모든 반복 trace/condition 오차가0이었다. GPU2와3의 독립32개 생성은 asset 순서·GT·noise·TP/FP/FN·출력 PNG가 exact했고, 이어128개 생성의 공통32개도 원래 exact-count gate를 통과했다. Peak allocated는32개6.038GiB,128개6.060GiB였다. Full128/noise0 IoU0.432198은 새 계산 모드의 단일 평가이며 baseline 재평가 전의 성능 결론으로 사용하지 않는다.

구 계산 모드와 새 모드는 GT/source/noise가 모두 같지만 Full32 중19개의 생성 TP/FP/FN이 달랐다. 이를 원래 epoch 결과의 exact replay로 이름 붙이지 않는다. 새 검증은 **동일 저장 checkpoint의 독립 결정적32개 생성 대 결정적128개 생성**의 재현이다. 원래 학습·검사 실패·평가 파일은 모두 보존하고 새 `deterministic-evaluation-v1/study-v1/`에 최종 프로토콜을 분리한다. 여섯 ray arm과 V1·기존 Top1+R·두 zero-shot을 모두 같은 결정적 모드로128개×3 noise 평가하고, 각 모델에 독립32개 exact replay를 요구한다. Value-address 진단도 새32개 정상 결과와 같은 새128개 기준을 사용한다.

## 이어서 확인할 사항

기존6개 raw480 학습 예산과 입력을 유지한다. 같은 ray의 Top-1 + R을 기존 R과 비교해 query/carrier 변경의 손실을 먼저 확인하고, Full−가중합/score-prior-off/no-decoder/native-spatial로 모듈 효과를 분리한다. 추가 에폭만으로 이 입력·전달 경로 차이가 해결된다고 가정하지 않는다. 기존 runtime source와 두 stage의 numerical model/data/trainer 파일은 유지한다. 새 프로토콜의55개 runtime binding과 계획·QA·실제32→128 증거83파일은 file4 SHA/크기를 직접 검증했다. 관련 CPU 검사38개를 통과했으며, 전체 여섯 arm의 최종 비교는 아직 완료되지 않았다.
