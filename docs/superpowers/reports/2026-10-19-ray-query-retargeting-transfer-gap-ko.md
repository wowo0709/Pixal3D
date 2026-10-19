# Ray query로 변경한 뒤 기존 R 이득이 유지되는가

2026-10-19 UTC. 현재 ray 구조가 기존 Top-1 + retargeting 기준선을 넘는 성능은 아직 확인되지 않았다. 기존 R의 두 seed 양성 결과는 유효하며, 이를 새 query/carrier로 옮긴 효과는 별도 검증해야 한다. 같은 ray의 Top-1 + R은 아직 학습 전이므로 현재 Full/가중합만으로 원인을 하나로 특정할 수 없다.

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

Normal−R-off는 +0.5540%p, source-cluster95% CI[−0.4325,+1.3333]%p다. 현재 모델이 R을 전혀 사용하지 않는 관측은 아니지만, CI는0을 포함한다. 이는 이미 R과 함께 학습한 모델의 조건부 의존성이지 R 없이 처음부터 재학습한 ablation이나 일반화된 효과 증명은 아니다. 생성값·입력 contact sheet·비교·실제 file4 SHA 검증은 `ray-group-study-v1/diagnostics/ray-retarget-off-v1/ray-weighted/`에 있다. Full의 같은 진단도 큐에 있다.

## 수치 재현성 검사에서 발견한 별도 문제

가중합의 첫 QA는 반복 계산한 `score/0.05`의2개 값이 절대1e-5를 넘는 차이로 실패했다. 실제 ray arithmetic을 GPU0에서12회 반복하면 후보 ID와 R 출력은 모두 같았고, score 최대 차이2.48e-5, weight5.96e-7, projected condition1.91e-6이었다. 별도 static-ray QA는 원래 scalar head score 단위에서 기존1e-5를 적용한다. 후보 ID exact와 weight/evidence/R/projection/RGB/BF16/CFG의 모든 기존 검사는 유지했고26개 CPU 검사와 실제 저장 가중합 GPU QA를 통과했다. 이 변경은 model forward를 바꾸지 않는다.

Full의 원래 QA에서는6,291,456개 evidence 원소 중36개가1e-5를 넘고 최대0.00097775 차이가 관측됐다. 또 가중합의 epoch32와 final128/noise0에는 GT/seed가 같은32개 중1개에서 TP+5/FP+44/FN−5의 차이가 있었다. 최종 aggregate32 IoU 차이는 약−0.00375%p다. 현재 strict replay gate는 이를 거부하므로 bit-exact 최종 재현이나 전체 QA 완료를 주장하지 않는다. 이 작은 반복 차이를 큰 성능 차이의 원인으로 단정하거나 검사 실패를 숨기지 않는다. Deterministic ray reduction 및 BF16 검사 범위를 별도로 확인해야 한다.

## 이어서 확인할 사항

기존6개 raw480 학습 예산과 입력을 유지한다. 같은 ray의 Top-1 + R을 기존 R과 비교해 query/carrier 변경의 손실을 먼저 확인하고, Full−가중합/score-prior-off/no-decoder/native-spatial로 모듈 효과를 분리한다. Full의 R-off, 두 모델의 수치 재현성 검사도 완료해야 한다. 추가 에폭만으로 이 입력·전달 경로 차이가 해결된다고 가정하지 않는다. 기존45개 runtime source와 두 stage의 numerical model/data/trainer 파일은 유지한다.
