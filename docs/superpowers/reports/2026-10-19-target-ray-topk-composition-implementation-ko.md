# Target-ray에서 semantic Top-K를 조합하는 구현과 검증 현황

목표는 정확한 Top-1 분류가 아니라 **유용한 correspondence를 Top-K에 확보하고, matching-aware attention이 이를 선택·조합해 SS 생성을 개선하는 것**이다. 2026-10-19 UTC 시점의 구현 및 소규모 진단 기록이며 성능 향상을 달성했다는 보고가 아니다.

## 현재 증거가 가리키는 문제

기존 source-camera pooled-ray Full480의 개발32개에서 관측 가능한 서로 다른 (source-camera ray, target patch) 관계50,509개를 평가했다. Frozen score의 exact Top-1 coverage는8.5688%, Top-64 후보 포함률은50.7731%였다. 같은 source view의 한 patch 이웃까지 허용하면 각각28.3237%,80.4807%다. 따라서 Top-1이 놓친 정보를 후보 집합에 보유하고 있다. 이 진단은 새 target-patch query의 측정값이나 SS 정확도가 아니다.

그러나 기존 learned attention은 scalar prior보다 정답 후보 집합의 평균 확률 질량을 늘리지 못했다. 같은128개×3 생성 noise에서 Full IoU0.433994, 고정 score 가중합0.434168, Top-1 0.434027로 추가 이득을 확인하지 못했다. 학습 seed는1개다. Source-ray query에서 수행한 auxiliary utility7epochs도 실제 SS32를 개선하지 못했다. 후보 포함률이나 보조 loss만으로 모델을 채택하지 않는다.

검증은 다음 세 단계를 연결한다.

| 단계 | 확인할 질문 | 지표 |
| --- | --- | --- |
| Semantic retrieval / matching decoder | 유효한 대응점이 후보에 들어오는가? | Top-K exact/near1 coverage, proposal miss, view별 coverage |
| Matching score / attention readout | 포함된 유효 후보를 선택·조합하는가? | positive-set probability mass, effective K, score prior 대비 변화 |
| Pixal3D 생성 | 그 조합이 실제 생성에 도움을 주는가? | 같은 입력·GT·noise의 SS IoU/F1, source-cluster paired CI |

단순히 모든 후보를 평균하는 방식은 목표가 아니다. Matching score는 prior로 사용하고 target-ray query·candidate feature·Omega context가 content utility를 학습한다. 최종 성공 조건은 생성 metric의 개선이다.

## 기존 양성 retargeting 경로를 유지한 새 구현

`AnchorRayCompositionConditioner`는 edited-target cam08 patch 하나를 query로 삼는다. 같은 target-anchor patch를 참조하는 voxel들은 같은 evidence를 받는다. 원래 `static-top1-c1`의 nearest gather, native condition, alpha validity, R2048 입력, fusion 및 residual 계수를 유지한다. 기존 source-camera3 ray에서 여러 target descriptor를 평균하던 query와 구분한다.

Evidence는 `e_top1 + lambda * (e_soft_topk - e_top1)`로 구성한다. Lambda는0으로 초기화하고 optimizer step 뒤[0,1]로 projection한다. 초기에는 기존 Top-1+R과 같고, 학습이 유익한 방향으로 gate를 열도록 한다. Gate0에서 utility gradient가0인 단계와 gate가 열린 뒤의 gradient를 분리해 검사했다. GT는 model input이나 query에 넣지 않는다.

외부 결과 root:
`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1`

완료한 실제 검증:

- `identity-v1`: 실제 cached RGB/camera 입력에서 원래 condition/R/BF16 및 source-disable 동작 exact.
- `SS32-initial-v1` / `initial-parity-v1`: 결정적 생성32개의 TP/FP/FN, GT/noise/source 및 입력·GT/Pred PNG가 기존 oldR480과 exact. 공통 IoU0.454022361676035는 이32개 초기화 검사 값이다.
- `SS-gradient-v1`: 실제 SS flow loss에서 gate gradient, gate 활성화 뒤 utility5의 gradient 및 임시 optimizer update 확인. NN/R state hash는 전후 exact. 과학적 SS 학습은 아니다.
- CPU 회귀8개는 query-domain 혼용·source overlap 거부 검사와 원래 양성 경로 보존 검사를 포함한다.

현재 기존 DDP와 동시 사용한 제한된30% allocator에서는 micro24만 통과했다(24.66GiB). Micro32/48/96의 해당 제한 내 OOM 기록도 보존한다. 이는 GPU를 단독 사용했을 때의 최대 batch가 아니다. Global96/accumulation4로 SS 학습을 끼워 넣지 않고, 기존 arm 종료 후 전용 GPU pair에서 micro48×2/accumulation1을 우선 검증한다.

## Target-patch query에 맞춘 auxiliary utility warm-up

이전 실패한 source-pooled-ray cache를 재사용하지 않는다. 같은 source-disjoint train512/calibration32 assets에서 **edited-target-patch-ray** 입력544개를 새로 만들었다. Train503개에 covered positive set이 있고 나머지9개도 selection 기록에 남긴다. DINO/Omega/semantic64 입력만 저장하며 GT geometry label은 별도 파일에 둔다.

544개 모두 현재 GT teacher fingerprint가 일치했고 cache에서 재계산한 실제 head 확률이 byte exact였다. 구축은83.96초, peak0.365GiB였다. 실제 file4의1,096개 파일/3,818,702,928bytes를 SHA256·크기로 검증했다(`patch-head-v1/cache-file4-closure.json`). 추가 raw 다운로드나 SS/R update는0이다.

`anchor_patch_head_training.py`는 query domain을 archive 내부까지 검사하며 이전 source-pooled-ray cache를 거부한다. Native query/context/candidate/prior만 head.forward에 전달한다. 별도 loss는 visible same-view near1 positive set의 합산 확률을 높인다. 후보에 정답이 없는 관계는 proposal miss로 계속 계수한다. 정확한 Top-1에만 맞추는 one-hot 분류가 아니다.

새 warm-up은 utility5만 고정7epochs, AdamW LR1e-4/weight-decay.01, seed101, 512query draws/asset로 진행한다. 실제 GPU profile을 통과한 query microbatch를 사용하고 accumulation은1이다. Epoch0 및 매 epoch calibration loss·exact/near1 probability mass·Top1/64 coverage를 W&B에 기록한다. Epoch 선택 없이 epoch7의 pure utility state5개만 내보낸다. R/SS는 이 보조 과정에서 로드하거나 갱신하지 않는다. 실행 상태와 결과는 외부 progress/completed 파일이 기준이다.

## SS 본실험 budget의 후속 구성

OldR480은 구조·초기화 동일성 검증용이다. 이 모델에서 SS를 더 학습한 뒤 기존 five-epoch V1과 비교하면 학습 budget이 달라진다. 본실험은 oldR epoch1/raw96의 동일 NN/R에서 시작하는 **R-only continuation 대조와 R+Top-K composer**를 함께 구성하고, 추가384updates로 총480updates/five nominal epochs를 맞춘다. 처음96updates는 두 arm의 공통 warm-up 이력이다.

새 utility5 및 lambda를 제외한 초기 NN/R, batch 순서, 남은 budget 및 optimizer/RNG 전환 조건을 두 arm에 동일하게 적용한다. Native misc checkpoint에는 RNG가 없으므로 과거 학습의 bit-exact resume를 주장하지 않는다. 새 stage의 공통 RNG reset과 optimizer state 처리 방식을 실행 recipe에 기록하고 실제 startup audit으로 확인한다. Auxiliary head 학습 비용은 별도 비용으로 명시한다.

현재6개 canonical arm의 source/config/480update budget과 최종128×3 평가는 변경하지 않는다. 새 target-ray composer의 성능 향상과 decoder·score·correspondence의 독립 효과는 이 구현에서 실제 SS 비교를 마친 뒤에 판단한다.
