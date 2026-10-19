# Ray-wise matching 분석을 위한 6개 arm

Retargeting adapter와 source-camera별 ray-group query를 모든 arm에 유지한다. 기존 No-score의 정의는 matched Top-1 feature 하나이며 uniform Top-K로 바꾸지 않는다. 추가하는 `ray-no-score-prior`는 별도 진단 arm이다.

| Arm | 조건 | 확인할 효과 |
| --- | --- | --- |
| ray-top1 | Semantic Top-1 + R | No-score 기준선 |
| ray-weighted | Semantic Top64 + frozen score 가중합 + R | Top-K와 soft weighting의 결합 효과 |
| ray-full | 동일 Top64/D1/score + 학습형 utility attention + R | 가중합 대비 학습형 attention 추가 효과 |
| ray-no-score-prior | Full에서 scalar score prior만 제거 | 같은 Top64/D1/utility/R에서 score prior의 기여 |
| ray-native-spatial | 원래 native spatial 3개 후보 + attention + R | Semantic correspondence 경로의 전체 효과 |
| ray-no-decoder | D1 block 물리적 제거 + 나머지 Full | 현재 고정 score head에서 D1 제거의 영향 |

Full과 추가 arm 사이의 유일한 trainer 설정 차이는 `deform_conditioner.task_attention.score_prior=False`이다. Full의 RAW64 후보, contextual decoder, 학습형 utility 5개 parameter tensor, R의 입력 2048차원, native projection, 초기 SS4210 base+LoRA를 보존한다. No-DQ는 모든 arm에서 추가 채널이 물리적으로 없다.

공통 예산은 DeformNeXt world-v4 train9149/val508, seed101, raw480/5에폭, global96/DDP2/micro48/acc1이다. 매96 step 고정32개 검증과 W&B scalar/PNG 및 실제 file4 checkpoint 검증, 최종 고정128개/93개 source cluster ×3개 생성 noise, 저장 모델 RGB/BF16/CFG 검사, correspondence value-address 진단을 연결한다. 생성 noise 3개는 학습 seed 반복이 아니다. 개선 후보는 별도 학습 seed와 reserved 평가가 필요하다.

Native-spatial의 3개 후보와 semantic의64개 후보 차이가 있으므로 semantic 품질만의 효과로 해석하지 않는다. 동일 candidate ID/score/weight/support를 유지한 value-address 진단을 함께 사용한다. No-decoder는 D1에서 선택한 scalar head를 고정한 조건부 ablation이며, 최적으로 재학습한 D0 대비 우위라는 의미가 아니다.

실행 중인 5개 arm의 작업 경로와13개 numerical source를 보존했다. `wowo0709/feature-retargeting` 개발 경로를 별도로 옮겼고, 새 arm의12개 모델·데이터·trainer source는 기존 실행과 byte 단위로 같다. 바뀐 numerical component는 arm/recipe validation registry 하나다. 최종6개 비교는 실제 arm별 source binding을 대조한다.

CPU 검사26개를 통과했다. Score prior를 끈 arm에서는 scalar score만 바꾸어도 weight와 map delta가 그대로이며, decoder context를 바꾸면 learned attention weight가 바뀐다. 이는 구현 경로 검증이며 SS 성능 개선의 증거가 아니다. 실제 GPU gate와 큐 설치 결과는 별도 실행 기록에 남긴다.
