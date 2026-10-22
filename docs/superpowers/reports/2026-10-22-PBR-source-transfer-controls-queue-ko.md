# PBR1900 source 전달 진단의 실제 연결

Top-1+R PBR의 첫 에폭 고정4개 평가에서 V1 대비 pooled PSNR이
0.265673dB 낮았고, Bed 한 자산의 하락이 컸다. 초기 step0의 실제 출력은
V1과 같았다. 이 모델에는 추가 QKV feature attention이 없으므로 이 하락을
QKV의 문제로 설명하지 않는다. 네 샘플의 첫 에폭 결과만으로 원인을 확정하거나
현재 학습 설정을 바꾸지 않는다.

원래 Top-1 PBR의 실제 1900 update 완료 snapshot 이후에 실행할 진단 일곱 개를
queue에 추가했다. GPU0·1에서 Normal을 독립 load하고 실제 입력·가중치·prediction·
GT/Pred pixel·metric·conditioning 소비가 정확히 같은지 CPU에서 먼저 검사한다.
통과한 뒤 같은 가중치에서 다음 세 개입을 실행한다.

- R 보정의 최종 전달만 제거한다.
- 직접 Top-1 feature shift의 최종 전달만 제거한다.
- 두 source contribution의 전달을 모두 제거해 native projected conditioning을 복원한다.

마지막 CPU 작업이 동일 입력의 Normal 대비 차이와 GT/Pred/Diff를 집계하고
`youngwoojeon-unist`의 W&B에 기록한다. Matching 후보·score·Top-1 carrier·계산된
R·source-use·gate 및 학습 가중치는 유지한다. 직접 Top-1 shift를 제거해도 R의
입력에는 기존 Top-1 정보가 남는다. 따라서 재학습한 R-only나 NoCorr/NoScore와
동일한 비교가 아니다. 이는 고정4개·GT Shape 조건의 전달 경로 진단이며,
source preservation이나 full-cascade 성능 평가를 대체하지 않는다.

현재 학습의 수치 설정을 바꾸지 않으며, 진단은 별도의 결정적 실행 조건을 사용한다.
원래 epoch 평가와의 prediction 일치 여부를 따로 기록하고, 세 개입의 차이는
새 Normal을 기준으로 계산한다. 두 Normal이 다르면 개입 세 개를 시작하지 않는다.
실패한 실행과 실제 출력은 보존한다.

## 실제 검증 범위

CPU 검사27개, public/worker CLI help와 구문 검사를 통과했다. 실제 공개된
380 checkpoint의 metadata 검사에서 NN1060 tensor·LoRA360 및 physical
PBR Top-1 schema와 R 입력3072 channel을 확인했다. 이 검사는 GPU forward나
미래1900 checkpoint 실행 검증이 아니다. 새 작업은 nonfinal380 weight에 의존하지 않는다.

준비·queue·적용 기록을 실제 file4 SHA256/size로 각각 검증했다.

| 기록 | 파일 수 | Bytes |
|---|---:|---:|
| Prelaunch | 15 | 219,658 |
| Queue | 2 | 7,792,776 |
| Applied | 1 | 968 |

Queue는3540개에서3547개로 늘었고, 기존3540개 descriptor는 이전 canonical
queue snapshot과 모두 같다. 전체 dependency DAG를 검사했다. Queue-state는
수정하지 않았으며, dispatcher가 추가된 일곱 작업을 모두 실제 pending으로 읽었다.
이 기록은 미래 GPU 결과의 완료를 의미하지 않는다.

Canonical 경로는 `PBR-native-science-runtime-v1/`
`Top1-native1900-source-transfer-controls-fixed4-v1/`이다.
소스는 `wowo0709/feature-retargeting`의8dc0106·5e090c6에 있다.

실제 NN355900·epoch observer360656·dispatcher2134868·file4 mirror3989158의
생존을 확인했다. 22:01 UTC의 학습은652/1900, world4/global24/micro6/accumulation1이며
물리 GPU0–3 모두 사용 중이다. 기존 자동 observer는760/1140/1520을 비교한다.
현재 학습 종료 뒤 역사적 SS 세 arm을 먼저 실행하는 순서는 유지한다.

두 후속 PBR arm의 학습·isolation128·공식 렌더 metric, SS/Shape/PBR cascade,
독립 학습 seed 확인, 실제 DEV 근거에 따른 모델 선택과 필요시 선택 조합 검증,
최종508개·10 metric은 계속 남아 있다. Ray-wise QKV의 필수성이나 semantic
Top-K의 추가 생성 성능 개선을 이번 연결만으로 주장하지 않는다.
