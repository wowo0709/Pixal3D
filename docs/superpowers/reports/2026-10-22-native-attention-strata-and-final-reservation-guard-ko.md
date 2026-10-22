# Native attention의 변형 유형별 결과와 최종 평가 GPU 예약 보완

2026-10-22 UTC. 기존 native SS480의 개발128개·93 source cluster·세 생성 noise
결과를 다시 분석했다. 새로운 모델 학습이나 test508 평가는 수행하지 않았다.

## 변형 유형별로 나눠도 Top-K의 추가 이득은 확정되지 않는다

기존 pair ID의 d1/d2/d3 label로 분류했다. 이 label을 변형 크기나 난이도의
순서라고 해석하지 않는다. 12개 실제 평가 JSON을 canonical receipt와 대조하고,
모든 arm/noise의 sample 순서·source cluster·GT voxel identity·생성 seed가 같은지
확인했다. 원래 네 모델의 전체128 mean-asset IoU/precision/recall/F1을 정확히 재현했다.

아래는 세 noise 평균 후 asset 평균한 IoU 차이이며 단위는 percentage point다.
R은 retargeting adapter, Feature는 feature attention 경로다.

| 그룹 | Asset 수 | Top-1+R − V1 | Feature+R − Top-1+R | Raw Top-K+Feature+R − Top-1+R |
|---|---:|---:|---:|---:|
| 전체 | 128 | +.5245 | −.2127 | −.0522 |
| d1 | 50 | +.4915 | −.2855 | +.0891 |
| d2 | 40 | +.5522 | −.0668 | +.0292 |
| d3 | 38 | +.5389 | −.2707 | −.3238 |

Raw Top-K 대 Top-1+R의 source-cluster bootstrap 95% CI는 d1에서
[−.3535,+.5437]pp, d2에서 [−.3209,+.3733]pp, d3에서
[−.7959,+.1198]pp다. 모두 0을 포함한다. 전체에서 raw Top-K가 더 좋은 자산과
나쁜 자산은 각각64개다. 단순히 전체 평균에 가려진 강한 유형별 Top-K 이득이
있다고 주장할 근거는 얻지 못했다.

d1에서 raw Top-K 대 Feature-only의 차이는 +.3745pp,
CI[+.0912,+.6636]pp였으나, 같은 d1의 Top-1+R을 넘는 이득은 불확실하다.
모든 label과 여섯 대조를 보고했고, 여러 유형/지표를 탐색한 보정 전 interval이므로
이 결과로 모델을 선택하거나 attention의 효과를 확정하지 않는다.

Bootstrap은 source cluster를 복원 추출하고 선택된 cluster의 모든 asset을 평균한다.
4000회, seed20261023이며 세 생성 noise는 독립 학습 seed가 아니다.
이 분석의 CI는 기존 최종 보고서의 seed와 다르므로, 원래 주 비교의 family 보정
구간이나 primary endpoint를 대체하지 않는다. Geometry 배열이나 새 NN forward는
읽거나 실행하지 않았다. 실제 변형량에 따른 분석과 native480 소비 경로 개입은 남아 있다.

Canonical 근거는 native-flow-cold3-v1 아래
`deformation-label-strata-dev128-CPU-v1/{audit.json,statistics.json,per-asset.json,file4-closure.json}`이다.
네 파일201,925 bytes를 실제 file4 SHA/size와 대조했다.

## 최종 평가의 GPU 예약 해제 조건 보완

기존 final508 supervisor는 실패 후 하위 작업을 기다리지만, 이 과정의 metadata
저장까지 실패하면 남은 하위 작업을 기다리지 못하는 경로가 있었다. 이미 SHA에
묶인 supervisor·case worker·모델·metric 코드는 수정하지 않았다.

새 guardian이 기존 supervisor를 독립 OS session으로 실행한다. Supervisor가
종료돼도 해당 session에 살아 있는 하위 프로세스가 있으면 네 GPU 예약을 유지한다.
진행 metadata 저장 실패도 대기를 중단시키지 않는다. 실제 OS 프로세스 검사에서
부모가 exit7로 끝난 뒤 남은 자식을 감지하고, simulated disk error 중에도 자식의
실제 종료까지 기다리는 동작을 확인했다. 이 검사는 CUDA 성능 검사가 아니다.

Pending factory 하나의 plan 경로와 requires만 canonical revision으로 바꿨다.
새 plan은 campaign executable, 추가 세 source SHA, source commit만 갱신한다.
기존 source pin, 모델·데이터·noise·metric·학습 recipe, 다른 job descriptor와
queue-state는 바꾸지 않았다. Revision prelaunch9개145,017 bytes와 적용 audit
1개1,167 bytes는 실제 file4 SHA/size 검증을 통과했다.

관련 CPU 검사11개 파일의 **202개 test가 4.45초에 통과**했고,
현재3523개 job의 dependency DAG에 누락이나 cycle이 없음을 확인했다.
전체 queue에는 보존 중인 과거 실패와 HELD 이력이 있으며 이를 성공으로 바꾸지 않았다.

## 남은 과학 실험

현재 PBR vanilla의 실제 parent238065/birth1792670094.76과 outer238032,
dispatcher2134868, mirror3989158을 확인했다. GPU0~3/global24/micro6/accumulation1로
1047/1900 update, 마지막 저장760 시점까지 진행했고 기존 학습을 재시작하지 않았다.
고정4개 GT-shape stage validation의 0/380/760 추이는 이전 epoch2 보고서에 기록했다.
이를 새로운 source-preservation 또는 full-cascade 성능 증거로 해석하지 않는다.

나머지 PBR source 세 arm, PBR isolation, native SS/PBR full-cascade 개발 비교,
독립 학습 seed202, 역사적 target-patch query 재현, 개발 결과에 근거한 구조 선택,
최종508개와 10 metric 평가 및 전체 결론은 아직 남아 있다. 개발 검토 완료 receipt는
agent가 실제 결과를 분석한 뒤 작성해야 하며 사용자 승인을 기다리는 조건이 아니다.
