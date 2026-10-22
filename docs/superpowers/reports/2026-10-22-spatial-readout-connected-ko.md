# Spatial480 readout 진단의 실제 연결

저장된 native spatial3 SS480에서 Normal·matching-prior-only·Top1-feature-only·
zero-transformed-value·wrong-projected-V와 최종 집계를 실행할6개 작업을 추가했다.
원래32개 prediction/GT latent와 voxel의 exact Normal 재현이 통과하기 전에는
네 개입을 시작하지 않는다. 실제 GPU 결과는 아직 없다.

Source code·직접 CPU 검사·[진단 프로토콜](2026-10-22-spatial-readout-diagnostic-protocol-ko.md)은
`wowo0709/feature-retargeting`의a014c19에 보존했다. Synthetic CPU 검사20개와
public/worker CLI help·구문 검사를 통과했다. CPU 검사로 실제 GPU source model의
forward나 품질 재현을 주장하지 않는다.

준비 시 실제 NN480/source480 파일2개를 원래 final file4 receipt의 SHA256과
대조했다. 앞선 두 준비 검사에서는 flat input map과 canonical receipt map의 차이,
원래 HELD job 대신 완료된 memory-v2 successor를 사용해야 하는 조건을 확인했다.
두 시도는 prelaunch 생성·queue 변경·GPU 실행 전 중단됐고 원래 로그를 보존했다.
성공한 연결은 실제 완료0 successor에 의존하며 원래 HELD state를 바꾸지 않는다.

| Canonical 기록 | 파일 | Bytes |
|---|---:|---:|
| Prelaunch | 13 | 316,500 |
| Queue | 2 | 7,804,179 |
| Applied | 1 | 760 |

각 기록을 실제 file4 SHA256/size로 검증했다. 전체 dependency DAG를 확인했고,
queue3547개가3553개가 된 뒤 기존3547개 descriptor가 이전 canonical snapshot과
모두 같은지 다시 확인했다. Queue-state는 수정하지 않았으며 dispatcher가 새6개를
모두 pending으로 읽었다.

현재 Top-1 PBR1900 완료 snapshot 뒤의 자원 구간에 연결했다. Historical SS 학습과
최종 평가의 우선순위를 유지하고 비는 GPU0–3에서 수행한다. 기존 train config·
학습 예산·진행 중인 NN은 바꾸지 않는다. 새로운 GPU worker는 own OS session을
정리한 뒤 예약을 반납하며, 실제 terminal과 원래32 재현 proof를 확인한 후 집계한다.

이번 선택의 근거인 후기 source-ray11개 SS의 실제480 update·33개128×3 평가와
32개 cascade prefix 검증도 완료했다. 원래 total11 semantic-search 모델과는 별도다.
후기11개 중 feature-spatial은 mean-asset IoU47.6667%로 강한 기준이며,
학습된 spatial feature stack의 일부가 유용할 가능성을 따로 확인해야 한다.
서로 다른 stack의 평균 차이를 QKV만의 독립적 효과로 해석하지 않는다.

Canonical root는 `feature-readout-v1/spatial-feature-readout-diagnostics-v1/`이다.
원래11개 SS coverage는 `native-flow-cold3-v1/eleven-source-ray-SS-final-coverage-audit-v1/`에
기록했고,211개 metadata 파일13,399,408bytes의 실제 remote SHA256/크기를 재확인했다.
모든 current PBR·historical 재현·독립 seed·적절한 SS/PBR 조합 DEV 비교와
최종508개10 metric 범위는 계속 남아 있다. 모델 채택이나 전체 목표 완료가 아니다.
