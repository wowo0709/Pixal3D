# GT scale 미제공 TEST508 실시간 진행표

갱신: 2026-10-27 23:16 UTC. 이 표는 진행률이며 최종 성능표가 아니다.

| 단계 | 검증된 진행률 |
|---|---:|
| Ours V1 full cascade, file4 SHA | 213/508 |
| Ours Selected full cascade, file4 SHA | 208/508 |
| Ours Selected Top-1 PBR, file4 SHA | 208/508 |
| 같은 사례의 두 arm 모두 완료 | 208/508 |
| Official Pixal3D RGB1 source-M0 surface | 501/501 |
| Official Pixal3D RGB1 render mesh | 501/501 |
| Official render handoff | waiting on external metrics (PID 1552286) |
| External seven-method learned results | 3/7 |
| External learned evaluator (PID/state/I/O) | 2492093/R/88.9 GiB read |
| Unified nine-method learned evaluation | pending |
| Ours four-condition learned results | 0/4 |
| Local primary cascade workers (rank: PID) | 0:678576, 1:687195, 2:676710, 3:669224 |

| Completion gate | Receipt |
|---|---|
| Ours cascade | pending |
| Ours surface | pending |
| Official Cycles render | pending |
| Ours Cycles render | pending |
| Ours 10-metric evaluation | pending |
| Common 10-method table | pending |
| Ours paired uncertainty | pending |
| Official RGB1 paired uncertainty | pending |
| Fair common-success teaser | pending |

The main table uses the final common-success intersection, camera-given where supported, no target-derived object scale at generation, and post-generation source-M0-only Sim(3). The older M1-derived-scale table is an internal diagnostic.
