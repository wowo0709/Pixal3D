# 기존 query 실험 종료와 ray 학습 시작

사용자는 기존 softprior attention을 새 ray-wise query 기준에서 끝까지 실행할 필요가 있는지 지적했다. 기존 arm은 anchor-patch query와 retargeter를 유지하며 T0.05의 완만한 matching-score prior와 learned utility attention의 효과를 확인하기 위한 실험이었다. 동일 온도의 static weighted 및 Top1-R 기준이 있어 온도 변경과 learned utility의 차이를 볼 수 있었다.

그러나 camera-ray group query가 모든 matching 실험의 공통 기준이 된 이후,기존 anchor-patch/individual-voxel arm을 raw480까지 유지할 우선순위는 없다. 효과 판단은 같은 ray-group query/R을 공유하는 Top1→score-weightedTopK→Full을 통해 수행한다. Native spatial 및 physical No-decoder arm도 같은 ray 기준을 쓴다. 기존 두 학습은 native STOP으로 완전한 state를 저장하고 종료했다.

| 종료한 기존 arm | 저장 step | 보존한 내용 |
| --- | ---: | --- |
| train-voxel-query-c1 | 377 | denoiser/conditioner/optimizer/sampler/LoRA,finite optimizer moments |
| train-utility-softprior-c1 | 408 | denoiser/conditioner/optimizer/sampler/LoRA,finite optimizer moments |

중단 전 epoch 검증과 모든 기존 산출물을 보존했다. 중단 checkpoint6개 및 정리 기록은 실제 file4 SHA256/size readback을 통과했다. Native stage는 stopped다. 기존 wrapper는480 완료를 요구하여 exit1/queue FAILED를 남겼으므로 이를 의도된 protocol 변경으로 기록했다. 과거 queue-state를 완료로 바꾸지 않았고,종료한 descriptor2개와 미시작 후속41개를 활성 큐에서 retire했다. OOM/발산에 의한 중단이나 raw480 최종 결과로 해석하지 않는다.

## 실제 시작한 ray 학습

| GPU | Arm | 실제 DDP48 peak, GiB | 실제 main steady step |
| --- | --- | ---: | --- |
| 0,1 | ray-weighted | 49.862 | 약18초 |
| 2,3 | ray-full | 50.897 | 약20초 |

각 arm은 원래SS4210에서 fresh continuation한다. Benchmark에서 학습한2 update의 가중치를 본 학습에 이어 쓰지 않는다. Actual main의 두 rank 첫3 update와 실제2-updateDDP48 benchmark를 별도로 확인했다. Initial common32의 모든 개별 GT hash/seed/TP/FP/FN/4 metric은 V1 및 기존 Top1-R의 학습 전 결과와 exact하다. Initial IoU는0.42671546671444294다. Prepared model/data/trainer/validation recipe도 같다. 이 startup audit과 실제 file4 readback이 두 arm 모두 PASS다.

Global96/DDP2/micro48/accumulation1,raw480/5nominalepochs,epoch마다 quick32 validation,최종128×3 generation protocol을 유지한다. Full−weighted가 learned utility attention의 추가 효과,weighted−Top1이 score-weighted multi-candidate fusion의 효과다. Full−native spatial은 semantic correspondence,Full−No-decoder는 선택된 frozen scalar 아래 decoder context의 효과를 확인한다. Static/Full의 query 형태는 모두 동일하며 retargeter도 공통이다.

이 기록은 신규 ray 모델의 실행·정합·배치 검증 결과다. 성능 개선이나 모델 채택을 의미하지 않는다.
