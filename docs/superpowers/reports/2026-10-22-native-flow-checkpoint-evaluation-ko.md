# Native SS 학습 체크포인트 검증과 paired 평가

기존 frozen-denoiser source-feature teacher 실험은 native SS 학습을 대신하지 않는다.
DEV32에서 관측했던 작은 개선은 기존 source-disjoint72×3에서 재현되지 않았다.
따라서 그 teacher를 더 학습하거나 성공 모델로 채택하지 않고, 새 cold native-task
3-arm의 실제 SS flow loss로 LoRA와 source 모듈을 함께 학습하는 실험을 진행한다.

## 진행 중인 모델과 평가 목적

세 모델은 `vanilla`, `R_feature`, `R_raw_topk`이다. 마지막 두 모델은 source-camera
ray group query와 R-only retargeting을 사용한다. R_feature에도 learned Top-K feature가
들어가므로 **NoScore/Top1-only ablation으로 해석하지 않는다**. R_raw_topk는 동일한
attention posterior로 가중합한 raw appearance를 R 입력에 추가하는 비교다.
이 세 모델만으로 decoder·score·semantic correspondence의 독립 효과가 입증되지는 않는다.

TRAIN9149 전체, seed101, world1, global/micro96, accumulation1, 480 updates를 유지한다.
source-only teacher initializer는 사용하지 않는다. 학습 중인 코드·config·optimizer는
평가 준비 과정에서 변경하거나 재시작하지 않는다.

## 저장된 실제 모델 검증

새 `evaluate_r_native_checkpoint.py`는 이 학습의 실제 recipe를 직접 사용한다.
원래 multi-view SS4210이나 teacher112 checkpoint를 학습된 native SS480으로 간주하지 않는다.
동일 variant·step의 complete denoiser+LoRA와 source checkpoint, native config,
저장 후 완료된 validation row를 하나의 bundle로 고정한다. Vanilla에는 source가 없다.
각 입력은 파일서버4의 SHA256/size 검증 영수증으로 확인한다.

첫 96 step checkpoint는 학습 중 validation과 같은 DEV32·seed map·generation batch8,
12 sampling steps·CFG3으로 독립 프로세스에서 다시 평가한다. 학습 validation과 같은
deterministic-algorithm 설정(False)을 유지한다. 모든 per-asset GT hash·seed·TP/FP/FN·metric과
전체 metric이 정확히 일치해야 replay PASS로 기록한다. 차이가 나면 실제 출력과 실패를
남기고 원인을 조사하며, 학습을 임의 재시작하거나 허용 오차로 숨기지 않는다.

## 최종 DEV 평가와 cascade 입력

최종 480 step을 세 모델 모두 고정하고, 같은 DEV128과 세 개의 generation seed map으로
각각 평가한다. 세 generation noise는 독립 training seed 세 개가 아니다.
실험 최종 체크포인트 비교 전에 DEV32 최고 epoch를 골라 비교하지 않는다.

분석은 같은 asset·source cluster·GT hash·seed를 짝지어 계산한다. 각 asset의 세 noise
결과를 먼저 평균하고 source cluster 단위 bootstrap으로 CI를 계산한다.
mean asset voxel IoU가 primary metric이며 native global occupancy metric은 별도로 보존한다.
R_feature−vanilla, R_raw_topk−vanilla, R_raw_topk−R_feature의 세 primary IoU 대비에는
Bonferroni 보정 CI도 함께 기록한다. Recall·precision·F1은 보조 분석이다.

DEV128 noise0의 첫 32 asset을 같은 조건으로 재생성하여 실제 predicted SS latent와
occupancy를 저장한다. 원래 128 평가의 해당 32개 기록과 정확히 일치한 archive만
후속 fixed Shape512/Shape1024/PBR cascade 입력으로 사용할 수 있다. GT SS는 scoring에만 쓴다.
따라서 SS 평가의 IoU 개선과 rendering/source-preservation 개선을 혼동하지 않는다.

이는 DEV 평가 연결이며 최종 test508·독립 training seed202·native PBR port·10개 metric의
최종 비교가 완료된다는 의미는 아니다. Formal NoScore=Top1, NoCorr=native spatial,
실제 channel을 제거한 NoDQ에 대한 별도 재학습/비교 역시 필요한 범위로 남아 있다.

## 구현 검증 범위

CPU tests는 checkpoint variant/step/teacher 혼입, vanilla의 source 혼입, panel·seed·평가
protocol 변경, 평균 metric만 같고 개별 기록이 다른 replay, noise pseudoreplication,
paired GT/seed/source cluster 변경을 거부하는지 확인한다. 기존 cold recipe/runtime tests와
함께 42개가 통과했다. 실제 GPU checkpoint replay는 96 step 저장 뒤 실행하며, CPU test
통과를 실제 GPU replay 완료로 보고하지 않는다.
