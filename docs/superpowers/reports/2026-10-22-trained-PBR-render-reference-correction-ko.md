# PBR replay 렌더 정합 검증의 reference 수정

[후속 평가 설계](2026-10-22-trained-PBR-generated32-render-protocol-ko.md)의 첫 GPU
proof에서 reference 범위를 잘못 잡았다. 기존 zero-export proof는 **동일 replay의
vanilla와 zero 모듈들**에서 texture/latent가 같음을 보인 것이다. 과거 cascade의
texture와 같음을 보인 것이 아니다. 실제로 source geometry·camera·GT는 같았지만
zero replay의 attrs/texture_feats는 과거 cascade와 달랐다. 이는 common replay
sampling 조건과 과거 cascade 조건의 차이를 포함하므로 wrapper 정합 기준이 될 수 없다.

V1 proof는 이 잘못된 비교에서 실패했고160개 후속 case는 모두 시작 전 HELD됐다.
기존 실패·계획·출력을 보존한다. 학습이나 이미 생성된1905 prediction은 변경하지 않는다.

V2 proof에서는 같은 zero replay의 appearance/voxel data와 같은 native predicted
LOD를 사용하는 두 실행을 비교한다. 한쪽은 새 LOD 재사용 wrapper, 다른 쪽은
기존 native renderer다. Native reference에는 기존 predicted LOD만 담은 engineering
packet을 전달해 native의100만-face 이하 분기를 사용한다. Hook·추가 simplification·
학습 모델 호출 없이 같은 geometry·texture·camera·AO의 렌더와 metric을 비교한다.
이 packet은 model prediction 또는 품질 평가 결과가 아니며 raw CD/F-Score도 읽지 않는다.

실제3개 camera의 모든 array와 자산별 metric이 정확히 같은 경우에만160개 후속
렌더를 진행한다. Scientific case는 원래 raw prediction을 읽고 이미 만들어진
native LOD를 재사용한다. Source preservation/target alignment, 동일 M1 material GT,
official metric와 원래 raw surface CD/F-Score라는 평가 범위는 유지한다.
