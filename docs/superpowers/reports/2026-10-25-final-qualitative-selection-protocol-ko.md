# 최종 정성 비교의 사전 사례 선정

2026-10-25. 최종 그림에는 같은 사례/view의 GT·두 zero-shot·V1·R·Ours와
성공·실패를 포함해야 한다. 기존 DEV figure는 관측된 가장 큰 개선/손실의
설명용 그림이다. 이를 최종 TEST 사례의 선정 규칙으로 소급하지 않는다.

## 실제 완료

`prepare_outcome_blind_qualitative_CPU_v355.py`로 기존 native TEST508
manifest의 metadata만 읽었다. 원래 닫힌 split audit의 manifest SHA256/size와
현재 로컬 및 실제 file4의 값이 모두 일치한다.508개 sample/134개 source cluster를
확인하고 source hash 순 첫 두 cluster, 각 cluster의 독립 case hash 순 첫
sample을 선정했다. 예측·quality metric·geometry array는 읽지 않았다.

고정된 두 sample ID는 다음과 같다.

- `shard003_6e4005b2a9ac445c8fb4295bc5d29f9a_d2_fast300fpssupplyv1_PA_target08`
- `shard001_b5e62e1a306e4cb7846bd51a75ac0d9a_d1_fast300fpssupplyv1_balanced_JN_L_target08`

`final-qualitative-outcome-blind-plan-v355`의 selection/policy/preparer3개,
159,948bytes가 실제 file4와 SHA256/size 일치했다. 이는 실제 metadata 선정의
증거이며 모델 동결·TEST inference·최종 이미지 완료의 증거가 아니다.

## 이후 적용할 규칙

모델 선택이 끝난 뒤 Ours 대 R을 비교하고 Ours=R이면 V1을 비교 대상으로 둔다.
TEST metric을 보고 comparator를 바꾸지 않는다. Joint improvement는
CD↓/FScore↑/source LPIPS↓/DreamSim↓가 함께 성립하는 경우다. Geometry loss는
CD↑/FScore↓, appearance loss는 LPIPS↑/DreamSim↑다. 각 eligible group에서
geometry는 ΔFScore, appearance는 ΔLPIPS 순 lower median을 고르고 case hash로
tie를 푼다. 조건을 만족하는 사례가 없으면 빈 category로 남긴다. 중복 sample은
label을 합치고, 그림을 채우려고 다른 사례로 바꾸지 않는다.

실제 camera8/9/10의 동일 scored-image archive를 사용하며 모든 model의
full frame·camera·배경·conversion을 맞춘다. 최종4/5개 모델과 동일 GT를 나란히
보여준다. 추가 generation/render나 model별 crop/alignment는 없다.

정확한 알고리즘·ID·빈 category와 중복 처리는
[공개 선정 protocol](../paper/qualitative-selection-protocol.md)에 기록했다.
Manuscript의 qualitative protocol과 claim 원장에도 두 선정 범위를 구분했다.
실제 full TEST 이미지·conditional case·최종 figure·caption은 아직 남아 있다.
기존5개 SS 대조·독립 seed·cascade·동결/TEST/최종 writing 범위는 유지하며
새 학습 arm을 추가하지 않았다.

## 실제 manuscript 반영

`paper-draft-qualitative-selection-policy-v356`에서14쪽/24references 개발
초안을 실제 compile했다. Warning/overfull/underfull은0이고 qualitative protocol의
6쪽과 실제 ray 그림의7쪽을 PNG로 직접 검수했다. 공개 TeX/선정 protocol의
현재 bytes와 compile source가 같다.32개 artifact/850,933bytes가 실제 file4와
일치한다. [개발 PDF](/root/dev/.cache/deformnext-retargeting/paper-preview-20261025-v356.pdf)는
현재 미완료 결과를 명시한 초안이며 최종 제출 형식/분량의 논문이 아니다.
실제 평가 표와 정성 결과를 넣은 뒤 최종 편집·분량 조정·검수가 남아 있다.

최종 freeze를 만들 때 이 실제 policy와 metadata-only selection의 SHA를
evaluation protocol에 함께 남기고 TEST 결과에 따라 변경하지 않는다. 이는
final model/protocol freeze를 생성했거나 TEST 실행을 허용했다는 뜻이 아니다.
