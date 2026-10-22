# 기존 SS 후보의 공통 PBR32 비교와14그룹 분석

이미 연결한3개 engineering prefix의 actual gate가 통과하면 나머지93개 case를
진행한다. 기존 geometry96개를 읽기 전용으로 재사용하고 original PBR16850 및
공통 native vanilla PBR1900을 적용해6그룹192개 texture 결과를 만든다. Prefix를
제외한 추가 SS/Shape inference는 없으며, prefix도 SS/Shape를 새로 생성하지 않는다.

최대4개 순차 lane을 사용한다. 새 own texture raw의 file4 SHA/size·pending consumer·
open handle·실제 child 종료·소스 ancestry 검증 후 local duplicate만 정리해야 다음
case를 시작한다. Prefix3개에도 같은 정리를 적용한다. 원래 geometry96개와 canonical
remote raw, metadata, render pixel, GT/Pred/Diff panel 및 checkpoint는 삭제 대상이 아니다.

새 공식 DreamSim/CLIP576 view는 FP32·TF32 off·no AMP와 같은 registry를 사용한다.
CPU neural forward를 사용하지 않으며, GPU 자식 session이 모두 종료될 때까지 예약을
유지한다. 기존 native8그룹768 view와 합쳐 총14그룹·32asset·1344 view를 분석한다.
저장된 native8개 statistics를 원래 집계로 다시 계산해 모든 role·CI·per-asset 값의
차이가1e-12 이하인지 검사한 뒤 새 결과와 비교한다.

Target8은 PSNR·SSIM·M-IoU, source9/10은 LPIPS·DreamSim·CLIP, geometry는
source-frame20k surface의 CD/F@.02로 구분한다. Cross-model M1 원래 material RGB/alpha,
GT-only crop 및 변환된 reference pixel identity도 검사한다. Native8개와 legacy6개의
공식 평가 report는 같은 metric weight/precision을 사용해야 한다.

Native8개는 각 SS의 original PBR 및 자체 학습한 PBR 조합이고, legacy6개는 공통
vanilla PBR1900을 사용한다. 따라서 전체14개 표의 모든 차이가 SS 하나만의 효과인 것은
아니다. Prior 그룹끼리 및 legacy-matched 대 native vanilla-matched 비교가 downstream
PBR을 고정한 비교다. 서로 다른 학습 계열의 순위를 QKV 하나의 독립적 효과로 해석하지
않는다. 동일 geometry의 pair 안에서만 실제 texture initial noise가 같다고 주장한다.

91개 pairwise 비교의 cluster bootstrap4000회/seed20261023 CI는 반복 사용한 DEV32의
보정하지 않은 탐색적 분석이다. 이 표만으로 Full을 채택하지 않는다. 독립 학습 seed,
선택된 SS/PBR 조합의 적절한 추가 확인 및 frozen TEST508의10 metric까지 남아 있다.
FID/KID는 DEV32에 새로 추정하지 않는다. W&B에는14개 모델의 role별 표와 각 첫 asset의
GT/Pred/Diff를 기록하고 실패하면 실제 통계를 보존하며 logging 실패를 따로 표시한다.

추가 범위는93개 GPU case·96개 CPU raw 정리·공식576개 view의 GPU 작업1개·14그룹
CPU 집계1개다. 기존4개 prefix 작업과 합쳐195개 작업으로 구성한다. 준비·검사·큐 연결과
실제 GPU 평가 완료는 별도로 기록한다.
