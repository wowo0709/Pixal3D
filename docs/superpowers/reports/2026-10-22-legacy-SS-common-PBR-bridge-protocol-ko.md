# 강한 기존 SS 후보를 동일 PBR 조건에서 비교

Retargeting의 개선 근거와 ray-wise QKV의 필수성은 다른 주장이다. 현재 native
Top-1+R의128개·3생성 noise 평균 IoU는47.2600%, raw Top-K+feature+R은47.2078%다.
후기 source-ray의 feature-spatial은47.6667%로 강한 기준이지만 query·carrier·feature
변환·전달도 달라 QKV 하나의 효과가 아니다. Member posterior의 후보 recall 개선도
최종 생성 품질 개선과 구분한다. 새로운 비교에서 좋은 기존 후보를 제외하지 않는다.

기존14개 DEV cascade는 SS만 바꾸고 Shape5124210·Shape102416850·V1 PBR762를
고정했다. 앞으로의 native8개 그룹은 original PBR16850 또는 각자의 native PBR1900을
사용한다. PBR이 다른 두 표에서 모델 순위를 직접 정하지 않도록 추가 비교를 준비한다.

## 비교 범위와 재사용

기존 retargeting(C1), feature-spatial, feature-DQ 각각 DEV32의 생성 geometry를
그대로 읽어 original PBR16850과 공통 native vanilla PBR1900으로 texture를 다시 생성한다.
Retargeting(C1)을 다른 과거 target-patch Top-1+R 모델과 같은 것으로 부르지 않는다.
3개 SS×32개×2개 PBR의6그룹이며, native8개와 함께14그룹을 비교할 계획이다.
이는 서로 다른 SS 학습 계열의 배포 결과 비교이며 순수 attention ablation이 아니다.

기존96개 generation의 metadata·prediction·surface평가·SS archive/artifact 및
원래 generation plan을 확인했다.549개 입력이 기존 file4 receipt와 일치했다.
원래 Shape config와 checkpoint의 기록된 digest도 공통 native template과 일치한다.
기존 geometry를 새로 생성했다고 기록하지 않으며, 원래 PBR762를 사용해 생성된
provenance도 유지한다. 기존 대형 Shape checkpoint를 다시 load한 검사는 아니다.

각 geometry 안에서는 RGB4장·camera·sampler·texture seed20261020을 고정하고,
actual initial texture noise와 raw geometry가 정확히 같은지 확인한다. Prior의 native
predicted LOD를 matched texture에도 사용한다. 서로 다른 SS geometry 사이에서는
support가 다를 수 있으므로 texture noise tensor가 동일하다고 주장하지 않는다.
원래 geometry 폴더는 읽기 전용 입력으로 사용하며 새 결과 폴더에 덮어쓰지 않는다.

## 실행 순서와 판정

현재4GPU PBR 학습 및 historical SS 재현을 유지한다. 먼저 예정된 native8개 DEV32
cascade의 실제 완료 summary를 기다린다. 같은 첫 asset을3개 기존 SS에서 재사용하는
engineering prefix3개를 GPU2/3에서 실행하고,6개 texture replay·18개 render view의
계약을 CPU에서 검사한다. GPU 자식은 각각 별도 OS session으로 실행하고 모든 자식이
종료된 후 예약을 반납한다. CPU synthetic 검사는 실제 GPU 재현을 대체하지 않는다.

Prefix gate는 품질 향상을 요구하지 않는다. 실제 child terminal0, 원래 geometry/SS
binding, 공통 PBR lineage, pair 안의 noise/LOD, 모델 사이의 M1 원래 material GT를
검사한다. 한 asset 결과로 모델을 채택하지 않는다. 실패나 부분 결과도 보존한다.

이후 남은93개 case를 최대4개의 순차 lane으로 연결하고, 새로 만든 own raw duplicate만
실제 file4 SHA/size·미완료 consumer·open handle·소스 ancestry 검증 후 정리해야 한다.
원래96개 입력 geometry는 정리 대상이 아니다.6그룹576개 official perceptual view와
native8개를 합친 paired 분석은 후속 구현·실행 범위로 남아 있다. Prefix만으로32개
품질 비교가 완료됐다고 보고하지 않는다.

Source9/10은 LPIPS·DreamSim·CLIP, target8은 PSNR·SSIM·M-IoU, raw geometry는
CD/F-Score를 평가한다. GT는 M1 geometry에 원래 source material을 적용한 공통 기준이다.
반복 사용한 DEV32의 탐색적 CI, 독립 학습 seed 확인, 최종508개10 metric을 구분한다.
FID/KID는 작은 prefix에서 추정하지 않는다. Full 구조나 최종 모델 채택은 아직 미정이다.
