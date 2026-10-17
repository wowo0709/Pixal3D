# 이전 양의 모듈의 재현 구현과 실행 계약

사용자의 “이전에 성능 향상을 보였던 모듈들을 재현” 요청에 따른 실행 문서다. [차이 분석](2026-10-18-historical-module-transfer-gap-analysis-ko.md)의 확인된 불일치를 복구하며, 현재14개 구성의 결과를 삭제하거나 이름을 바꿔 성공한 재현으로 취급하지 않는다.

## 재현 대상

| 대상 | 이전에 유효했던 정확한 비교 | 재현 계약 |
| --- | --- | --- |
| Source spatial routing | 세 source-view map의 GT 주소 대 wrong 주소/동일 projection-source-off | 3×32² write slot, 정확한 read address, view별 value multiset·support 고정, 각 source의 native ProjGrid, global/negative 불변 |
| Q2C | 같은64 candidate·task RGB loss의 D0 pairwise 대 D1 query→candidate context | Top1/soft 양쪽에 Q2C가 들어간 현 대비와 구분; candidate·objective·seed·fixed-final checkpoint를 맞춤 |
| Geometry content | 3D predicted O/D logit·occupancy·SDF9채널을 residual content에 넣은 효과 | 16³ native xyz, target/mismatch/difference, frozen SS full rollout→decoder regional SS64/32 BCE, correction penalty0.001 |

이전 retargeter는 모든 DEV source-use gate를 통과하지 않았다. 따라서 과거의 geometry cleanup 상대 개선과 실제 source-retargeting 성공을 구분한다. RAW Omega2048 feature delta를 이전9채널 occupancy/SDF geometry와 같다고 취급하지 않는다.

## 이식한 reference 구현

`historical_reproduction.py`와 `historical_geometry.py`는 이전 원본에서26개 함수/클래스의 AST를 그대로 옮겼다. `historical_reproduction_authority.json`에 각 원본 파일 SHA256·line·함수 AST hash를 고정했다. 실행 시 과거 worktree나 `runtime_bundle`을 import하지 않는다. 저수준 Q2C/scalar는 기존 `r9_matching.py`에 이미 같은 AST로 이식돼 있다.

Reference G2의 네 mode는 과거 수식의 감사용 API다. 사용자가 요청한 production의 No-query arm을 zero channel로 대체하거나 새로운2×2 학습을 실행하는 뜻이 아니다. Production query ablation은 입력 채널의 물리적 유무를 유지한다.

CPU 검증은 native identity/fallback, soft validity 단일 가중, identity sampling, view별 wrong-address derangement/value multiset, pixel-ray 왕복, geometry 축/채널, zero adapter, frozen rollout의 gradient, regional loss, scoped hook 복원, frame fingerprint를 포함한다. 관련30개 테스트가 통과했다. 이 PASS는 수식·통합 계약의 검증이며 생성 성능 PASS가 아니다.

## 1차 실제 검증: 현재 common32의 per-view oracle

Data는 기존 DeformNeXt validation common32·world-v4를 유지한다.32개 모두 M0 geometry·M1 vertices·RGB가 로컬 file4 mirror에 있었으며 추가 원본 다운로드는 하지 않았다. 과거와 같은 Open3D raycast·barycentric route·pixel7.5+16n·visibility tolerance0.002를 사용해 세 source-camera route를 구축했다. CPU affinity를2 core로 제한했고 실제 계산은약3초였다.32개 모두 각 source-view에서2개 이상 active slot이 있어 wrong-address control이 정의됐다.

Route-v2는 RGB/mesh fingerprint 외에 mesh scale·render-world schema·center/extent도 고정한다. 기존 v1 route는 보존하며 v2를 실행에 쓴다. 모든 GT cache는 `diagnostic_only`이며 inference-safe matcher cache와 namespace가 다르다.

실제 encoder의 native per-view ProjGrid 입출력을 scoped hook으로 캡처한다.4개 view의 native online mean이 원래 joint projection과 bit-exact인지를 검사하고 global/negative는 유지한다. Source-on case는 세 source map을 각각 교정해 각 카메라의 원래 ProjGrid로 투영한다. Anchor-camera map 하나로 합친 뒤 voxel residual을 더하는 이전 구현을 재사용하지 않는다.

Cases는 `native`, `source-off`, `gt-corr`, `wrong-corr`다. 여기서 source-off는 **source projection만 제거하고 네 RGB로 얻은 native global/negative를 유지**하는 과거 reference control이다. 현재 primary study의 추가 모듈 전체 OFF와 정의가 다르므로 별도 reproduction output에 기록한다. Target-only 모델 학습 arm을 추가하지 않는다.

Checkpoint는 원본 multi-view mixed385 complete base+LoRA SS64이며, loader·architecture는 현재 dev-jaehyeok 기반 runtime을 사용한다. 같은32개·같은 sample noise·12 steps·CFG3·generation batch8에서 frozen consumer를 비교한다. 먼저1개 sample의 encoder preflight를 통과해야 전체 생성을 시작한다. GPU0~3만 사용하고, queue가 비는GPU3에 이 작업을 배정한다. 실행 중인 C2와 충돌시키지 않는다.

초기 metric은 global SS IoU/F1과 `route_sampled_hidden_gt_recall64/32`다. 후자는 raycast로 샘플링한 hidden surface cell과 decoded target SS의 교집합이며, **과거의 full source-exclusive SS region authority를 재구축한 metric은 아니다.** 결과 파일에 이 한계를 명시한다. 이전 regional primary의 정확한 재현에는 full region authority를 별도로 구성해야 한다. 생성 검증을 과거 수치의 재현 완료로 확대하지 않는다.

## 후속 연결

1. Actual GPU preflight와 full32 correct/wrong/source-off를 완료해 native per-view 경로의 headroom을 판별한다.
2. 같은 upstream/task loss에서 D0/D1을 직접 비교하는 작은 head 학습·evaluation runner를 연결한다. 기존 H3를 Top1과soft 양쪽에 쓰는 비교로 이 검증을 대체하지 않는다.
3. Predicted SS O/D geometry·full region authority를 준비하고, 옮긴 full-rollout regional objective의 실제 native GPU gradient·zero-regression을 확인한다.
4. 재현 가능한 source-use/geometry gain이 확인된 경로에 adapter와 physical query를 순서대로 연결한다. 기존 데이터의 전체 재전처리나 장기 fine-tuning을 먼저 반복하지 않는다.

현재 성능 gate가 열린 것으로 간주하지 않는다. 준비·수식 검증·GPU preflight·생성 endpoint·모듈 채택은 별도 상태로 기록한다. 기존14개 비교의 완료와 위 faithful reproduction·필요한 후속의 완료도 구분하며, Full/PBR 결정까지 추적한다.
