# 필수 실험 계획 실행과 논문 초안 진행

2026-10-24 15:20 UTC 전후 실제 상태. 사용자가 지정한 module-effects 보고서와 essential-remaining 계획을 실행한다. 최종 모델 선정·독립 확인·TEST와 최종 writing은 아직 미완료다.

## 완료한 실제 작업

### 최종 평가의 render 경로 검증

기존 DEV32의 Top-1 R 사례 한 개를 같은 physical GPU2에서 실행했다. `selected_final_phase`가 원래 renderer를 호출하면서 in-memory NumPy proxy로 canonical owner의 render mesh만 재사용했다. Source59 스냅샷/lock과 실행 계획, 실제 결과를 file4 SHA256/size로 닫았다.

- 원래 sampled prediction 및 재질·sparse support 보존.
- 실제 render vertices/faces 동일.
- 세 view의 GT RGB/alpha와 카메라 행렬 동일.
- 세 view의 shaded/base_color/alpha/mask 배열 전부 byte-identical, 최대 차이0.
- 추가 SS·Shape·PBR 생성 없음; 새 prediction archive 없음.
- 실제 child returncode0과 owned session drain 확인.

이는 한 DEV 사례의 실제 integration 검증이다. TEST508 전체 실행이나 최종 모델의 선택을 뜻하지 않는다. 기존 source57 preflight를 덮어쓰지 않았으며, 이후 최종 freeze는 새 source59 identity를 사용해야 한다.

Canonical 결과: `unified-module-pipeline-v1/final-render-compatibility-and-source-lock-v208/compatibility-proof-v1/audit.json`.

### E_FIXED_RAW joint 평가 완료

같은32개 DEV asset·source cluster와 단일 training seed101/noise에서 실제8개 metric을 집계했다. V1/R baseline의 asset별 결과가 D1-cos study와 동일함을 확인한 뒤 아래 표를 만들었다.

| 실제 SS/PBR 조합 | Source LPIPS↓ | DreamSim↓ | CLIP↑ | Target PSNR↑ | SSIM↑ | M-IoU↑ | CD↓ | F-Score↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1/V1 | 0.053099 | 0.155692 | 0.942434 | 24.47068 | 0.950488 | 0.880945 | 0.012452 | 0.849149 |
| C_DECODER/Top-1 R | 0.052104 | 0.152435 | 0.940636 | 24.57466 | 0.951284 | 0.885678 | 0.012413 | 0.851420 |
| D1-cos/Top-1 R | 0.053362 | 0.155373 | 0.942157 | 24.57729 | 0.951170 | 0.886600 | 0.012504 | 0.851308 |
| E_FIXED_RAW/Top-1 R | 0.053290 | 0.157828 | 0.941692 | 24.54433 | 0.951563 | 0.886574 | 0.012328 | 0.856566 |

E_FIXED_RAW 대 R에서 CD −0.00008579, F-Score +0.005146인 반면 source LPIPS +0.001186, DreamSim +0.005393이다. Source LPIPS의 보정 전 paired95% CI는 [+0.000224,+0.002275], F-Score는 [−0.000163,+0.011128]이다. Geometry/appearance tradeoff이며 일반적인 우위를 입증한 결과가 아니다. Main/primary metric을 사후에 CLIP으로 바꾸지 않는다.

Canonical 결과: `consistent-fixed-Top1-joint-native32-v200/summary-v1/audit.json`. 실제 W&B: [E_FIXED_RAW joint DEV32](https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/consistent-fixed-Top1-joint-native32-v200).

## 연결한 다음 평가

PBR 선택과 SS 선택의 상호작용을 분리하기 위해 E_FIXED_RAW에서 **빠진 vanilla PBR counterpart만** 추가한다. 기존 Top-1 R32개와 SS/Shape·guide를 재사용하므로 추가 SS/Shape 생성이나 새 학습은 없다. 기존 native texture RNG를 그대로 비교에 섞지 않고 같은 seed-reset replay noise를 사용한다. 렌더링은 기존 E_FIXED_RAW+R의 실제 triangles를 고정해 재사용한다.

첫 GPU2 사례에서 raw geometry·native guide·초기 texture noise·실제 render mesh·M1 material GT가 모두 일치함을 확인하고 canonical pilot gate를 닫았다. 이후31개는 이 성공에 의존해 GPU2/3에서 실행한다. 평가 chain도 등록했다.

1. Vanilla PBR32개 실제 생성·render·shared-reference scoring.
2. Coverage 확인 뒤 새96 view의 official DreamSim/CLIP GPU 평가.
3. 기존 V1/R의192개 official view는 closure를 확인해 재사용; 실제 신규 forward와 구별해 기록.
4. 세 조합의 joint 집계와 **같은 E_FIXED_RAW geometry에서 vanilla 대 R PBR**의 paired 통계·W&B.

Canonical 계획: `fixed-evidence-SS-paired-PBR-native32-v209`. 후보 문맥의 stage 선택 확인이며 최종 Ours 채택이나 TEST가 아니다. 모든 SS×PBR 조합으로 확대하지 않는다.

## 학습과 writing

E_LEARNED_RAW는 실제480-update science 학습과 file4 closure를 마쳤다. 다음 final128×3 평가는 기존 queue에서 이어진다. FULL_MEAN_MATCH와 FULL_VOXEL_Q는 원래 PID로 계속 학습 중이다. 독립 seed202 vanilla SS도 GPU1에서 실행 중이다. 이들의 실제 endpoint를 확인하기 전 query/member/learned refinement의 필수성을 확정하지 않는다.

[영문 논문 초안](../paper/retargeting-evidence-draft.tex)에 실제 구현 수식, native MV4 carrier, member posterior, fixed/anchored attention, retargeting/fallback, 데이터와 stage별 학습·평가 protocol을 작성했다. [Claim 원장](../paper/claim-evidence-ledger.md), 실제 DEV 표2개, 최종10-metric 표와 ablation 표 양식도 작성했다. 최종 표에는 수치를 추정해 넣지 않았다.

TeX environment 균형과 input 파일 존재, 최종 template에 수치가 없음을 확인했다. TeX compiler는 설치되어 있지 않아 PDF 컴파일은 아직 검증하지 못했다. 기존 논문 PDF의5개 페이지를 읽어 문제 정의와 metric 구성을 확인했으며 원본 PDF를 수정하지 않았다. Related Work의 primary citation 검증과 최종 Abstract/Results/Discussion/Conclusion은 남아 있다.

남은 우선순위는 기존3개 SS 비교의 최종 평가, 후보 문맥의 PBR 짝비교, 최종 구조에서 빠진 matched ablation, 필요한 독립 확인·VAL36, 선택/protocol freeze, TEST508·10개 metric, 표·그림·최종 writing이다.

## 15:30 UTC 실행 갱신

E_LEARNED_RAW의 final128 noise0 평가가 완료됐고 noise1이 실제 실행 중이다. 아직3개 noise 전체의 최종 비교로 해석하지 않는다. Mean 대조군은475/480, voxel query 대조군은325/480이다. PBR 짝비교는11/32 완료·2개 실행 중·19개 대기이며 실패/HELD는0이다. 이 수치는 해당 시점의 관측이다.

Seed202 R의 기존 job이 vanilla 학습/cache 완료에 직렬 의존하던 연결을 점검했다. R에 필요한14개 canonical 입력과 engineering gate가 이미 준비돼 있고, 초기화 checkpoint는 원본 MV SS4210이었다. 실제 disk admission도 통과했다. 같은 기존 job의 source/config/command/output/seed/batch는 유지하면서, **GPU0의 Mean science 종료 뒤 R202를 시작**하도록 scheduling 의존성과 GPU assignment만 변경했다. GPU1의 vanilla202와 겹쳐 진행하며, 진행 중인 학습은 중단하지 않는다. 실제 시작 여부는 다음 관측에서 확인해야 한다.

준비 근거와 변경 전/후 job은 `required-SS202-parallel-scheduling-v212/prelaunch-v1/audit.json`에 있으며 file4에서 SHA256/size를 확인했다. 각 실행의 기존 checkpoint budget reservation은 계속 적용된다.

핵심6개 문헌을 primary abstract/metadata와 대조한 Related Work 및 BibTeX도 추가했다. 전체 문헌의 exhaustive review나 외부 편집법 대비 우위를 검증한 것은 아니다. 갱신 문서는 `paper-writing-related-work-v211`, 가독성을 확인한 DEV tradeoff PNG/PDF는 `paper-writing-and-figure-v210`에 canonical closure로 보관했다.
