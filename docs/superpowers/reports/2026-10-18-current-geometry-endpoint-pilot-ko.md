# 원본 regional objective를 현재 SS consumer로 이식하는 검증

Q2C 원본24-run과G2 기존checkpoint의144개 endpoint는재현됐다. 그러나현재dataset에서geometry module의이득을검증하려면입력9채널뿐아니라학습objective도맞춰야한다. 현재flow-matching loss의2-update gradient PASS는이효과검증을대체하지않는다.

## 고정 범위

기존320개RGB task population의role별앞32개를사용한다. Category/hash selection은성능을보기전에고정됐으며source cluster가겹치지않는다. Source-exclusive/stale 영역의크기나현재metric으로asset을고르지않는다. Train32·calibration32의GT label과cached native input condition을별도로준비한다. Validation508의이미지/label을이pilot의학습에사용하지않는다.

두arm만비교한다: retargeting adapter, 같은adapter에predicted O/D9채널query를추가한구조다. No-query는encoder와입력채널이물리적으로없다. Source-projection OFF·query tuple shuffle·native는동일checkpoint의진단control이며별도2×2학습을하지않는다. Query는frozen mixed385 SS의source4-view O /target1-view D 예측이다. GT labels는query 입력으로전달하지않는다.

원래의frozen SS12-step differentiable rollout→raw decoder logits64→동일SS64/32 regional BCE와0.001 correction penalty를사용한다. 原본과같은128update·pair/epoch hash order·AdamW LR1e-4/WD0.01·clip1을기준으로한다. Consumer는사용자가지정한mixed385 complete base+LoRA step4210이며frozen이다. 현재native의12step/CFG3/time rescale1을명시적으로고정하고,과거full-FT20k/CFG7.5 consumer의수치와합쳐집계하지않는다. Physical No-query의구조도과거zero-input R2와구분한다.

이것은이전frozen-controller 실험의transfer pilot이다. 본SS LoRA fine-tuning의global96·DDP rank micro48·accumulation1·5epoch를변경하거나V1을재학습하는것이아니다. 실제native loss/gradient preflight후짧은동일protocol의효과를검증하고,그근거로full-data 학습과추가seed 확인을결정한다. Calibration은fixed-final checkpoint freeze뒤에만평가하며best-epoch선택을하지않는다. 다른데이터/목적의작은실험으로현재14개본비교를대체하지않는다.

## GT region의정합과보존

`historical_region_authority.py`에원본native flexible64 voxelization·center+8corner cube-ray 분류·projection·pooling·validator7개함수를AST 그대로이식했다. SHA authority와입력검증을포함한CPU3개검증을통과했다. 현재mesh를기존world-v4 camera/mesh scale로변환하고,만든target64가저장된Shape encoder입력support와정확히같아야통과한다. 원래CAGE의GT pair-union scale로현재frame을바꾸지않는다.

Source-state visible face barycentric을M1의동일triangle에적용해matched source-exclusive label을만든다. Target 관측RGBAforeground는visible-positive를,sourceRGBAforeground는source evidence를제한한다. Conservative FREE는원래cube-ray 함수와opaque geometry z-buffer로계산하고GT target support를항상제외한다. Open3D의top-down pixel 배열을raw OpenGL bottom-up 순서로변환해원래classifier의y축과맞춘다. Current frame밖source point는boundary로clip하지않고제외·기록한다. Empty source/stale label도성능을보고대체하지않고count와함께보존한다.

실제첫train asset의physical64는encoder 입력support2179개와좌표까지정확히일치했다. Full-region preflight에서SS64 source-positive1022/stale162/visible-positive829/free239996, SS32 source-positive330/stale14/visible-positive266/free29250이확인됐다. Native target/mismatch projection·loss 초기값은두actual GPU모듈모두exact이고query encoder의두번째update gradient L1은0.0001064372였다.

Native voxelization은CPUkernel을사용하고Open3D/BLAS도2core/1thread로제한한다. 설치된`o_voxel`의상위package import가CUDA metadata를조회하므로허용된GPU3이보이는환경에서실행한다. 계산은CPU이며다른GPU를찾거나조회하지않는다. 첫full-region asset은로딩이후2.94초였다. 최초CPU-only import실패와pooling validator 누락실패는보존하고,의존함수까지복구한`preflight-assigned3-v2`를새로통과시켰다.

파일은file4의`rgb_matching_v2/current-geometry-regions-v1`에저장한다. 이문서의준비/연결PASS는새module의학습완료·regional metric 향상·최종Full채택을뜻하지않는다.
