# 원본 regional objective를 현재 SS consumer로 이식하는 검증

Q2C 원본24-run과G2 기존checkpoint의144개 endpoint는재현됐다. 그러나현재dataset에서geometry module의이득을검증하려면입력9채널뿐아니라학습objective도맞춰야한다. 현재flow-matching loss의2-update gradient PASS는이효과검증을대체하지않는다.

## 고정 범위

기존320개RGB task population의role별앞32개를사용한다. Category/hash selection은성능을보기전에고정됐으며source cluster가겹치지않는다. Source-exclusive/stale 영역의크기나현재metric으로asset을고르지않는다. Train32·calibration32의GT label과cached native input condition을별도로준비한다. Validation508의이미지/label을이pilot의학습에사용하지않는다.

두arm만비교한다: retargeting adapter, 같은adapter에predicted O/D9채널query를추가한구조다. No-query는encoder와입력채널이물리적으로없다. Source-projection OFF·query tuple shuffle·native는동일checkpoint의진단control이며별도2×2학습을하지않는다. Query는frozen mixed385 SS의source4-view O /target1-view D 예측이다. GT labels는query 입력으로전달하지않는다.

원래의frozen SS12-step differentiable rollout→raw decoder logits64→동일SS64/32 regional BCE와0.001 correction penalty를사용한다. 원본과같은128update·pair/epoch hash order·AdamW LR1e-4/WD0.01·clip1을기준으로한다. Consumer는사용자가지정한mixed385 complete base+LoRA step4210이며frozen이다. 현재native의12step/CFG3/time rescale1을명시적으로고정하고,과거full-FT20k/CFG7.5 consumer의수치와합쳐집계하지않는다. Physical No-query의구조도과거zero-input R2와구분한다.

이것은이전frozen-controller 실험의transfer pilot이다. 본SS LoRA fine-tuning의global96·DDP rank micro48·accumulation1·5epoch를변경하거나V1을재학습하는것이아니다. 실제native loss/gradient preflight후짧은동일protocol의효과를검증하고,그근거로full-data 학습과추가seed 확인을결정한다. Calibration은fixed-final checkpoint freeze뒤에만평가하며best-epoch선택을하지않는다. 다른데이터/목적의작은실험으로현재14개본비교를대체하지않는다.

## GT region의정합과보존

`historical_region_authority.py`에원본native flexible64 voxelization·center+8corner cube-ray 분류·projection·pooling·validator7개함수를AST 그대로이식했다. SHA authority와입력검증을포함한CPU3개검증을통과했다. 현재mesh를기존world-v4 camera/mesh scale로변환한다. Target64의authority는현재실제SS encoder입력support이며SS latent에기록된source_coords_sha256과정확히일치해야한다. Source/target 모두현재native1024 voxelization→integer pool64로재구축하고target은실제encoder입력과좌표set까지exact여야한다. 직접64 voxelization의근접도임계값으로대체하지않는다. 원래CAGE의GT pair-union scale로현재frame을바꾸지않는다.

Source-state visible face barycentric을M1의동일triangle에적용해matched source-exclusive label을만든다. Target 관측RGBAforeground는visible-positive를,sourceRGBAforeground는source evidence를제한한다. Conservative FREE는원래cube-ray 함수와opaque geometry z-buffer로계산하고GT target support를항상제외한다. Open3D의top-down pixel 배열을raw OpenGL bottom-up 순서로변환해원래classifier의y축과맞춘다. Current frame밖source point는boundary로clip하지않고제외·기록한다. Empty source/stale label도성능을보고대체하지않고count와함께보존한다.

실제첫train asset의physical64는encoder 입력support2179개와좌표까지정확히일치했다. Full-region preflight에서SS64 source-positive1022/stale162/visible-positive829/free239996, SS32 source-positive330/stale14/visible-positive266/free29250이확인됐다. Native target/mismatch projection·loss 초기값은두actual GPU모듈모두exact이고query encoder의두번째update gradient L1은0.0001064372였다.

Native voxelization은CPUkernel을사용하고Open3D/BLAS도2core/1thread로제한한다. 설치된`o_voxel`의상위package import가CUDA metadata를조회하므로허용된GPU3이보이는환경에서실행한다. 계산은CPU이며다른GPU를찾거나조회하지않는다. 첫full-region asset은로딩이후2.94초였다. 최초CPU-only import실패와pooling validator 누락실패는보존하고,의존함수까지복구한`preflight-assigned3-v2`를새로통과시켰다.

파일은file4의`rgb_matching_v2/current-geometry-regions-v1`에저장한다. 이문서의준비/연결PASS는새module의학습완료·regional metric 향상·최종Full채택을뜻하지않는다.

`train_current_geometry_endpoint`는위의128-update 학습과fixed-final 평가를실행한다. Seed101/202/303에서두additive arm을같은pair/noise hash schedule로각각학습한다. Train32는epoch마다한번씩사용하고총4회순회한다. Native flow/decoder/LoRA는frozen이고controller만최적화한다. Native/current/projected-source-off와query arm의joint-tuple shuffle을같은32개calibration cluster·3noise에서평가한다. Projected-source-off는native4RGB global/negative를유지한다.

No-query는query 파일자체를열지않고training loader는calibration role을파일개방전에거부한다. Final checkpoint/SHA freeze후calibration GT를연다. Empty source/stale metric은0으로채우지않고None과유효population을보존한다. GT encoder support/predicted voxel SHA와TP/FP/FN, regional recall/stale/visible/BCE, first4 고정asset의Y voxel-projection GT/Pred/Diff를W&B에기록한다. 여기의macro asset/noise IoU·GT encoder support와본14개표의micro IoU·decoded-latent GT는같은표로직접합치지않는다. Evaluation batch32는같은cal32의native 설정으로고정한다.

자원은첫seed의두arm을GPU2/3에각각배치한다. 본14개구성의최종report가완료되면seed202는GPU0/1을사용하고,seed303은첫seed가끝난GPU2/3에서실행한다. 최종Full/PBR판단은본비교와이source-use/geometry 결과및필요한training-seed 확인을함께사용한다.

## 전체검사에서확인한target authority 차이

두번째train asset에서직접flexible64는8395개, 실제SS encoder입력은8400개였다. 추가5개는모두인접1voxel이내이고IoU0.99940476이다. 실제encoder입력의byte hash는SS latent의source_coords_sha256과exact다. 차이의상세root cause를확정한것은아니며한asset의100%일치를전체로확대하지않는다. Region label은이미학습한GT를지키도록실제encoder support를authority로사용한다. 두support와차이통계·GTfile SHA·frame contract를모두보존한다. V1의strict-equality실패출력은보존하고`deformnext-current-encoder-authority-regions64-v2` schema/새namespace로검증한다. 입력RGB·query·기존latent를고쳐PASS시키지않는다.

## Native1024→64의exact authority로복구

V2의24번째asset에서직접64는6382개, 실제encoder입력은6433개였고거리3cell 예외가확인됐다. 같은mesh/camera/scale로native1024 voxelization→pool64를수행하면6433개와정확히일치했다. 직접64 voxelization과고해상voxelization후pooling은같은연산이아니다. 이것을확인한뒤source와target 모두실제1024 pipeline으로맞췄다. V3는`deformnext-current-native1024-authority-regions64-v3`이며target set exact·SS source_coords hash exact를모두요구한다. 1/3cell 거리로허용범위를넓혀통과시키지않는다. V1/V2의실패와관측provenance는보존한다.

현재native condition64 cache도완료됐다. 실제4-view projection mean과target first projection을고정했고로딩이후15.4초였다. GT geometry/latent 값을condition 구축에열지않았다. V2 one-pair full regional gradient proof도초기native regional loss55.82744598 exact, 두모듈2update·frozen SS/decoder gradient없음을통과했다. Query encoder의두번째gradient L1은0.004031518이었다. 새native1024 label의V3 preflight와본64개구축은별도로판정한다.
