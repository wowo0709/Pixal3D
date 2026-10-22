# Native PBR의 실제 생성·렌더 정합과 고정32개8metric 후속

2026-10-22 UTC. 네PBR 모델의학습성능을입증한결과가아니다. 정상SS학습과
dispatcher/mirror는 유지했고 물리GPU0–3만 사용한다.

## 이번에 실제 통과한 검사

네모드의cold32 stage 검사가 완료되고 CPU gate가file4에 닫혔다.
별도로 원본PBR16850과각cold source를실제predicted support에서sampling·decode한
네모드도모두완료했다. 첫texture noise,texture latent,decoded coordinates/attributes가
vanilla와정확히같고,원래geometry8개array도bytes가같음을검사했다.
전처리조건을바꾸거나모듈을최종모델로채택한것은아니다.

새cold vanilla prediction의실제target8/source9/source10 렌더링·고정shared GT채점도
완료했다. 원래native predicted LOD를재사용했고,GT는원래material을M1 geometry에
적용한동일reference다. Actualcamera matrices와GT RGB/alpha 일치,3camera panel을검사했다.

첫asset의원본PBR targetPSNR은22.3986dB,source9는29.3518dB,source10은23.4691dB다.
GT/Pred/Diff panel에는특히target의material 밝기차이가보인다. TargetforegroundPSNR은
8.2076dB로globalPSNR과다르므로검사통과를시각적재구성성공이나fine-tuning 개선으로
해석하지않는다. 모델간품질차이는앞으로같은reference로비교한다.
Training config와pipeline의shape/texture32-channel normalization전체도정확히같다.

실측은 `PBR-native-generated-render-v1/cold-render-preview-v1/`의
`audit.json`, `GT-Pred-Diff.png`, `render/`, `shared/`와각file4 receipt에있다.

## 연결한 최종개발평가

네native1900 PBR을같은32개predicted geometry에서비교한다.
Vanilla,physicalTop-1+R,Top-1+feature+R,rawTop-K+feature+R이다.
각case의생성완료후동일LOD·camera8/9/10·originalmaterialM1 GT로렌더·채점한다.
128case와384view공식DreamSim/CLIP GPU평가,CPU paired집계를연결했다.

Target8은PSNR/SSIM/M-IoU,source9/10은LPIPS/DreamSim/CLIP으로읽는다.
CD/F-Score는같은geometry를고정한PBR isolation이므로조건간정확히같아야한다.
ForegroundPSNR과GT-onlycrop DreamSim/CLIP도보조값으로유지한다.
Official metric은기존같은registry/weights와FP32·TF32off·noAMP를사용한다.
W&B에역할별수치·source-cluster CI와네모드GT/Pred/Diff를기록하는후속을포함했다.

새CPU join은actualnative1900 NN/source pair,sampling seed,geometry/rawsurface metrics,
동일GT RGB/alpha,camera와공식metric대상pixel SHA,공식registry/precision,
384개고유view를검사한다. 관련CPU23개검사가통과했다.
차이는6개모델쌍에대해source-cluster bootstrap4000회로계산한다.
반복개발panel·trainingseed101 하나의비보정CI이므로independent seed와test508을대체하지않는다.

`PBR-native-generated-render-v1/{prelaunch-v1,queue-v1}`에
521개준비물12,714,451byte와실제131개후속descriptor를file4 SHA/size로닫았다.
이것은실제128case scoring이나최종모델선택이완료됐다는주장이아니다.

## 로컬저장공간과GPU사용순서

기존32개raw prediction archive합계는3.8358GiB다. 네모드가모두남으면약15.3431GiB로,
동시NN checkpoint예산과별도관리가필요하다. 다음조치를실행대기열에연결했다.

1. 각GPU쌍의마지막PBR science와own completed-cache 정리가끝난뒤그쌍의모델평가를
   시작한다. 평가가먼저한GPU를차지해다음DDP쌍의시작을막는것을피한다.
2. 모델별raw archive를두case window로제한한다. 다음case는두slot전case의실제
   렌더·shared score·canonical 검증·로컬중복정리가끝난뒤시작한다.
3. 렌더를추가생성보다먼저실행하고,렌더후의로컬prediction.npz/geometry-evaluation.npz만
   정리한다. 자체run과sample/mode,정상terminal,source/geometry불변,실제완료render,
   pending/running/futureconsumer와열린filehandle,clean committedworktree,
   **fresh 실제file4 SHA/size**를확인하고삭제전ledger도file4에닫는다.
4. Canonical file4의raw array,모든checkpoint·원본데이터·generation report·render pixel은
   유지한다. Summary의generation.json reader와rawshape reader를구분하되,
   알수없는새root consumer는보수적으로raw reader로보호한다.

관련CPU2개검사가통과했고128개CPU정리작업을추가했다. 실제로raw cache를지운것은
아직없다. 원래131개평가범위와모든command/plan/output을유지하며시작전dependency와
renderpriority만변경했다. 기록은 `local-raw-cache-scheduling-v1/`에있다.

별도로원래faithfultarget-patch/R96의세SS학습도유지한다. 처음두DDP쌍이완료되고
own unused192/288/384 checkpoint가검증·정리된뒤세번째DQ학습이시작되게했다.
부모wrapper에native하위결과를닫는원래receipt구조는새cachehelper version에서명시했다.
원래R96입력,모든final480과file4 checkpoint는유지한다. 이3개CPU후속도연결했다.
현재queue는3066개다. 모든기존state를성공으로바꾸거나정상학습을재시작하지않았다.

## 남은과학적판단

Vanilla와Top-1+R SS는실제480에완료했고,Top-1은final128×3평가중이다.
나머지두SS의최종480·128×3 비교,actualPBR micro12 capacity와1900학습,
새SS의fixedShape512/1024 cascade와SS/PBR fullstack,independentTRAINseed202,
모듈기여분리와final508/전체10metric(FID/KID포함)이남는다.
Matching-aware attention의추가이득이나ray-wiseQKV의필수성은여전히확정하지않는다.
