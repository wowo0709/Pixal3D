# Pixal3D 데이터 전처리 실행 순서

이 문서는 사용자가 직접 전처리를 실행할 때 사용하는 runbook이다.

## 0. 작업 위치와 환경

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-preprocess
conda activate pixal3d
export PYTHONPATH=.
```

필수 경로:

- 프로젝트: `/root/dev/Pixal3D`
- data2: `/root/data2/pixal3d`
- data3: `/root/data3/pixal3d`
- local preprocess: `/root/node17/data/pixal3d`

검증된 실행 환경(2026-07-18):

- PyTorch `2.8.0+cu128`
- PyTorch CUDA `12.8`
- CUDA 사용 가능, RTX PRO 6000 Blackwell GPU 7개

## 1. 시작 전 점검

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli preflight \
  --config data_toolkit/configs/multiview_preprocess.yaml

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli hardware-preflight \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --bootstrap-peak-local-gib 350

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli report \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --hardware-check
```

모든 source가 `ready`인지 확인한다. `not ready`가 있으면 해당 source만 보류하고 원인을 해결한다.

### 3D-FUTURE 해시 규칙

3D-FUTURE의 canonical `sha256`은 `image.jpg`의 SHA-256이며 asset identity로
사용한다. `raw_model.obj`의 해시로 canonical registry를 변경하면 안 된다.
다운로드 metadata의 `content_sha256`은 OBJ 자체를 검증하고,
`companion_files`는 `model.mtl`, texture 및 같은 asset directory의 동반 파일을
경로별 SHA-256으로 검증한다. staging과 raw archive에는 이 전체 파일 묶음이
포함되어야 한다.

Blender 4.x에서는 OBJ condition render가 `bpy.ops.wm.obj_import`를 사용해야
한다. 이 호환성 수정이 포함된 최초 commit은
`b52edb16847b30a96933f8a043a59611dc2e832f`이다.

### Frozen shard의 commit 규칙

이미 pack이나 checkpoint가 생성된 frozen shard는 해당 산출물을 만든 정확한
commit으로만 `resume`하고 `audit`한다. 현재 코드에서 과거 commit 문자열만
덮어써서 실행하면 안 된다. producing commit을 사용할 수 없으면 기존 제어 상태와
산출물을 먼저 백업한 뒤, 현재 commit으로 shard 전체를 처음부터 다시 만든다.

과거 commit audit 예시:

```bash
SOURCE=ObjaverseXL_github
SHARD=ObjaverseXL_github-00000
COMMIT=480999ac1b2f77e751bf46b596283f300a7eaac7
AUDIT_ROOT=$(mktemp -d /tmp/pixal3d-audit.XXXXXX)

git clone --shared --quiet --no-checkout \
  /root/dev/Pixal3D "$AUDIT_ROOT/repo"
git -C "$AUDIT_ROOT/repo" checkout --quiet --detach "$COMMIT"

cd "$AUDIT_ROOT/repo"
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source "$SOURCE" --shard "$SHARD"
```

### 2026-07-18 frozen smoke 복구 결과

- `ObjaverseXL_sketchfab`: producing commit `db605ec3c9bf1d1b79d93d785e18518459a0f472`,
  frozen 20개 중 completed 18개와 terminal failure 2개, exact-commit audit exit `0`.
- `ObjaverseXL_github`: producing commit `480999ac1b2f77e751bf46b596283f300a7eaac7`,
  7개 batch/20개 중 completed 12개와 quarantine 8개, exact-commit audit exit `0`.
  이 중 clone 불가는 `kaktu5/Nascar`와 `RetroJohn86/Pogo-APK`에 대해 GitHub가
  `Repository not found`를 반환한 경우다. 삭제·비공개·이름 변경 여부는 구분할 수
  없지만, 다른 저장소는 정상 처리되었으므로 전역 GitHub 인증 문제는 아니다.
- `3D-FUTURE`: producing commit `b52edb16847b30a96933f8a043a59611dc2e832f`,
  3개 batch/9개 모두 completed, quarantine 0개, audit exit `0`.

감사 로그와 백업 증거:

- `/root/data2/pixal3d/control/recovery/objaversexl-20260718-xI6nPr`
- `/root/data2/pixal3d/control/recovery/3d-future-20260718-e3Cv8D`
- `/root/data3/pixal3d/recovery/3d-future-20260718-e3Cv8D`

위 결과는 frozen smoke 범위에 대한 결과이며 전체 production 데이터 처리가 끝났다는
뜻은 아니다. 또한 위 수치는 family-scoped eligibility 도입 전 old-contract 결과다.
새 실행에서는 unsupported standard-PBR asset을 전역 quarantine으로 세지 않고 PBR
family에서만 제외하므로, 같은 asset을 재실행한 결과와 직접 비교하면 안 된다.

### 2026-07-18 current-contract smoke 결과

- frozen 범위: `3D-FUTURE` 9, `ABO` 9, `HSSD` 9,
  `ObjaverseXL_github` 20, `ObjaverseXL_sketchfab` 20, 총 67개.
- global completed: 65개. GitHub provider에서 사라진 2개만 durable quarantine.
- common/SS/shape 포함: 65개. PBR 포함: 60개. 나머지 PBR 5개는
  `unsupported_shader` family-only 제외이며 geometry family는 유지한다.
- schema failure 0, 전체 failure rate 2.99%, source별 최저 성공률 90%.
- hardware report: PyTorch `2.8.0+cu128`, CUDA `12.8`, OPTIX GPU 7개,
  CPU fallback 없음, decision `passed`.
- smoke report: decision `passed`, checksum 215개 검증/실패 0.
- smoke report SHA-256:
  `adda2930b80154f82eecd6ee06065cbe2103b7c81fdccc51dce6d33ebab8880d`.
- evidence manifest SHA-256:
  `aaf7d8c8cefd6d294403d6cb5ee8805ce1b9a567a3ce835d890a571549f56142`.
- hardware report SHA-256:
  `4d13dc2f3b6c40ca48408eba9746af0bd9523f035ddb5634a2d75cf7051a4a8a`.

current-contract recovery root:

- `/root/data2/pixal3d/control/recovery/hssd-current-contract-20260718T131131Z-74cee4e`
- `/root/data2/pixal3d/control/recovery/abo-current-contract-20260718T133134Z-74cee4e`
- `/root/data2/pixal3d/control/recovery/objaversexl-sketchfab-current-contract-20260718T135201Z-74cee4e`
- `/root/data3/pixal3d/recovery/hssd-current-contract-20260718T131131Z-74cee4e`
- `/root/data3/pixal3d/recovery/abo-current-contract-20260718T133134Z-74cee4e`
- `/root/data3/pixal3d/recovery/objaversexl-sketchfab-current-contract-20260718T135201Z-74cee4e`
- `/root/node17/data/pixal3d/recovery/hssd-current-contract-20260718T131131Z-74cee4e`
- `/root/node17/data/pixal3d/recovery/abo-current-contract-20260718T133134Z-74cee4e`
- `/root/node17/data/pixal3d/recovery/objaversexl-sketchfab-current-contract-20260718T135201Z-74cee4e`

## 2. Smoke 범위 계획

새 smoke는 source별 첫 9개 asset, 즉 3개 batch로 실행한다. 단, 이미 frozen된
ObjaverseXL historical smoke는 20개 범위를 그대로 유지하며 다시 계획하지 않는다.

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli plan \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source SOURCE --shard SOURCE-00000
```

전체 registry를 출력하므로, 실제 smoke 실행은 `--count 9`를 사용한다.

## 3. Source별 실행 순서

권장 순서는 다음과 같다.

1. 이미 로컬 archive가 준비된 `3D-FUTURE`
2. 이미 로컬 archive가 준비된 `Toys4k`
3. `HSSD`
4. `ABO`
5. `ObjaverseXL_sketchfab` 및 `ObjaverseXL_github`는 현재 결과를 유지하고 source gate에서 별도 판정

각 source의 최초 실행:

```bash
SOURCE=3D-FUTURE
SHARD=3D-FUTURE-00000

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli run \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source "$SOURCE" --shard "$SHARD" --count 9
```

다른 source도 `SOURCE`와 `SHARD`만 바꿔 동일하게 실행한다.

## 4. 중단 후 재개

실행 중 중단되었거나 resource stop이 발생하면 같은 frozen shard를 재개한다.

```bash
SOURCE=3D-FUTURE
SHARD=3D-FUTURE-00000

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli resume \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source "$SOURCE" --shard "$SHARD"
```

ABO의 전체 `abo-3dmodels.tar` 약 154GB는 이미 다운로드되어 있다. 이후 실행이
중단되더라도 archive를 삭제하지 말고 위 `resume`을 사용한다.

## 5. Source audit

source의 frozen smoke batches가 모두 완료된 뒤 audit한다.

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli audit \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate smoke --source "$SOURCE" --shard "$SHARD"
```

검사 대상:

- pack/raw archive manifest의 SHA와 tool commit
- manifest의 frozen `asset_sha256s`와 family별 exact
  `included_asset_sha256s`가 일치하는지
- `PBR-R`이 `shape-R`의 부분집합이고 `SS-64`가 `shape-1024`의
  부분집합이며 `common`이 non-common family 합집합인지
- 전역 quarantine asset이 모든 family에서 빠지고, family 제외 asset이 해당
  family에서만 빠지는지
- quality ledger의 `quarantine` 및 `family_exclusions`에
  category/stage/reason이 기록되었는지
- source 성공률이 90% 이상인지

## 6. 전역 quarantine과 family 제외 정책

다음은 usable training family가 남지 않는 asset 자체 문제이므로 전역
quarantine하고 이후 모든 family에서 제외한다.

- missing render transform 또는 필수 metadata
- provider에서 특정 asset을 더 이상 제공하지 않음
- 모든 shape/SS/PBR family가 자체 검증에 실패함

TRELLIS.2 공식 metallic-roughness PBR parser가 `Material is not supported`를
반환하는 경우는 전역 quarantine이 아니다. 해당 asset은 PBR family에서만
제외한다. shape와 SS는 각자의 validator를 통과하면 그대로 사용한다. shader
graph를 bake, rewrite 또는 자동 변환하지 않는다.

pack manifest schema 2는 frozen scope인 `asset_sha256s`와 실제 family 포함 범위인
`included_asset_sha256s`를 모두 기록한다. manifest의 `completed_count`와
`quarantined_count`는 그 family의 included/excluded 개수다. 특히 PBR pack의
excluded 개수를 전역 quarantine 개수로 해석하면 안 된다.

다음은 quarantine하지 않는다.

- 네트워크/인증 장애
- GPU, 디스크, RAM/resource 문제
- checkpoint, process-control, pipeline infrastructure 오류

전역 quarantine과 family 제외 ledger 확인:

```bash
LEDGER=/root/data2/pixal3d/control/qualification/smoke/quality/SOURCE/SOURCE-00000.json
conda run -n pixal3d python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({"quarantine":d.get("quarantine", {}), "family_exclusions":d.get("family_exclusions", {})}, indent=2, ensure_ascii=False))' \
  "$LEDGER"
```

## 7. Smoke evidence/report와 800개 pilot

모든 smoke source의 run/audit가 끝나면 다음 순서를 지킨다. FP32 설정에서는
`fp16.csv`가 header-only인 것이 정상이다.

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli evidence \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli report \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate smoke
```

smoke report가 `passed`일 때 현재 production source당 200개, 총 800개 pilot을 순차 실행한다.
`plan`은 preview이고, 실제 frozen scope는 첫 `run --count 200`이 만든다.

```bash
for SOURCE in \
  ObjaverseXL_sketchfab ABO HSSD 3D-FUTURE
do
  SHARD="${SOURCE}-00000"
  conda run --no-capture-output -n pixal3d \
    python -m data_toolkit.pipeline.cli plan \
    --config data_toolkit/configs/multiview_preprocess.yaml \
    --gate pilot --source "$SOURCE" --shard "$SHARD" --count 200
  conda run --no-capture-output -n pixal3d \
    python -m data_toolkit.pipeline.cli run \
    --config data_toolkit/configs/multiview_preprocess.yaml \
    --gate pilot --source "$SOURCE" --shard "$SHARD" --count 200
  conda run --no-capture-output -n pixal3d \
    python -m data_toolkit.pipeline.cli audit \
    --config data_toolkit/configs/multiview_preprocess.yaml \
    --gate pilot --source "$SOURCE" --shard "$SHARD"
done

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli evidence \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli report \
  --config data_toolkit/configs/multiview_preprocess.yaml --gate pilot
```

중단된 pilot source는 같은 `run --count 200` 명령 또는 `resume --gate pilot`로
동일 frozen scope를 재사용한다. source 성공률 90% 미만, schema failure 5% 초과,
capacity/checksum/hardware/audit 실패가 있으면 production을 시작하지 않는다.

## 7-1. 병렬 처리 성능 gate

production 전에는 기존 single-worker ABO 기준 `119.46 assets/hour` 대비 최소
`1.8x`, 즉 `215.03 assets/hour`를 64개 고정 범위에서 통과해야 한다. 벤치마크는
condition image를 기존 계약 그대로 asset당 8장, `512 x 512`로 렌더링하며,
aligned view `0/1`, `256/512/1024`, `SS-64`, FP32 latent를 변경하지 않는다.

먼저 범위와 출력 경로만 읽기 전용으로 확인한다.

```bash
cd /root/dev/Pixal3D/.worktrees/parallel-preprocessing

PYTHONPATH=. conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli benchmark-parallelism \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --source ABO --shard ABO-00000 --count 64 --dry-run
```

실제 벤치마크를 한 번 실행한다. 재시작된 측정은 정확한 처리율 증거로 인정하지
않으므로, 중단되면 checkpoint는 보존하되 원인을 해결한 뒤 새로운 held 범위를
사용한다.

```bash
PYTHONPATH=. conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli benchmark-parallelism \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --source ABO --shard ABO-00000 --count 64

PYTHONPATH=. conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli report \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --parallelism-check
```

증거는 다음 두 파일에 원자적으로 기록된다.

- `/root/data2/pixal3d/control/reports/parallelism.json`
- `/root/data2/pixal3d/control/reports/parallelism.md`

`decision`이 `passed`가 되려면 처리율과 1.8x 기준 외에도 audit 통과, GPU peak
90% 이하, GPU 평균 메모리 80% 이하, CPU 할당 44 physical core 이하, 유효한
7-GPU telemetry가 모두 필요하다. `held`이면 production을 시작하지 않는다.

production 전체 계획은 다음 읽기 전용 명령으로 확인한다. 각 publication batch는
최대 256개, 실행 chunk는 최대 64개이며 이 명령은 scope를 freeze하거나 worker를
실행하지 않는다.

```bash
PYTHONPATH=. conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli full-run \
  --config data_toolkit/configs/multiview_preprocess.yaml --dry-run
```

성능 저하나 resource stop이 나면 checkpoint, pack, raw archive를 삭제하지 않는다.
새 admission을 중단하고 마지막으로 audit된 worker profile로 되돌린 뒤 동일 frozen
scope를 `resume`한다.

### 두 노드 공용 work queue

두 노드는 source를 고정 배정하지 않는다. NFS의
`control/runtime/work_queue`에서 최대 256개인 frozen publication batch 하나를
원자적으로 claim하고, 그 batch 안의 최대 64개 chunk를 해당 노드 CPU/GPU 전체로
처리한다. 먼저 끝난 노드는 다른 source를 포함한 다음 남은 batch를 바로 가져간다.
stage 중간 파일은 각 노드 local scratch에 두고, 검증된 pack과 raw archive만 공유
data2/data3에 원자적으로 게시한다.

모든 노드는 동일한 Git commit, config hash, `pixal3d` conda 환경(CUDA 12.8,
PyTorch 2.8 이상), Blender 4.5.1/OptiX, canonical raw 접근 권한을 가져야 한다.
현재 등록값은 node17 CPU 44/GPU 1-6, node16 CPU 40/GPU 0-3이다. node17에서
공유 registry에 다음과 같이 등록한다. 비밀번호는 명령이나 로그에 저장하지 않는다.

```bash
CONFIG=data_toolkit/configs/multiview_preprocess.yaml

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli workers --config "$CONFIG" \
  --action register --node-id node17 --ssh-target local://node17 \
  --cpu-limit 44 --gpus 1,2,3,4,5,6 \
  --data2-root /root/data2/pixal3d --data3-root /root/data3/pixal3d \
  --local-root /root/node17/data/pixal3d

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli workers --config "$CONFIG" \
  --action register --node-id node16 \
  --ssh-target youngwoo@n16.unist.info:55555 \
  --cpu-limit 40 --gpus 0,1,2,3 \
  --data2-root /file2/youngwoo/pixal3d \
  --data3-root /file3/youngwoo/pixal3d \
  --local-root /home/youngwoo/data/pixal3d
```

worker는 queue lease를 잡기 전에 PyTorch/CUDA 버전, 등록 GPU, Blender 실행 파일,
네 source adapter와 `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`를 모두
검사한다. 하나라도 없으면 asset 실패로 잘못 기록하지 않고 worker 시작 자체가
실패한다. node16 컨테이너는 CUDA 12.8 및 Ubuntu 22.04 이상이어야 한다. 저장소의
`docker/production-worker-cu128.Dockerfile`이 재현 가능한 기준 이미지다. 기본 Python
의존성을 설치한 다음 TRELLIS.2와 동일하게 CuMesh, FlexGEMM, O-Voxel을 현재
PyTorch 2.8/CUDA 12.8 환경에서 빌드하고, 다음 점검이 모두 성공해야 등록한다.

```bash
python -c 'import torch, cumesh, flex_gemm, o_voxel, nvdiffrast; assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 8); assert torch.version.cuda == "12.8"; assert torch.cuda.is_available()'
$LOCAL_PATH/tools/blender-4.5.1-linux-x64/blender --version
```

같은 `node-id`/`local-root`에 worker를 두 번 실행하면 두 번째 프로세스는 비차단
파일 잠금에서 즉시 실패한다. 비정상 종료 시 커널이 잠금을 자동 해제하므로 별도
lock 파일 삭제는 하지 않는다.

기존 single-process `full-run`을 종료하고 미완료 child가 없음을 확인한 뒤 큐를 딱
한 번 초기화한다. 이 명령은 네 source의 모든 production batch를 freeze하고, 기존
pack과 raw archive가 모두 audit되는 완료 batch만 `completed`로 채택한다.

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli queue --config "$CONFIG" --action init
```

production에서는 단발 `worker`를 직접 상주시켜 두지 않고 `supervisor`를 실행한다.
supervisor는 child worker가 비정상 종료해도 registry가 `active`이고 queue가 남아
있으면 10초 뒤 다시 실행한다. `draining` 동안에는 새 worker를 실행하지 않고
대기하며, 같은 node-id를 `activate`하면 자동으로 이어서 실행한다.

```bash
# node17
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli supervisor --config "$CONFIG" \
  --node-id node17

# node16
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli supervisor --config "$CONFIG" \
  --node-id node16 \
  --worker-registry /root/data2/pixal3d/control/runtime/workers.json
```

실시간 상태에는 node별 현재 source/shard/batch, attempt, stage, 마지막 heartbeat와
전체 pending/running/completed/failed 수가 표시된다.

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli queue --config "$CONFIG" --action status
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli workers --config "$CONFIG" --action status
```

노드를 빼려면 `drain`한다. 진행 중 batch는 끝내되 새 batch를 claim하지 않고 child
worker가 종료하며 supervisor는 대기한다. 잠시 뒤 다시 쓸 노드는 `activate`만 하면
되고, 완전히 뺄 노드는 `remove`한다. 새 노드는 `register` 후 supervisor process를
실행하면 즉시 다음 batch부터 참여한다. CPU/GPU 구성을 바꿀 때는 `drain -> child
worker 종료 확인 -> 동일 node-id register -> activate` 순서를 쓴다. 다른 노드는
멈추지 않는다.

```bash
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli \
  workers --config "$CONFIG" --action drain --node-id node16
conda run --no-capture-output -n pixal3d python -m data_toolkit.pipeline.cli \
  workers --config "$CONFIG" --action remove --node-id node16
```

worker가 비정상 종료되면 마지막 heartbeat 5분 뒤 해당 미완료 batch만 다른 노드가
처음부터 재시작한다. 완료 marker는 보존된다. infrastructure 실패는 batch별 최대
3회이며, 3회째에도 실패한 batch만 `failed`로 격리되고 나머지 큐는 계속 진행한다.

## 7-2. 전체 데이터 다운로드/전처리

전체 처리는 smoke와 pilot report가 모두 `passed`인 경우에만 실행한다. 전체
실행에서는 `--count`를 절대 사용하지 않는다. 권장 실행은 위의 공용 queue와
노드별 worker다. 아래 `full-run`은 단일 노드 호환 및 수동 복구용이며 분산 worker와
동시에 실행하면 안 된다.

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli full-run \
  --config data_toolkit/configs/multiview_preprocess.yaml
```

중단 후에도 위의 동일한 `full-run` 명령을 다시 실행한다. runner는
`ABO -> HSSD -> 3D-FUTURE -> ObjaverseXL_sketchfab`
순서로 각 canonical shard에 대해 `run(production, count=None)` 직후 audit한다.
이미 frozen/completed된 shard는 재검증하고, 실패한 shard보다 뒤의 작업은 예약하지
않는다.

특정 shard를 수동 복구해야 할 때만 다음 명령을 사용한다.

```bash
SOURCE=3D-FUTURE
SHARD=3D-FUTURE-00000

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli resume \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate production --source "$SOURCE" --shard "$SHARD"
```

전체 처리 전에는 다음을 확인한다.

- data2/data3와 local에 필요한 여유 공간이 있는지
- source 전체 다운로드에 필요한 시간과 네트워크 사용량
- pilot에서 관측한 p95 처리시간과 GPU/RAM peak
- 전역 quarantine 예상량, family별 included/excluded 수, 최종 training handoff
  asset 수

ABO의 약 154GB `abo-3dmodels.tar` 전체 archive는
`/root/data2/pixal3d/raw/ABO/raw/abo-3dmodels.tar`에 다운로드가 완료되어 있다.
ABO frozen smoke 9개도 completed 상태이므로 production에서 이 archive를 재사용한다.

full runner는 source별 다운로드를 필요할 때 시작한다. ABO는 기존 tar를 재사용하며,
ObjaverseXL sketchfab/HSSD/3D-FUTURE는 canonical registry reference에 따라
다운로드와 전처리를 같은 shard 실행 안에서 이어간다.

ObjaverseXL GitHub source는 repository 단위 크기와 다운로드 지연 문제로 현재
production source에서 제외한다. 별도 registry와 download 정책을 확정하기 전에는
현재 4-source queue에 다시 넣지 않는다.

## 7-3. ABO 승인 valid-subset materialization 및 handoff

ABO `ABO-00000`의 원래 source-level 90% production gate는 통과하지 않았다. 그러나
사용자는 검증된 valid subset의 사용을 승인했다. 이 예외의 고정 수치는 frozen 4,485,
global quarantine 825, shape-512 추가 family exclusion 29이다. pack 교집합 후보 수는
`ss64=3,660`, `shape512=3,631`, `shape1024=3,660`, `pbr1024=3,660`이고, training
eligibility 적용 뒤 최종 수는 `ss64=3,660`, `shape512=3,628`, `shape1024=3,634`,
`pbr1024=3,598`이다. training 제외 수는 각각 `0/3/26/62`이며, 이 수는 global
quarantine 또는 pack-family exclusion과 합치거나 바꾸지 않는다. 따라서 이 handoff는
`valid_subset_user_waiver`를 명시하며 원래 90% gate가 통과했다는 뜻이 아니다.

Training eligibility는 모델 설정을 바꾸지 않는다. Shape-512 token 한도는 8,192,
Shape-1024 및 PBR-1024 token 한도는 32,768이며 두 anchor 모두 검사한다. PBR/Shape
coordinate는 정확히 동일해야 하고, finite positive float32 `total_scale`만
`rtol=0`, `atol=2e-7`까지 허용한다. 제외 이유는 다음 고정 식별자만 사용한다.

- `shape_tokens_view00_exceed_<limit>`, `shape_tokens_view01_exceed_<limit>`
- `pbr_tokens_view00_exceed_32768`, `pbr_tokens_view01_exceed_32768`
- `pbr_shape_coords_view00_mismatch`, `pbr_shape_coords_view01_mismatch`
- `pbr_shape_scale_view00_mismatch`, `pbr_shape_scale_view01_mismatch`

입력은 다음의 변경 불가 production index와 pack만 사용한다.

```text
/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json
/root/data2/pixal3d/prepared/{common,ss/64,shape/512,shape/1024,pbr/1024}/ABO/ABO-00000/
```

출력 stage는 서로 격리된
`/root/node17/data/pixal3d/train/production/abo/{ss64,shape512,shape1024,pbr1024}/active`
이며, 기존 `active` root는 절대로 덮어쓰지 않는다. 다음 두 명령은 낮은 CPU/I/O
우선순위로 실행하고 GPU를 사용하지 않는다.

실행 전에는 ABO publisher 또는 materializer/preflight가 active 상태가 아닌지 확인하고,
`/root/node17/data/pixal3d/train/production`에 최소 70 GiB의 여유 공간이 있는지
확인한다. 현재 실패한 두 번째 시도는 같은 filesystem의
`/root/node17/data/pixal3d/train/production/rejected/` 아래 고유 child로 **move**하여
보존한다. `rm`, 다른 filesystem으로의 copy, 또는 기존 `active`/shared 문서의 덮어쓰기는
금지한다. shared report/handoff/local manifest가 이미 있으면 먼저 중단하고 operator가
byte digest와 recovery 상태를 확인한다.

다음 recovery block은 위 failed current attempt에만 한 번 사용한다. 세 publication
artifact가 모두 없고 source가 정확한 ABO production root일 때만 같은 filesystem 안에서
rename한다. 어느 guard라도 실패하면 중단하고 아무 것도 삭제하지 않는다.

```bash
set -euo pipefail

SOURCE=/root/node17/data/pixal3d/train/production/abo
REJECTED=/root/node17/data/pixal3d/train/production/rejected
REPORT=/root/data2/pixal3d/control/reports/gates/ABO/ABO-00000-valid-subset.json
HANDOFF=/root/data2/pixal3d/control/splits/ABO/ABO-00000-valid-subset-handoff.json
LOCAL_MANIFEST=/root/node17/data/pixal3d/train/production/abo/training_data.json

for artifact in "$REPORT" "$HANDOFF" "$LOCAL_MANIFEST"; do
  if [ -e "$artifact" ] || [ -L "$artifact" ]; then
    echo "refusing recovery: published artifact exists: $artifact" >&2
    exit 1
  fi
done
if [ ! -d "$SOURCE" ] || [ -L "$SOURCE" ]; then
  echo "refusing recovery: source is not the expected real directory: $SOURCE" >&2
  exit 1
fi
mkdir -p -- "$REJECTED"
if [ -L "$REJECTED" ]; then
  echo "refusing recovery: rejected root must not be a symlink: $REJECTED" >&2
  exit 1
fi
if [ "$(stat -c %d -- "$SOURCE")" != "$(stat -c %d -- "$REJECTED")" ]; then
  echo "refusing recovery: source and rejected root are on different filesystems" >&2
  exit 1
fi

TARGET="$REJECTED/abo-rejected-$(date -u +%Y%m%dT%H%M%SZ)-$$"
while [ -e "$TARGET" ] || [ -L "$TARGET" ]; do
  TARGET="$REJECTED/abo-rejected-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
done
printf 'guarded rename: %s -> %s\n' "$SOURCE" "$TARGET"

# --no-clobber prevents replacing a target created after the uniqueness check.
mv -T -n -- "$SOURCE" "$TARGET"
if [ -e "$SOURCE" ] || [ -L "$SOURCE" ] || [ ! -d "$TARGET" ] || [ -L "$TARGET" ]; then
  echo "recovery rename did not produce the expected source/target state" >&2
  exit 1
fi
printf 'preserved failed attempt at %s\n' "$TARGET"
```

위 block 성공 후에도 `SOURCE`가 없고 출력된 `TARGET`만 존재하는지 다시 확인한 뒤 아래
process/free-space check와 CPU-only materializer/preflight 명령으로 진행한다.

```bash
pgrep -af 'materialize_multiview_production|preflight_multiview_production' || true
df -BG /root/node17/data/pixal3d/train/production
```

```bash
CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/materialize_multiview_production.py

CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py
```

엄격한 네 stage preflight가 모두 통과한 뒤에만 다음 immutable shared 문서와 local
convenience manifest가 생성된다.

```text
/root/data2/pixal3d/control/reports/gates/ABO/ABO-00000-valid-subset.json
/root/data2/pixal3d/control/splits/ABO/ABO-00000-valid-subset-handoff.json
/root/node17/data/pixal3d/train/production/abo/training_data.json
```

공유 report/handoff는 create-only다. 이미 존재하면 byte-identical 내용만 재실행으로
허용하며, 다른 내용은 덮어쓰지 않고 실패한다. local `training_data.json`은 두 공유
문서가 모두 성공한 뒤에만 원자적으로 쓴다. 이 명령들은 training을 시작하지 않으며
training-input 사용만 승인한다. 실패 시 명령으로 기존 `active`나 문서를 삭제하지 말고,
operator가 evidence, digest, stage scope를 먼저 점검한 뒤 복구 방법을 결정한다.
네 final scope의 structural/direct-loader 검증, report/handoff/training-data cross-digest,
그리고 CUDA context 또는 training process가 생성되지 않았음을 확인한 뒤에도 여기서
멈춘다. fine-tuning 실행 명령은 이 runbook의 이 단계에 포함하지 않는다.

## 7-4. 3D-FUTURE 및 ABO 결합 training publication

이 절은 operator용 절차와 2026-07-28에 완료한 실제 publication 측정값을 함께 기록한다.
3D-FUTURE는 두 frozen index의 모든 stage-eligible asset을 ABO와 함께 사용하며 내부
train/validation/test split이나 source weight를 만들지 않는다. 결합 sampling 표식은
`proportional-unweighted-concatenation`이다.

먼저 다른 materializer, preflight, training이 실행 중이지 않고 production filesystem에
최소 150 GiB가 남아 있는지 읽기 전용으로 확인한다. 다음 create-only 3D-FUTURE
artifact가 하나라도 있으면 새 실행을 시작하지 말고 아래 digest inspection으로 이동한다.

```bash
pgrep -af 'materialize_multiview_production|preflight_multiview_production|preflight_multisource_training|python train.py' || true
df -h /root/node17/data/pixal3d/train/production /root/data2 /root/data3
test ! -e /root/node17/data/pixal3d/train/production/3d-future
test ! -e /root/data2/pixal3d/control/reports/gates/3D-FUTURE/3D-FUTURE-production-training.json
test ! -e /root/data2/pixal3d/control/splits/3D-FUTURE/3D-FUTURE-production-training-handoff.json
```

3D-FUTURE materialization과 source preflight는 CUDA를 숨기고 낮은 CPU/I/O 우선순위로
실행한다. 두 명령이 모두 성공해야 네 `active` root, source report/handoff, local
`training_data.json`이 완성된다.

```bash
CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/materialize_multiview_production.py --profile 3d-future

CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py --profile 3d-future
```

중간 실패 또는 preflight 거부가 발생하면 정확한 failed attempt를 삭제하거나 성공한
ABO evidence를 바꾸지 않는다. 3D-FUTURE shared report/handoff/local manifest가 모두
없는 것을 확인한 경우에만 exact
`/root/node17/data/pixal3d/train/production/3d-future` directory를 같은 filesystem의
`production/rejected` 아래 고유 timestamp child로 `mv -T -n`하여 보존한다. symlink,
다른 filesystem, 기존 target, 또는 publication artifact가 있으면 중단한다. 결합
manifest 검증이 실패했다면 그 파일도 덮어쓰기 전에 digest를 기록하고, 같은 방식으로
결합 directory 전체를 고유 rejected child로 rename한 뒤 원인을 조사한다. `rm`,
cross-filesystem copy, shared evidence 덮어쓰기는 recovery가 아니다.

다음 guarded recovery block은 publication artifact가 전혀 없는 rejected
3D-FUTURE attempt에만 사용한다. 어느 guard라도 실패하면 아무 것도 이동하지 않는다.

```bash
set -euo pipefail

SOURCE=/root/node17/data/pixal3d/train/production/3d-future
REJECTED=/root/node17/data/pixal3d/train/production/rejected
REPORT=/root/data2/pixal3d/control/reports/gates/3D-FUTURE/3D-FUTURE-production-training.json
HANDOFF=/root/data2/pixal3d/control/splits/3D-FUTURE/3D-FUTURE-production-training-handoff.json
LOCAL_MANIFEST="$SOURCE/training_data.json"

for artifact in "$REPORT" "$HANDOFF" "$LOCAL_MANIFEST"; do
  if [ -e "$artifact" ] || [ -L "$artifact" ]; then
    echo "refusing recovery: publication artifact exists: $artifact" >&2
    exit 1
  fi
done
if [ ! -d "$SOURCE" ] || [ -L "$SOURCE" ]; then
  echo "refusing recovery: source is not the exact real directory" >&2
  exit 1
fi
mkdir -p -- "$REJECTED"
if [ -L "$REJECTED" ]; then
  echo "refusing recovery: rejected root is a symlink" >&2
  exit 1
fi
if [ "$(stat -c %d -- "$SOURCE")" != "$(stat -c %d -- "$REJECTED")" ]; then
  echo "refusing recovery: different filesystems" >&2
  exit 1
fi

TARGET="$REJECTED/3d-future-rejected-$(date -u +%Y%m%dT%H%M%SZ)-$$"
while [ -e "$TARGET" ] || [ -L "$TARGET" ]; do
  TARGET="$REJECTED/3d-future-rejected-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
done
mv -T -n -- "$SOURCE" "$TARGET"
if [ -e "$SOURCE" ] || [ -L "$SOURCE" ] || [ ! -d "$TARGET" ] || [ -L "$TARGET" ]; then
  echo "recovery rename did not produce the expected state" >&2
  exit 1
fi
printf 'preserved rejected attempt at %s\n' "$TARGET"
```

source publication 뒤에는 두 source의 기존 artifact를 쓰기 없이 다시 검증하고 digest를
기록한다.

```bash
conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py \
  --profile abo --verify-existing

conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multiview_production.py \
  --profile 3d-future --verify-existing

sha256sum \
  /root/node17/data/pixal3d/train/production/abo/training_data.json \
  /root/node17/data/pixal3d/train/production/3d-future/training_data.json
```

두 source digest가 승인된 evidence와 일치할 때만 local combined manifest를 원자적으로
publication한다. 이어지는 verification은 CUDA/model/trainer/W&B를 초기화하지 않고 네
configured Dataset의 exact disjoint union, source count, boundary instance direct load,
실제 cross-source `collate_fn`을 zero-worker DataLoader로 검사한다.

```bash
CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/publish_multisource_training.py

TRAINING_DATA=/root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json
CUDA_VISIBLE_DEVICES="" nice -n 15 ionice -c 2 -n 7 \
  conda run --no-capture-output -n pixal3d \
  python scripts/preflight_multisource_training.py \
  --training-data "$TRAINING_DATA"

sha256sum "$TRAINING_DATA"
```

### 2026-07-28 production 측정값

3D-FUTURE source materialization과 strict source preflight를 CPU-only로 완료했다. candidate,
training exclusion, 최종 active asset 수는 다음과 같다.

| stage | candidate | training exclusion | final active |
| --- | ---: | ---: | ---: |
| `ss64` | 8,495 | 0 | 8,495 |
| `shape512` | 8,513 | 11 | 8,502 |
| `shape1024` | 8,495 | 37 | 8,458 |
| `pbr1024` | 8,495 | 89 | 8,406 |

각 `active/materialization.json`의 SHA-256은 다음과 같다.

- `ss64`: `ead55bb04e82f3c4767197895ba97de15b77fce1f1dd19469d1cd95563ffdbc5`
- `shape512`: `e7acff6325cdebd4660754a623251826d00f33939d3ed2cf1067bf3822dc2f4d`
- `shape1024`: `0c0b7ee8665655a74ea32523e1a8cc3bbfabc2519379f59edb255a02d1dbc1c3`
- `pbr1024`: `cca3db6935028421ebb3c19e734a5f21dccfe60a57cfaf7403af2f894d87e282`

source publication artifact와 SHA-256은 다음과 같다.

- report
  `/root/data2/pixal3d/control/reports/gates/3D-FUTURE/3D-FUTURE-production-training.json`:
  `a6d62c367eca0b4bf9306533a9cc4f175234eedf06dd97dcec986759aef21f03`
- handoff
  `/root/data2/pixal3d/control/splits/3D-FUTURE/3D-FUTURE-production-training-handoff.json`:
  `0c36ef981b4426b408323910c13e8faf9b630b7c0c21d97512722552cd72059c`
- local training data
  `/root/node17/data/pixal3d/train/production/3d-future/training_data.json`:
  `88b2ccffe41bad2fc83c543ca2acbf8c490e08a0f7a698145d260493b612d6a4`

ABO와 결합한 create-only manifest
`/root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json`의
SHA-256은 `1ee3ed0c54cfaf3597f4cd2f9ce25a9ea16d6fb1dbb6b228efa0597474d85a13`이다.
결합 final count는 `ss64` 12,155, `shape512` 12,130, `shape1024` 12,092,
`pbr1024` 12,004이다. 네 configured Dataset은 각각 ABO와 3D-FUTURE의 first/last
boundary instance 네 개를 direct-load했고, zero-worker DataLoader가 두 source를 실제
`collate_fn`으로 함께 묶는 검증을 exit `0`으로 완료했다. 이 측정 중 CUDA context,
training process, W&B run은 시작하지 않았다.

combined verification이 exit 0으로 끝난 뒤에만 training을 고려한다. operator는 먼저
`nvidia-smi`와 운영자별 할당 기록으로 GPU ownership 및 여유 memory를 확인하고, 사용할
여섯 physical GPU ID를 직접 선택해 `CUDA_VISIBLE_DEVICES`에 설정해야 한다.
`CUDA_VISIBLE_DEVICES=0,...` 같은 기본값은 없으며 GPU 0을 가정하지 않는다.

```bash
# 아래 placeholder를 실제로 확인한 여섯 physical GPU ID로 바꾼 뒤에만 export한다.
export CUDA_VISIBLE_DEVICES="<operator-selected-id-1>,<operator-selected-id-2>,<operator-selected-id-3>,<operator-selected-id-4>,<operator-selected-id-5>,<operator-selected-id-6>"

TRAINING_DATA=/root/node17/data/pixal3d/train/production/abo-3d-future/training_data.json

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb

conda run --no-capture-output -n pixal3d python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json \
  --training_data "$TRAINING_DATA" --num_gpus 6 --use_wandb
```

네 명령은 독립 launch다. 한 명령의 checkpoint/output/W&B 상태와 GPU allocation을
operator가 확인하고 종료 또는 승인한 뒤 다음 stage를 시작한다. 이 절의 preflight나
publication 명령 자체는 training 시작 권한이 아니다.

### HSSD completed-source boundary와 Node16 3-source 준비

위 절의 2026-07-28 측정값과 launch는 ABO + 3D-FUTURE **두 source** 증거다. HSSD를
추가했다고 해서 기존 final count, digest, 또는 loader 결과를 세 source 결과로 다시
표기하지 않는다.

HSSD의 Node16 source boundary는 `HSSD-00000` 20 batch와 `HSSD-00001` 7 batch, 총
frozen 6,670 asset이다. 네 stage의 candidate contract는 각각 6,078이고, HSSD는
`production_gate`를 통과한 경우에만 training input으로 허용한다. final active count와
digest는 과거 두-source 기록에서 추정하지 않고 Node16 preparation report의 실제 evidence로
확인한다.

Node16에서는 공유 `/file2/youngwoo/pixal3d`를 source로 읽고 local
`/home/youngwoo/data/pixal3d`에 ABO, 3D-FUTURE, HSSD를 create-only로 준비한다. driver는
HSSD-only configured Dataset/DataLoader preflight와 three-source combined preflight를
모두 성공시킨 뒤 report를 쓴다. 정확한 CPU-only 준비 command, report/digest/runtime-config
검사, one-at-a-time six-GPU launch, tmux monitoring, same-command resume, partial-artifact
보존 규칙은 [Node16 HSSD 3-source 학습 runbook](hssd_node16_training_runbook_ko.md)을
따른다. 이 준비 workflow는 training을 자동 시작하지 않는다.

## 8. 모델 구현 시점

데이터 단계에서 최소한 다음 조건을 만족한 뒤 모델 구현으로 이동한다.

- source별 smoke/pilot audit 완료
- 사용 가능한 asset만 training handoff에 포함
- 전역 quarantine 및 family exclusion ledger와 실패 category가 정리됨
- multi-view 입력 계약(K 가변, calibrated camera, first-view anchor)이 고정됨

모델은 기존 single-view Pixal3D 구조를 유지하면서 `3D alignment -> feature fusion` 순서로 확장한다.
