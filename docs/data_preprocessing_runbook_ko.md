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

## 1. 시작 전 점검

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli preflight \
  --config data_toolkit/configs/multiview_preprocess.yaml

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli hardware-preflight \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --bootstrap-peak-local-gib 120
```

모든 source가 `ready`인지 확인한다. `not ready`가 있으면 해당 source만 보류하고 원인을 해결한다.

## 2. Smoke 범위 계획

Smoke는 source별 첫 9개 asset, 즉 3개 batch로 실행한다.

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

ABO는 최초 실행 전에 전체 `abo-3dmodels.tar` 약 154GB를 다운로드한다. 부분 다운로드가 이미 있으면 삭제하지 말고 위 `resume`을 사용한다. 전체 archive 다운로드를 허용할 충분한 용량과 시간을 먼저 확인한다.

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
- completed asset만 pack member인지
- quarantine asset이 pack에 들어가지 않았는지
- quality ledger의 category/stage/reason
- source 성공률이 90% 이상인지

## 6. Quarantine 정책

다음은 asset 자체 문제이므로 quarantine하고 이후 단계에서 제외한다.

- unsupported shader/format
- missing render transform 또는 필수 metadata
- provider에서 특정 asset을 더 이상 제공하지 않음

다음은 quarantine하지 않는다.

- 네트워크/인증 장애
- GPU, 디스크, RAM/resource 문제
- checkpoint, process-control, pipeline infrastructure 오류

quarantine ledger 확인:

```bash
jq '.quarantine' \
  /root/data2/pixal3d/control/qualification/smoke/quality/SOURCE/SOURCE-00000.json
```

## 7. Smoke 통과 후

source 성공률이 90% 미만이면 production으로 진행하지 말고 source를 `non-admitted`로 기록한다. 90% 이상인 source만 pilot으로 올린다.

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli run \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate pilot --source "$SOURCE" --shard "$SHARD"
```

pilot audit가 통과한 뒤에만 전체 production preprocessing을 검토한다.

## 7-1. 전체 데이터 다운로드/전처리

전체 처리는 smoke와 pilot audit가 통과한 source에 대해서만 실행한다. 전체 실행에서는 `--count`를 절대 사용하지 않는다.

```bash
SOURCE=3D-FUTURE
SHARD=3D-FUTURE-00000

conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli run \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate production --source "$SOURCE" --shard "$SHARD"
```

중단 후 전체 production을 재개할 때:

```bash
conda run --no-capture-output -n pixal3d \
  python -m data_toolkit.pipeline.cli resume \
  --config data_toolkit/configs/multiview_preprocess.yaml \
  --gate production --source "$SOURCE" --shard "$SHARD"
```

전체 처리 전에는 다음을 확인한다.

- data2/data3와 local에 필요한 여유 공간이 있는지
- source 전체 다운로드에 필요한 시간과 네트워크 사용량
- pilot에서 관측한 p95 처리시간과 GPU/RAM peak
- quarantine 예상량과 최종 training handoff asset 수

ABO는 production 실행 시에도 최초 단계에서 약 154GB `abo-3dmodels.tar` 전체 archive를 다운로드한다. 현재 ABO는 이 archive 다운로드를 시작하다가 중단된 상태이므로, 사용자가 전체 다운로드를 허용한 뒤 production 또는 resume을 실행해야 한다.

현재 source 상태상 전체 production을 아직 실행하지 않는 이유는 smoke/pilot gate를 통과하지 않은 source를 곧바로 대규모 처리하지 않기 위해서다. 이 조건을 충족한 source만 위 명령으로 전체 다운로드와 전처리를 진행한다.

## 8. 모델 구현 시점

데이터 단계에서 최소한 다음 조건을 만족한 뒤 모델 구현으로 이동한다.

- source별 smoke/pilot audit 완료
- 사용 가능한 asset만 training handoff에 포함
- quarantine ledger와 실패 category가 정리됨
- multi-view 입력 계약(K 가변, calibrated camera, first-view anchor)이 고정됨

모델은 기존 single-view Pixal3D 구조를 유지하면서 `3D alignment -> feature fusion` 순서로 확장한다.
