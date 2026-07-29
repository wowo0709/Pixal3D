# Node16 HSSD 3-source 학습 준비 및 실행 Runbook

이 문서는 Node16에서 ABO + 3D-FUTURE + HSSD의 검증된 결합 입력을 준비하고, 네
multi-view 모델을 **한 번에 하나씩** 여섯 GPU로 실행하는 절차다. 준비 단계는 CPU-only,
create-only이며 학습, 모델, CUDA context, W&B를 시작하지 않는다. 기존 ABO +
3D-FUTURE 두 source 증거와 launch는 [기존 전처리 runbook](data_preprocessing_runbook_ko.md)의
기록이며, 이 문서의 세 source 증거로 소급해서 부르지 않는다.

## 1. Node16 shell과 CPU-only 준비

아래 경로와 빈 `CUDA_VISIBLE_DEVICES`는 필수다. 준비 CLI는 변수가 없거나 비어 있지
않으면 Python 모듈을 import하기 전에 실패한다.

```bash
source /home/youngwoo/miniconda3/etc/profile.d/conda.sh
conda activate pixal3d
cd /home/youngwoo/Pixal3D-training-hssd
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""

python scripts/prepare_node16_training.py \
  --data2-root /file2/youngwoo/pixal3d \
  --local-root /home/youngwoo/data/pixal3d \
  --repo-root /home/youngwoo/Pixal3D-training-hssd

python scripts/prepare_node16_training.py \
  --data2-root /file2/youngwoo/pixal3d \
  --local-root /home/youngwoo/data/pixal3d \
  --repo-root /home/youngwoo/Pixal3D-training-hssd \
  --execute
```

첫 명령은 read-only plan이다. disk admission과 canonical root 계약을 확인하지만 local
production root를 만들지 않는다. 두 번째 명령만 실행한다. 실행은 다음 순서로 ABO,
3D-FUTURE, HSSD를 local root에 create-only로 materialize/publish하고, HSSD 단독
loader preflight, 세 source 결합 manifest publish, 세 source loader preflight, 최종
evidence report 작성을 수행한다.

기존 완전 artifact는 검증 후 재사용할 수 있지만 partial artifact, 다른 content의 runtime
config, 다른 content의 report는 자동으로 고치거나 덮어쓰지 않고 실패한다. 실패했거나
partial artifact를 발견하면 **절대 자동 삭제하지 않는다**. 우선 해당 절대 경로, 오류,
digest를 보존·보고하고 recovery 여부를 결정한다.

## 2. 준비 결과와 경계 검증

`--execute`가 성공하면 다음 report만 세 source 실행의 승인 증거다.

```bash
REPORT=/home/youngwoo/data/pixal3d/train/production/node16-preparation-evidence/report.json
HSSD_DATA=/home/youngwoo/data/pixal3d/train/production/hssd/training_data.json
TRAINING_DATA=/home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json

test -f "$REPORT"
test -f "$HSSD_DATA"
test -f "$TRAINING_DATA"
python -m json.tool "$REPORT"
sha256sum "$HSSD_DATA" "$TRAINING_DATA"
```

report에서 다음을 함께 점검한다.

- `cpu_only`가 `true`이고 `paths`의 `data2_root`, `local_root`, `repo_root`,
  `hssd_training_data`, `combined_training_data`가 위의 정확한 canonical path인지 확인한다.
- `sources.hssd`의 report/handoff/training-data digest와 네 stage의 asset count,
  `asset_scope_sha256`, eligibility exclusion count를 기록한다. HSSD boundary는
  `HSSD-00000`의 20 batch와 `HSSD-00001`의 7 batch(총 frozen 6,670), 각 stage의
  candidate 6,078이며 `production_gate`를 통과한 source만 허용한다. final active count는
  추정하지 않고 이 report의 실제 stage evidence를 사용한다.
- `hssd_standalone_preflight.stages` 네 항목이 HSSD-only configured Dataset/DataLoader
  검증 결과인지 확인한다. 이는 HSSD-only 검증이며 combined 검증을 대체하지 않는다.
- `combined.path`와 `combined.sha256`, 그리고 `combined.stages`의 source count,
  total count, `union_scope_sha256`를 기록한다. `combined.preflight.stages` 네 항목은
  ABO + 3D-FUTURE + HSSD three-source configured Dataset/DataLoader 검증 결과다.
- `runtime_configs`와 `launch_commands`가 네 stage 모두를 포함하는지 확인한다.

runtime config는 source config의 parsed JSON에서 `trainer.args.num_workers`만 `1`로
바꾼 create-only 복사본이다. 실행 전 정확한 네 파일과 이 값, report의 SHA-256을
대조한다.

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/home/youngwoo/data/pixal3d/runtime-configs")
for path in (
    root / "ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.node16-workers1.json",
    root / "slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.node16-workers1.json",
    root / "slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.node16-workers1.json",
    root / "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.node16-workers1.json",
):
    with path.open() as stream:
        config = json.load(stream)
    print(path, config["trainer"]["args"]["num_workers"])
PY
```

출력은 네 절대 path 각각에 대해 `1`이어야 한다. report의 `runtime_configs`는 또한
stage별 global batch (`48`, `48`, `12`, `12`), `max_steps=20000`,
`save_interval=2000`, retained checkpoints `5`, snapshots disabled를 기록한다.

## 3. GPU 선택, tmux, 로그

준비 명령의 빈 GPU visibility를 학습 명령에 재사용하지 않는다. `nvidia-smi`와 운영자
할당 기록으로 서로 비어 있는 physical GPU 여섯 개를 확인한 뒤에만 아래 placeholder를
그 여섯 ID로 바꾼다. GPU 0이나 임의의 기본 six-pack을 가정하지 않는다.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES="<operator-selected-id-1>,<operator-selected-id-2>,<operator-selected-id-3>,<operator-selected-id-4>,<operator-selected-id-5>,<operator-selected-id-6>"

LOG_DIR=/file3/youngwoo/pixal3d/logs/hssd-node16
mkdir -p "$LOG_DIR"
```

학습 output/checkpoint는 각 config의 기존 `/file3/youngwoo/pixal3d/ckpts/` path를
그대로 사용한다. 따라서 `--output_dir`나 `--load_dir`로 다른 path를 지정하지 않는다.
한 tmux session이 끝나고 checkpoint/output/W&B/GPU 상태를 확인한 뒤에만 다음 model
session을 시작한다.

## 4. 네 개의 one-at-a-time six-GPU launch

각 명령은 준비 report에서 검증한 combined manifest를 `--training_data`에 직접 넘기고,
해당 Node16 runtime config를 사용한다. `--num_gpus 6`은 visible GPU 여섯 개를 사용한다.

```bash
tmux new-session -d -s node16-ss64 \
  "source /home/youngwoo/miniconda3/etc/profile.d/conda.sh && conda activate pixal3d && cd /home/youngwoo/Pixal3D-training-hssd && PYTHONPATH=. CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" python train.py --config /home/youngwoo/data/pixal3d/runtime-configs/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.node16-workers1.json --training_data /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json --num_gpus 6 --use_wandb 2>&1 | tee -a \"$LOG_DIR/ss64.log\""

tmux new-session -d -s node16-shape512 \
  "source /home/youngwoo/miniconda3/etc/profile.d/conda.sh && conda activate pixal3d && cd /home/youngwoo/Pixal3D-training-hssd && PYTHONPATH=. CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" python train.py --config /home/youngwoo/data/pixal3d/runtime-configs/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.node16-workers1.json --training_data /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json --num_gpus 6 --use_wandb 2>&1 | tee -a \"$LOG_DIR/shape512.log\""

tmux new-session -d -s node16-shape1024 \
  "source /home/youngwoo/miniconda3/etc/profile.d/conda.sh && conda activate pixal3d && cd /home/youngwoo/Pixal3D-training-hssd && PYTHONPATH=. CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" python train.py --config /home/youngwoo/data/pixal3d/runtime-configs/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.node16-workers1.json --training_data /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json --num_gpus 6 --use_wandb 2>&1 | tee -a \"$LOG_DIR/shape1024.log\""

tmux new-session -d -s node16-pbr1024 \
  "source /home/youngwoo/miniconda3/etc/profile.d/conda.sh && conda activate pixal3d && cd /home/youngwoo/Pixal3D-training-hssd && PYTHONPATH=. CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" python train.py --config /home/youngwoo/data/pixal3d/runtime-configs/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.node16-workers1.json --training_data /home/youngwoo/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json --num_gpus 6 --use_wandb 2>&1 | tee -a \"$LOG_DIR/pbr1024.log\""
```

이 네 명령은 queue나 병렬 실행 명령이 아니다. 첫 `node16-ss64`만 시작하고 그 명령이
종료·검토된 뒤 `node16-shape512`, `node16-shape1024`, `node16-pbr1024` 순으로 하나씩
시작한다. config가 지정한 output은 각각 다음과 같다.

```text
/file3/youngwoo/pixal3d/ckpts/ss64
/file3/youngwoo/pixal3d/ckpts/shape512
/file3/youngwoo/pixal3d/ckpts/shape1024
/file3/youngwoo/pixal3d/ckpts/pbr1024
```

이 runtime copy는 `num_workers` 외에는 source config를 바꾸지 않으므로, 다음 기존
single-view finetune checkpoint path도 그대로 보존한다.

```text
/file3/youngwoo/pixal3d/train/checkpoints/single_view/ss_flow_img_dit_1_3B_64_bf16.pt
/file3/youngwoo/pixal3d/train/checkpoints/single_view/slat_flow_img2shape_dit_1_3B_512_bf16.pt
/file3/youngwoo/pixal3d/train/checkpoints/single_view/slat_flow_img2shape_dit_1_3B_1024_bf16.pt
/file3/youngwoo/pixal3d/train/checkpoints/single_view/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt
```

## 5. 실행 중 관측과 resume

현재 session과 log, GPU, W&B CLI 상태, checkpoint directory를 주기적으로 점검한다.
아래의 `SESSION`, `LOG`, `OUTPUT`은 현재 하나만 실행 중인 stage에 맞춘다.

```bash
SESSION=node16-ss64
LOG=/file3/youngwoo/pixal3d/logs/hssd-node16/ss64.log
OUTPUT=/file3/youngwoo/pixal3d/ckpts/ss64

tmux has-session -t "$SESSION"
tmux capture-pane -pt "$SESSION" -S -200
tail -n 200 "$LOG"
nvidia-smi
wandb status
find "$OUTPUT/ckpts" -maxdepth 1 -type f -name 'misc_*.pt' -printf '%f %s bytes\n' | sort
```

`train.py`는 `--load_dir`가 생략되면 resolved output directory를 load directory로 쓰고,
`--ckpt`가 생략되면 `latest`의 `ckpts/misc_*.pt`를 선택한다. 중단 뒤에는 output,
runtime config, combined manifest, GPU selection, W&B option을 바꾸지 말고 해당 stage의
**같은 tmux launch 명령**을 다시 실행한다. 이것이 checkpoint resume 명령이다.
`--ckpt none`으로 새로 시작하지 않는다. W&B console URL/run ID는 log와 운영 기록에
보존하고, 기존 run ID를 명시적으로 재사용해야 하는 경우에만 그 승인된 ID를
`--wandb_id`로 추가한다.

partial output, unexpected checkpoint, tmux 오류, W&B 오류, GPU contention이 있으면
다음 model을 시작하지 않는다. 관련 `/file3` output, log, report path, checkpoint filename,
`nvidia-smi` 결과를 보존·보고한 뒤 recovery 결정을 받는다. partial artifacts를 이
runbook의 명령으로 자동 제거하거나 덮어쓰지 않는다.
