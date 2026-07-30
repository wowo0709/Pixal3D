# Node17 3개 소스 학습 준비 및 실행 런북

이 절차는 Node17에서 ABO, 3D-FUTURE, HSSD의 검증된 학습
publication을 결합하고, 네 stage의 설정 기반 CPU DataLoader 검증 및
불변 readiness report 생성을 수행한다. 준비 실행은 학습을 시작하지
않으며 W&B에 접속하지 않는다. ABO와 3D-FUTURE는 기존 불변
publication만 검증하고 다시 materialize하거나 전체 population
preflight를 수행하지 않는다.

모든 준비 명령은 깨끗한 Git worktree에서 실행해야 한다.

## 1. 쓰기 없는 계획 확인

다음 명령은 root, 현재 Git HEAD, 원본 및 기존 runtime config, 최소
여유 공간만 확인한다. 디렉터리 생성, Node16 접속, HSSD 전송,
publication, CUDA import, 학습 또는 W&B 호출을 하지 않는다.

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py \
  --data2-root /root/data2/pixal3d \
  --local-root /root/node17/data/pixal3d \
  --repo-root /root/dev/Pixal3D/.worktrees/multiview-model-extension \
  --source-host youngwoo@n16.unist.info \
  --source-port 55555 \
  --source-root /home/youngwoo/data/pixal3d/train/production/hssd
```

출력 JSON에서 `"execute": false`, 40자리 `revision`, disk admission,
예상 runtime/evidence/combined 경로를 확인한다.

## 2. 준비 실행

Node16의 완료된 HSSD만 resume 가능한 staging으로 전송하고 검증한
뒤 canonical HSSD publication으로 승격한다. 이어서 ABO와
3D-FUTURE의 불변 chain을 검증하고, HSSD 단독 및 3개 소스 결합
DataLoader preflight를 수행한다.

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py \
  --data2-root /root/data2/pixal3d \
  --local-root /root/node17/data/pixal3d \
  --repo-root /root/dev/Pixal3D/.worktrees/multiview-model-extension \
  --source-host youngwoo@n16.unist.info \
  --source-port 55555 \
  --source-root /home/youngwoo/data/pixal3d/train/production/hssd \
  --execute
```

성공 출력의 report 경로는 다음과 같다.

```text
/root/node17/data/pixal3d/train/production/node17-preparation-evidence/report.json
```

## 3. 중단 후 재개

전송이나 검증이 중단되면 부분 파일을 직접 이동하거나 수정하지
않는다. 같은 깨끗한 revision에서 다음의 동일한 create-only 명령을
다시 실행한다. 안전한 HSSD staging은 resume하고, 이미 완료된
동일 publication/runtime config/report는 모든 불변 조건을 다시
검증한 뒤 inode를 교체하지 않고 재사용한다.

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py \
  --data2-root /root/data2/pixal3d \
  --local-root /root/node17/data/pixal3d \
  --repo-root /root/dev/Pixal3D/.worktrees/multiview-model-extension \
  --source-host youngwoo@n16.unist.info \
  --source-port 55555 \
  --source-root /home/youngwoo/data/pixal3d/train/production/hssd \
  --execute
```

partial combined 또는 evidence 디렉터리 오류가 발생하면 자동
복구하지 말고 해당 경로를 보존하여 운영자가 내용을 조사한다.

## 4. 불변 evidence 재검증

다음 명령은 report가 참조하는 runtime/source/combined artifact를
다시 hash하고, 세 source chain과 네 combined stage를 다시
resolve한다. DataLoader preflight나 전송을 다시 실행하지 않는다.

이 검증은 세 revision을 명시적으로 구분한다.

- evidence revision
  `34949b1422d4799eef68ab757430051cbf8da604`: 준비 실행 및 report가
  기록한 revision
- delivery revision
  `c3423fd23340b301940cd9c6ea084744caa77bd6`: readiness report만
  추가한 최초 전달 revision
- validator revision
  `cdc13bae59fcdbf3b7eab052a94f4f62e5995300`: historical validator
  구현과 테스트를 고정한 revision

검증기는 clean worktree, 세 commit의 존재 및 ancestry를 확인한다.
evidence에서 delivery까지는 이 Task의 readiness/runbook 문서만,
delivery에서 validator까지는 고정된 validator core/CLI/test 세
경로만, validator에서 현재 HEAD까지는 readiness/runbook 문서 두
경로만 허용한다. 다른 문서, 학습 코드, config, test 변경은
거부한다. 일반 plan/execute 및 기존 evidence validator의
exact-current-revision 규칙은 그대로 유지된다.

```bash
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/opt/conda/envs/pixal3d/bin/python scripts/prepare_node17_training.py \
  --validate-evidence \
  --delivery-revision c3423fd23340b301940cd9c6ea084744caa77bd6 \
  --validator-revision cdc13bae59fcdbf3b7eab052a94f4f62e5995300

sha256sum \
  /root/node17/data/pixal3d/train/production/node17-preparation-evidence/report.json
```

## 5. 네 학습 stage 실행

먼저 공통 launch 환경을 준비한다.

```bash
mkdir -p /root/node17/data/pixal3d/training-logs
cd /root/dev/Pixal3D/.worktrees/multiview-model-extension
export PYTHONPATH=.
```

아래 네 명령 중 필요한 stage를 하나씩 실행한다. 이 시점부터
`--use_wandb`에 의해 W&B를 사용하고 6-GPU 학습을 시작한다.

### ss64

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/ss64.log
```

### shape512

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/shape512.log
```

### shape1024

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/shape1024.log
```

### pbr1024

```bash
/opt/conda/envs/pixal3d/bin/python train.py \
  --config /root/node17/data/pixal3d/train/runtime-configs/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.node17.json \
  --training_data /root/node17/data/pixal3d/train/production/abo-3d-future-hssd/training_data.json \
  --num_gpus 6 --use_wandb \
  2>&1 | tee -a /root/node17/data/pixal3d/training-logs/pbr1024.log
```
