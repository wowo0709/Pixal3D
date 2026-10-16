# 2026-10-16: target RGB 및 네 카메라 입력 정정

## 잘못된 구현과 결과의 유효 범위

사용자가 의도한 입력은 edited target RGB였으나 구현자가 이를 supervision-only 이미지로 잘못 분류하고 geometry-depth 입력으로 대체했다. 이는 사용자 요구가 아니었다. 기존 depth-conditioned V1과 B1–C2의 수치로 RGB 실험의 성능이나 모듈 효과를 주장할 수 없다. 따라서 “V1을 다시 학습할 필요 없다”는 기존 판단도 철회한다. Depth 실험의 모델·결과·캐시는 file4에 보존하고 modality invalidation으로 재개를 차단한다.

05:16 UTC에 후속 큐를 중지하고 SS C1 / PBR B2에 STOP을 요청했다. 각각 step69 / step860에서 체크포인트를 저장하고 정지했다. 큐 controller도 종료했다. 이후 학습은 원래 mixed385 multi-view base+LoRA initializer에서 새 RGB namespace로 시작한다. Depth fine-tuned weights나 depth-trained D1 head를 이어 쓰지 않는다. Clean RGB object 데이터로 학습했던 Omega epoch7과 정합을 검증한 world-v4 M1 geometry/latents는 재사용할 수 있다.

## 확정 입력: 두 조건 모두 K=4

현재 export의 target camera는 8이다. 원본 role manifest에는 M1 camera8과 M0 cameras8–14가 들어 있다. 이전 코드가 고른 세 M0 source cameras8/9/10에는 target camera가 중복되어 있었다. 이를 M0 cameras9/10/11로 정정한다.

| 슬롯 | Source-only zero-shot | Edited-target zero-shot / V1 / arms |
|---|---|---|
| 0 | M0 camera8 RGB (`view01.png`) | M1 camera8 RGB (`view00.png`) |
| 1 | M0 camera9 RGB (`view02.png`) | 동일 |
| 2 | M0 camera10 RGB (`view03.png`) | 동일 |
| 3 | M0 camera11 RGB (`view04.png`) | 동일 |

두 조건의 view 수, 카메라, 나머지 세 source, latent frame, noise seed, sampling steps와 guidance는 같다. Source-only는 3장이 아니며 M0 camera8을 복제해서 네 슬롯을 채우지도 않는다. 새 loader는 RGB modality, 명시적 target RGB 경로와 네 개의 서로 다른 camera ID를 검사한다. 평가에서 실제 모델 입력 텐서를 `actual_inputs.png`로 저장한다.

## 재사용과 재구축

9,149 train / 508 validation의 asset ID, split과 M1 anchor camera는 유지한다. PBR의 기존 32,768 token cap도 유지한다. RGB 파일과 camera11 source 파일은 기존 file4 export에서 복사하며 재렌더링하지 않는다. M1 SS/Shape/PBR latent, canonical geometry와 target-anchor projection maps는 재사용한다.

반면 target modality와 source camera 집합이 모두 달라졌으므로 Omega joint-view descriptor 네 stream, native DINO/NAF values, visibility teacher, RAW matching, D1 학습과 learned matching cache는 다시 만든다. 이미지와 카메라의 content fingerprint를 cache에 기록하며, sample ID가 같아도 이전 depth 캐시를 사용할 수 없다. D1은 새 RGB teacher/features로 seed42017부터 fixed-final 7 epochs를 실행한다. 원본 file4 캐시는 보존한다. 로컬 캐시 교체로 디스크가 두 배가 되지 않도록 기존 copy의 file4 checksum 일치와 새 RGB cache의 atomic completion을 확인한 경우에만 해당 owned local copy를 evict한다.

## 실행과 검증

새 데이터/결과 namespace는 `/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_v1`이며 runtime metadata는 `/root/dev/.cache/deformnext-retargeting/production-r9-rgb-v1`에 둔다. W&B run ID도 `deformnext-rgb-v1-*`로 분리한다. Hardware monitoring은 비활성화하고 loss, epoch validation, quick metrics 및 이미지는 계속 기록한다.

`run_rgb_restart.py`는 paired four-view native zero-shot stage controls, fresh RGB V1, 이후 strict RGB cache/actual native-start 검증을 통과한 B1–C2 순서로 실행한다. SS는 GPU0, PBR DDP는 GPUs1/2/3이며 matching preparation은 명시적으로 할당된 GPU7에서 실행한다. 모든 fine-tuning arm은 independently 같은 original mixed385 initializer에서 시작한다. Native-zero 보존 검증과 one-epoch execution/quality gate는 유지하며 총 학습 목표는 5 epochs다.

현재 zero-shot 도구의 SS는 voxel stage, PBR은 **GT target shape를 고정한 texture stage** 평가다. 이를 전체 cascade의 3D 복원 성능으로 표현하지 않는다. Quick PBR anchor metric도 입력 target camera의 진단값이며 novel-view 평가가 아니다. Main target-view 목록에서는 입력 M1 camera8을 제외하고 남은 M1 views를 사용해야 한다. 전체 dataset challenge와 fine-tuning 필요성의 주장은 향후 동일 입력의 전체 cascade 결과와 M0/M1 GT를 함께 비교해야 한다.

## 05:46 UTC 실행 확인

RGB manifest 감사는 train9,149 / validation508개 전체에서 통과했다. 두 조건 모두 카메라8/9/10/11의 네 슬롯이며 실제 입력 텐서 contact sheet를 각각 확인했다. Frozen SS32개 panel의 M1 GT IoU는 source-only0.396067 / edited-RGB0.415866이다. Frozen PBR16개 GT-shape-conditioned panel의 foreground PSNR은 각각17.740654 /17.901609 dB다. 이는 paired input 효과의 stage 진단값이며, 기존 depth-run 수치와 비교하지 않는다. 또한 학습 중 quick panel은 SS4/PBR2개이므로 larger zero-shot panel과 학습 곡선의 aggregate를 직접 빼서 fine-tuning 개선량으로 표현하지 않는다. 이후 공통 panel로 재평가해야 한다.

새 SS RGB V1은 원래 initializer에서 실제 optimizer update를 진행했다. PBR은 최초 시작 시 resolved JSON의 integer master_port가 native distributed runtime의 문자열 요구와 달라 optimizer update 전에 종료됐다. Config 값을 문자열로 정정했고 SS process를 보존한 채 controller를 rebind했다. 새 `--attach-native-runs`와 PID 기록으로 진행 중 native 학습의 재시작을 피한다. 이후 PBR의 3-rank 초기화와 RGB validation 진행을 확인했다. 관련 artifact는 `rgb-launch.json`, `native-attach-recovery.json`, `rgb-pipeline-state.json`이다. 이미지·camera·RGB 캐시 fencing·기존 runtime 검사를 포함한 직접 CPU 테스트 82개가 통과했다.
