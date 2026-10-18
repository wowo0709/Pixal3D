# 고정 matching head의 precision–coverage와 남은 병목

원래 SS 계획8.2의 confidence coverage 곡선을 실제 fixed-final H2/H3 저장 값으로 완성했다. 모델·threshold·calibration parameter를 다시 학습하거나 validation으로 선택하지 않았다. `summarize_matching_confidence.py`는 저장된 logit에 기존 train-calibration의 slope/bias를 적용하고, 동일 확률의 query는 모두 함께 accept하는 곡선을 만든다. Frozen operating point의 accepted count·precision·coverage는 실제 GPU head audit와 정확히 같았다.

Canonical 산출물은 file4 `rgb_matching_v2/head-audit/confidence-curves-v1/`의 `summary.json`, `precision-coverage.npz/.png/.pdf`다. CPU2개·CUDA 비가시 상태에서 기존 저장 값만 사용했고 optimizer update는0이었다. GPU0/1의 B3 학습은 유지했다.

## 후보와 선택을 구분한 기존508개 audit

모든 행의 source-visible query 분모는22,893개다. Foreground38,325개와 visibility가 없는 query도 별도로 보존했다. H0/H1/H2/H3는 동일 GT contract의 fixed7-epoch head이며 SS 생성 지표가 아니다.

| Head | TopK | 정답 후보 recall | Exact Top1 | Near1 accuracy |
| --- | ---: | ---: | ---: | ---: |
| H0 replay | 64 | 63.86% | 20.71% | 49.26% |
| H1 | 256 | 93.16% | 21.79% | 51.64% |
| H2 | 256 | 93.16% | 22.78% | 53.50% |
| H3 | 256 | 93.16% | 23.71% | 53.95% |

H3는22,893개 중21,326개에서 후보에 정답이 있지만 선택이 맞은 것은5,428개였다. 따라서 현재 candidate miss만이 유일한 병목은 아니다. 후보 확대 이후 선택과 consumer에 유용한 evidence 전달이 남는다. Near1 정확도가 exact보다 높고 Q2C RGB task 개선도 별도로 존재하므로 exact Top1 하나로 soft evidence의 유용성을 판정하지 않는다. H2 confidence loss는 ranking branch에 detach돼 있으므로 H2−H1의 Top1 변화도 inference rejection의 인과 효과로 해석하지 않는다.

## Confidence 곡선의 분모와 실제 operating point

이 곡선의 분모는 양쪽 모두38,044개의 **trustworthy foreground query**다. 가시 query의 proposal miss와 certified-unobservable query를 포함한다. Positive label은 선택한 Top1의 exact correctness다. 위22,893개 visible-query 정확도나 SS inference cache의38,366 foreground와 다른 분모다. 본 곡선으로 기존 cache의 실제acceptance11.33%를 대체하지 않는다. H0/H1은 confidence를 학습 목표로 사용하지 않아 같은 의미의 곡선을 만들지 않았다.

| Fixed head | 잠근 threshold | Coverage | Precision | 남긴 correct / 전체 correct |
| --- | ---: | ---: | ---: | ---: |
| H2 | 0.262521315 | 11.2606% | 32.9599% | 1,412 / 5,216 |
| H3 | 0.276316212 | 11.4052% | 33.1643% | 1,439 / 5,428 |

H3의 operating point는 query11.41%를 남기면서 전체 correct match의26.51%를 유지한다. Acceptance가 낮아도 선택된 대응의 exact precision이 높다고 볼 수 없는 상태다. 기존 calibration은 confidence 확률의 calibration이지, correspondence를 정확하게 만드는 연산은 아니다.

곡선의 H3 참고 지점은 coverage1%에서 precision45.93%,5%에서38.78%,10%에서34.27%,20%에서29.99%,50%에서22.13%였다. 이는 validation 진단점이며 새 threshold 후보로 사용하지 않았다. Threshold를 낮추면 더 많은 정보를 읽지만 더 많은 잘못된 exact match도 읽으므로, threshold 조정만으로 현재 문제를 해결했다고 주장할 수 없다. 반대로 exact label은 soft evidence의 task utility와 같지 않아 이 precision을 SS 생성 실패의 단독 원인으로 단정하지도 않는다.

## 검증과 재실행

NPZ의 모든 threshold group에서 accepted/true-positive count와 precision·coverage를 재계산했다. 실제 checkpoint·validation/calibration/recipe·selected-head fingerprint도 대조했다. W&B `ss-matching-v2-confidence-coverage-v1`의 실제 finished/PASS·12개 numeric field·PNG 한 장·4개 artifact 파일을 서버에서 내려받아 비교했다. `provenance/head-confidence-curves-complete-and-remote-v1-file4.json`은 input checkpoint를 포함한23개 산출물20,570,833bytes의 actual-file4 SHA256/크기 검증이다.

재실행은 새 output 경로로 다음을 사용한다. 기존 output은 덮어쓰지 않는다. `--wandb`는 최초 publication용 fixed run ID를 사용하므로 같은 결과를 재현할 때 생략한다.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.deformnext_retargeting.summarize_matching_confidence \
  --root /file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2 \
  --output /file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/head-audit/confidence-curves-replay
```

저장된 split은 head validation이며 해당 결과를 선택에 사용하면 unseen test라고 부르지 않는다. 곡선은 query-level 서술 통계이며 독립 query에 대한 population CI를 추가하지 않았다. 이 결과를 본 뒤 고정 H3 policy나 진행 중 SS arm을 바꾸지 않았다.
