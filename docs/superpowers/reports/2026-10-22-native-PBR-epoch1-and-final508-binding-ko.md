# Native PBR 첫 에폭과 최종 평가 연결 상태

2026-10-22 UTC. 실제 첫 에폭 검증과 완료된 CPU 검사, 설치된 대기 작업을 구분한다.
아래의 모델 고정·최종 평가 작업은 아직 실행 전이다. 전체 실험 완료나 full 모델 채택을 의미하지 않는다.

## 실제 PBR vanilla 첫 에폭

원본 multi-view PBR16850의 base+LoRA에서 fresh optimizer로 시작한 vanilla가
380 update를 마쳤고, NN·Adam·config·checkpoint descriptor 네 파일의 실제 file4
SHA256/size 검증을 통과했다. 이후 고정4 렌더 validation을 마치고 학습을 재개했다.
학습은 physical GPU0–3, world4/global24/micro6/accumulation1, 총1900 update다.
동일한 학습 예산의 source 세 모델은 뒤에 순차 실행한다.

| 고정4 평균 | Step0 | Step380 |
|---|---:|---:|
| Render PSNR ↑ | 26.05731 | 26.34439 |
| Render SSIM ↑ | .991635 | .992008 |
| Render LPIPS ↓ | .010105 | .009688 |
| Foreground PSNR ↑ | 14.94644 | 15.23352 |
| GT foreground crop LPIPS ↓ | .095126 | .092427 |

각 asset의 실제 초기 noise, 입력 RGB, GT PBR latent, GT Shape condition, camera hash가
step0과380에서 같다. Validation 중 optimizer update는0이다. PSNR 변화는 순서대로
+0.1263/+0.7076/+0.0022/−0.0017 dB로, 평균 개선은 주로 앞의 두 사례에서 나온다.
GT|Pred|Diff PNG도 보존했다. 두 번째 사례를 직접 확인했으며 변화는 작다.

이 평가는 **GT Shape를 고정한 PBR stage 평가**다. Alpha IoU1은 같은 GT shape를
사용한 결과이며 geometry 성능 향상이 아니다. Source preservation이나 실제 예측
SS→Shape→PBR full-cascade 품질 향상, 넓은 데이터에서의 유의한 개선을 주장하지 않는다.

423 update까지 읽은 실제 runtime prefix에서 마지막10 update의 rank별 median compute는
약13.16초다. 최대 allocated는 rank0 43.652 GiB, 다른 rank는43.409–43.415 GiB였으며,
최대 reserved는46.615/48.461 GiB다. 해당 prefix와 현재 live process를 확인했으며
관측된 OOM은 없다. 이것은 모든 후속 source 모델의 실제 학습 완료를 보증하지 않는다.

원본 live 로그는 수정하지 않았다. Newline으로 끝나는 validation/runtime prefix,
두 시점의 PNG, 실제 checkpoint 검증 기록을 별도 소유 폴더로 복사해
**17개 파일·1,308,923 bytes**의 실제 file4 SHA/size를 검증했다. Canonical 결과는
`PBR-native-science-runtime-v1/epoch380-actual-evidence-v1/`의
`audit.json`, `validation-prefix.jsonl`, `validation-panels/`, `file4-closure.json`이다.

## 최종508 SS 평가를 위한 native 연결부

기존 legacy evaluator용 factory는 새 cold-native SS/R checkpoint schema와 다르다.
새 `evaluate_native_bound_population.py`는 원래 native bundle/config 검사와 모델
로딩을 사용하고, panel 선택 지점에서 validation dataset의 manifest/split/사용 중인
root만 바꾼다. 모델·LoRA·source 모듈·trainer·normalization·sampler는 유지한다.
특히 원래 `match_root=None`인 online matching을 GT correspondence cache로 바꾸지 않는다.
실제 생성은 batch8/12steps/CFG3/deterministic이며 optimizer update0이다.

508개는 batch8의 마지막 batch가4개다. Test quality를 열기 전에 원래 validation508의
prefix36으로 두 번 독립 model load를 실행한다. 앞32개는 기존 final128의 실제 기록과
정확히 일치해야 하고, 마지막4개를 포함한36개의 metric·GT/noise·predicted occupancy·
실제 prediction/target latent array SHA가 두 실행에서 일치해야 한다. ZIP container의
압축 시각 차이를 latent 차이로 판정하지 않는다. 네 모델의 실제 GPU 반복8개와 CPU
판정4개를 연결했다. GPU 검사는 아직 pending이며 기존 PBR 학습을 중단하지 않는다.

## Development-only freeze와 다음 실행

새 `native_final_freeze.py`는 vanilla, physical Top-1+R, feature+R,
raw Top-K+feature+R 네 모델을 모두 보존한다. 결과가 음수인 모델도 표에 남기며,
기존 three-role schema에 맞추려고 같은 NN을 별도 candidate로 복제하지 않는다.
Matching-aware attention을 성공한 full 모델로 자동 채택하지 않는다.

실제 모델 고정 전에 아래 evidence가 모두 필요하다.

1. 완료된 native SS480 네 모델의 DEV128×생성 noise3 평가.
2. Native SS480·PBR1900의 8개 full-cascade 개발 조합·32개·768 view·8 metric 실제 완료.
3. 학습 seed101/202의 native SS480 비교와 실제 학습 trace·frozen base 검사 완료.
   독립 seed에서 성능 차이가 음수여도 결과를 보존하며 고정 자체를 막지 않는다.
4. 위 네 모델 각각의 validation36/tail4 독립 반복 검사 통과.

SS480 NN/source/config와 실제 개발 평가의 path 및 SHA를 대조하고,
PBR1900 NN/source/config가 실제 stage completion·개발 cascade에서 사용한 것과 같은지
검사한다. 원본 Shape5124210/Shape102416850와 SS-only용 원본 PBR16850 prior도 고정한다.
Test508/134 source cluster는 원래 TRAIN9149·validation508과 sample/sourceUID/cluster가
겹치지 않아야 한다. 실제 test cache QA·manifest SHA와 전체 ordered508도 검사한다.

CPU activator 한 개를 실제 대기열에 추가했다. 위 dependency가 모두 완료되면
development-only freeze를 저장하고, 동일한 noise map으로 네 모델의 실제 SS508 GPU
평가를 추가한다. 현재는 activator가 pending이며 freeze·test 추론·test 점수는 없다.
사전 준비2파일22,967 bytes와 append-only queue snapshot2파일7,704,225 bytes의
실제 file4 SHA/size를 검증했다. 전체3515개 dependency DAG에 cycle/누락 참조가 없다.

원본 zero-source4(M0 RGB 네 장)와 zero-edited(M1 target8+M0 source9/10/11) 비교를
최종 프로토콜에 유지했다. **두 zero control의 최종508 실행과 native cascade508,
공식 perceptual/Inception, 전체10 metric을 연결·실행하는 작업은 여전히 남아 있다.**
Source LPIPS/DreamSim/CLIP, global FID/KID, target PSNR/SSIM/M-IoU/CD/F-Score의
역할은 사용자가 지정한 대로 보존한다. CPU freeze/schema/population 회귀 검사는
**79 passed in0.81s**다. 실제 GPU 반복 검사나 최종508 품질 통과로 대신 해석하지 않는다.

## 모듈 기여 해석

Retargeting의 과거 두 학습 seed 분리 비교는 개선 근거가 있다. 최신 native SS에서는
Top-1+R이 V1보다 높지만 feature attention과 raw Top-K의 추가 이득은 미확인이다.
Member/decoder/score의 correspondence 진단 개선을 생성 품질 개선과 합쳐 주장하지
않으며 ray-wise QKV의 필수성도 아직 입증되지 않았다. 전체 목표는 진행 중이다.
