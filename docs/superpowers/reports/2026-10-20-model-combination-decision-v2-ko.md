# 현재 효과가 보이는 조합과 다음 실험

확인 시각: 2026-10-20 07:01 UTC. 현재 수치가 가장 높은 완성 후보는 Direct이며, 이후 비교의 핵심 기준은 Top1/R이다. Semantic Top-K와 matching-aware attention이 retargeting보다 독립적으로 개선된다는 결론은 아직 없다. Full 모델은 확정하지 않는다.

## 완료된 SS 비교

동일128개·93 source cluster·생성 noise3회, 최종480 update의 비교다. 학습 seed는101 하나이며 생성 noise3회는 독립 학습3회가 아니다.

| 조합 | Voxel IoU(%) | 해석 |
|---|---:|---|
| Zero-shot:4장 모두 source M0 |38.0711|변형 전 target-camera RGB까지 포함한4장|
| Zero-shot:edited target M1 RGB1장+source M0 RGB3장 |41.9275|동일 입력을 사용하는 학습 전 기준|
| Vanilla V1 |43.2650|추가 conditioner 없이 fine-tuning|
| Semantic Top1 feature+retargeting, matched R |44.0294|새 실험과 시작점·추가 budget을 맞춘 R 기준|
| Top-K scalar 가중합+retargeting |44.0329|R 대비+.00352pp,CI[−.16781,+.15520]pp;추가 이득 미확인|
| Feature attention |43.9242|R 대비 추가 이득 미확인|
| Feature attention+physical deformation query |44.2249|feature 대비+.30071pp,CI[+.05724,+.54286]pp|
| Reliability-gated candidate-paired retargeting |44.1322|R 대비+.10279pp,CI[−.03580,+.25454]pp|
| Direct:semantic Top-K ray attention+physical deformation query+residual retargeting |44.3692|완료 후보 중 가장 높은 수치|

Direct−V1은 **+1.10421pp,95% source-cluster CI[+.61198,+1.63009]pp**다. Direct−matched R은 **+.33986pp,CI[−.05564,+.74928]pp**이므로 retargeting에 더한 matching 모듈의 추가 이득은 불확실하다. 이전 positive Top1/R의44.2757%도 유지한다. 이 기준은 post96 학습 stream이 달라 matched R과 합치지 않으며 Direct−이전R은+.09349pp,CI[−.26941,+.48361]pp다.

Physical deformation query는 feature-attention 기준에서는 개선됐지만 강한 R보다 나아졌다는 결론은 없다. Decoder·score·correspondence의 독립 효과가 분리된 것도 아니다. 고정 checkpoint의 score-prior-off·decoder-bypass·wrong-value 진단을 재학습 ablation으로 해석하지 않는다.

근거:

- [Direct 및 강한 기준 비교](/root/dev/.cache/deformnext-file4/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/direct-retargeting-v1/matched-final-v1/strong-reference-summary-v1/summary.json)
- [Top-K 가중합 최종 결과](/root/dev/.cache/deformnext-file4/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/matched-final-v1/all-three-summary-v1/summary.json)
- [Candidate-paired 최종 결과](/root/dev/.cache/deformnext-file4/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1/prior-anchored-QK-v1/reliability-head-v1/consistent-retargeting-v1/final128x3-comparison-v1/summary.json)

## PBR의 현재 근거

기존762-step32개 GT-Shape-fixed texture-stage 평가에서 B2의 semantic Top1+D1/learned-score bundle은 B1보다 foreground PSNR+.07232dB,crop LPIPS−.0006063,crop SSIM+.0009518이었다. B3의 Top-K 가중합은 B2보다 추가 이득을 입증하지 못했다. 작은 개발 패널·학습 seed1·native exact replay 미확정이므로 이 결과만으로 full 구조를 결정하지 않는다. 이 crop LPIPS는 논문의 source-preservation LPIPS와 구분한다.

새 PBR 비교는 **V1 / retargeting-only Top1 / Direct**의3군이다. 공통 V1 step762와 NN Adam360을 이어받고, 각 군은1143 update를 추가해1905로 맞춘다. Global24,physical GPU0–3의DDP4,micro6,accumulation1이다. V1과 Direct의 기존 수치 설정은 유지한다. R-only는 attention과 RAW deformation query 채널·projection을 물리적으로 포함하지 않는다.

R-only의 자체 native micro1 learnability가 실제 통과했다. 임시3 update뿐이며 성능 향상 근거는 아니다. NN360+R6 Adam 상태, frozen 상태 유지와 R6 gradient를 확인했고 peak19.0998GiB였다. 초기32 exact identity→자체DDP4/micro6 capacity→3군 비교 학습·최종32 평가·paired 요약을 연결했다. 본학습3군은 아직 시작하지 않았다.

이전 micro1 시도는 allocator17.09GiB 제한으로 실패했고, 다음 시도는 unconditional CFG update에 모든 R gradient가 양수여야 한다는 검사를 적용해 멈췄다. 새 미세 검증은 조건부 학습 가능성만 검사하며 임시3 update에만p_uncond0을 쓴다. 실제 capacity와 본학습은 원래p_uncond.1을 유지한다. 잘못 생성된 한 CLI 계획은 모델 초기화 전 parser exit2였으며 함께 보존했다. 실패한 기존 activation을 초기화하지 않고 자체 검증을 요구하는 새 activation을 연결했다.

## 다음 실험의 순서와 판단 기준

1. **TaskRay SS 최종 비교.** 현재GPU2/3에서212/480까지 진행했다. 같은 R96·Adam366,global96/world2/micro48/accumulation1을 유지한다. Ray 단위 query로 semantic Top-K 후보를 결합하고 생성 loss로 attention을 학습하는 후보를 고정480·128×noise3에서 V1,R,Direct,C와 비교한다. 초기32 출력 보존과 자체369-state DDP capacity는 실제 통과했다. 새 후보의 성능 결과는 아직 없다. 현재 두 완료 후보인 scalar와 C는 반복 학습을 새로 추가하지 않는다.
2. **동일 mesh/render 조건의 최종 출력 비교.** V1·matched R·Direct SS가 만든 예측을 동일 Shape5124210/Shape102416850와 V1 PBR762로 이어가고32개×3카메라를 평가한다. 첫3군 사례 검사와 나머지 패널을 실제 대기열에 연결했다. 이 비교는 SS 선택이 최종 결과에 주는 효과다. 학습된 PBR conditioner까지 사용한 full-stack 비교는 이후 별도로 구현·검증한다. 생성된 shape 좌표에서 conditioner를 적용할 때 GT 좌표용 projection cache를 그대로 쓰지 않는다.
3. **PBR3군 비교.** R-only 자체 초기32·4GPU 배치 검증,TaskRay 최종 평가와 현재32개 cascade 평가가 끝나면 V1→R-only→Direct를 자동 실행한다. 결과의 부호는 실행 gate가 아니다. 최종1905의 GT-Shape texture-stage 비교로 texture 학습의 이득을 먼저 분리하고, 이어 실제 생성 shape를 쓰는 full-stack 평가로 확인한다.
4. **개선 후보에서 모듈 효과 분리.** matching decoder/context,semantic correspondence 대 native spatial correspondence,Top1 payload 대 Top-K 결합,matching score를 쓰는 attention,physical deformation query를 동일 초기화·budget의 학습 arm으로 비교한다. NoScore는 Top1 correspondence feature만 사용하고,NoCorr는 원래 native spatial correspondence를 사용한다. NoDQ는 채널·projection 자체를 제거한다. 사용자 지정대로 retargeting-only 다음 retargeting+deformation query의 추가 순서를 유지하며2×2 조합이나 target-only는 추가하지 않는다. Direct−R만으로 attention과 query의 독립 효과를 주장하지 않는다.
5. **별도 학습 seed와 최종 split 확인.** 추가 matching 모듈이 강한 R보다 좋아지는지와 texture/geometry의 손실 여부를 확인한 후 양성 후보를 새 학습 seed로 재현한다. 같은 작은 개발 패널에서 최고 epoch를 고르는 방식으로 full 모델을 확정하지 않는다.

## IoU 외 metric의 상태

| 목적 | Metric | 현재 상태 |
|---|---|---|
| Source preservation |LPIPS,DreamSim,CLIP|paired textured M1 reference와 동일 camera/material/lighting을 사용한다. LPIPS native 경로는 연결했고,DreamSim·원본 OpenAI CLIP 공식 가중치는 별도 평가 의존성 경로와 file4에 준비·CPU load 검증을 완료했다. 실제 GPU metric sanity 및 전체 패널 로깅은 남아 있다.|
| Target alignment |PSNR,SSIM,2D M-IoU|실제 생성 cascade와 공유 GT pixel을 사용하는 평가 경로 연결|
| Target alignment |mesh CD,surface F-Score|실제 생성 mesh에서 측정하는 경로 연결;32개 전체 결과는 실행 중|
| Global naturalness |FID,KID|충분한 이미지 수를 갖춘 최종 평가에서 보고하며32개 개발 패널로 모델을 선정하지 않는다.|

32개 전체 paired 표·source-cluster CI·W&B metric/image 로깅을 연결했다. 기존 texture-stage 평가의 거의 고정된 GT-Shape mask IoU를 새 형상 성능으로 보고하지 않는다. Global 환경은 변경하지 않았고 평가 전용 DreamSim.2.1/PEFT.17.1 및 공식 가중치를 별도 경로에 두었다. 사용자는 기존 평가 가중치 경로가 없다고 확인했다.

새80k나 raw dataset을 추가하지 않는다. 구축된 DeformNeXt split/cache와 제공된 LoRA checkpoint/config를 사용하고,계산은 사용자 지정GPU0–3만 사용한다. File4를 canonical 저장소로 유지한다. 현재 목표는 진행 중이며 최종 모델 채택과 독립 재현은 남아 있다.
