# 새 PBR source 구조의 초기 생성 동등성 검사

원본 multi-view PBR16850의 frozen base와 LoRA를 모두 불러온 vanilla와
source 세 구조를 같은32개 개발 asset에서 비교한다. Source 구조는 물리적
Top-1+R, Top-1+feature+R, raw Top-K+feature+R이다. 앞선 micro1 proof에서
세 번 갱신한 임시 weight나 Adam state는 이 검사에 사용하지 않는다.

`evaluate_r_mediated_pbr_initial.py`는 원본 config와 checkpoint에서 매번
새로 구성한다. TRAIN9139를 trainer constructor에 유지하고 실제 평가는
고정 validation32에서 수행한다. 데이터와 latent normalization은 원본과 같고,
source 구조는 실제 DINO+NAF2048 및 Omega1024 cache를 읽는다.

각 asset의 실제 RGB·camera·GT PBR latent·GT Shape condition·생성 noise와
**예측 PBR latent**의 SHA256을 기록한다. 같은 이름의 panel이나 aggregate
metric이 같다는 사실만으로 동등성을 인정하지 않는다. Renderer는 기존 native
anchor-camera 경로를 사용하며 sampling12/CFG3, BF16/TF32 off, AO seed0이다.
PSNR은 pixel MSE를 합친 뒤 계산하며 foreground PSNR은 foreground 면적으로
가중한다. 각 asset의 GT/Pred/Diff도 보존한다.

이 평가는 GT Shape를 고정한 PBR stage 검사다. Source preservation이나
predicted-geometry full-cascade 성능을 평가하지 않는다. 모델·source·buffer 전체의
평가 전후 hash와 optimizer state0을 검사한다. 네 실행의 실제 LoRA·constructor RNG·
공통 R10 초기값·asset/noise/input·prediction·rendering metric을 CPU에서 비교한다.
Sampler 또는 renderer 예외가 나더라도 process-local audit hook은 복구한다.

네 결과가 실제로 완료되고 file4와 SHA/size가 일치할 때만 초기 generated32
동등성을 인정한다. CPU 테스트나 queue 등록만으로 GPU 검사가 통과했다고
기록하지 않는다. 이 검사 통과만으로 full microbatch capacity 또는 본 학습을
허용하지 않는다. Global24의 실제 최대 microbatch와 native dropout.1 검사는
별도로 필요하다. 기존 SS 학습과 source72 HOLD 및 최종508개/10 metric 범위는 유지한다.

해석상 R의 두 학습 seed 개선과 QKV+R 조합의 V1 대비 개선을 구분한다.
QKV+R이 강한 Top-1+R을 유의하게 넘었다는 근거는 아직 없으므로 ray-wise QKV,
matching-aware attention 또는 member 후보 개선을 필수 모듈로 확정하지 않는다.
