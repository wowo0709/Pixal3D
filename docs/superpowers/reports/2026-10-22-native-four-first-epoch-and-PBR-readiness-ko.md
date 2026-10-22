# Native SS 네 모델 첫 epoch와 PBR integration 준비 결과

2026-10-22 06:46 UTC 기준. 실제 training child process와 첫96-update validation,
완료된 CPU paired summary를 확인했다. 네 학습은 중단하거나 재시작하지 않았다.

## SS 첫96 update의 공정한 대조

네 모델 모두 같은9149 train asset·학습 seed101·original MV SS4210+fresh optimizer,
world1/global96/micro96/accumulation1이다. 첫96개 asset/noisy latent/timestep/global
condition trace가 모두 같다. 초기 native32 IoU는 모두0.4267154667이었다.

| 모델 | Native global occupancy IoU | Mean asset IoU |
|---|---:|---:|
| Vanilla | .433978862 | .463800914 |
| Physical Top1+R | .430990571 | .461618760 |
| Learned feature+R | .431369326 | .461107488 |
| Raw Top-K+feature+R | .430619123 | .460971375 |

두 IoU 열은 집계가 다르다. Global count로는 feature+R이 Top1+R보다 높고,
asset 평균으로는 Top1+R이 높다. 작은 차이를 혼용하거나 한 집계의 최고를 골라 채택하지 않는다.

32 source cluster의 mean asset IoU에 대한 paired bootstrap CI95는 다음과 같다.
모든 차이와 CI 단위는 IoU 자체이며 percentage point가 아니다.

| 대비 | Mean asset IoU 차이 | CI95 |
|---|---:|---|
| Top1+R − vanilla | −.002182154 | [−.010194913, +.005050589] |
| Feature+R − Top1+R | −.000511272 | [−.003767900, +.002615855] |
| Raw Top-K+R − Top1+R | −.000647385 | [−.004016404, +.002592863] |
| Raw Top-K+R − feature+R | −.000136113 | [−.001159088, +.001087442] |

첫 epoch에서는 source 세 모델 모두 vanilla보다 낮고, 모든 위 CI가0을 포함한다.
모두 초기 checkpoint보다 좋아졌으므로 native 초기 출력 붕괴가 관측된 상황은 아니다.
Top1+R도 낮아서 차이를 learned attention에만 돌릴 근거가 없다. 반대로 attention의
추가 이득이나 확정적 악화도 현재32개·한 generation noise로 주장하지 않는다.

Vanilla의192-update IoU는.435746760이다. 나머지 모델의192 결과가 없으므로
이 수치를 그 모델들의96 결과와 같은 epoch 비교에 사용하지 않는다.
Source 모델들의 saved-checkpoint GPU replay와 signal observer는 자원을 기다린다.
Vanilla96만 독립 GPU replay의 실제 terminal0·개별 기록 동등성·file4 검증까지 완료됐다.

완료 근거는 native-flow-cold3-v1 아래
`checkpoint-evaluation-v2/epoch96-CPU-evidence-v1/R-native-four-epoch96-in-process-paired32-CPU-v1/`
`{audit.json,file4-closure.json}`이다. 이는 GPU replay나 final128/test508의 대체가 아니다.
최종480·128개×세 generation noise·독립 training seed·후속 cascade를 유지한다.

## PBR의 실제 준비 진척

새 source2048 모듈, sparse native trainer, original-PBR GPU proof entry를 구현했고
관련 CPU37개 검사가 통과했다. [구조와 검증 범위](2026-10-22-PBR-R-source-native-integration-ko.md).

실제 dataset을 CPU에서 읽어 vanilla와 feature dataset의 eligible9139개 asset 순서가 같고,
선언한31900-token sample의 RGB·shape/PBR latent·camera·mesh scale가 정확히 같은 것을 확인했다.
Source values는3×1024×2048, Omega key는1024차원이며 실제 NAF 절반을 포함한다.
PBR의 기존32768-token 제한으로 제외된10개를 유지하므로 SS9149와 PBR9139를 구분한다.
학습 arm별로 다른 subset을 선택하지 않았다. 새 데이터나 latent 다운로드는 하지 않았다.

원본 PBR16850 checkpoint는 기존 로컬 복사본을 재사용한다. Actual file4 origin receipt의
SHA256과 현재 로컬 SHA256이 같고, 대응 원본 config의 모델·두 latent normalization을 대조했다.
이전 V1 deformation step762의 weight/Adam을 새 baseline으로 쓰지 않는다.

PBR geometry engineering job 다음에 **vanilla와 source 세 mode의 micro1 GPU proof4개 및
그 결과의 CPU 비교1개를 실제 연결**했다. 기존2591개 job descriptor를 보존해 총2596개다.
모두 GPU0–3의 기존 학습 종료 후 빈 자원에서 실행된다. 실제 PBR GPU proof는 아직 완료되지 않았다.
Micro1 proof의 임시3-update 가중치는 폐기하며 본 학습으로 이어 쓰지 않는다.

계획·config·실제 CPU input audit·검사 log의 file4 검증은15파일355304byte,
queue append 근거는3파일5395270byte로 완료됐다. Canonical root는
`rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/`
`feature-readout-v1/source-feature-teacher-pilot-v1/r-mediated-source-pilots-v1/PBR-native-source-micro1-v1/`다.
GPU proof 통과 후에도 초기32개 generation, global24의 최대 microbatch/capacity 및 native dropout.1,
학습/빠른 rendering validation 준비가 별도로 필요하다. 이를 완료된 PBR fine-tuning으로 보고하지 않는다.
