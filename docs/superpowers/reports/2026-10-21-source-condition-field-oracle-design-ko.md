# 부분 RGB oracle의 global/projection 경로 분리

2026-10-21. 추가 GT source RGB9·10을 사용한 native V0 SS32 진단에서는
IoU가0.426715에서0.458495로 개선됐다. 하지만 RGB 교체는 global과 projection을
함께 바꾸므로 현재 projection 전용 모듈의 개선 여력을 확정하지 못한다.

## 고정한 비교

기존32개 전체, 같은 SS4210 checkpoint, 카메라/scale/GT latent/noise,
sampling12/CFG3/generation batch8을 유지한다. 추가 학습, GT source11,
matching/Omega 또는 teacher loss는 사용하지 않는다.

| 조건 | Global CLS/register | Projected grid |
|---|---|---|
| 기존 native V0 | 원래 M1 target8 + M0 source9/10/11 | 원래 입력 |
| 완료한 RGB oracle | M1 source9/10 교체 | M1 source9/10 교체 |
| Projected-only | 원래 입력의 실제 tensor | M1 source9/10의 실제 native 출력 |
| Global-only | M1 source9/10의 실제 native 출력 | 원래 입력의 실제 tensor |

두 입력을 같은 실제 native encoder로 인코딩한다. Cache feature의 근사 대체나
새 interpolation을 만들지 않고, 원래 encoder가 반환한 dict의 한 field만
선택한다. Target8과 M0 source11 RGB는 두 encoding에서 정확히 같다.
선택하지 않은 field와 negative conditioning은 원래 tensor를 보존하고,
추가 encoding이 CPU/CUDA RNG를 소비하면 실패한다.

## 구현 검증

기존 extra-GT 입력 어댑터와 평가기는 수정하지 않았다. 별도 process-local
context가 native `_encode_deform_views`를 감싸며 예외 발생 시에도 복원한다.
원래 입력은 실제 four-RGB tensor SHA로 식별해 같은32개 원본 row에서 읽는다.
Runtime은 실제 global[5,1024]/projection[4096,1024] 크기,32개 전체 실행,
선택/비선택 field의 tensor SHA와 CPU preflight RGB의 동일성을 확인한다.

CPU6개 테스트는 두 field의 독립 교체, negative와 RNG 보존, target 입력 변경
거부, RNG를 소비하는 encoder 거부, 실패 시 method 복구 및 기존 RGB adapter를
검증했다. 이 테스트가 실제 GPU 결과나 성능 향상을 증명하는 것은 아니다.

두 새 GPU32 진단과 CPU 집계를 기존 RGB oracle 뒤에 연결한다. 기존 training,
queue descriptor/state와 PBR4-GPU drain을 유지하고 admission lease를 사용한다.
현재 실행 중인 cascade case는 중단하지 않는다. 원래 baseline과 완료한 RGB
oracle 결과를 재사용하며 GPU 재실행을 추가하지 않는다.

## 결과에 따른 설계

Projected-only가 개선되면 현재 source feature 보정의 방향·크기·coverage를
실제 M1 source DINO와 비교할 근거가 된다. Global-only가 주로 개선되면
현 projection 전용 interface가 중요한 global adaptation을 바꾸지 못한다는
설명을 우선 검토한다. 결과를 보기 전에 teacher loss나 새 global retargeter를
학습하지 않는다.

모든 조건은 추가 GT 입력 진단이며 competitive 모델, matching 모듈 효과,
최종 full 모델 선정 결과로 취급하지 않는다. 같은32개의 paired voxel count와
source-cluster bootstrap을 보고하며 원래 최종508개/10개 metric 범위도 유지한다.
