# 학습된 attention의 실제 활용 관측과 논문 protocol 준비

2026-10-24. 기존 필수 계획의 retrieval→attention 활용→출력 연결 검증을 구현했다. 새 학습 arm·생성 조합을 추가하지 않았다. 실제 GPU 관측 결과는 아직 없으며 완료된 matching/renders의 기존 품질 수치를 바꾸지 않았다.

## 이번에 완료한 작업

- 기존 matching 진단과 실제 cascade32가 **정확히 같은32개 asset·32개 source cluster**임을 확인했다.
- 사용할 topology label32개의 실제 파일서버4 SHA256/size를 원격에서 읽고 local mirror와 대조했다.648,104 bytes의 원본 label을 canonical input 폴더에 복사하고 실제 file4 byte closure를 완료했다. 새 label 구축은0이다.
- 학습된 E_FIXED_RAW/E_LEARNED_RAW480의 원래 native trainer와 batch8 conditioning을 그대로 호출하는 관측 entry를 구현했다. 기존 NN module/trainer·generation·metric numerical code는 수정하지 않았다.
- Evaluator-only 통계 코드의 테스트9개가 통과했다. 반복 voxel의 ray/patch 중복 투표, 보이지 않는 GT, 후보 support miss와 Top-1 ambiguity의 구분, 확률/마스크 오류, 빈 subset과 source-cluster bootstrap 집계를 검사했다. Observer CLI syntax 검사도 통과했다.
- GPU1의 현재 V1 seed202 **final128 noise0/1/2 실제 완료 후** 실행하는 job을 queue에 연결했다. GPU0/1 학습과 GPU2/3/6 DEV128, GPU7 voxel 대조는 유지했다. CPU NN forward/학습을 실행하지 않는다.

## 실제 GPU 실행에서 요구할 검사

첫 batch에서 관측 hook이 없는 conditioning과 hook이 있는 conditioning의 전체 tensor가 정확히 같아야 한다. Hook은 output을 반환/교체하지 않는다. 두 모델의 전체 forward를 끝낸 뒤 topology label 배열을 연다.

Fixed/Learned의 실제 native query·source value/key·카메라 ray/target patch·관측 mask의 tensor SHA256이 같아야 한다. 후보 주소·valid mask·frozen prior도 같아야 한다. 초기화 모델 대신 각 모델의 실제480 LoRA/source pair를 로드하고 관측 전후 trainable state hash가 같고 optimizer update가0임을 검사한다. 이 검사들은 **실제 실행에서 통과해야 하는 조건**이며 준비만으로 통과했다고 보고하지 않는다.

## 측정할 핵심 질문

관측 단위는 기존과 같은 `(source-camera ray, target patch)`의 고유 관계다. 같은 ray에 여러 target patch가 있을 수 있으며, voxel 개수를 표본 수로 부풀리지 않는다.

| Subset | 확인할 내용 |
|---|---|
| Prior Top-1 miss, retained Top-K hit | Matching prior·실제 attention·Uniform-K가 정답 후보에 주는 mass |
| Prior Top-1 hit | 이미 잘 찾은 대응을 attention이 유지하는가? |
| Support miss | 후보 탐색 실패와 후보 결합 실패를 분리 |
| 전체 observable 관계 | 전체 경향 및 위 subset의 분모 |

각 subset에서 source-use probability와 실제 native source-map correction RMS/상대 RMS를 함께 읽는다. Correction은 원래 전달식 `source_numerator - semantic_weight * native_source`로 관측한다. GT visibility는 집계 분모에만 사용하며 correction을 바꾸거나 새로운 visibility gate로 쓰지 않는다.

같은 사례의 기존 source/target render metric과 surface metric을 asset/source ID로 연결한다. 이 연결은 설명용 association이다. 다른 학습된 R 파라미터까지 포함한 두 모델의 차이를 attention 하나의 causal effect로 해석하지 않는다.

Uniform-K의 확률 mass는 **진단 기준**이다. 최종 구조의 matching score 효과를 확인할 matched Uniform-K 학습 대조는 계속 필요하다. 이 관측을 실제 Top-1 학습 모델의 성능 비교로도 대체하지 않는다.

## 산출물과 실행 상태

결과 root `unified-module-pipeline-v1/trained-ray-utilization-mechanism-v259` 아래:

- `inputs-v1`: actual file4 label source byte 검증과32개 canonical label.
- `prelaunch-v1`: source/weight/reference binding, 실제 job과 protocol snapshot. File4 closure 완료.
- `native`: 실제 GPU conditioning observations·GT 활용 통계·기존 downstream join이 기록될 위치. 아직 미측정.
- `wrapper`: assigned GPU admission·owned child terminal/session drain·실제 완료 audit가 기록될 위치. 아직 미실행.

Job 이름은 `trained-Fixed-Learned-ray-utilization-DEV32-GPU1-v259`다.32개 source cluster에서 ambiguity subset의 attention-minus-prior 및 attention-minus-uniform mass에 대해 paired cluster bootstrap을 계산한다. 탐색적 CI이며 다중 비교 보정이나 TEST 확인으로 표현하지 않는다. 빈 subset bootstrap draw는 수를 명시하고, 관측되지 않은 효과를0으로 채우지 않는다.

## 논문에 반영한 것

- Method/Protocol에 실제 trained conditioning을 관측하는 절차와 GT label 배열을 나중에 여는 규칙, 고정 후보/prior·기존 downstream join·인과 해석 한계를 추가했다. 결과는 pending으로 명시했다.
- Main ablation 양식에 **Uniform weighting on the same Top-K** 행을 명시적으로 추가했다. 사용자가 정의한 No-score=실제 Top-1과 구분했다.
- 새 `paper-draft-utilization-protocol-v260`은 실제8쪽으로 빌드했다. Protocol이 있는4쪽과 ablation이 있는8쪽을 렌더링해 글자·표·caption의 clipping 없이 읽을 수 있음을 확인했다. TeX warning/미해결 citation0이다.

이 revision도 DEV 초안이다. 새 utilization 수치, 전체 DEV128 pipeline 결과, 최종 TEST/Main table 성능을 포함했다고 주장하지 않는다. 최종 문맥 matched ablation, 독립 seed, protocol freeze·TEST508의10개 metric과 최종 writing은 계속 진행한다.

근거: [핵심 실험·ablation·writing 계획](2026-10-24-paper-critical-remaining-experiments-and-ablation-design-ko.md), [실제 DEV128 metric 통합 검증](2026-10-24-DEV128-metric-real-output-integration-ko.md).
