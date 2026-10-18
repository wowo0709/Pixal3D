# 실제 C1 weight의 cache·AMP·CFG 검증

최초 계획의 active-path cache parity와 dropout/AMP 검증 근거를 현재 최종 후보에 연결한다. 기존 R9 보고서는 adapter를 활성화한 BF16 SS/PBR16개 사례에서 cached/online evidence 오차0을 기록했지만, 사용한 head/모델 weight의 hash를 보고서 자체에 담지 않았다. 과거 검사를 현재 checkpoint의 직접 증명으로 확대하지 않고 별도 검증을 추가한다.

현재 matched seed101 C1의 실제 final denoiser·conditioner·frozen Q2C head를 그대로 읽는다. 고정32에서 실제 target/source validity를 갖는 첫 두 사례를 선택하고, cached/online의 RGB 네 장·camera 및 transport tensor bytes를 fingerprint한다. Head·recipe·manifest·panel·weight도 SHA256으로 고정한다. Learned retargeter와 최종 projection의 native 대비 변화가 실제 nonzero여야 하며, zero initializer 상태로 검사를 통과할 수 없다.

BF16에서 candidate index exact, evidence/logit/weight/residual·최종 condition의 cached/online 일치를 검사한다. 이어서 실제 `get_cond`의 CFG 확률0과1, seed를 고정한 확률0.5의 sample별 mask를 확인한다. Global과 projection에 같은 mask가 적용돼야 하고 확률1에서는 모든 추가 조건도 완전히0이어야 한다. 같은 입력의 training p0/inference condition도 일치해야 한다. 실제 LoRA·base·conditioner를 고정한 no-grad 검사이며 optimizer state/update는0이다. 생성 품질 평가나 전체9149개 cache의 GPU parity 검사는 아니다.

CPU 검증5개는 활성 경로 누락, online/cache 차이, CFG 뒤 semantic projection 누출, mixed mask의 branch 불일치를 거부한다. 실제 GPU proof는 원본 network-gate6개가 끝난 GPU2에서 실행한다. GPU3의 후속 sampler 진단은 병행할 수 있고, GPU2의 sampler는 이 검사가 끝난 뒤 시작한다. 현재 GPU0/1 B3 학습과 완료된 V1/C1 weight는 유지한다. Canonical 결과는 file4 `rgb_matching_v2/conditioning-contract-v1/summary.json`이다. 설정/CPU 검증을 실제 GPU PASS로 표시하지 않는다.
