# 기존 zero64 및 PBR3 완료 결과 재검증

기존 original multi-view zero-shot64개와 legacy PBR3개의 완료 기록을 실제 file4
원본1537개·347,297,882bytes와 대조했다.7개 모델·224개 생성 case·672개 official
perceptual view에서 기존 role별 평균·paired CI·per-asset 통계 차이는0이었다.
추가 GPU inference나 optimizer update는 수행하지 않았다.

Zero-source4는 변형 전 camera8과 source9/10/11의 RGB4장, zero-edited-source3은
변형 후 target camera8 RGB와 변형 전 source3장을 사용했다. 실제 입력256개 항목의
hash를 대조했다. 원래 SS4210·Shape5124210·Shape102416850·PBR16850 lineage이며
deformation fine-tuning이나 depth target 입력을 사용한 결과가 아니다.

Legacy PBR3개는 공통762 prefix 뒤 각각1143 update를 추가한1905 모델이다. 각 arm의
실제4rank runtime1143개, global24·micro6·accumulation1, finite loss 및 완료 terminal0을
확인했다. GT shape 기반32개 stage 평가의 집계가 기존 publication과 정확히 같고,
3모델의96개 GT/Pred/Diff panel에서 GT 영역 pixel도 같았다. 현재 fresh native1900과는
별도 학습 계열이며, 이 stage 평가를 source-view rendering metric으로 해석하지 않는다.

첫 감사 시도는 V1/Direct의 원래 final receipt에 runtime 로그가 없어서 중단됐다.
원래 실패 로그와 worker를 보존했다. 새 v2 감사는 해당 로그를 새로 hash한 뒤 실제
remote 파일과 직접 대조했다. 기존 receipt에 없던 항목을 있었다고 바꾸지 않았다.
R-only runtime은 원래 receipt와 새 remote 양쪽을 대조했다.

Canonical 결과는 `native-flow-cold3-v1/zero64-legacy-PBR3-completed-coverage-audit-v2/`다.
새 감사4파일1,767,412bytes도 실제 file4 SHA/size 검증을 통과했다. 기존 결과 보존과
coverage 확인이며, 진행 중인 PBR·SS 재현·독립 seed·DEV 선택·최종508개10 metric의
완료를 뜻하지 않는다.
