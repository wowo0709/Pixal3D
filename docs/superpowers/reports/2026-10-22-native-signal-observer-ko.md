# Native SS의 matching → attention → R → denoiser 신호 진단

다음 native epoch96 평가에서는 성능 숫자와 별개로 신호가 어디서 약해지는지 확인한다.
학습 코드와 저장 모델을 바꾸지 않고 별도의 read-only inference에서 관측한다.

* matching posterior의 Top1 질량·유효 후보 수를 측정한다. Top-K 안에 후보가 있더라도
  frozen prior가 사실상 한 후보에 집중하면 attention이 다른 후보를 활용하기 어려울 수 있다.
* 학습된 네 head의 유효 후보 수·prior 대비 KL·matching Top1 밖의 확률 질량을 측정한다.
  gradient가 0이 아니라는 사실과 실제 후보 재조합이 일어난다는 사실을 구분한다.
* source ray의 matching support·source-use probability·member retained mass를 기록한다.
  matching이 없는 영역의 처리는 native multiview prior를 유지하는 현재 구현으로 관측하며
  새로운 visibility write gate를 추가하지 않는다.
* R 입력 appearance/R 출력 RMS와 source correction의 projection 전후 전달을 확인한다.
  native condition과 비교한 실제 correction 비율, BF16 addition 후 실제로 달라진 channel의
  비율을 측정한다. 계산된 작은 correction이 dtype 반올림으로 사라지는 경우도 구분한다.

계측은 model tensor·attention probability·RNG·학습 optimizer를 수정하지 않는다. 계측한
동일 epoch96 모델의 32개 asset·noise·GT·TP/FP/FN·metric이 앞선 normal replay와 모두
일치해야 결과를 배포한다. context는 실패하더라도 원래 function binding으로 복구된다.

이 관측만으로 semantic correspondence의 정확도, matching decoder/score의 인과 효과,
최종 성능 향상을 입증하지 않는다. Native task 비교 결과와 함께 확인하여, 실패 원인이
retrieval support인지, prior에 묶인 attention인지, R의 전달 약화인지 구분하는 데 쓴다.
Formal NoScore=Top1-only, NoCorr=native spatial의 재학습 ablation을 대체하지 않는다.
