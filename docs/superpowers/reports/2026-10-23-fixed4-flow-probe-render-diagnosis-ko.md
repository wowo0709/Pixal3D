# Bed의 flow probe와 렌더링 지표가 다르게 움직임

저장된 fixed4 validation을 추가로 읽었다. Bed의 Top-1+R은3→4에폭에서 고정-seed
flow loss가 조금 줄었지만 실제12-step 생성 후 target 렌더링 지표는 악화됐다.

| 같은 Bed·Top-1+R | 3에폭 | 4에폭 |
|---|---:|---:|
| 기록된 flow loss ↓ | .12699378 | .12696853 |
| Target PSNR ↑ | 26.603889 | 25.600025 |
| Target LPIPS ↓ | .00650782 | .00946909 |

같은4에폭 V1과 비교해도 Bed flow loss는−.00011703으로 낮으나, PSNR은−1.094776dB,
target LPIPS는+.00248275로 나쁘다. 단일 flow loss를 렌더링 품질의 대리 판정으로
사용하기 어렵다는 관측이다. 일반적인 과적합이나 R의 잘못을 입증한 결과는 아니다.

실제 validation 함수는 asset별 CPU/CUDA seed를 reset한 뒤 training_losses를 실행하고,
별도로 실제 PBR 생성·렌더링 지표를 계산한다. 최초 step0의 flow loss도 V1과4개 모두
정확히 같다. 다만 flow probe의 x_t와 t hash는 별도로 저장되지 않았다. 기록된 생성
sampling noise hash를 이 flow probe의 bit-exact 증명으로 대신하지 않는다.

샘플별 source 보정 RMS·fusion gate·transport event도 이 평가 row에 없다. 따라서 이전
리포트의 TRAIN batch RMS 증가를 Bed 고정 입력 보정량의 변화로 해석할 수 없다.
예정된 source 전달/R 제거 진단과 실제 DEV rendering 비교에서 기여를 분리해야 한다.

이 작업은 저장된4개·5epoch row의 metadata/hash/통계 재검토이며 추가 neural forward,
optimizer update 또는 학습 중단이 없다. Canonical
`PBR-native-science-runtime-v1/fixed4-flow-probe-render-mismatch-CPU-v1/`의2파일19,141bytes를
실제 file4 SHA256/size로 검증했다.1900까지의 동일 예산, 후속 arm·역할별 DEV 비교·독립
TRAIN seed·최종508개10 metric 범위는 유지한다.
