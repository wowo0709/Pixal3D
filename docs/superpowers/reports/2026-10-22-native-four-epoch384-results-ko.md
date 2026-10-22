# Native SS 384 update: raw Top-K는 feature보다 높지만 Top-1 대비 이득은 불확실

Original MV SS4210에서 fresh optimizer, TRAIN9149, seed101,
world1/global96/micro96/accumulation1로 시작한 네 arm의 같은384 update를 비교했다.
실제 첫384 update의 asset/noise/timestep/global condition trace가 모두 같다.
평가는 기존 고정32개와 생성 noise1회이며, 학습 중 비결정적 ray reduction을 사용한 기록이다.
저장 가중치의 독립 GPU 재현성 검사나 최종128×3 비교의 대체가 아니다.

| 모델 | Global occupancy IoU |
|---|---:|
| Vanilla V1 | .432526757 |
| Physical Top-1+R | .436983125 |
| Feature+R | .430150648 |
| Raw Top-K+feature+R | .440634819 |

아래 차이는 pooled global IoU의 차이가 아니라 **mean asset IoU의 paired 차이**다.
Source-cluster bootstrap4000회, 단위는 percentage point다.

| 비교 | 차이 (%p) | 탐색적 95% CI (%p) |
|---|---:|---|
| Top-1+R − V1 | +.187472 | [−.711031, +1.113758] |
| Feature+R − V1 | −.162890 | [−1.080952, +.714711] |
| Raw Top-K+feature+R − V1 | +.410868 | [−.625675, +1.414529] |
| Feature+R − Top-1+R | −.350362 | [−.798007, +.075256] |
| Raw Top-K+feature+R − feature+R | +.573758 | [+.005086, +1.191632] |
| Raw Top-K+feature+R − Top-1+R | +.223396 | [−.446276, +.935222] |

Raw Top-Kはこの時点でfeature-onlyを上回るが、強いTop-1+RやV1を上回る根拠は
まだ不十分だ。Featureの192 updateでの正の差は288で反転し、384でもTop-1を
上回らない。同じ開発パネルを繰り返し見た多重比較補正前の結果であり、
このepochを選んでfullモデルを確定しない。固定480と128×3を維持する。

根拠は `native-flow-cold3-v1/checkpoint-evaluation-v2/epoch384-CPU-evidence-v1/`
の `actual-inputs-v1/` と `four-native-paired32-v1/audit.json`。
実際のNN/source/Adam384、config、validation snapshot、共通TRAIN traceの
24ファイル11,472,049,689 bytesをfile4の実際のSHA/sizeと照合した。
集計も別のfile4 receiptで閉じた。関連CPU検査8件が通過した。

Sourceを持つ二armのepoch96 exact GPU replayが失敗した件は別途保持している。
実入力8個の診断では非決定的index-addによるray平均の微小差とattention出力の差を
確認し、平均だけ決定的にすると同じ保存sourceの3回の出力が一致した。
これは数値再現性の原因の証拠であり、生成品質が改善しなかった原因の確定ではない。
新たな決定的独立load反復を接続した。元の失敗をPASSに変更していない。
