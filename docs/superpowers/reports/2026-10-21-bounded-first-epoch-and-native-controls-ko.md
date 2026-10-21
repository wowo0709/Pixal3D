# Bounded attention 첫 epoch와 native source control의 중간 검증

2026-10-21 11:55 UTC. 다음 실제 validation 완료를 추적해 bounded96,
spatial192, deformation-query192 결과를 확인했다. 실행 중인 학습을
중단하거나 새 training arm을 추가하지 않았다.

## 같은 step의 결과

모두 같은 validation32와 noise를 사용한 pooled voxel IoU다. 각 arm은
동일한 step의 기존 unbounded member-feature NoDQ와 비교한다.

| Arm | Step | IoU | 같은 step NoDQ | 차이 | Paired 95% 구간 |
|---|---:|---:|---:|---:|---:|
| Bounded prior | 96 | 0.435350 | 0.436409 | −0.001060 | [−0.002877, +0.000674] |
| Native spatial | 192 | 0.440074 | 0.434217 | +0.005857 | [−0.006402, +0.019541] |
| Deformation query | 192 | 0.434763 | 0.434217 | +0.000546 | [−0.000171, +0.001419] |
| Physical Top-1 | 288 | 0.432298 | 0.434628 | −0.002331 | [−0.007706, +0.002838] |

各比較で32個のGT voxel hash、source UID、noise seedが一致した。
Source-cluster bootstrapは4,000回で、複数比較の補正はしていない。
すべての区間が0を含む。Training seedは1個で、これは480-stepの
native128 × 3-noise評価を代替する結果ではない。

Spatialの192-step point estimateは高いが、15個で改善、17個で低下
している。Per-asset IoUの平均差は+0.006518であり、pooled値とは
区別する。1個ずつ除いたpooled差は+0.001467〜+0.009221だった。
特定assetを除外せず全32個を保持し、不確実性もそのまま報告する。

## 実際に学習が動いているか

Bounded96の保存されたconditionerをCPUでstrict loadし、実GPUの
capacity検査で記録した初期状態と照合した。

- Frozen matching parameter 24個と初期基準buffer 8個はビット単位で
  一致し、conditioner全体のfrozen SHA256も一致した。
- Headの9個のparameter tensorはすべて初期値から更新されている。
  Q/K/Vの相対L2変化はそれぞれ約3.03%、2.39%、4.17%。ゼロ初期化した
  output weightも非ゼロになり、全trainable source tensorはfiniteだった。
- 初期Q/K/Vの比較には、固定seedで作るheadと同じ7-epoch utilityを
  使用した。親retargeterのランダム初期値を更新判定に流用していない。
- 96-step optimizer checkpointのstateは379個で、すべてstep96。
  LoRA 360個のLRは`1e-5`、conditioner 19個のLRは`1e-4`だった。
- 検査はCPUの保存ファイル読み取りだけで、GPU推論やoptimizer update、
  実行中trainerの変更は行っていない。

Headのbufferだけをhashした値は、frozen matcherを含むconditioner
全体のhashとは異なる。照合には同じfrozen範囲を使い、全体で一致を
確認した。Headのみの部分hashの違いは学習中の状態変化を示さない。

これらは、重みが更新されない、基準関数が書き換わる、optimizerや
LRが欠落するという説明を支持しない。ただし、candidateが正しいか、
attentionが有効に使えているか、一般化性能が改善したかを証明する
検査ではない。実際のattention entropyやpriorからのKLもここでは
測っていない。初期出力一致と、学習後の性能改善を区別する。

## 継続する判定

現在の結果でfullを選ばず、計画した480-stepまで継続する。最終
native128 × 3、projected-V診断、cascade32のtarget/source別8指標、
pilot source UIDを除外した72個のprior-only診断を接続済みである。
独立training seed、PBR stage、最終test508とFID/KIDを含む10指標の
範囲も維持する。学習を延ばすだけで改善すると仮定しない。

同じモデルの色を左右の図で統一した中間曲線、CSV、元validation
prefix、head更新表と照合結果をW&Bおよびファイルサーバ4に保存した。

- W&B: https://wandb.ai/youngwoojeon-unist/pixal3d-deformnext-retargeting/runs/ss-member-native-controls-first-bounded96-s101-v1
- 中間曲線・paired結果: `member-feature-controls-v1/paired-first-bounded96-observation-v1`
- 実checkpointのhead検査: `bounded-prior-member-source-ray-v1/checkpoint96-head-audit-v1`

上記相対パスのcanonical親は
`/file4/youngwoo/deformation/experiments/retargeting-v1/rgb_matching_v2/semantic-topk-search-v1/ray-group-study-v1/anchor-ray-composition-v1/feature-readout-v1`。
今回のpartial観測を、最終モデル採用や全実験の完了として扱わない。
