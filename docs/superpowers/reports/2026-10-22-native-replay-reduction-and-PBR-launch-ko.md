# SS saved replayの数値診断とPBR本学習の接続

2026-10-22 UTC。正常なSS学習、dispatcher、file4 mirrorを再起動していない。
物理GPU0–3のみを使用する。目標は継続中で、fullモデルは採択していない。

## 元の失敗と実入力での原因切り分け

Vanillaとphysical Top-1+Rの保存epoch96は学習中32個の記録を正確に再現した。
Featureとraw Top-Kの保存epoch96は全32個を実際に生成した後、exact equalityで失敗した。
Featureのglobal IoUは学習中.431369326に対し再評価.430793205、raw Top-Kは
.430619123に対し.430694048だった。失敗測定、PNG、terminal1を保持した。

同じ保存feature96と実際の8個のRGB/camera conditioningを用いて、GPUでray平均と
source forwardを反復した。入力とsource parameters/buffersは変えていない。

| 実際に反復した処理 | 結果 |
|---|---|
| native DINO queryの通常ray平均、6回 | 全比較で不一致、最大絶対差7.6294e−6 |
| 同じ平均をdeterministicで6回 | 全要素が一致 |
| 通常source forward、3回 | attention確率・source numeratorが不一致 |
| **ray平均だけ**deterministic、source forward3回 | 確率・numerator・semantic weightが全要素一致 |

この診断はoptimizer/samplerを一度も実行していない。Matmul TF32はFalse、
cuDNN TF32はTrue、PyTorch2.8.0+cu128だった。Floating-point index_addの
計算順に由来する差が実際のattention/consumer featureへ届くことを確認した。
ただし、この8個のconditioning診断だけで32個のgeneration全体や学習の差の全てを
説明したとは言わない。性能低下の原因がこれだけだったという主張もしない。

実測は `checkpoint-evaluation-v2/source-reduction-diagnosis-v1/actual-v1/`
のnative/audit.jsonとnative・wrapper両方のfile4 receiptに保存した。

## 正直な再現性プロトコル

過去の非決定的な学習中記録とのbit equalityを、許容誤差でPASSに変更しない。
新しい `epoch96-deterministic-saved-pair` プロトコルは、同じ保存NN/sourceを
別プロセスで2回ロードし、同じ32個・noise・GT・batch8で決定的に測定する。
2回目には不活性なmatching/attention/R/consumer observerを入れる。
両測定の全per-asset count/metric、aggregate、loaded trainable SHAが正確に
等しいことを要求する。元のstochastic epochを再現したというflagはFalseに保つ。

Feature/rawそれぞれ2回のGPU測定とCPUの厳密比較を新たに接続した。
既存のfinal480/128個×3noiseはもともと決定的評価であり、手順を変えていない。
止まっていたfinal128、archive32、3モデル/4モデル集計の10ジョブは、
**まだ実行されていない同じplan/command/output**を新aliasから実行する。
依存先は元の不可能なstochastic exact gateから新しいdeterministic repetitionへ置き換える。
元のFAILED/HELD descriptorとstateは保持する。Observerの必要範囲は2回目の実測に含めた。
古いepoch96 exact-summaryは履歴としてHELDのままで、成功した集計とは呼ばない。

接続は `checkpoint-evaluation-v2/deterministic-recovery-v1/{prelaunch-v1,queue-v1}`。
接続時点で16ジョブ追加、旧2608 descriptorは全てそのまま維持した。
新protocolのCPU検査5件は通過。独立GPU反復の成否は実行結果で確認する。

## PBR四armの本学習を接続

Original MV PBR16850のcomplete base+LoRAをfresh optimizerで読み、
DeformNeXt9139個、global24、DDP2、micro12、accumulation1、
380 update/epoch×5=1900 updateで比較する。

| GPU | 最初 | 完了後 |
|---|---|---|
| 0,1 | Vanilla | Physical Top-1+R |
| 2,3 | Top-1+feature+R | Raw Top-K+feature+R |

実際のmicro1 backward・cold32生成一致・DDP2/micro12 capacityの四arm gateを
全て要求する。Engineering中のweight/Adamを継続しない。実際のcapacityでOOMが
発生した場合はその失敗を保持し、別versionで必要なGPU配分を測る。
ここに記す1900 science接続だけでengineeringが通過したとは言わない。

本学習は初回と各epochに固定4個のGT-shape条件付きflow/render metricと
GT/Pred/DiffをW&B youngwoojeon-unistへ記録し、各epochのNN/source/Adamを
file4の実際のSHA/sizeで閉じる。Stage評価と最終predicted-geometry cascadeを区別する。

ローカル空き領域に対し、各runの将来checkpoint増分16GiBと共通workspace8GiBを
atomicに予約する。既に書いたcheckpointは空き容量に二重計上しない。
PIDが消えただけでは予約を解除しない。完了した自分のrunのcanonical wrapper receiptで解除する。

最初の保存容量確保では、4SSの正常終了とfinal480 closureの後、旧192/288/384の
ローカル重複だけを候補とする。元モデル・データ・最終480は対象にしない。
Queueのpending/running/未登録状態のdescriptorからconsumerを探し、実際のopen fileも確認する。
Own namespace、clean/committed worktree、実際のfresh remote SHA/sizeを確認し、
削除前ledgerをfile4に閉じてからローカルだけunlinkする。Canonical file4は残す。
PBR最初の二armも正常終了・canonical closure後に旧4epochのローカル重複だけを整理し、
それぞれ次armを同じGPUペアで開始する。Disk policyのCPU検査9件が通過した。

接続は `PBR-native-science-runtime-v1/{launch-preparation-v1,launch-queue-v1}`。
4本学習と3CPU保存容量ジョブを追加し、旧2624 descriptorを保持して計2631となった。
接続時点ではPBR本学習未開始。最終480評価、独立TRAIN seed202、必要なmatched module
比較、新source構造のgenerated-support PBR sampling、固定Shape cascade、final508と
全10metricは引き続き残る。途中の優位なepochでfullを決めたり、scopeを縮めたりしない。
