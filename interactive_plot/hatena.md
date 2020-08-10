# matplotlibだけでGUIを作る
## はじめに

たまに，特定のタスクのために変なアノテーターを作りたいときがある．
慣れているので，pythonでやりたい．
これまでこういうときは，opencvを使って作っていたが，最近，matplotlibを使っても同じようなことができると知ったので，調べて使ってみる．

[公式ドキュメント](https://matplotlib.org/3.1.1/users/event_handling.html)をeventを受け取る関数を作って，それを fig.canvas.mpl_connect という関数で登録することでマウスやキーボードのイベントを取得できるらしい．
以下ではよくある感じのアノテーターを作って，どんな感じでできるかを確認する．


## 画像にラベルをつける

- ←→が押されたら前の画像，次の画像を表示する．
- キーが押されたら，押されたキーを記録して次の画像へ移動
- qが押されたら結果をpickleに保存して終了

<script src="https://gist.github.com/kskkwn/0edaff6559b7b30a0c6c1a2f8254d374.js"></script>

## 矩形選択

- ←→が押されたら前の画像，次の画像を表示する．
- ドラッグで領域選択して座標を取得
- 選択中は四角を表示する
- cを押したら前回の結果をキャンセルする
- qが押されたら結果をpickleに保存して終了

<script src="https://gist.github.com/kskkwn/ef71b4aa6dbc379ea671e6b93a45b221.js"></script>

- keyのイベント，マウス押し込み，マウスドラッグ，マウスリリースのそれぞれのイベントを検出する関数を作って登録する．


## 特定の物体が写ってる画像を選択

- クリックされたら，クリックされた画像のインデックスを保存して，別の画像を表示する．

<script src="https://gist.github.com/kskkwn/09f90fc21de5b30e000438fb2795b3b4.js"></script>

- マウスイベントではevent.inaxesとして，axisのオブジェクトが返ってくる．subplotの番号がほしいときは，```self.plot_axes.index(event.inaxes)```とかやると番号を取得できる．


## まとめ
- matplotlibを使ってアノテーターを作ってみた．
- いくつかやってみた感じ，opencvを覚えてるなら，opencvで作るのと対して手間は変わらないかなという印象(subplot使えるぶんだけ有用かも)．
- plt.drawを使うと若干ラグが生じるときがあって，plt.pause()で適当に短い時間を指定したほうが軽快だった．
- ここに書いたようなものだったら何でもできそうだけど，matplotlibの機能を使ったアノテーター(scatterの点の位置をドラッグして補正するとか)するときは便利かもしれない．

