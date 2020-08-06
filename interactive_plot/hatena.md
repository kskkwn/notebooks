# matplotlibだけでGUIを作る


## はじめに

- 特定のタスク用の変わったアノテーターがほしいときなど，たまにguiを作りたいときがある．
- pythonを普段から使っているので，pythonで作りたい．

このようなとき，これまではopencvを使って画像の表示，マウスイベントの検知をやっていた．
最近，matplotlibでも同じようなことができると知ったので，調べて試してみる．

## matplotlibのevent handling

[公式ドキュメント](https://matplotlib.org/3.1.1/users/event_handling.html)をeventを受け取る関数を作って，それを fig.canvas.mpl_connect という関数で登録することでマウスやキーボードのイベントを取得できるらしい．

```python

```




