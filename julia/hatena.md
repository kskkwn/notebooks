この記事は[Julia Advent Calendar 2017](https://qiita.com/advent-calendar/2017/julialang)の17日目の記事です．
普段はpythonばかり書いていて，juliaは最近文法覚えてきたかなレベルなので色々許してください．

## 式or擬似コードに可能な限り近いプログラム

- 自分で解いた数式を実装するとき
- 論文に書いてある擬似コードを実装するとき

式or擬似コードに可能な限り近いプログラムを書くようにしている．
ここで"可能な限り近い"とは，関数の名前とかを合わせるとかだけでなく，$\alpha$などunicode文字をバンバン使うことを意味する．このようなプログラムを書くことで，

- デバッグがしやすい
- 頭を使わなくてもプログラミングできる

という利点がある．

例えば[No-U-Turn Sampler Hoffman+, 2011](https://arxiv.org/abs/1111.4246)の擬似コード(の一部，論文より引用)は

[f:id:ksknw:20171130224045p:plain]

これに対して書いたpythonコードは以下．

```python
def BuildTree(θ, r, u, v, j, ε):
    if j == 0:
        θd, rd = Leapfrog(x, θ, r, v * ε)
        if np.log(u) <= (L(*θd) - 0.5 * np.dot(rd, rd)):
            Cd_ = [[θd, rd]]
        else:
            Cd_ = []
        sd = int(np.log(u) < (Δ_max + L(*θd) - 0.5 * np.dot(rd, rd)))
        return θd, rd, θd, rd, Cd_, sd
    else:
        θ_minus, r_minus, θ_plus, r_plus, Cd_, sd = BuildTree(θ, r, u, v, j - 1, ε)
        if v == -1:
            θ_minus, r_minus, _, _, Cdd_, sdd = BuildTree(θ_minus, r_minus, u, v, j - 1, ε)
        else:
            _, _, θ_plus, r_plus, Cdd_, sdd = BuildTree(θ_plus, r_plus, u, v, j - 1, ε)
        sd = sdd * sd * int((np.dot(θ_plus - θ_minus, r_minus) >= 0) and (np.dot(θ_plus - θ_minus, r_plus) >= 0))
        Cd_.extend(Cdd_)

        return θ_minus, r_minus, θ_plus, r_plus, Cd_, sd
```

pythonではある程度unicodeを変数に使うことができ，例えばギリシャ文字などは全て思うように書ける．
しかし，一部の例えば$\theta ^+$や$\nabla$などの記号は使うことができないため，微妙に見難い感じになってしまっていた．
(あとpythonはまじで遅い)
探してみるとjuliaで同じことをしている人がいて，こちらのほうがだいぶ良さそうだった．

[http://bicycle1885.hatenablog.com/entry/2014/12/05/011256:embed:cite]

juliaでは↑であげたような文字に加えて$\hat$みたいな修飾文字も[変数名に使えるらしい](https://docs.julialang.org/en/release-0.4/manual/unicode-input/)．
さらに不等号の$\le$とかが定義されていて使えるらしい．
juliaすごい．あとなんか速いらしい．pythonには実装されていない()多重入れ子なfor文も使っていいらしい．

ということで以下では練習がてら，これまでpythonで実装したコードをjuliaで書きなおしてみて，数式/擬似コードの再現度と実行速度を比較する．

NUTSはもういいかなって感じなので

- Dynamic Time Warping
- Stochastic Block Model

について実装する．
juliaはたぶん型とかをちゃんと定義するともっと速くなるが，「そのまま実装する」という目的に反するのでやらない．

## Dynamic Time Warping
Dynamic Time Warping (DTW) は，２つの連続値系列データのいい感じの距離を求めるアルゴリズム．
動的計画法で解く．
pythonで実装したときの記事はこっち
[http://ksknw.hatenablog.com/entry/2017/03/26/234048:embed:cite]


(DTW自体の論文ではないけど) [A global averaging method for dynamic time warping, with applications to clustering](http://www.francois-petitjean.com/Research/Petitjean2011-PR.pdf)
の擬似コードを参考にしてプログラムを書く．
擬似コードは以下．

[f:id:ksknw:20171130230716p:plain]

[f:id:ksknw:20171130230720p:plain]

ただしこの擬似コードは多分間違っているので，m[i,j]の遷移前のインデックスをsecondに入れるように変えた．


### python

```python
δ = lambda a,b: (a - b)**2
first = lambda x: x[0]
second = lambda x: x[1]

def minVal(v1, v2, v3):
    if first(v1) <= min(first(v2), first(v3)):
        return v1, 0
    elif first(v2) <= first(v3):
        return v2, 1
    else:
        return v3, 2 

def dtw(A, B):
    S = len(A)
    T = len(B)

    m = [[0 for j in range(T)] for i in range(S)]
    m[0][0] = (δ(A[0],B[0]), (-1,-1))
    for i in range(1,S):
        m[i][0] = (m[i-1][0][0] + δ(A[i], B[0]), (i-1,0))
    for j in range(1,T):
        m[0][j] = (m[0][j-1][0] + δ(A[0], B[j]), (0,j-1))

    for i in range(1,S):
        for j in range(1,T):
            minimum, index = minVal(m[i-1][j], m[i][j-1], m[i-1][j-1])
            indexes = [(i-1,j), (i,j-1), (i-1,j-1)]
            m[i][j] = (first(minimum)+δ(A[i], B[j]), indexes[index])
    return m
```

擬似コードや数式ではインデックスを1から始めることが多いが，pythonは0からなので，頭の中でインデックスをずらしながらプログラムを書く必要がある．
それ以外は割とそのまま書けた．


### julia

```julia
δ(a,b) = (a - b)^2
# first(x) = x[1] firstは元からあるのでいらない
second(x) = x[2]

function minVal(v₁, v₂, v₃)
#    if first(v₁) ≦ minimum([first(v₂), first(v₃)])
    if first(v₁) <= minimum([first(v₂), first(v₃)])
        return v₁, 1
    elseif first(v₂) <= first(v₃)
        return v₂, 2
    else
        return v₃, 3
    end
end

function DTW(A, B)
    S = length(A)
    T = length(B)
    m = Matrix(S, T)
    m[1, 1] = [δ(A[1], B[1]), (0,0)]
    for i in 2:S
        m[i,1] = [m[i-1, 1][1] + δ(A[i], B[1]), [i-1, 1]]
    end
    for j in 2:T
        m[1,j] = [m[1, j-1][1] + δ(A[1], B[j]), [1, j-1]]
    end
    for i in 2:S
        for j in 2:T
            min, index = minVal(m[i-1,j], m[i,j-1], m[i-1,j-1])
            indexes = [[i-1, j], [i, j-1], [i-1, j-1]]
            m[i,j] = first(min) + δ(A[i],B[j]), indexes[index]
        end
    end
    return m
end
```

endがある分pythonより長い．一方でpythonでは使えない$v\_1$とかが使えるので，より忠実な感じになっている．


実際に書いてみるとわかるけど，インデックスが擬似コードと同じなのは結構大事で，全然頭を使わずにそのまま書けた．


### 実行速度
juliaはpythonに比べてずっと早いらしいので実行速度を比較した．
シェルのtime を使って実行速度を比較した．
コードの全体は[ここ] (https://github.com/kskkwn/notebooks/tree/master/julia)にある．

結果
``` 
julia dtw_julia.jl  2.62s user 0.30s system 110% cpu 2.641 total
python dtw_python.py  2.76s user 0.11s system 99% cpu 2.873 total
```

期待していたよりも全然速くならなかった．
実行時間が短くてコンパイルのオーバヘッドが大きいのかなと思ったから，forで10回実行するようにした結果が以下．

``` 
julia dtw_julia.jl  21.97s user 0.66s system 101% cpu 22.355 total
python dtw_python.py  25.81s user 0.78s system 99% cpu 26.591 total
```

多少速い気がするけど，期待としては数十倍とかだったので，いまいち．
よくわかってないけど，リストに色々な型の変数を入れるやり方だとそこまで速くならないのかも?

## Stochastic Block Model
Stochastic Block Model (SBM)は非対称関係データのクラスタリング手法．
崩壊ギブスサンプリングで事後確率からサンプリングして解く． (TODO 説明ちゃんとする)

pythonでの実装したときの記事はこっち．
[http://ksknw.hatenablog.com/entry/2017/04/23/194223:embed:cite]


[関係データ学習という本](http://www.kspub.co.jp/book/detail/1529212.html)にのっているクラスタzの事後確率に関する数式は以下． (TODO 数式が微妙に違うので直す)

z\_{1,i}}], [tex:{ \displaystyle
z\_{2,j}}]をサンプリングする。

他の変数がgivenだとした時の、[tex:{ \displaystyle
z\_{1,i}}]の事後確率は、

<img src="https://latex.codecogs.com/gif.latex?p(z_{1,i}=k|{\bf X},{\bf Z}_1^{\backslash(i)}}, {\bf Z}_2)" />
<img src="https://latex.codecogs.com/gif.latex?\varpropto \prod_{l=1}^L 
\hat{\alpha}_{1,k} \frac{\Gamma(\hat{a}_{k,l}+\hat{b}_{k,l})}{\Gamma(\hat{a}_{k,l})\Gamma(\hat{b}_{k,l})}
\frac{\Gamma\left(\hat{a}_{k,l}+\sum_{j=1}^{N_2}x_{i,j}\mathbb{I}(z_{2,j}=l)\right)\Gamma\left(\hat{b}_{k,l}+\sum_{j=1}^{N_2}(1-x_{i,j})\mathbb{I}(z_{2,j}=l)\right)}{\Gamma(\hat{a}_{k,l}+\hat{b}_{k,l}+\sum_{j=1}^{N_2}\mathbb{I}(z_{2,j}=l))}" />
ここで、

<img src="https://latex.codecogs.com/gif.latex?\hat{\alpha}_{1,k}=\alpha_{1,k}+\hat{m}_{1,k}" />

<img src="https://latex.codecogs.com/gif.latex?\hat{a}_{k,l}=a_0 + \hat{n}_{k,l}^{(+)}" />

<img src="https://latex.codecogs.com/gif.latex?\hat{b}_{k,l}=b_0 + \hat{n}_{k,l}^{(-)}" />

<img src="https://latex.codecogs.com/gif.latex?\hat{m}_{1,k} = \sum_{i'\neqi, i'=1}^{N_1}\mathbb{I}(z_{1,i'}=k)" />

<img src="https://latex.codecogs.com/gif.latex?\hat{n}_{k,l}^{(+)} = \sum_{i'\neqi, i'=1}^{N_1}\sum_{j=1}^{N_2}x_{i',j}\mathbb{I}(z_{1,i'}=k)\mathbb{I}(z_{2,j}=l)" />

<img src="https://latex.codecogs.com/gif.latex?\hat{n}_{k,l}^{(-)} = \sum_{i'\neqi, i'=1}^{N_1}\sum_{j=1}^{N_2}(1-x_{i',j})\mathbb{I}(z_{1,i'}=k)\mathbb{I}(z_{2,j}=l)" />

<img src="https://latex.codecogs.com/gif.latex?\Gamma" />はガンマ関数で、
<img src="https://latex.codecogs.com/gif.latex?K,L,a_0,b_0,\alpha_{1,k}" />はパラメータ

この確率を求める部分をpythonとjuliaで比較する．

### python

```python
nb_k = 8
α = 6
a0 = b0 = 0.5

import numpy as np
from numpy import exp
from scipy.special import loggamma as logΓ
from numpy.random import choice

m = lambda z: z.sum(axis=0)
α1 = α2 = np.ones(nb_k) * α


def onehot(i, nb_k):
    ret = np.zeros(nb_k)
    ret[i] = 1
    return ret


def update_z1(X, z1, z2):
    N1, N2 = X.shape

    m1 = m(z1)
    m2 = m(z2)

    new_z1 = []

    for i in range(N1):
        n_pos = np.einsum("ikjl, ij", np.tensordot(z1, z2, axes=0), X)  # n_pos_kl = n_pos[k][l]
        n_neg = np.einsum("ikjl, ij", np.tensordot(z1, z2, axes=0), 1 - X)
        # hatつきはi番目
        m1_hat = lambda i: m1 - z1[i]  # m1_hat_k = m1_hat[k]

        n_pos_hat = lambda i: n_pos - np.einsum("kjl, j", np.tensordot(z1, z2, axes=0)[i], X[i])
        n_neg_hat = lambda i: n_neg - np.einsum("kjl, j", np.tensordot(z1, z2, axes=0)[i], 1 - X[i])

        α_1_hat = lambda i: α1 + m1_hat(i)
        a_hat = lambda i: a0 + n_pos_hat(i)
        b_hat = lambda i: b0 + n_neg_hat(i)

        aᵢhat = a_hat(i)
        bᵢhat = b_hat(i)

        p_z1ᵢ_left = logΓ(aᵢhat + bᵢhat) - logΓ(aᵢhat) - logΓ(bᵢhat)
        p_z1ᵢ_right_upper = logΓ(aᵢhat + np.dot(X[i], z2)) + logΓ(bᵢhat + np.dot((1 - X[i]), z2))
        p_z1ᵢ_right_lower = logΓ(aᵢhat + bᵢhat + m2)
        p_z1ᵢ = (α_1_hat(i) * exp(p_z1ᵢ_left + p_z1ᵢ_right_upper - p_z1ᵢ_right_lower)).prod(axis=1)
        p_z1ᵢ = p_z1ᵢ.real
        p_z1ᵢ = p_z1ᵢ / p_z1ᵢ.sum()
        new_z1.append(onehot(choice(range(nb_k), p=p_z1ᵢ), nb_k))
    return new_z1
```

数式には$\hat{a}$や$n^+$などが頻出するが，pythonではこれらの文字を使うことができない．
このため，



### julia

```julia
function update_z₁(X, 𝕀z₁, 𝕀z₂)
    N₁, N₂ = size(X)
    m₁ = m(𝕀z₁)

    for i in 1:N₁
        @einsum n⁺[k,l] := X[i,j] * 𝕀z₁[i,k] * 𝕀z₂[j,l]
        @einsum n⁻[k,l] := (ones(X)[i,j] - X[i,j]) * 𝕀z₁[i,k] * 𝕀z₂[j,l]

        m̂₁ = m(𝕀z₁) - transpose(𝕀z₁[i,:])
        @einsum Σ⁺x𝕀z₂[i,l] := X[i,j] * 𝕀z₂[j,l]
        @einsum Σ⁻x𝕀z₂[i,l] := (ones(X)[i,j] - X[i,j]) * 𝕀z₂[j,l]
        @einsum n̂⁺[k,l] := n⁺[k,l] - 𝕀z₁[i,k] * Σ⁺x𝕀z₂[i,l]
        @einsum n̂⁻[k,l] := n⁻[k,l] - 𝕀z₁[i,k] * Σ⁻x𝕀z₂[i,l]

        α̂₁ = α₁ + m̂₁
        â = a₀ + n̂⁺
        b̂ = b₀ + n̂⁻

        temp⁺ = zeros(â)
        temp⁻ = zeros(â)
        temp = zeros(â)
        for j in 1:size(temp⁺)[1]
            temp⁺[j,:] = Σ⁺x𝕀z₂[i,:]
            temp⁻[j,:] = Σ⁻x𝕀z₂[i,:]
            temp[j,:] = sum(𝕀z₂,1)
        end

        @einsum p_z₁[k,l] := exp(logΓ(â + b̂)-logΓ(â)-logΓ(b̂)
            + logΓ(â+temp⁺)+logΓ(b̂+temp⁻)-logΓ(â+b̂+temp))[k,l]
        p_z₁ = α̂₁ .* transpose(prod(p_z₁, 2))
        p_z₁ /= sum(p_z₁)

        𝕀z₁[i,:] = onehot(sample(1:K, Weights(p_z₁[:])), K)
    end
    return 𝕀z₁
end

```




## おわりに

|     | 書きやすさの改善                | 実行速度の改善     |
|-----|---------------------------------|--------------------|
| DTW | インデックスが1から．数字の添字 | そんなに変わらない |
|     |                                 |                    |
