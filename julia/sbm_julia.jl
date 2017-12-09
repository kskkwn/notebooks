using Einsum
using DataFrames
using CSV
using StatsBase

K = 8
α = 6
a₀ = b₀ = 0.5

α₁ = transpose(ones(K) * α) # =α₂
logΓ = lgamma

function m(x)
    return sum(x,1)
end

function onehot(i, K)
    ret = zeros(K)
    ret[i] = 1
    return ret
end

function update_z₁(X, 𝕀z₁, 𝕀z₂)
    N₁, N₂ = size(X)
    m₁ = m(𝕀z₁)

    for i in 1:N₁
        print(i)
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

data = readtable("./bi_data.csv")
X = hcat(data.columns...)
𝕀z₁ = zeros(size(X)[1],K)
𝕀z₁[:,1] = 1
𝕀z₂ = zeros(size(X)[1],K)
𝕀z₂[:,1] = 1

samples_𝕀z₁ = update_z₁(X, 𝕀z₁, 𝕀z₂)
samples_𝕀z₂ = update_z₁(transpose(X), 𝕀z₂, 𝕀z₁)
