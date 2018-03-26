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
        m̂₁ = m(𝕀z₁) - transpose(𝕀z₁[i,:])
        n⁺ = zeros(K, K)
        n⁻ = zeros(K, K)

        Σ⁺x𝕀z₂ = zeros(K)
        Σ⁻x𝕀z₂ = zeros(K)
        for l in 1:K
            for j in 1:N₂
                Σ⁺x𝕀z₂[l] += X[i,j] * 𝕀z₂[j,l]
                Σ⁻x𝕀z₂[l] += (1 - X[i,j]) * 𝕀z₂[j,l]
                for k in 1:K
                    n⁺[k,l] += 𝕀z₁[i,k] * X[i,j] *  𝕀z₂[j,l]
                    n⁻[k,l] += (1 - X[i,j]) * 𝕀z₁[i,k] * 𝕀z₂[j,l]
                end
            end
        end

        n̂⁺ = zeros(K,K)
        n̂⁻ = zeros(K,K)
        for k in 1:K
            for l in 1:K
                n̂⁺[k,l] += n⁺[k,l] - 𝕀z₁[i,k] * Σ⁺x𝕀z₂[l]
                n̂⁻[k,l] += n⁻[k,l] - 𝕀z₁[i,k] * Σ⁻x𝕀z₂[l]
            end
        end
        α̂₁ = α₁ + m̂₁
        â = a₀ + n̂⁺
        b̂ = b₀ + n̂⁻
        p_z₁ = α̂₁
        for k in 1:K
            for l in 1:K
                p_z₁[k] *= exp(logΓ(â[k,l] + b̂[k,l])-logΓ(â[k,l])-logΓ(b̂[k,l]) \
                    + logΓ(â[k,l]+Σ⁺x𝕀z₂[l])+logΓ(b̂[k,l]+Σ⁻x𝕀z₂[l])-logΓ(â[k,l]+b̂[k,l]+sum(𝕀z₂,1)[l]))
            end
        end
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
