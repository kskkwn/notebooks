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
