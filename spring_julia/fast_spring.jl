using CSV

function dist(x::Float64, y::Float64)::Float64
    (x-y)^2
end

function get_min(m₀::Float64, m₁::Float64, m₂::Float64,
                 i::Int64, j::Int64)::Tuple{Int64, Int64, Float64}
    if m₀ < m₁
        m₀ < m₂ && return i-1, j, m₀
        return i-1, j-1, m₂
    else
        m₁ < m₂ && return i, j-1, m₁
        return i-1, j-1, m₂
    end
end

function spring(x::Array{Float64,1}, y::Array{Float64,1}, ϵ::Int64)
    pathes = []

    Tx = length(x)
    Ty = length(y)

    C = zeros(Float64, (Tx, Ty))
    B = ones(Int64, (Tx, Ty, 2))
    S = ones(Int64, (Tx, Ty))

    @inbounds C[1,1] = dist(x[1], y[1])

    for j in 2:Ty
        @inbounds C[1, j] = C[1, j-1] + dist(x[1], y[j])
        @inbounds S[1, j] = 1
        @inbounds B[1, j, :] = [1, j-1]
    end

    for i in 2:Tx
        @inbounds C[i, 1] = dist(x[i], y[1])
        @inbounds S[i, 1] = i
        @inbounds B[i, 1, :] = [1, 1]

        for j in 2:Ty
            pi, pj, m = get_min(C[i-1, j],
                                C[i, j-1],
                                C[i-1, j-1],
                                i, j)
            @inbounds C[i,j] = dist(x[i], y[j]) + m
            @inbounds B[i,j,:] = [pi, pj]
            @inbounds S[i,j] = S[pi, pj]
        end

        imin = argmin(C[1:i, end])
        dmin = C[imin, end]

        dmin > ϵ && continue

        for j in 2:Ty
            C[i,j] < dmin && S[i,j] < imin && @goto yet
        end

        path = []
        push!(path, [imin, Ty])
        tempᵢ = imin
        tempⱼ = Ty

        while (B[tempᵢ, tempⱼ, 1] != 1 || B[tempᵢ, tempⱼ, 2] != 1)
            push!(path, B[tempᵢ, tempⱼ, :])
            tempᵢ, tempⱼ = B[tempᵢ, tempⱼ, :]
        end

        @inbounds C[S .<= imin] .= Inf
        push!(pathes, path)
        @label yet
    end
    return pathes, C
end


function fast_spring()
    df = CSV.read("./data.csv", header=false, delim=",")
    data = df[!, 2]
    x = data[1:4:1000]
    y = data[1000:4:end]
    ϵ = 80

    pathes, C = spring(x, y, ϵ)
    println(pathes)
end

fast_spring()
