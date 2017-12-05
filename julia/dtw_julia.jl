T = 1000
A = sin((1:T)/10)
B = sin((1:T)/10 + pi*0.4)

δ(a,b) = (a - b)^2
# first(x) = x[1] firstは元からあるのでいらない
second(x) = x[2]

function minVal(v₁, v₂, v₃)
    if first(v₁) ≤ minimum([first(v₂), first(v₃)])
        return v₁, 1
    elseif first(v₂) ≤ first(v₃)
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

for i in 1:10
    DTW(A,B)
end
