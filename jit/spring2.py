import numpy as np


def dist(x, y):
    return (x - y)**2


def get_min(m0, m1, m2, i, j):
    if m0 < m1:
        if m0 < m2:
            return i - 1, j, m0
        else:
            return i - 1, j - 1, m2
    else:
        if m1 < m2:
            return i, j - 1, m1
        else:
            return i - 1, j - 1, m2


def partial_dtw(x, y, epsilon):
    Tx = len(x)
    Ty = len(y)

    M = np.zeros((Tx, Ty))
    S = np.zeros((Tx, Ty, 2), int)
    B = np.zeros((Tx, Ty, 2), int)

    M[0, 0] = dist(x[0], y[0])
    for i in range(Tx):
        M[i, 0] = dist(x[i], y[0])
        S[i, 0] = [i, 0]
        B[i, 0] = [0, 0]

    for j in range(1, Ty):
        M[0, j] = M[0, j - 1] + dist(x[0], y[j])
        S[0, j] = S[0, j - 1]
        B[0, j] = [0, j - 1]

    for i in range(1, Tx):
        for j in range(1, Ty):
            pi, pj, m = get_min(M[i - 1, j],
                                M[i, j - 1],
                                M[i - 1, j - 1],
                                i, j)
            M[i, j] = dist(x[i], y[j]) + m
            S[i, j] = S[pi, pj]
            B[i, j] = [pi, pj]

    dmin = epsilon
    paths = []
    costs = []

    t_end = None
    for t in range(Tx):
        if M[t, -1] < dmin:
            dmin = M[t, -1]
            t_end = t

        if t_end is not None and (S[t, -1][0] > t_end) or t == (Tx - 1):
            costs.append(dmin)
            path = [[t_end, Ty - 1]]
            i = t_end
            j = Ty - 1
            while (sum(B[i, j] != [0, 0]) > 0):
                path.append(B[i, j])
                i, j = B[i, j].astype(int)
            paths.append(np.array(path))
            costs.append(dmin)
            dmin = epsilon
            t_end = None

    return costs, paths


if __name__ == '__main__':

    T = 500
    t = .4

    x = np.sin(np.array(range(T)) / 10)
    y = np.sin((np.array(range(T)) / 10 + t * np.pi))
