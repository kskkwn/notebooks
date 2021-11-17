from sklearn.neighbors import NearestNeighbors
import numpy as np
import ot
from itertools import combinations


def wass_distance(x, y):
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]

    M = np.sum((x[None] - y[:, None])**2, axis=2)
    return ot.emd2(a, b, M)**0.5


X = np.array([[[-1, -1], [-2, -1]], [[-3, -2], [1, 1]], [[2, 1], [3, 2]]])
C = np.zeros((X.shape[0], X.shape[0]))

for i, j in combinations(range(X.shape[0]), 2):

    [C[i, j]= wass_distance(x, y) for]

    [wass_distance(x, y) for x, y in permutate]


nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric=wass_distance).fit(X)
distances, indicies = nbrs.kneighbors(X)
print(indicies)
