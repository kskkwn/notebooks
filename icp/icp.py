import numpy as np
import pylab as plt
from scipy.linalg import svd
from scipy.spatial import KDTree
from torch_geometric.datasets import ShapeNet, ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader


def calcTransformation(X, Y):
    centroid_X = X.mean(axis=0)
    centroid_Y = Y.mean(axis=0)

    cov_XY = (X - centroid_X).T @ (Y - centroid_Y)

    U, S, Vt = svd(cov_XY)
    R = Vt.T @ U.T
    T = -R @ centroid_X + centroid_Y

    return R, T


def icp(X, Y):
    kdtree_X = KDTree(X)
    ids = kdtree_X.query(Y)[1]

    for i in range(100):
        R, T = calcTransformation(X[ids], Y)
        ids = kdtree_X.query(Y @ R + T)[1]

    return Y @ R + T


def main(npoints=500, path="./data/", categories=["Airplane"],
         split="train"):
    pre_transform, transform = T.NormalizeScale(), T.FixedPoints(npoints)
    path += '/ShapeNet'
    dataset = ShapeNet(path, categories=categories,
                       split=split,
                       transform=transform,
                       pre_transform=pre_transform)
    batch_size = 2
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    X = next(iter(loader))
    points_X = X.pos.reshape(-1, npoints, 3)[0].detach().numpy()
    points_Y = X.pos.reshape(-1, npoints, 3)[1].detach().numpy()

    transformed_Y = icp(points_X, points_Y)


if __name__ == '__main__':
    main()
