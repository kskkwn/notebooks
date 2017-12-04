# -*- coding: utf-8 -*-

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
