import numpy as np
from numpy import exp
from copy import deepcopy
import pylab as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.stats import norm, gamma
from tqdm import tqdm
from numpy import random
sns.set_style("white")

true_μ = 3
true_σ = 1
# nb_data = 1000
nb_data = 500

x = np.random.normal(true_μ, true_σ, nb_data)

print(x.mean(), x.std())

norm_lpdf = lambda μ, σ: np.sum(norm.logpdf(x, μ, σ))
gamma_lpdf = lambda a: np.sum(gamma.logpdf(x, a))

Δ_max = 1000
θ0 = np.array([random.randn(1), random.gamma(1)])
ε = 0.1
L = norm_lpdf
M = 100
list_θₘ = [θ0]


def log_dh(μ, σ):
    return np.array([-np.sum(x - μ) / σ**2,
                     len(x) / (2 * σ**2) - np.sum((x - μ)**2) / (2 * σ**4)])


def H(θₜ, p):
    return -norm_lpdf(θₜ[0], θₜ[1]) + 0.5 * np.dot(p, p)


def Leapfrog(x, θ, r, ε):
    θ_d = deepcopy(θ)
    r_d = deepcopy(r)
    r_d -= 0.5 * ε * log_dh(θ_d[0], θ_d[1])
    θ_d[0] = θ_d[0] + ε * r[0]
    θ_d[1] = θ_d[1] + ε * r[1]
    r_d -= 0.5 * ε * log_dh(θ_d[0], θ_d[1])
    return θ_d, r_d


def BuildTree(θ, r, u, v, j, ε):
    if j == 0:
        θd, rd = Leapfrog(x, θ, r, v * ε)
        if np.log(u) <= L(*θd) - 0.5 * np.dot(r, r):
            Cd = [[θd, rd]]
        else:
            Cd = []
        sd = (np.log(u) < Δ_max + L(*θd) - 0.5 * np.dot(r, r)).astype(float)
        return θd, rd, θd, rd, Cd, sd
    else:
        θ_minus, r_minus, θ_plus, r_plus, Cd, sd = BuildTree(θ, r, u, v, j - 1, ε)
        if v == -1:
            θ_minus, r_minus, _, _, Cdd, sdd = BuildTree(
                θ_minus, r_minus, u, v, j - 1, ε)
        else:
            _, _, θ_plus, r_plus, Cdd, sdd = BuildTree(
                θ_plus, r_plus, u, v, j - 1, ε)
        sd = sdd * sd * ((np.dot(θ_plus - θ_minus, r_minus) >= 0).astype(float)) \
            * ((np.dot(θ_plus - θ_minus, r_plus) >= 0).astype(float))
        Cd.extend(Cdd)

        return θ_minus, r_minus, θ_plus, r_plus, Cd, sd


for m in tqdm(range(M)):
    r0 = random.randn(2)
    u = random.uniform(0, exp(L(*θ0) - 0.5 * np.dot(r0, r0)))

    θ_minus = deepcopy(list_θₘ[-1])
    θ_plus = deepcopy(list_θₘ[-1])
    r_minus = r0
    r_plus = r0
    j = 0
    C = [[list_θₘ[-1], r0]]
    s = 1

    while s == 1:
        v = random.choice([-1, 1])
        if v == -1:
            θ_minus, r_minus, _, _, Cd, sd = BuildTree(θ_minus, r_minus, u, v, j, ε)
        else:
            _, _, θ_plus, r_plus, Cd, sd = BuildTree(θ_plus, r_plus, u, v, j, ε)
        if sd == 1:
            C.extend(Cd)
        s = sd * ((np.dot(θ_plus - θ_minus, r_minus) >= 0).astype(float)) \
            * ((np.dot(θ_plus - θ_minus, r_plus) >= 0).astype(float))
        j += 1
        print(r"%d" % j)

    index = random.choice(list(range(len(C))))
    list_θₘ.append(C[index][0])

for i in list_θₘ:
    print(i)
print(exp(L(*θ0) - 0.5 * np.dot(r0, r0)))
