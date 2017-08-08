import numpy as np


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1]]
    stds = [[0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x

N = 500  # number of data points
D = 2  # dimensionality of data

x_train = build_toy_dataset(N)


# %matplotlib inline
import pylab as plt
import matplotlib.cm as cm

plt.style.use('ggplot')

plt.scatter(x_train[:, 0], x_train[:, 1])
plt.axis([-3, 3, -3, 3])
plt.show()

import tensorflow as tf
import edward as ed
from edward.models import Categorical, InverseGamma, Mixture, \
    MultivariateNormalDiag, Normal, Dirichlet
K = 2  # number of components

pi = Dirichlet(alpha=tf.ones(K))
mu = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))
sigma = InverseGamma(alpha=tf.ones([K, D]), beta=tf.ones([K, D]))
cat = Categorical(logits=tf.zeros([N, K]))
components = [
    MultivariateNormalDiag(mu=tf.ones([N, 1]) * tf.gather(mu, k),
                           diag_stdev=tf.ones([N, 1]) * tf.gather(sigma, k))
    for k in range(K)]
x = Mixture(cat=cat, components=components)


# qpi = Dirichlet(
#     alpha=tf.Variable(tf.random_normal([K]))
# )

qmu = Normal(
    mu=tf.Variable(tf.random_normal([K, D])),
    sigma=tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))
qsigma = InverseGamma(
    alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))),
    beta=tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

data = {x: x_train}
# inference = ed.KLqp({pi: qpi, mu: qmu, sigma: qsigma}, data)
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data)
inference.run(n_iter=4000, n_samples=20)

# Average per-cluster and per-data point likelihood over many posterior samples.
log_liks = []
for _ in range(100):
    mu_sample = qmu.sample()
    sigma_sample = qsigma.sample()
    # Take per-cluster and per-data point likelihood.
    log_lik = []
    for k in range(K):
        x_post = Normal(mu=tf.ones([N, 1]) * tf.gather(mu_sample, k),
                        sigma=tf.ones([N, 1]) * tf.gather(sigma_sample, k))
        log_lik.append(tf.reduce_sum(x_post.log_prob(x_train), 1))

    log_lik = tf.pack(log_lik)  # has shape (K, N)
    log_liks.append(log_lik)

log_liks = tf.reduce_mean(log_liks, 0)

clusters = tf.argmax(log_liks, 0).eval()

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()
