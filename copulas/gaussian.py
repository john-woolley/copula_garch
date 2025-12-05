import numpy as np
import numba as nb
from marginals.gaussian import cdf, ppf

@nb.njit()
def validate_corr_matrix(corr_matrix):
    if not np.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("Correlation matrix must be symmetric.")
    if np.linalg.eigvalsh(corr_matrix).min() <= 0:
        raise ValueError("Correlation matrix must be positive definite.")

@nb.njit()
def gaussian_copula_sample(corr_matrix, n):
    dim = corr_matrix.shape[0]
    mean = np.zeros(dim)
    std = np.ones(dim)
    L = np.linalg.cholesky(corr_matrix)
    z = np.random.randn(n, dim)
    mvn_samples = z @ L.T
    uniform_samples = np.empty((n, dim))
    for i in range(dim):
        uniform_samples[:, i] = cdf(mvn_samples[:, i], mean[i], std[i])
    return uniform_samples

@nb.njit()
def gaussian_copula_logpdf(u, corr_matrix):
    n, dim = u.shape
    z = np.empty_like(u)
    for i in range(dim):
        z[:, i] = ppf(u[:, i], 0, 1)
    mean = np.zeros(dim)
    L = np.linalg.cholesky(corr_matrix)
    inv_L = np.linalg.inv(L)
    det_corr = np.prod(np.diag(L)) ** 2
    log_joint = np.empty(n)
    for i in range(n):
        x = z[i]
        temp = inv_L @ x
        log_joint[i] = -0.5 * (np.sum(temp ** 2) + dim * np.log(2 * np.pi) + np.log(det_corr))
    log_marginals = np.sum(-0.5 * (z ** 2 + np.log(2 * np.pi)), axis=1)
    return log_joint - log_marginals

class GaussianCopula:
    def __init__(self, corr_matrix):
        validate_corr_matrix(corr_matrix)
        self.corr_matrix = corr_matrix

    def sample(self, n):
        return gaussian_copula_sample(self.corr_matrix, n)

    def logpdf(self, u):
        return gaussian_copula_logpdf(u, self.corr_matrix)

if __name__ == "__main__":
    corr_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    copula = GaussianCopula(corr_matrix)
    samples = copula.sample(1000)

    # Transform copula samples to a normal marginal distribution
    mean, var = 0, 1  # Example marginal parameters
    transformed_samples = np.empty_like(samples)
    for i in range(samples.shape[1]):
        transformed_samples[:, i] = ppf(samples[:, i], mean, var)

    print("Transformed samples (first 5):", transformed_samples[:5])
