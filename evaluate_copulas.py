import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import math

from garch import GarchCopulaModel
from copulas.gaussian import gaussian_copula_logpdf


def clayton_logpdf_multivariate(u, theta):
    """
    Log-pdf of the d-dimensional Clayton copula (θ > 0) for samples u in (0,1)^d.
    """
    if theta <= 0:
        return -np.inf
    n, d = u.shape
    term = np.sum(np.log(u), axis=1)
    s = np.sum(u ** (-theta), axis=1) - d + 1.0
    if np.any(s <= 0):
        return -np.inf
    log_c = (theta * (d - 1) + 1.0) * term - (d + 1.0 / theta) * np.log(s) + np.log(math.factorial(d - 1))
    return np.sum(log_c)


def gumbel_logpdf_multivariate(u, theta):
    """
    Log-pdf of the d-dimensional Gumbel copula (θ >= 1) for samples u in (0,1)^d.
    """
    if theta < 1.0:
        return -np.inf
    n, d = u.shape
    x = -np.log(u)
    s = np.sum(x ** theta, axis=1)
    t = s ** (1.0 / theta)
    log_c = np.empty(n)
    for i in range(n):
        log_c[i] = -t[i]
    log_c += (theta - 1.0) * np.sum(np.log(x), axis=1)
    log_c += (2.0 - d * theta) * np.log(t)
    log_c += np.log(math.factorial(d - 1))
    log_c += np.log(theta - 1.0 + np.sum(x ** theta, axis=1))
    log_c -= np.sum(np.log(u), axis=1)
    return np.sum(log_c)


def evaluate_family(uniforms, family, gaussian_corr=None):
    n, dim = uniforms.shape
    if family == "gaussian":
        if gaussian_corr is None:
            raise ValueError("gaussian_corr must be provided for Gaussian copula evaluation")
        ll = np.sum(gaussian_copula_logpdf(uniforms, gaussian_corr))
        k = dim * (dim - 1) // 2
        return ll, k

    # Use a single θ estimated from Kendall's tau of the first pair (assumes exchangeability).
    tau = kendalltau(uniforms[:, 0], uniforms[:, 1]).correlation
    if family == "clayton":
        theta = 2.0 * tau / (1.0 - tau)
        ll = clayton_logpdf_multivariate(uniforms, theta)
        k = 1
        return ll, k
    elif family == "gumbel":
        theta = 1.0 / (1.0 - tau)
        ll = gumbel_logpdf_multivariate(uniforms, theta)
        k = 1
        return ll, k
    else:
        raise ValueError(f"Unsupported family: {family}")


def main():
    data = pd.read_csv("data.csv", index_col=0)
    log_returns = np.log(data).diff().dropna().values
    log_returns_std = (log_returns - log_returns.mean(axis=0)) / log_returns.std(axis=0)

    # Get PIT uniforms via IFM marginals
    model = GarchCopulaModel(log_returns_std, dist="t", copula="clayton")
    model.fit()
    uniforms = model.uniforms
    gaussian_corr = model.get_copula_correlation_matrix()

    results = {}
    for family in ["gaussian", "clayton", "gumbel"]:
        ll, k = evaluate_family(uniforms, family, gaussian_corr=gaussian_corr)
        aic = 2 * k - 2 * ll
        bic = k * np.log(len(uniforms)) - 2 * ll
        results[family] = {"loglik": ll, "AIC": aic, "BIC": bic}

    print("Goodness-of-fit (higher loglik, lower AIC/BIC is better):")
    for fam, stats in results.items():
        print(f"{fam:8s}  loglik={stats['loglik']:.2f}  AIC={stats['AIC']:.2f}  BIC={stats['BIC']:.2f}")


if __name__ == "__main__":
    main()
