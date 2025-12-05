import numpy as np
import numba as nb
from math import gamma, sqrt, pi

@nb.njit(fastmath=True)
def t_pdf(x, nu):
    """
    Probability density function of the univariate t-distribution.
    """
    coef = gamma((nu + 1) / 2.0) / (sqrt(nu * pi) * gamma(nu / 2.0))
    return coef * (1.0 + (x**2) / nu) ** (-(nu + 1) / 2.0)

@nb.njit(fastmath=True, parallel=True)
def t_cdf(x, nu, n_points=1000):
    """
    CDF of the univariate t-distribution using numerical integration.
    """
    if x < 0:
        return 1.0 - t_cdf(-x, nu, n_points)

    # Integration bounds
    a, b = -10.0, x
    dx = (b - a) / n_points
    cdf_val = 0.0
    for i in nb.prange(n_points):
        t1 = a + i * dx
        t2 = a + (i + 1) * dx
        cdf_val += (t_pdf(t1, nu) + t_pdf(t2, nu)) * 0.5 * dx

    return cdf_val

@nb.njit(fastmath=True)
def rvs(mean, scale, df, skew, max_tries=1000):
    """
    Generate a random sample from the univariate skew-t distribution.
    """
    for _ in range(max_tries):
        # Generate a univariate t sample
        chi2_sample = np.random.chisquare(df)
        t_sample = np.random.randn() * sqrt(df / chi2_sample)

        # Apply skew
        latent_sample = t_sample + skew
        if latent_sample > 0:
            return mean + scale * t_sample

    # If not accepted after max_tries
    return np.nan

@nb.njit(fastmath=True)
def pdf(x, mean, scale, df, skew):
    """
    PDF of the univariate skew-t distribution.
    """
    z = (x - mean) / scale
    t_val = t_pdf(z, df) / scale

    # Skew adjustment
    w = skew * z / sqrt(1 + (z**2) / df)
    cdf_val = t_cdf(w, df + 1)
    return 2.0 * t_val * cdf_val

@nb.njit(fastmath=True)
def logpdf(x, mean, scale, df, skew):
    """
    Log of the PDF for the univariate skew-t distribution.
    """
    z = (x - mean) / scale
    log_t = np.log(t_pdf(z, df)) - np.log(scale)

    # Skew adjustment
    w = skew * z / sqrt(1 + (z**2) / df)
    cdf_val = t_cdf(w, df + 1)
    if cdf_val < 1e-15:
        cdf_val = 1e-15
    return np.log(2.0) + log_t + np.log(cdf_val)

@nb.njit(parallel=True, fastmath=True)
def rvs_batch(mean, scale, df, skew, n_samples):
    """
    Generate a batch of random samples from the univariate skew-t distribution.
    """
    samples = np.empty(n_samples)
    for i in nb.prange(n_samples):
        samples[i] = rvs(mean, scale, df, skew)
    return samples

@nb.njit(parallel=True, fastmath=True)
def pdf_batch(x, mean, scale, df, skew):
    """
    Batch evaluation of the PDF for the univariate skew-t distribution.
    """
    result = np.empty(x.shape[0])
    for i in nb.prange(x.shape[0]):
        result[i] = pdf(x[i], mean, scale, df, skew)
    return result

@nb.njit(parallel=True, fastmath=True)
def logpdf_batch(x, mean, scale, df, skew):
    """
    Batch evaluation of the log-PDF for the univariate skew-t distribution.
    """
    result = np.empty(x.shape[0])
    for i in nb.prange(x.shape[0]):
        result[i] = logpdf(x[i], mean, scale, df, skew)
    return result

@nb.njit(fastmath=True)
def cdf(x, mean, scale, df, skew, n_points=2000):
    """
    Numerical CDF for the skew-t distribution using trapezoidal integration of the PDF.
    """
    # Adaptive bounds centered at the point of interest.
    span = 12.0 * scale
    a = -span + mean
    b = x
    if b < a:
        return 0.0

    dx = (b - a) / n_points
    area = 0.0
    for i in range(n_points):
        t1 = a + i * dx
        t2 = a + (i + 1) * dx
        area += (pdf(t1, mean, scale, df, skew) + pdf(t2, mean, scale, df, skew)) * 0.5 * dx
    return area

@nb.njit(fastmath=True)
def ppf(u, mean, scale, df, skew, tol=1e-6, max_iter=100):
    """
    Inverse CDF via bisection for u in (0,1).
    """
    if u <= 0.0:
        return -np.inf
    if u >= 1.0:
        return np.inf

    low = mean - 20.0 * scale
    high = mean + 20.0 * scale

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        cmid = cdf(mid, mean, scale, df, skew)
        if abs(cmid - u) < tol:
            return mid
        if cmid < u:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
