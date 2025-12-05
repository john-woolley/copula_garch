import numpy as np
import numba as nb
from math import gamma, sqrt, pi

@nb.njit(fastmath=True)
def pdf(x, mean, scale, df):
    """
    PDF of the symmetric Student-t distribution with location/scale.
    """
    z = (x - mean) / scale
    coef = gamma((df + 1) / 2.0) / (sqrt(df * pi) * gamma(df / 2.0))
    return coef * (1.0 + (z * z) / df) ** (-(df + 1) / 2.0) / scale

@nb.njit(fastmath=True)
def logpdf(x, mean, scale, df):
    """
    Log-PDF of the symmetric Student-t distribution with location/scale.
    """
    z = (x - mean) / scale
    log_coef = np.log(gamma((df + 1) / 2.0)) - np.log(np.sqrt(df * np.pi)) - np.log(gamma(df / 2.0)) - np.log(scale)
    return log_coef - ((df + 1) / 2.0) * np.log(1.0 + (z * z) / df)

@nb.njit(parallel=True, fastmath=True)
def pdf_batch(x, mean, scale, df):
    out = np.empty(x.shape[0])
    for i in nb.prange(x.shape[0]):
        out[i] = pdf(x[i], mean, scale, df)
    return out

@nb.njit(fastmath=True)
def cdf(x, mean, scale, df, n_points=2000):
    """
    Numerical CDF via trapezoidal integration of the PDF.
    """
    z = (x - mean) / scale
    if z < 0:
        return 1.0 - cdf(2 * mean - x, mean, scale, df, n_points)

    span = 12.0
    a = -span
    b = z
    dx = (b - a) / n_points
    area = 0.0
    for i in range(n_points):
        t1 = a + i * dx
        t2 = a + (i + 1) * dx
        area += (pdf(t1 * scale + mean, mean, scale, df) + pdf(t2 * scale + mean, mean, scale, df)) * 0.5 * dx
    return area

@nb.njit(fastmath=True)
def rvs(mean, scale, df, size=1):
    """
    Random variates from the symmetric Student-t distribution.
    """
    out = np.empty(size)
    for i in range(size):
        chi2_sample = np.random.chisquare(df)
        z = np.random.randn() * sqrt(df / chi2_sample)
        out[i] = mean + scale * z
    return out

@nb.njit(fastmath=True)
def ppf(u, mean, scale, df, tol=1e-6, max_iter=100):
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
        cmid = cdf(mid, mean, scale, df)
        if abs(cmid - u) < tol:
            return mid
        if cmid < u:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
