import numpy as np
import numba as nb

@nb.njit()
def rvs(mean, var, size=1):
    """
    Generate random samples from a univariate normal distribution.

    Parameters:
        mean: Mean of the distribution (scalar).
        var: Variance of the distribution (scalar, > 0).
        size: Number of samples to generate (int, default=1).
    
    Returns:
        Array of random samples (1D array of length 'size').
    """
    std = np.sqrt(var)
    z = np.random.randn(size)  # Standard normal samples
    return mean + std * z

@nb.njit()
def cdf(x, mean, var):
    """
    Compute the CDF of a univariate normal distribution.

    Parameters:
        x: Points at which to evaluate the CDF (1D array).
        mean: Mean of the distribution (scalar).
        var: Variance of the distribution (scalar, > 0).
    
    Returns:
        CDF values (1D array).
    """
    std = np.sqrt(var)
    z = (x - mean) / std
    return 0.5 * (1 + erf(z / np.sqrt(2)))

@nb.njit()
def ppf(u, mean, var):
    """
    Compute the percent point function (inverse CDF) of a univariate normal distribution.

    Parameters:
        u: Array of probabilities (1D array in [0, 1]).
        mean: Mean of the distribution (scalar).
        var: Variance of the distribution (scalar, > 0).
    
    Returns:
        Inverse CDF values (1D array).
    """
    std = np.sqrt(var)
    return mean + std * np.sqrt(2) * erfinv(2 * u - 1)

@nb.njit()
def erf(x):
    """
    Approximation of the error function for a vector of values.

    Parameters:
        x: 1D numpy array of input values.

    Returns:
        1D numpy array of error function values.
    """
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        t = 1.0 / (1.0 + 0.5 * np.abs(x[i]))
        tau = t * np.exp(-x[i]**2 - 1.26551223 +
                         1.00002368 * t +
                         0.37409196 * t**2 +
                         0.09678418 * t**3 -
                         0.18628806 * t**4 +
                         0.27886807 * t**5 -
                         1.13520398 * t**6 +
                         1.48851587 * t**7 -
                         0.82215223 * t**8 +
                         0.17087277 * t**9)
        result[i] = 1.0 - tau if x[i] >= 0 else tau - 1.0
    return result

@nb.njit()
def erfinv(y):
    """
    Approximation of the inverse error function for a vector of values.

    Parameters:
        y: 1D numpy array of input values in [-1, 1].

    Returns:
        1D numpy array of inverse error function values.
    """
    result = np.empty_like(y)
    for i in range(y.shape[0]):
        if y[i] == 0:
            result[i] = 0
        elif y[i] == 1:
            result[i] = np.inf
        elif y[i] == -1:
            result[i] = -np.inf
        ln_term = np.log(1 - y[i]**2)
        x = np.sqrt(np.sqrt(ln_term**2 - (2 * ln_term / 0.147)) - ln_term)
        result[i] = x if y[i] > 0 else -x
    return result
