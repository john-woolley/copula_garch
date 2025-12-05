import numpy as np
import pandas as pd

from garch import GarchCopulaModel


def test_copula_matches_residual_correlation():
    """
    IFM property: the fitted copula correlation should align with the empirical
    correlation of Gaussian scores derived from the PIT uniforms.
    """
    data = pd.read_csv("data.csv", index_col=0)
    log_returns = np.log(data).diff().dropna().values
    log_returns_std = (log_returns - log_returns.mean(axis=0)) / log_returns.std(axis=0)

    model = GarchCopulaModel(log_returns_std, dist="gaussian")
    model.fit()

    cop_corr = model.get_copula_correlation_matrix()
    emp_corr = model.empirical_corr
    lower = np.tril_indices_from(cop_corr, k=-1)
    diff = np.abs(cop_corr[lower] - emp_corr[lower])

    # Allow modest deviation; we only assert the copula matches residual dependence, not raw returns.
    assert diff.max() < 0.15, f"Copula correlation deviates from residual correlation by {diff.max():.3f}"
