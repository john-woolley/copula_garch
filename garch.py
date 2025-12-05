import numpy as np
import numba as nb
from tqdm import tqdm
from marginals.gaussian import cdf as gaussian_cdf, ppf as gaussian_ppf
from marginals.skewt import cdf as skewt_cdf, logpdf as skewt_logpdf
from marginals.student_t import cdf as student_t_cdf, logpdf as student_t_logpdf
from copulas.gaussian import gaussian_copula_logpdf
from copulas.vine import DVineCopula
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import t as student_t

@nb.njit()
def compute_garch_variance(data, omega, alpha, beta):
    """
    Compute the conditional variances for a GARCH(1, 1) model.

    Parameters:
        data: 1D numpy array.
            Observed time series.
        omega: Scalar.
            GARCH omega parameter.
        alpha: Scalar.
            GARCH alpha parameter.
        beta: Scalar.
            GARCH beta parameter.

    Returns:
        1D numpy array of conditional variances.
    """
    n = len(data)
    var = np.empty(n)
    denom = 1.0 - alpha - beta
    # Use unconditional variance when the process is stationary; fall back to sample variance otherwise.
    if denom > 1e-6 and omega > 0:
        var[0] = omega / denom
    else:
        var[0] = np.var(data)
    for t in range(1, n):
        var[t] = omega + alpha * (data[t - 1] ** 2) + beta * var[t - 1]
    return var

def fit_ar1(series):
    """
    Fit AR(1) with intercept via OLS and return (mu, phi).
    """
    if len(series) < 2:
        return 0.0, 0.0
    y = series[1:]
    X = np.column_stack((np.ones(len(y)), series[:-1]))
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return params[0], params[1]

def ar1_residuals(series, mu, phi):
    """
    Compute AR(1) residuals for a series.
    """
    n = len(series)
    res = np.empty(n)
    res[0] = series[0] - mu
    res[1:] = series[1:] - (mu + phi * series[:-1])
    return res

def t_copula_logpdf(u, corr_matrix, df):
    """
    Log-pdf of a t-copula for uniforms u, correlation matrix, and degrees of freedom df.
    """
    from numpy.linalg import cholesky, solve

    n, dim = u.shape
    z = student_t.ppf(u, df)  # shape (n, dim)
    L = cholesky(corr_matrix)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    inv_L_z = solve(L, z.T).T
    quad = np.sum(inv_L_z ** 2, axis=1)

    log_const = (
        np.log(np.math.gamma((df + dim) / 2.0))
        - np.log(np.math.gamma(df / 2.0))
        - 0.5 * log_det
        - (dim / 2.0) * (np.log(df) + np.log(np.pi))
    )
    log_joint = log_const - ((df + dim) / 2.0) * np.log(1.0 + quad / df)
    log_marginals = np.sum(student_t.logpdf(z, df), axis=1)
    return log_joint - log_marginals

def clayton_logpdf(u, theta):
    """
    Log-pdf of a d-dimensional Clayton copula with parameter theta>0.
    """
    if theta <= 0:
        return -np.inf
    n, d = u.shape
    term = np.sum(np.log(u), axis=1)
    s = np.sum(u ** (-theta), axis=1) - d + 1.0
    if np.any(s <= 0):
        return -np.inf
    log_c = (theta * (d - 1) + 1.0) * term - (d + 1.0 / theta) * np.log(s) + np.log(np.math.factorial(d - 1))
    return np.sum(log_c)

def build_corr_matrix(cholesky_params, dim):
    """
    Build a correlation matrix from unconstrained Cholesky parameters using a tanh transform.
    """
    L = np.eye(dim)
    index = 0
    for i in range(1, dim):
        for j in range(i):
            L[i, j] = np.tanh(cholesky_params[index])
            index += 1
    corr_matrix = L @ L.T
    diag = np.sqrt(np.diag(corr_matrix))
    return corr_matrix / np.outer(diag, diag)

@nb.njit()
def garch_loglikelihood_nb(params, data):
    """
    Compute the negative log likelihood of a univariate GARCH(1, 1) model with Numba.

    Parameters:
        params: Array-like, [omega, alpha, beta].
            GARCH(1, 1) parameters.
        data: 1D numpy array.
            Observed time series.

    Returns:
        Negative log likelihood.
    """
    omega, alpha, beta = params
    var = compute_garch_variance(data, omega, alpha, beta)
    ll = 0.0
    for t in range(1, len(data)):
        ll += -0.5 * (np.log(2 * np.pi) + np.log(var[t]) + (data[t] ** 2) / var[t])
    return -ll

@nb.njit()
def transform_to_uniform(residuals, variances):
    """
    Standardize residuals and transform to uniform distribution using Gaussian CDF.

    Parameters:
        residuals: 1D numpy array.
            Residuals of the time series.
        variances: 1D numpy array.
            Conditional variances from the GARCH model.

    Returns:
        1D numpy array of transformed uniform values.
    """
    std_residuals = residuals / np.sqrt(variances)
    return gaussian_cdf(std_residuals, 0, 1)

def transform_to_uniform_t(residuals, variances, df, cdf_func, vectorized=False):
    """
    Transform standardized residuals to uniforms using a symmetric t CDF.
    """
    std_residuals = residuals / np.sqrt(variances)
    if vectorized:
        return cdf_func(std_residuals, 0.0, 1.0, df)
    u = np.empty_like(std_residuals)
    for i in range(len(std_residuals)):
        u[i] = cdf_func(std_residuals[i], 0.0, 1.0, df)
    return u

def transform_to_uniform_skewt(residuals, variances, df, skew):
    """
    Transform standardized residuals to uniforms using a skew-t CDF.
    """
    std_residuals = residuals / np.sqrt(variances)
    u = np.empty_like(std_residuals)
    for i in range(len(std_residuals)):
        u[i] = skewt_cdf(std_residuals[i], 0.0, 1.0, df, skew)
    return u

@nb.njit()
def joint_loglikelihood_nb(params, data, dim):
    """
    Compute the joint log likelihood of GARCH marginals and the copula.

    Parameters:
        params: Array-like, flattened GARCH and Cholesky parameters.
            - First `3 * dim` values are GARCH parameters (omega, alpha, beta for each marginal).
            - Remaining values are the off-diagonal elements of the Cholesky factor.
        data: 2D numpy array.
            Observed time series for each dimension (shape: [n, dim]).
        dim: int.
            Dimensionality of the data (number of marginals).

    Returns:
        Negative joint log likelihood.
    """
    n = data.shape[0]

    # Extract GARCH and Cholesky parameters
    garch_params = params[:3 * dim].reshape(dim, 3)
    cholesky_params = params[3 * dim:]

    # Reconstruct the Cholesky factor
    L = np.zeros((dim, dim))
    index = 0
    for i in range(dim):
        L[i, i] = 1  # Set diagonal elements to 1
        for j in range(i):
            L[i, j] = np.tanh(cholesky_params[index])
            index += 1

    # Reconstruct the correlation matrix
    corr_matrix = L @ L.T
    diag = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(diag, diag)

    # Fit GARCH marginals and compute CDF-transformed residuals
    residuals = np.empty_like(data)
    for i in range(dim):
        omega, alpha, beta = garch_params[i]
        variances = compute_garch_variance(data[:, i], omega, alpha, beta)
        residuals[:, i] = transform_to_uniform(data[:, i], variances)

    # Compute copula log likelihood
    copula_ll = np.sum(gaussian_copula_logpdf(residuals, corr_matrix))

    # Compute joint negative log likelihood
    joint_nll = -copula_ll
    for i in range(dim):
        joint_nll += garch_loglikelihood_nb(garch_params[i], data[:, i])

    return joint_nll  # Negative log likelihood for minimization



class GarchCopulaModel:
    def __init__(self, data, dist="gaussian", df=8.0, skew=0.0, copula="gaussian", copula_df=8.0, t_backend="numba"):
        """
        Initialize the GARCH-Copula model.

        Parameters:
            data: 2D numpy array.
                Observed time series for each dimension (shape: [n, dim]).
            dist: str.
                Marginal distribution for PIT ('gaussian', 't', or 'skewt').
            df: float.
                Degrees of freedom for t / skew-t (when used).
            skew: float.
                Skew parameter for skew-t (when used).
            copula: str.
                Copula family ('gaussian', 't', or 'vine').
            copula_df: float.
                Degrees of freedom for t copula (when used).
        """
        self.data = data
        self.n, self.dim = data.shape
        self.dist = dist
        self.df = df
        self.skew = skew
        self.copula = copula
        self.copula_df = copula_df
        self.t_backend = t_backend
        if t_backend == "scipy":
            # SciPy vectorized CDF/logpdf
            self.student_t_cdf = lambda x, mean, scale, d: student_t.cdf(x, d, loc=mean, scale=scale)
            self.student_t_logpdf = lambda x, mean, scale, d: student_t.logpdf(x, d, loc=mean, scale=scale)
            self.student_t_vectorized = True
        else:
            self.student_t_cdf = student_t_cdf
            self.student_t_logpdf = student_t_logpdf
            self.student_t_vectorized = False

    def fit(self):
        """
        IFM two-stage fit: (1) univariate GARCH + marginal tail/skew per series, (2) Gaussian copula on PIT uniforms.

        Returns:
            Optimized parameters for GARCH and copula.
        """
        # Fit each marginal GARCH independently
        garch_params = np.empty((self.dim, 3))
        ar_params = np.empty((self.dim, 2))
        for i in range(self.dim):
            mu, phi = fit_ar1(self.data[:, i])
            ar_params[i] = (mu, phi)
            residual_series = ar1_residuals(self.data[:, i], mu, phi)
            initial = np.array([0.01, 0.1, 0.85])
            bounds = [(1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]

            result = minimize(
                garch_loglikelihood_nb,
                initial,
                args=(residual_series,),
                bounds=bounds,
                options={'maxiter': 500}
            )
            if not result.success:
                raise RuntimeError(f"GARCH marginal {i} failed to converge: {result.message}")
            garch_params[i] = result.x

        self.garch_params = garch_params
        self.ar_params = ar_params
        df_params = np.full(self.dim, self.df)
        skew_params = np.full(self.dim, self.skew)

        if self.dist in ("t", "skewt"):
            for i in range(self.dim):
                omega, alpha, beta = garch_params[i]
                residual_series = ar1_residuals(self.data[:, i], ar_params[i, 0], ar_params[i, 1])
                variances = compute_garch_variance(residual_series, omega, alpha, beta)
                std_resid = residual_series / np.sqrt(variances)

                if self.dist == "t":
                    def nll_df(d):
                        if d <= 2.1:
                            return 1e6
                        if self.student_t_vectorized:
                            ll_vals = self.student_t_logpdf(std_resid, 0.0, 1.0, d)
                            return -np.sum(ll_vals)
                        ll = 0.0
                        for r in std_resid:
                            ll -= self.student_t_logpdf(r, 0.0, 1.0, d)
                        return ll
                    res = minimize_scalar(nll_df, bounds=(2.1, 80.0), method="bounded", options={"maxiter": 200})
                    if res.success:
                        df_params[i] = res.x
                else:  # skewt
                    def nll_skewt(x):
                        d, s = x[0], x[1]
                        if d <= 2.1:
                            return 1e6
                        ll = 0.0
                        for r in std_resid:
                            ll -= skewt_logpdf(r, 0.0, 1.0, d, s)
                        return ll
                    res = minimize(
                        nll_skewt,
                        x0=np.array([self.df, self.skew]),
                        bounds=[(2.1, 80.0), (-10.0, 10.0)],
                        method="L-BFGS-B",
                        options={"maxiter": 300}
                    )
                    if res.success:
                        df_params[i] = res.x[0]
                        skew_params[i] = res.x[1]

        self.df_params = df_params
        self.skew_params = skew_params

        # Build pseudo-observations from standardized residuals
        uniforms = np.empty_like(self.data)
        eps = 1e-10
        for i in range(self.dim):
            omega, alpha, beta = garch_params[i]
            residual_series = ar1_residuals(self.data[:, i], ar_params[i, 0], ar_params[i, 1])
            variances = compute_garch_variance(residual_series, omega, alpha, beta)
            if self.dist == "skewt":
                u = transform_to_uniform_skewt(residual_series, variances, self.df_params[i], self.skew_params[i])
            elif self.dist == "t":
                u = transform_to_uniform_t(
                    residual_series,
                    variances,
                    self.df_params[i],
                    self.student_t_cdf,
                    vectorized=self.student_t_vectorized,
                )
            else:
                u = transform_to_uniform(residual_series, variances)
            uniforms[:, i] = np.clip(u, eps, 1 - eps)
        self.uniforms = uniforms

        # Initial copula parameters from empirical Gaussian scores
        z = np.empty_like(uniforms)
        for i in range(self.dim):
            z[:, i] = gaussian_ppf(uniforms[:, i], 0, 1)
        emp_corr = np.corrcoef(z, rowvar=False)
        self.empirical_corr = emp_corr
        initial_theta = []
        for i in range(1, self.dim):
            for j in range(i):
                rho = np.clip(emp_corr[i, j], -0.95, 0.95)
                initial_theta.append(np.arctanh(rho))
        initial_theta = np.array(initial_theta)

        if self.copula == "vine":
            vine = DVineCopula()
            vine.fit(uniforms)
            self.vine = vine
            self.optimized_params = garch_params.ravel()  # vine params stored separately
            return self.optimized_params

        penalty_lambda = 5.0

        def copula_nll(theta):
            corr_matrix = build_corr_matrix(theta, self.dim)
            if self.copula == "t":
                ll = -np.sum(t_copula_logpdf(uniforms, corr_matrix, self.copula_df))
            else:
                ll = -np.sum(gaussian_copula_logpdf(uniforms, corr_matrix))
            diff = corr_matrix - emp_corr
            ridge = penalty_lambda * np.sum(diff * diff)
            return ll + ridge

        best = None
        rng = np.random.RandomState(42)
        for scale in [0.0, 0.1]:
            start = initial_theta + rng.randn(*initial_theta.shape) * scale
            res = minimize(
                copula_nll,
                start,
                method="L-BFGS-B",
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
            if best is None or (res.success and res.fun < best.fun) or (not best.success and res.fun < best.fun):
                best = res

        if best is None or not best.success:
            raise RuntimeError(f"Copula fit failed to converge: {best.message if best is not None else 'no result'}")

        self.copula_theta = best.x
        self.optimized_params = np.concatenate([garch_params.ravel(), self.copula_theta])
        return self.optimized_params

    def get_copula_correlation_matrix(self):
        """
        Extract the copula correlation matrix from the optimized parameters.

        Returns:
            Copula correlation matrix.
        """
        if self.copula == "vine":
            return getattr(self, "empirical_corr", None)

        if not hasattr(self, 'optimized_params'):
            raise RuntimeError("Model must be fitted before accessing the copula correlation matrix.")

        if self.copula == "clayton":
            # No correlation matrix parameter; return empirical PIT correlation for reference.
            return getattr(self, "empirical_corr", None)

        cholesky_params = self.optimized_params[3 * self.dim:]
        return build_corr_matrix(cholesky_params, self.dim)

if __name__ == '__main__':
    np.random.seed(42)
    import pandas as pd
    data = pd.read_csv('data.csv', index_col=0)
    log_returns = np.log(data).diff().dropna().values
    log_returns_std = (log_returns - log_returns.mean(axis=0)) / log_returns.std(axis=0)

    true_corr_matrix = np.corrcoef(log_returns.T)
    print("True correlation matrix:", true_corr_matrix)


    # Initialize and fit the GARCH-Copula model
    model = GarchCopulaModel(log_returns_std, dist="t", df=8.0, skew=0.0)
    optimized_params = model.fit()

    # Extract the copula correlation matrix
    copula_corr_matrix = model.get_copula_correlation_matrix()

    # Print results
    print("Optimized parameters:", optimized_params)
    print("Copula correlation matrix:", copula_corr_matrix)
