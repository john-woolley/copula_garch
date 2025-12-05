import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, t as student_t, norm


def _gaussian_logpdf(u, v, rho):
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)
    det = 1 - rho ** 2
    quad = (z1 ** 2 + z2 ** 2 - 2 * rho * z1 * z2) / det
    return -0.5 * (np.log(det) + quad - z1 ** 2 - z2 ** 2)


def _t_logpdf(u, v, rho, df):
    z1 = student_t.ppf(u, df)
    z2 = student_t.ppf(v, df)
    det = 1 - rho ** 2
    quad = (z1 ** 2 + z2 ** 2 - 2 * rho * z1 * z2) / det
    log_const = (
        np.log(student_t.pdf(z1, df))
        + np.log(student_t.pdf(z2, df))
    )
    log_joint = (
        np.log(student_t.pdf(z1, df))
        + np.log(student_t.pdf(z2, df))
        - 0.5 * np.log(det)
        - ((df + 2) / 2.0) * np.log(1 + quad / df)
        + ((df + 1) / 2.0) * (np.log(1 + z1 ** 2 / df) + np.log(1 + z2 ** 2 / df))
    )
    return log_joint - log_const


def _hfunc_gaussian(u, v, rho):
    z = norm.ppf(u)
    t = norm.ppf(v)
    num = z - rho * t
    den = np.sqrt(1 - rho ** 2)
    return norm.cdf(num / den)


def _hfunc_t(u, v, rho, df):
    z = student_t.ppf(u, df)
    t_ = student_t.ppf(v, df)
    alpha = (df + 1) / (df + t_ ** 2)
    return student_t.cdf(z * np.sqrt(alpha) - rho * t_ * np.sqrt(alpha), df + 1)


def fit_pair(u, v, family_options=("gaussian", "t")):
    # Start from tau estimate
    tau = kendalltau(u, v).correlation
    rho0 = np.clip(np.sin(np.pi * tau / 2), -0.95, 0.95)

    best = None
    if "gaussian" in family_options:
        def nll_rho(r):
            if r <= -0.99 or r >= 0.99:
                return 1e6
            return -np.sum(_gaussian_logpdf(u, v, r))
        res = minimize(lambda x: nll_rho(x[0]), x0=np.array([rho0]), bounds=[(-0.99, 0.99)])
        if res.success:
            best = ("gaussian", res.fun, {"rho": res.x[0]})

    if "t" in family_options:
        def nll_t(params):
            r, df = params
            if r <= -0.99 or r >= 0.99 or df <= 2.1 or df >= 80:
                return 1e6
            return -np.sum(_t_logpdf(u, v, r, df))
        res = minimize(
            nll_t,
            x0=np.array([rho0, 8.0]),
            bounds=[(-0.99, 0.99), (2.1, 80.0)],
            method="L-BFGS-B"
        )
        if res.success:
            if best is None or res.fun < best[1]:
                best = ("t", res.fun, {"rho": res.x[0], "df": res.x[1]})

    if best is None:
        raise RuntimeError("Pair copula fit failed")
    return best


class DVineCopula:
    """
    Simple D-vine with Gaussian/t pair-copulas.
    """
    def __init__(self, families=("gaussian", "t")):
        self.families = families
        self.edges = []

    def fit(self, u):
        n, d = u.shape
        order = self._order_by_tau(u)
        current = u[:, order]
        h_values = [current]

        edges = []
        for level in range(d - 1):
            next_level = []
            for i in range(d - 1 - level):
                u1 = h_values[level][:, i]
                u2 = h_values[level][:, i + 1]
                family, nll, params = fit_pair(u1, u2, self.families)
                edges.append({"level": level, "i": i, "j": i + 1, "family": family, "params": params})

                if family == "gaussian":
                    h1 = _hfunc_gaussian(u1, u2, params["rho"])
                    h2 = _hfunc_gaussian(u2, u1, params["rho"])
                else:
                    h1 = _hfunc_t(u1, u2, params["rho"], params["df"])
                    h2 = _hfunc_t(u2, u1, params["rho"], params["df"])
                next_level.append(h1)
            h_values.append(np.column_stack(next_level))
        self.edges = edges
        self.order = order
        return self

    def _order_by_tau(self, u):
        d = u.shape[1]
        tau_mat = np.zeros((d, d))
        for i in range(d):
            for j in range(i):
                tau_mat[i, j] = np.abs(kendalltau(u[:, i], u[:, j]).correlation)
                tau_mat[j, i] = tau_mat[i, j]
        # simple greedy path: start with highest degree node
        scores = tau_mat.sum(axis=0)
        order = [int(np.argmax(scores))]
        remaining = set(range(d)) - set(order)
        while remaining:
            last = order[-1]
            next_node = max(remaining, key=lambda x: tau_mat[last, x])
            order.append(next_node)
            remaining.remove(next_node)
        return order
