import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from garch import GarchCopulaModel
from copulas.gaussian import gaussian_copula_sample


def tail_dependence(u, q):
    """
    Estimate lower/upper tail dependence coefficients for all pairs.
    lambda_L(q) = P(U1<q, U2<q) / q
    lambda_U(q) = P(U1>q, U2>q) / (1-q)
    """
    dim = u.shape[1]
    idx = np.tril_indices(dim, k=-1)
    lower = []
    upper = []
    for i, j in zip(*idx):
        joint_low = np.mean((u[:, i] < q) & (u[:, j] < q))
        joint_high = np.mean((u[:, i] > 1 - q) & (u[:, j] > 1 - q))
        lower.append(joint_low / q if q > 0 else np.nan)
        upper.append(joint_high / (q if q > 0 else np.nan))
    return idx, np.array(lower), np.array(upper)


def main():
    data = pd.read_csv("data.csv", index_col=0)
    log_returns = np.log(data).diff().dropna().values
    log_returns_std = (log_returns - log_returns.mean(axis=0)) / log_returns.std(axis=0)

    # Fit IFM with Gaussian copula and t marginals
    model = GarchCopulaModel(log_returns_std, dist="t", copula="clayton")
    model.fit()
    uniforms = model.uniforms
    corr = model.get_copula_correlation_matrix()

    # Empirical tail dependence from PIT uniforms
    q = 0.1
    idx, emp_lower, emp_upper = tail_dependence(uniforms, q)

    # Simulated tail dependence from fitted Gaussian copula
    sim_u = gaussian_copula_sample(corr, n=uniforms.shape[0] * 10)
    _, sim_lower, sim_upper = tail_dependence(sim_u, q)

    labels = [f"({i},{j})" for i, j in zip(*idx)]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    axes[0].bar(x - width / 2, emp_lower, width, label="Empirical")
    axes[0].bar(x + width / 2, sim_lower, width, label="Gaussian copula")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title(f"Lower tail λ_L at q={q}")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, emp_upper, width, label="Empirical")
    axes[1].bar(x + width / 2, sim_upper, width, label="Gaussian copula")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title(f"Upper tail λ_U at q={q}")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Tail dependence: empirical PIT vs fitted Gaussian copula")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "tail_dependence.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved tail dependence plot to {out_path}")


if __name__ == "__main__":
    main()
