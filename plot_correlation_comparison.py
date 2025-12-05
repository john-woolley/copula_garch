import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from garch import GarchCopulaModel


def main():
    data = pd.read_csv("data.csv", index_col=0)
    log_returns = np.log(data).diff().dropna().values
    log_returns_std = (log_returns - log_returns.mean(axis=0)) / log_returns.std(axis=0)

    model = GarchCopulaModel(log_returns_std, dist="t", copula="clayton")
    model.fit()

    raw_corr = np.corrcoef(log_returns_std, rowvar=False)
    residual_corr = model.empirical_corr
    cop_corr = model.get_copula_correlation_matrix()

    idx = np.tril_indices_from(raw_corr, k=-1)
    labels = [f"({i},{j})" for i, j in zip(idx[0], idx[1])]

    width = 0.25
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, raw_corr[idx], width, label="Raw corr")
    ax.bar(x, residual_corr[idx], width, label="PIT residual corr")
    ax.bar(x + width, cop_corr[idx], width, label="Copula corr")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Correlation")
    ax.set_title("Dependence: raw vs PIT residual vs fitted copula")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "correlation_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
