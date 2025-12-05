import os
import time

import io
import numpy as np
import pandas as pd

from garch import GarchCopulaModel
from cProfile import Profile
import pstats

def main():
    copula = os.getenv("COPULA", "gaussian")
    dist = os.getenv("MARGINAL_DIST", "t")
    data = pd.read_csv("data.csv", index_col=0)
    log_returns = np.log(data).diff().dropna().values
    log_returns_std = (log_returns - log_returns.mean(axis=0)) / log_returns.std(axis=0)

    model = GarchCopulaModel(log_returns_std, dist=dist, copula=copula, t_backend="scipy")
    model.fit()
    profiler = Profile()
    profiler.enable()
    model.fit()
    profiler.disable()
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)
    print(buf.getvalue())

if __name__ == "__main__":
    main()
