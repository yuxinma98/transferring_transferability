import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from model import DeepSet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    d = 5
    n_samples = 1000
    M = int(1e8)  # reference set size, set to a large number as estimated cts limit
    log_n_range = np.arange(1, 4, 0.2)
    log_dir = os.path.join(CURRENT_DIR, "log/transferability")

    # fix a 2-d gaussian distribution
    L = torch.randn(d, d)
    mu = torch.randn(d)
    cov = L @ L.T
    multivariate_normal = torch.distributions.MultivariateNormal(mu, cov)

    # fix a model with random weights
    model = DeepSet(in_channels=d, output_channels=1, normalized=True)
    model.eval()

    # compute estimated limit
    X = multivariate_normal.sample((M,)).unsqueeze(0)
    with torch.no_grad():
        limit = float(model(X).mean(dim=0))

    # compute errors
    n_range = np.power(10, log_n_range).astype(int)
    errors_mean = np.zeros_like(n_range, dtype=float)
    errors_std = np.zeros_like(n_range, dtype=float)
    for i, n in enumerate(n_range):
        X = multivariate_normal.sample(
            (
                n_samples,
                n,
            )
        )
        with torch.no_grad():
            y = model(X)
        error = torch.abs(y - limit)
        errors_mean[i] = float(error.mean(dim=0).squeeze())
        errors_std[i] = float(error.std(dim=0).squeeze())

    # plot
    plt.figure()
    plt.errorbar(n_range, errors_mean, errors_std, fmt="o", capsize=3, markersize=5)
    reference = n_range ** (-0.5) * n_range[0] ** (0.5) * errors_mean[0]
    plt.plot(n_range, reference, label="$n^{-0.5}$")
    plt.xlabel("Set size $n$")
    plt.ylabel("$|f_n(x) - f_m(x)|$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(os.path.join(log_dir, "transferability.png"))
