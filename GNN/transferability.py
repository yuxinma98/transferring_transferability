import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from model import GNN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    d = 5
    n_samples = 100
    p = 0.5
    M = int(1e4)  # reference graph size, set to a large number as estimated cts limit
    log_n_range = np.arange(1, 3, 0.1)
    log_dir = os.path.join(CURRENT_DIR, "log/transferability")

    # fix a Gaussian distribution for features
    L = torch.randn(d, d)
    mu = torch.randn(d)
    cov = L @ L.T
    multivariate_normal = torch.distributions.MultivariateNormal(mu, cov)
    bernoulli = torch.distributions.Bernoulli(p)

    # fix a model with random weights
    model = GNN(in_channels=d, out_channels=1, hidden_channels=5, num_layers=3, reduced=True)
    model.eval()

    # compute estimated limit
    references = np.zeros(n_samples)
    for i in range(n_samples):
        A = bernoulli.sample((M, M)).float()
        A = A.tril(diagonal=-1) + A.tril(diagonal=-1).T
        A = A.unsqueeze(0)
        X = multivariate_normal.sample((M,)).unsqueeze(0)
        with torch.no_grad():
            references[i] = model(A, X).mean()
    reference = float(references.mean())

    # compute errors
    n_range = np.power(10, log_n_range).astype(int)
    errors_mean = np.zeros_like(n_range, dtype=float)
    errors_std = np.zeros_like(n_range, dtype=float)
    for i, n in enumerate(n_range):
        A = bernoulli.sample((n_samples, n, n)).float()
        A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-1, -2)
        X = multivariate_normal.sample(
            (
                n_samples,
                n,
            )
        )
        with torch.no_grad():
            y = model(A, X).mean(dim=1).squeeze()
        error = torch.abs(y - reference)
        errors_mean[i] = float(error.mean().squeeze())
        errors_std[i] = float(error.std().squeeze())

    # plot
    plt.figure()
    plt.errorbar(n_range, errors_mean, errors_std, fmt="o", capsize=3, markersize=5)
    y = n_range ** (-0.5) * n_range[0] ** (0.5) * errors_mean[0]
    plt.plot(n_range, y, label="$n^{-1/2}$")
    plt.xlabel("Set size $n$")
    plt.ylabel("$|f_n(x) - f_m(x)|$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(os.path.join(log_dir, "transferability.png"))
