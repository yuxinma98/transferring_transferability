import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch_geometric as pyg
from torch_geometric.datasets import planetoid

from model import GNN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def define_distribution(graph_model):
    if graph_model == "ER_Gaussian":
        p = torch.distributions.Uniform(0.1, 0.9).sample()
        L = torch.randn(d, d)
        mu = torch.randn(d)
        cov = L @ L.T
        return {"p": p, "mu": mu, "cov": cov}
    if graph_model == "ER_Multinomial":
        p = torch.distributions.Uniform(0.1, 0.9).sample()
        dirichlet = torch.distributions.Dirichlet(torch.ones(d))
        p_multinomial = dirichlet.sample()
        return {"p": p, "p_multinomial": p_multinomial}
    if graph_model == "SBM_Gaussian":
        uniform_sampler = torch.distributions.Uniform(0.1, 0.9)
        ps = uniform_sampler.sample((3, 3))
        L = torch.randn(3, d, d)
        mu = torch.randn(3, d)
        cov = torch.matmul(L, L.transpose(-1, -2))
        return {"ps": ps, "mu": mu, "cov": cov}
    if graph_model == "Sociality":
        kernel = lambda x, y: 1 / (1 + torch.exp(-10 * x * y))
        return {"kernel": kernel}
    if graph_model == "cora":
        pass
        # dataset = planetoid.Planetoid(
        #     root="data/", name="Cora", split="public", transform=pyg.transforms.NormalizeFeatures()
        # )
        # data = dataset[0]
        # A = pyg.utils.to_dense_adj(data.edge_index).squeeze(dim=0)
        # X = data.x
        # self.n = self.A.shape[0]
        # self.graphon = lambda i, j: self.A[int(i * self.n), int(j * self.n)]
        # self.signal = lambda i: self.X[i, :]


# def define_signal_sampler(signal_model, ):
#     if signal_model == "Gaussian":

#     if signal_model == "Multinomial":
#         dirichlet = torch.distributions.Dirichlet(torch.ones(d))
#         p = dirichlet.sample()
#         return torch.distributions.Multinomial(1, p)


def sample_graph_signal(graph_model, n_samples, n, parameters):
    if graph_model == "ER_Gaussian":
        A = sample_ER(parameters["p"], n_samples, n)
        X = sample_gaussian_features(parameters["mu"], parameters["cov"], n_samples, n)
        return A, X
    if graph_model == "ER_Multinomial":
        A = sample_ER(parameters["p"], n_samples, n)
        X = sample_multinomial_features(parameters["p_multinomial"], n_samples, n)
        return A, X
    if graph_model == "SBM_Gaussian":
        A, X = sample_SBM(parameters["ps"], parameters["mu"], parameters["cov"], n_samples, n)
        return A, X
    if graph_model == "SBM_Multinomial":
        pass
    if graph_model == "Sociality":
        A, X = sample_graphon(parameters["kernel"], n_samples, n)


def sample_ER(p, n_samples, n):
    sampler = torch.distributions.Bernoulli(p)
    A = sampler.sample((n_samples, n, n)).float()
    A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-1, -2)
    return A


def sample_gaussian_features(mu, cov, n_samples, n):
    X = torch.distributions.MultivariateNormal(mu, cov).sample((n_samples, n))
    return X


def sample_multinomial_features(p, n_samples, n):
    sampler = torch.distributions.Multinomial(100, p)
    X = sampler.sample((n_samples, n))
    return X


def sample_SBM(ps, mu, cov, n_samples, n):
    z = torch.randint(0, 3, (n_samples, n))
    z_one_hot = torch.nn.functional.one_hot(z, num_classes=3).float()
    prob_matrix = torch.einsum("nij,jk,nkl->nil", z_one_hot, ps, z_one_hot.transpose(-1, -2))
    A = torch.distributions.Bernoulli(prob_matrix).sample()
    X_distributions = torch.distributions.MultivariateNormal(mu[z], cov[z])
    X = X_distributions.sample()
    # X = torch.zeros(n_samples, n, d)
    # for i in range(n_samples):
    #     for j in range(n):
    #         X[i, j, :] = torch.distributions.MultivariateNormal(mu[z[i, j]], cov[z[i, j]]).sample()
    return A, X


def sample_graphon(kernel, n_samples, n, d):
    uniform_sampler = torch.distributions.Uniform(0, 1)
    z = uniform_sampler.sample((n_samples, n))
    # p = torch.zeros(n_samples, n, n)
    # for i in range(n_samples):
    #     for j in range(n):
    #         for k in range(n):
    #             p[i, j, k] = kernel(z[i, j], z[i, k])
    p = kernel(z.unsqueeze(2), z.unsqueeze(1))
    A = torch.distributions.Bernoulli(p).sample()
    X = z.unsqueeze(2).pow(torch.arange(1, d + 1).float())
    return A, X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_model",
        type=str,
        default="ER_Gaussian",
        choices=[
            "ER_Gaussian",
            "ER_Multinomial",
            "SBM_Gaussian",
            "SBM_Multinomial",
            "Sociality",
            "cora",
        ],
    )
    parser.add_argument("--n_samples", type=int, default=100)

    args = parser.parse_args()
    n_samples = args.n_samples
    d = 5
    M = int(1e4)  # reference graph size, set to a large number as estimated cts limit
    log_n_range = np.arange(1, 3, 0.1)
    log_dir = os.path.join(CURRENT_DIR, "log/transferability")
    graph_model = args.graph_model

    # fix distributions for graph and signal
    parameters = define_distribution(graph_model)

    # fix a model with random weights
    model = GNN(in_channels=d, out_channels=1, hidden_channels=5, num_layers=3, reduced=True)
    model.eval()
    model_full = GNN(in_channels=d, out_channels=1, hidden_channels=5, num_layers=3, reduced=False)
    model_full.eval()

    # compute estimated limit
    references = np.zeros(n_samples)
    references_full = np.zeros(n_samples)
    for i in range(n_samples):
        A, X = sample_graph_signal(graph_model, 1, M, parameters)
        with torch.no_grad():
            references[i] = model(A, X).mean()
            references_full[i] = model_full(A, X).mean()
    reference = float(references.mean())
    reference_full = float(references_full.mean())

    # compute errors
    n_range = np.power(10, log_n_range).astype(int)
    errors_mean = np.zeros_like(n_range, dtype=float)
    errors_std = np.zeros_like(n_range, dtype=float)
    errors_mean_full = np.zeros_like(n_range, dtype=float)
    errors_std_full = np.zeros_like(n_range, dtype=float)
    for i, n in enumerate(n_range):
        A, X = sample_graph_signal(graph_model, n_samples, n, parameters)
        with torch.no_grad():
            y1 = model(A, X).mean(dim=1).squeeze()
            y2 = model_full(A, X).mean(dim=1).squeeze()
        error = torch.abs(y1 - reference)
        error_full = torch.abs(y2 - reference_full)
        errors_mean[i] = float(error.mean().squeeze())
        errors_std[i] = float(error.std().squeeze())
        errors_mean_full[i] = float(error_full.mean().squeeze())
        errors_std_full[i] = float(error_full.std().squeeze())

    # plot
    plt.figure()
    plt.errorbar(
        n_range, errors_mean, errors_std, fmt="o", capsize=3, markersize=5, label="Reduced model"
    )
    plt.errorbar(
        n_range,
        errors_mean_full,
        errors_std_full,
        fmt="o",
        capsize=3,
        markersize=5,
        label="Full model",
    )
    y = n_range ** (-0.5) * n_range[0] ** (0.5) * errors_mean[0]
    plt.plot(n_range, y, label="$n^{-1/2}$")
    plt.xlabel("Set size $n$")
    plt.ylabel("$|f_n(x) - f_m(x)|$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(os.path.join(log_dir, f"transferability_{graph_model}.png"))
