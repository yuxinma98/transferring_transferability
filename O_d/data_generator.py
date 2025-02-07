import pickle
import jax.numpy as jnp
import jax
from jax import vmap
from ott.geometry import segment, pointcloud
from ott.geometry import costs, distrib_costs
from ott.solvers.linear import univariate
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import lower_bound
from . import data_dir


def dist_function(xx, yy, wxx, wyy):
    geom_xx = pointcloud.PointCloud(xx)
    geom_yy = pointcloud.PointCloud(yy)
    problem = quadratic_problem.QuadraticProblem(geom_xx=geom_xx, geom_yy=geom_yy, a=wxx, b=wyy)
    distrib_cost = distrib_costs.UnivariateWasserstein(
        solve_fn=univariate.uniform_solver, ground_cost=costs.SqEuclidean()
    )
    solver = jax.jit(lower_bound.third_lower_bound)
    out = solver(
        prob=problem,
        distrib_cost=distrib_cost,
        epsilon=0.01,
    )
    return out.reg_ot_cost, out.converged


def sample_pointclouds(xtrain, ytrain, xval, yval, classes, num_pointclouds, num_pointclouds_test):
    xtrain_s = []
    ytrain_s = []
    xtest_s = []
    ytest_s = []
    idxs_out = []
    idxs_out_test = []
    for i in classes:
        indices = [s for s in range(len(ytrain)) if ytrain[s] == i]
        indices_test = [s for s in range(len(yval)) if yval[s] == i]
        # indices = slice(*idxs[0:num_pointclouds])
        if num_pointclouds == -1 or num_pointclouds > len(indices):
            num_pointclouds = len(indices)
        if num_pointclouds_test == -1 or num_pointclouds_test > len(indices_test):
            num_pointclouds_test = len(indices_test)
        for s in range(num_pointclouds):
            xtrain_s.append(xtrain[indices[s]])
            ytrain_s.append(ytrain[indices[s]])
            idxs_out = indices[s]
        for s in range(num_pointclouds_test):
            xtest_s.append(xval[indices_test[s]])
            ytest_s.append(yval[indices_test[s]])
            idxs_out_test = indices_test[s]
    return xtrain_s, ytrain_s, xtest_s, ytest_s, idxs_out, idxs_out_test


if __name__ == "__main__":
    vectorized_dist_function_single_loop = jax.jit(vmap(dist_function, (0, 0, 0, 0), 0))
    classes = [2, 7]
    num_pointclouds = 40
    for num_points in [20, 100, 200, 300, 500]:
        xtrain, ytrain, xval, yval = pickle.load(
            open(f"{data_dir}/ModelNet_np_{num_points}.pkl", "rb")
        )

        xtrain_s, ytrain_s, xtest_s, ytest_s, indices, indices_test = sample_pointclouds(
            xtrain,
            ytrain,
            xval,
            yval,
            classes=[2, 7],
            num_pointclouds=num_pointclouds,
            num_pointclouds_test=num_pointclouds,
        )
        total_points, total_weights = segment.segment_point_cloud(
            jnp.concatenate(xtrain_s + xtest_s),
            num_per_segment=[cloud.shape[0] for cloud in xtrain_s + xtest_s],
        )
        idxs_train = jnp.array(
            [
                [i, j]
                for i in range(num_pointclouds)
                for j in range(num_pointclouds, num_pointclouds * 2)
            ]
        )
        idxs_test = jnp.array(
            [
                [i, j]
                for i in range(num_pointclouds * 2, num_pointclouds * 3)
                for j in range(num_pointclouds * 3, num_pointclouds * 4)
            ]
        )
        kernel_train, converged = vectorized_dist_function_single_loop(
            total_points[idxs_train[:, 0], :, :],
            total_points[idxs_train[:, 1], :, :],
            total_weights[idxs_train[:, 0], :],
            total_weights[idxs_train[:, 1], :],
        )
        print(kernel_train.shape, converged)
        kernel_test, converged = vectorized_dist_function_single_loop(
            total_points[idxs_test[:, 0], :, :],
            total_points[idxs_test[:, 1], :, :],
            total_weights[idxs_test[:, 0], :],
            total_weights[idxs_test[:, 1], :],
        )
        print(kernel_test.shape, converged)
        pickle.dump(
            (xtrain_s, ytrain_s, xtest_s, ytest_s, kernel_train, kernel_test),
            open(f"{data_dir}/GWLB_points{num_points}_classes{classes}.pkl", "wb"),
        )
