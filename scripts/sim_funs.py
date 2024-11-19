#!/usr/bin/env python
# coding: utf-8

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
scripts_dir = os.path.abspath(os.path.join(parent_dir, "scripts"))
sys.path.append(scripts_dir)
import ipyparallel as ipp
import metrics as met
import models as mod
import numpy #as np
import pandas #as pd
import scipy.stats as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from plotnine import *
from scipy.integrate import quad
from scipy.stats import multivariate_normal
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from typing import Dict, Tuple


class mvnorm:
    def __init__(self, mean=[0], cov_factor=None, seed=None):
        self.mean = numpy.asarray(mean)
        self.cov_factor = numpy.asarray(cov_factor)
        self.cov = cov_factor @ cov_factor.T
        self.seed = seed
        if seed is None:
            self.rng = numpy.random.default_rng()
        elif isinstance(seed, (int, numpy.random.RandomState, numpy.random.Generator)):
            self.rng = numpy.random.default_rng(seed)
        else:
            raise ValueError("seed must be None, an int, or a RandomState/Generator instance.")

    def rvs(self, size=1, random_state=None):
        r = self.cov_factor.shape[1]
        return multivariate_normal.rvs(mean=numpy.zeros(r), size=size, random_state=random_state) @ self.cov_factor.T + self.mean


def calc_opt_precision(x_sd: float, epsilon_sd: float, p: float) -> float:
    """
    Calculate P(Y > F_Y^{-1}(p) | X > F_X^{-1}(p)), the precison of {X > F_X^{-1}(p)}, the optimal predictor of {Y > F_Y^{-1}(p)} when Y = X + epsilon,
    with X and epsilon being independent mean zero normal random variables.

    :param x_sd: Float; standard deviation of X.
    :param epsilon_sd: Float; standard deviation of epsilon.
    :param p: Float; the quantile level that defines extremeness.
    :return: Float; the precision of the optimal predictor.
    """
    if not isinstance(x_sd, float):
        raise TypeError("x_sd must be a float")
    if x_sd <= 0:
        raise ValueError("x_sd must be positive")
    if not isinstance(epsilon_sd, float):
        raise TypeError("epsilon_sd must be a float")
    if epsilon_sd <= 0:
        raise ValueError("epsilon_sd must be positive")

    q_x = st.norm.ppf(p, scale=x_sd)
    q_y = st.norm.ppf(p, scale=numpy.sqrt(x_sd ** 2 + epsilon_sd ** 2))
    def calc_integrand(x):
        return st.norm.cdf(x - q_y, scale=epsilon_sd) * st.norm.pdf(x, scale=x_sd)
    numer_prob, _ = quad(calc_integrand, q_x, numpy.inf)
    return numer_prob / (1 - p)


def make_dataset(x_dist, epsilon_dist: rv_continuous_frozen, beta: numpy.ndarray, n: int, rng=None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Generate a random sample from the joint distribution of X and Y, where Y = X.T @ beta + epsilon.

    :param x_dist: Frozen multivariate random vector; the distribution of X.
    :param epsilon_dist: Frozen random variable; the distribution of epsilon.
    :param beta: NumPy array of shape (d,); the vector of model coefficients.
    :param n: Integer; the sample size.
    :param rng: the random state.
    :return: Tuple of NumPy arrays of shapes (n, d) and (n,); the matrix of observations on X and the vector of observations on Y.
    """
    if not isinstance(epsilon_dist, rv_continuous_frozen):
        raise TypeError("epsilon_dist must be an instance of scipy.stats._distn_infrastructure.rv_continuous_frozen")
    if not isinstance(beta, numpy.ndarray):
        raise TypeError("beta must be a NumPy array")
    if beta.ndim != 1:
        raise TypeError("beta must be 1D")
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be positive")

    x = x_dist.rvs(size=n, random_state=rng) # could try uniform as well? look at lasso literature
    x = numpy.atleast_2d(x)
    if x.shape[1] != beta.size:
        x = x.reshape(-1, beta.size)
    epsilon = epsilon_dist.rvs(size=n, random_state=rng) # could try different distributions here, like t
    y = x @ beta + epsilon

    return x, y


def plot_dataset(beta: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> ggplot:
    """
    Plot Y versus X.T @ beta for observations generated from the model Y = X.T @ beta + epsilon.    

    :param beta: NumPy array of shape (d,); the vector of model coefficients.
    :param x: NumPy array of shape (n, d); the matrix containing the observations on X.
    :param y: NumPy array of shape (n,); the vector containing the observations on Y.
    :return: ggplot; the plot of Y versus X.T @ beta.
    """
    if not isinstance(beta, numpy.ndarray):
        raise TypeError("beta must be a NumPy array")
    if beta.ndim != 1:
        raise TypeError("beta must be 1D")
    if not isinstance(x, numpy.ndarray):
        raise TypeError("x must be a NumPy array")
    if x.ndim != 2:
        raise TypeError("x must be 2D")    
    if not isinstance(y, numpy.ndarray):
        raise TypeError("y must be a NumPy array")
    if y.ndim != 1:
        raise TypeError("y must be 1D")

    return (
        ggplot(data=pandas.DataFrame({"lin_pred": x @ beta, "y": y}), mapping=aes(x="lin_pred", y="y"))
        + geom_point()
        + geom_abline(slope=1, intercept=0)
        + labs(x=r"$X^{\top}\beta$", y="Y", caption=r"$Y = X^{\top}\beta$ along the line")
        + theme_bw()
    )


def calc_true_quantiles(x_dist, epsilon_dist: rv_continuous_frozen, beta: numpy.ndarray, n: int, p: float, rng=None) -> Tuple[float, float]:
    """
    Calculate (estimate, really) the true pth quantiles of the linear predictor and the response for the model Y = X.T @ beta + epsilon.

    :param x_dist: Frozen multivariate random vector; the distribution of X.
    :param epsilon_dist: Frozen random variable; the distribution of epsilon.
    :param beta: NumPy array of shape (d,); the vector of model coefficients.
    :param n: Integer; the sample size.
    :param p: Float; the quantile level that defines extremeness.
    :param rng: the random state.
    :return: Tuple of floats; the pth quantiles of the linear predictor and the response.
    """
    x, y = make_dataset(x_dist, epsilon_dist, beta, n, rng=rng)
    lin_pred = x @ beta
    lin_pred_quantile = numpy.quantile(lin_pred, p, method="inverted_cdf")
    y_quantile = numpy.quantile(y, p, method="inverted_cdf")
    return lin_pred_quantile, y_quantile


def do_1_run(dists, beta: numpy.ndarray, sizes: Dict, quantiles: Dict, mods, metrics, rng=None) -> pandas.DataFrame:
    """

    :param dists:
    :param beta:
    :param sizes:
    :param quantiles:
    :param mods:
    :param metrics:
    :param rng:
    :return: Pandas DataFrame
    """
    x_dist, epsilon_dist = dists["x"], dists["epsilon"]
    train_size, test_size = sizes["train"], sizes["test"]
    lin_pred_quantile, y_quantile = quantiles["lin_pred"], quantiles["y"]

    x, y = make_dataset(x_dist, epsilon_dist, beta, train_size + test_size, rng)
    x_train, y_train = x[:train_size], y[:train_size] > y_quantile
    x_test, y_test = x[train_size:], y[train_size:] > y_quantile

    mod_summaries = []
    for mod in mods:
        mod.fit(x_train, y_train)
        mod_preds = mod.predict(x_test)
        mod_summary = {metric.__class__.__name__: metric.evaluate(y_test, mod_preds) for metric in metrics}
        mod_summary = pandas.DataFrame([mod_summary])
        mod_summary.insert(0, "mod", mod.__class__.__name__)
        mod_summaries.append(mod_summary)

    oracle_preds = x_test @ beta >= lin_pred_quantile
    oracle_summary = {metric.__class__.__name__: metric.evaluate(y_test, oracle_preds) for metric in metrics}
    oracle_summary = pandas.DataFrame([oracle_summary])
    oracle_summary.insert(0, "mod", "oracle")
    mod_summaries.append(oracle_summary)

    return pandas.concat(mod_summaries).reset_index(drop=True)


def simulate(num_runs, max_workers, dists: Dict, beta: numpy.ndarray, sizes: Dict, quantiles: Dict, mods, metrics_, rng) -> pandas.DataFrame:
    child_rngs = rng.spawn(num_runs)
    with ipp.Cluster(n=max_workers) as rc:
        e_all = rc[:]
        e_all.use_dill()
        with e_all.sync_imports():
            import os
            import sys
        e_all.execute("parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))", block=True)
        e_all.execute("scripts_dir = os.path.abspath(os.path.join(parent_dir, 'scripts'))", block=True)
        e_all.execute("sys.path.append(scripts_dir)", block=True)
        with e_all.sync_imports():
            import numpy
            import pandas
            from scipy.stats import multivariate_normal
            from scipy.stats._distn_infrastructure import rv_continuous_frozen
            from typing import Dict, Tuple
        e_all.push({
            "make_dataset": make_dataset, "do_1_run": do_1_run,
            "dists": dists, "beta": beta, "sizes": sizes, "quantiles": quantiles, "mods": mods, "metrics_": metrics_,
            "child_rngs": child_rngs
        })
        ar = e_all.map_sync(lambda i: do_1_run(dists, beta, sizes, quantiles, mods, metrics_, child_rngs[i]).assign(run_num=i), range(num_runs))
    results = pandas.concat(ar).reset_index(drop=True)
    run_nums = results.pop("run_num")
    results.insert(0, "run_num", run_nums)
    return results


def simulate2(num_runs, max_workers, dists: Dict, beta: numpy.ndarray, sizes: Dict, quantiles: Dict, mods, metrics_, rng) -> pandas.DataFrame:
    child_rngs = rng.spawn(num_runs)
    summaries = []
    with ProcessPoolExecutor(max_workers, mp_context=get_context("fork")) as executor:
        futures = [executor.submit(do_1_run, dists, beta, sizes, quantiles, mods, metrics_, child_rngs[i]) for i in range(num_runs)]
        for future in as_completed(futures):
            summaries.append(future.result())
            print(summaries.__len__())
    return pandas.concat(summaries).reset_index(drop=True)


def simulate3(num_runs, max_workers, dists: Dict, beta: numpy.ndarray, sizes: Dict, quantiles: Dict, mods, metrics_, rng) -> pandas.DataFrame:
    child_rngs = rng.spawn(num_runs)
    summaries = []
    for i in range(num_runs):
        summary = do_1_run(dists, beta, sizes, quantiles, mods, metrics_, child_rngs[i])
        summaries.append(summary)
        print(summaries.__len__())
    return pandas.concat(summaries).reset_index(drop=True)


def plot_results(results: pandas.DataFrame, opt_precision: float = None) -> ggplot:
    results2 = pandas.melt(results, id_vars=["run_num", "mod"], var_name="metric", value_name="val")
    plot = (
        ggplot(data=results2, mapping=aes(x="mod", y="val"))
        + facet_wrap("metric")
        + geom_boxplot()
        + labs(x="Model", y="Value", caption="Dashed line is at optimal precision")
        + theme_bw()
    )
    if opt_precision is not None:
        plot = plot + geom_hline(yintercept=opt_precision, linetype="dashed")
    return plot


def calc_epsilon_sd(a_, beta, snr):
    signal_var = beta.T @ a_ @ a_.T @ beta
    return (signal_var * (1 / snr - 1)) ** 0.5

