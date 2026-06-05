"""
bbo/surrogates.py — Surrogate models

Currently implements:
  - Gaussian Process with Matern kernel

Future additions:
  - CMA-ES (evolutionary, no surrogate needed)
  - Neural network surrogate
  - Different GP kernels per function
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings
warnings.filterwarnings("ignore")


def fit_gp(inputs, outputs, n_dim, kernel_nu=2.5, alpha=1e-6):
    """
    Fit a Gaussian Process surrogate to the observed data.

    WHAT: Builds a probabilistic model of the function. At every unsampled
    point, the GP predicts both a mean (best guess) and variance (uncertainty).

    ASSUMPTIONS:
      - Function is smooth and continuous (Matern kernel)
      - Nearby inputs produce similar outputs (stationarity)
      - kernel_nu=2.5: function is twice-differentiable. Use nu=1.5 for
        rougher functions or nu=0.5 for very rough/spiky functions.
      - alpha: observation noise. 1e-6 = essentially noiseless. Increase
        to 1e-4 or 1e-2 if the GP is overconfident.

    Args:
        inputs:    (n_points, n_dims) array of observed inputs
        outputs:   (n_points,) array of observed outputs
        n_dim:     number of input dimensions
        kernel_nu: Matern smoothness parameter (0.5, 1.5, or 2.5)
        alpha:     noise/regularisation term

    Returns:
        gp: fitted GaussianProcessRegressor
    """
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=np.ones(n_dim), nu=kernel_nu
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=alpha,
        normalize_y=True
    )
    gp.fit(inputs, outputs)
    return gp


def analyse_length_scales(gp, n_dim):
    """
    Extract and interpret the GP's learned length scales.

    WHAT: After fitting, the Matern kernel has a learned length scale
    per dimension. Short length scale = the function changes rapidly
    in that dimension (important). Long length scale = the function
    barely changes (irrelevant).

    This is the GP's version of feature importance — similar to
    looking at gradients in a neural network.

    Args:
        gp:    fitted GaussianProcessRegressor
        n_dim: number of dimensions

    Returns:
        dict with length_scales, importance ranking, and interpretation
    """
    # Extract length scales from the fitted kernel
    # Kernel structure: ConstantKernel * Matern
    # The Matern's length scales are in kernel_.k2.length_scale
    try:
        ls = gp.kernel_.k2.length_scale
        if np.isscalar(ls):
            ls = np.array([ls] * n_dim)
    except AttributeError:
        return None

    # Shorter length scale = more important
    importance = 1.0 / ls
    ranking = np.argsort(importance)[::-1]  # most important first

    return {
        "length_scales": ls,
        "importance": importance,
        "ranking": ranking,
    }