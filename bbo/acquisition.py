"""
bbo/acquisition.py — Acquisition functions and space-filling methods

Acquisition functions decide WHERE to query next, given a fitted surrogate.

Currently implements:
  - Expected Improvement (EI): balances explore/exploit via xi
  - Upper Confidence Bound (UCB): direct kappa control
  - Latin Hypercube Sampling (LHS): model-free space-filling

Each acquisition function returns NEGATIVE values because scipy.minimize
finds minimums, but we want the maximum acquisition value.
"""

import numpy as np
from scipy.stats import norm, qmc


# ============================================================
# EXPECTED IMPROVEMENT (EI)
# ============================================================
#
# FORMULA: EI(x) = (mu - y_best - xi) * Phi(z) + sigma * phi(z)
#   where z = (mu - y_best - xi) / sigma
#
# xi controls exploration:
#   xi = 0.01 → mostly exploitation
#   xi = 0.05 → moderate exploration
#   xi = 0.10 → heavy exploration
# ============================================================

def expected_improvement(X, gp, y_best, xi=0.01):
    X = X.reshape(1, -1)
    mu, sigma = gp.predict(X, return_std=True)
    sigma = max(sigma[0], 1e-9)
    z = (mu[0] - y_best - xi) / sigma
    ei = (mu[0] - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return -ei


# ============================================================
# UPPER CONFIDENCE BOUND (UCB)
# ============================================================
#
# FORMULA: UCB(x) = mu(x) + kappa * sigma(x)
#
# kappa controls exploration:
#   kappa = 0.5 → mostly exploitation (trust the mean)
#   kappa = 1.0 → moderate balance
#   kappa = 2.0 → heavy exploration (chase uncertainty)
# ============================================================

def upper_confidence_bound(X, gp, kappa=1.0):
    X = X.reshape(1, -1)
    mu, sigma = gp.predict(X, return_std=True)
    return -(mu[0] + kappa * sigma[0])


# ============================================================
# LATIN HYPERCUBE SAMPLING (LHS)
# ============================================================
#
# WHAT: Model-free space-filling. Generates candidates that
# evenly tile the input space, then picks the one furthest
# from all existing data (maximin criterion).
#
# WHEN TO USE: When the GP has no signal to learn from
# (e.g. function 1 early on, all outputs near zero).
#
# n_candidates: how many LHS points to generate. More = better
# coverage but slower.
# ============================================================

def lhs_next_query(inputs, n_dim, n_candidates=500):
    sampler = qmc.LatinHypercube(d=n_dim)
    candidates = sampler.random(n=n_candidates)
    candidates = candidates * 0.998 + 0.001  # scale to [0.001, 0.999]

    # Maximin: pick the candidate furthest from all existing points
    best_dist = -1
    best_x = None
    for c in candidates:
        dists = np.linalg.norm(inputs - c, axis=1)
        min_dist = dists.min()
        if min_dist > best_dist:
            best_dist = min_dist
            best_x = c

    return best_x
