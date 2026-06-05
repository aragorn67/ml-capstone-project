"""
bbo/optimizer.py — Query optimisation and strategy routing

This is the core module. It:
  1. Reads the per-function config
  2. Routes to the correct strategy (GP+EI, GP+UCB, LHS)
  3. Searches for the best query point
  4. Returns portal-ready strings

The generate_all_queries function is the main entry point called
from the notebook.
"""

import numpy as np
from scipy.optimize import minimize

from .data import load_function_data
from .surrogates import fit_gp
from .acquisition import expected_improvement, upper_confidence_bound, lhs_next_query
from .utils import format_portal_string, print_function_header, print_summary


# ============================================================
# SEARCH: find the point that maximises an acquisition function
# ============================================================
#
# HOW: Random restarts + local optimisation (L-BFGS-B).
# Two phases:
#   1. Exploration: random starting points across the space
#   2. Exploitation: starting points near the current best
#
# The exploit_std parameter controls how tightly the exploitation
# restarts cluster around the best point:
#   0.05 = standard (default)
#   0.01 = very tight (for narrow peaks like function 1)
#   0.15 = loose (for breaking out of local regions)
# ============================================================

def search_acquisition(acq_func, acq_args, inputs, best_idx, n_dim,
                       n_restarts=100, exploit_std=0.05):
    """
    Search for the point maximising an acquisition function.

    Args:
        acq_func:    the acquisition function (EI or UCB)
        acq_args:    tuple of extra args to pass to acq_func
        inputs:      all observed inputs (for exploitation restarts)
        best_idx:    index of the current best point
        n_dim:       number of dimensions
        n_restarts:  number of random exploration restarts
        exploit_std: std of Gaussian noise for exploitation restarts

    Returns:
        best_x: the recommended query point
    """
    bounds = [(0.001, 0.999)] * n_dim
    best_x = None
    best_val = np.inf

    # Phase 1: Exploration — random restarts across the space
    for _ in range(n_restarts):
        x0 = np.random.uniform(0.001, 0.999, n_dim)
        try:
            result = minimize(acq_func, x0, args=acq_args,
                              bounds=bounds, method='L-BFGS-B')
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            continue

    # Phase 2: Exploitation — restarts near current best
    n_exploit = max(50, n_restarts // 2)
    for _ in range(n_exploit):
        x0 = inputs[best_idx] + np.random.normal(0, exploit_std, n_dim)
        x0 = np.clip(x0, 0.001, 0.999)
        try:
            result = minimize(acq_func, x0, args=acq_args,
                              bounds=bounds, method='L-BFGS-B')
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            continue

    return best_x


# ============================================================
# STRATEGY HANDLERS
# ============================================================

def run_lhs(inputs, n_dim, config):
    """Latin Hypercube Sampling — no model, pure exploration."""
    query = lhs_next_query(inputs, n_dim)
    print(f"  LHS query (maximin)")
    return query


def run_gp_ei(inputs, outputs, n_dim, config):
    """GP surrogate + Expected Improvement."""
    xi = config.get("xi", 0.01)
    n_restarts = config.get("n_restarts", 100)
    exploit_std = config.get("exploit_std", 0.05)
    kernel_nu = config.get("kernel_nu", 2.5)
    alpha = config.get("alpha", 1e-6)

    gp = fit_gp(inputs, outputs, n_dim, kernel_nu=kernel_nu, alpha=alpha)
    y_best = outputs.max()
    best_idx = np.argmax(outputs)

    query = search_acquisition(
        expected_improvement, (gp, y_best, xi),
        inputs, best_idx, n_dim,
        n_restarts=n_restarts, exploit_std=exploit_std
    )

    mu, sigma = gp.predict(query.reshape(1, -1), return_std=True)
    print(f"  EI (xi={xi}) | mean={mu[0]:.4e} std={sigma[0]:.4e}")
    return query


def run_gp_ucb(inputs, outputs, n_dim, config):
    """GP surrogate + Upper Confidence Bound."""
    kappa = config.get("kappa", 1.0)
    n_restarts = config.get("n_restarts", 100)
    exploit_std = config.get("exploit_std", 0.05)
    kernel_nu = config.get("kernel_nu", 2.5)
    alpha = config.get("alpha", 1e-6)

    gp = fit_gp(inputs, outputs, n_dim, kernel_nu=kernel_nu, alpha=alpha)
    best_idx = np.argmax(outputs)

    query = search_acquisition(
        upper_confidence_bound, (gp, kappa),
        inputs, best_idx, n_dim,
        n_restarts=n_restarts, exploit_std=exploit_std
    )

    mu, sigma = gp.predict(query.reshape(1, -1), return_std=True)
    print(f"  UCB (κ={kappa}) | mean={mu[0]:.4e} std={sigma[0]:.4e}")
    return query


# ============================================================
# STRATEGY ROUTER
# ============================================================
# Maps strategy names to handler functions.
# Available strategies:
#   "lhs"      — Latin Hypercube Sampling (pure exploration)
#   "gp_ei"    — GP + Expected Improvement
#   "gp_ucb"   — GP + Upper Confidence Bound
#   "cma_es"   — CMA-ES optimising the GP's predicted mean
# ============================================================


def run_cma_es(inputs, outputs, n_dim, config):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    optimising the GP's predicted mean.

    WHAT: CMA-ES is an evolutionary optimiser that adapts a
    multivariate Gaussian search distribution. It samples candidates,
    evaluates them, and reshapes the distribution toward better regions.

    WHY: CMA-ES was found to be the single best individual method
    in the HPFSO paper (Ansotegui et al., 2021). It handles
    non-separable, ill-conditioned landscapes better than random
    restarts of L-BFGS-B.

    THE TRICK: We can't evaluate the real function (one query per week).
    Instead, CMA-ES optimises the GP's predicted mean as a cheap proxy.
    This combines:
      - GP's learned landscape model
      - CMA-ES's superior search strategy

    WHEN TO USE: For functions where GP+UCB/EI has stalled — the GP
    has a reasonable model but L-BFGS-B search isn't finding the peak.

    Config parameters:
      sigma0:     initial step size (default 0.3)
      popsize:    population size (default 20)
      maxiter:    max generations (default 100)
      kernel_nu:  Matern smoothness for the GP
      alpha:      GP noise term
    """
    import cma

    kernel_nu = config.get("kernel_nu", 2.5)
    alpha = config.get("alpha", 1e-6)
    sigma0 = config.get("sigma0", 0.3)
    popsize = config.get("popsize", 20)
    maxiter = config.get("maxiter", 100)

    # Fit GP surrogate
    gp = fit_gp(inputs, outputs, n_dim, kernel_nu=kernel_nu, alpha=alpha)
    best_idx = np.argmax(outputs)

    # CMA-ES minimises, so we negate the GP mean
    def neg_gp_mean(x):
        x = np.clip(x, 0.001, 0.999)
        return -gp.predict(x.reshape(1, -1))[0]

    # Start from the current best point
    x0 = inputs[best_idx].copy()
    bounds_lo = [0.001] * n_dim
    bounds_hi = [0.999] * n_dim

    opts = {
        'bounds': [bounds_lo, bounds_hi],
        'popsize': popsize,
        'maxiter': maxiter,
        'verbose': -9,  # silent
        'seed': 42,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(neg_gp_mean)
    query = np.clip(es.result.xbest, 0.001, 0.999)

    mu, sigma = gp.predict(query.reshape(1, -1), return_std=True)
    print(f"  CMA-ES (σ0={sigma0}) | mean={mu[0]:.4e} std={sigma[0]:.4e}")
    return query


STRATEGIES = {
    "lhs": run_lhs,
    "gp_ei": run_gp_ei,
    "gp_ucb": run_gp_ucb,
    "cma_es": run_cma_es,
}


def run_ensemble(inputs, outputs, n_dim, config):
    """
    Ensemble strategy — generate candidates from multiple methods,
    then pick the best one as scored by the GP's predicted mean.

    WHAT: Instead of committing to one search method, we run several
    in parallel and let the GP judge which candidate is most promising.

    WHY: No single optimisation method dominates across all function
    types (no free lunch theorem). The NeurIPS 2020 BBO winners all
    used ensemble approaches. This is our simplified version — same
    core insight, lighter implementation.

    HOW:
      1. Fit one GP to all data
      2. Generate candidates from GP+EI, GP+UCB, CMA-ES
      3. Optionally add polynomial regression's predicted max
      4. Score all candidates by GP predicted mean
      5. Pick the highest-scoring candidate

    ADVANTAGE: Gets diversity of search strategies without the risk
    of betting on the wrong one. If CMA-ES finds a better optimum
    than UCB, we use it. If UCB is better, we use that instead.

    Config parameters:
      kappa:      UCB exploration param (default 0.5)
      xi:         EI exploration param (default 0.01)
      n_restarts: restarts for EI/UCB search (default 100)
      exploit_std: perturbation for exploitation (default 0.05)
      kernel_nu:  Matern smoothness (default 2.5)
      alpha:      GP noise (default 1e-6)
      sigma0:     CMA-ES step size (default 0.3)
      use_poly:   include polynomial candidate (default True)
    """
    import cma
    import math

    # Read config
    kappa = config.get("kappa", 0.5)
    xi = config.get("xi", 0.01)
    n_restarts = config.get("n_restarts", 100)
    exploit_std = config.get("exploit_std", 0.05)
    kernel_nu = config.get("kernel_nu", 2.5)
    alpha = config.get("alpha", 1e-6)
    sigma0 = config.get("sigma0", 0.3)
    use_poly = config.get("use_poly", True)

    # Fit one shared GP
    gp = fit_gp(inputs, outputs, n_dim, kernel_nu=kernel_nu, alpha=alpha)
    y_best = outputs.max()
    best_idx = np.argmax(outputs)

    candidates = {}

    # --- Candidate 1: GP + UCB ---
    try:
        c_ucb = search_acquisition(
            upper_confidence_bound, (gp, kappa),
            inputs, best_idx, n_dim,
            n_restarts=n_restarts, exploit_std=exploit_std
        )
        candidates["UCB"] = c_ucb
    except Exception:
        pass

    # --- Candidate 2: GP + EI ---
    try:
        c_ei = search_acquisition(
            expected_improvement, (gp, y_best, xi),
            inputs, best_idx, n_dim,
            n_restarts=n_restarts, exploit_std=exploit_std
        )
        candidates["EI"] = c_ei
    except Exception:
        pass

    # --- Candidate 3: CMA-ES on GP mean ---
    try:
        def neg_gp_mean(x):
            x = np.clip(x, 0.001, 0.999)
            return -gp.predict(x.reshape(1, -1))[0]

        x0 = inputs[best_idx].copy()
        opts = {
            'bounds': [[0.001] * n_dim, [0.999] * n_dim],
            'popsize': 20,
            'maxiter': 100,
            'verbose': -9,
            'seed': 42,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(neg_gp_mean)
        candidates["CMA-ES"] = np.clip(es.result.xbest, 0.001, 0.999)
    except Exception:
        pass

    # --- Candidate 4: Polynomial max (if enabled and enough data) ---
    if use_poly and len(outputs) >= 10:
        try:
            from .diagnostics import polynomial_diagnostic
            n_features_d3 = int(math.factorial(n_dim + 3) /
                                (math.factorial(n_dim) * math.factorial(3)))
            degree = 2 if len(outputs) < n_features_d3 * 2 else 3
            poly_result = polynomial_diagnostic(inputs, outputs, n_dim, degree=degree)
            if poly_result["r_squared"] > 0.8:
                candidates["Poly"] = poly_result["max_point"]
        except Exception:
            pass

    # --- Score all candidates by GP predicted mean ---
    if not candidates:
        # Fallback: return current best
        return inputs[best_idx]

    best_name = None
    best_mean = -np.inf
    print(f"  Ensemble candidates:")
    for name, candidate in candidates.items():
        mu = gp.predict(candidate.reshape(1, -1))[0]
        marker = ""
        if mu > best_mean:
            best_mean = mu
            best_name = name
            best_query = candidate
        print(f"    {name:8s} → mean={mu:.4e} at {np.round(candidate, 3)}")

    mu, sigma = gp.predict(best_query.reshape(1, -1), return_std=True)
    print(f"  WINNER: {best_name} | mean={mu[0]:.4e} std={sigma[0]:.4e}")
    return best_query


# Add ensemble to available strategies
STRATEGIES["ensemble"] = run_ensemble


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def generate_all_queries(new_data, function_config, base_path, seed=42):
    """
    Generate query recommendations for all 8 functions.

    Args:
        new_data:        dict of weekly results per function
        function_config: dict of per-function strategy configs
        base_path:       path to initial_data folder
        seed:            random seed for reproducibility

    Prints portal-ready strings for submission.
    """
    np.random.seed(seed)
    portal_strings = {}

    for i in range(1, 9):
        inputs, outputs = load_function_data(i, new_data, base_path)
        n_dim = inputs.shape[1]
        best_idx = np.argmax(outputs)
        config = function_config[i]
        strategy = config["strategy"]

        print_function_header(i, n_dim, inputs.shape[0], strategy,
                              outputs[best_idx], inputs[best_idx])

        # Route to the correct strategy handler
        handler = STRATEGIES[strategy]
        if strategy == "lhs":
            query = handler(inputs, n_dim, config)
        else:
            query = handler(inputs, outputs, n_dim, config)

        portal_str = format_portal_string(query)
        portal_strings[i] = portal_str
        print(f"  >>> {portal_str}")
        print()

    print_summary(portal_strings)
    return portal_strings