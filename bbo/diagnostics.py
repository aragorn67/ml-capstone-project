"""
bbo/diagnostics.py — Secondary models for cross-checking GP predictions

The GP is our primary surrogate. These diagnostics fit simpler models
to the same data and flag disagreements — if the polynomial thinks the
peak is in a different region than the GP, that's worth investigating.

This is NOT a replacement for the GP. It's a sanity check.
"""

import math

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def polynomial_diagnostic(inputs, outputs, n_dim, degree=3, alpha=0.1,
                          grid_resolution=200, search_bounds=None):
    """
    Fit a polynomial regression and find its predicted maximum.

    WHAT: Fits a polynomial surface to the data, then grid-searches
    for the maximum. Compares this to the GP's best-known point.

    WHY: The GP assumes a specific kernel structure (Matern). If the
    true function doesn't match that structure, the GP might miss
    the peak. A polynomial has different biases — if it finds the
    peak in a different region, that's a signal to investigate.

    LIMITATIONS:
      - Polynomials extrapolate badly near boundaries
      - High degree + few points = overfitting
      - No uncertainty estimates
      - Grid search is coarse in high dimensions

    Args:
        inputs:           (n_points, n_dims) observed inputs
        outputs:          (n_points,) observed outputs
        n_dim:            number of dimensions
        degree:           polynomial degree (2, 3, or 4)
        alpha:            Ridge regularisation (higher = smoother)
        grid_resolution:  points per dimension in grid search
        search_bounds:    optional (n_dim, 2) array of [lo, hi] per dim.
                          Defaults to [0.001, 0.999] for all.

    Returns:
        dict with:
          - r_squared: fit quality (1.0 = perfect, <0.5 = poor)
          - predicted_max: highest predicted value
          - max_point: input where the max occurs
          - model: the fitted pipeline (for further inspection)
    """
    # Fit polynomial
    model = make_pipeline(
        PolynomialFeatures(degree, interaction_only=False),
        Ridge(alpha=alpha)
    )
    model.fit(inputs, outputs)
    r_squared = model.score(inputs, outputs)

    # For high dimensions, grid search is too expensive.
    # Use random sampling + local refinement instead.
    if n_dim <= 3:
        # Full grid search for low dimensions
        if search_bounds is None:
            bounds = [(0.001, 0.999)] * n_dim
        else:
            bounds = search_bounds

        grids = [np.linspace(lo, hi, grid_resolution) for lo, hi in bounds]
        mesh = np.meshgrid(*grids, indexing='ij')
        X_grid = np.column_stack([m.ravel() for m in mesh])

        preds = model.predict(X_grid)
        top_idx = np.argmax(preds)
        predicted_max = preds[top_idx]
        max_point = X_grid[top_idx]

    else:
        # Random sampling for higher dimensions
        n_samples = min(500000, grid_resolution ** min(n_dim, 4))
        X_random = np.random.uniform(0.001, 0.999, (n_samples, n_dim))

        # Also include perturbations around the known best
        best_idx = np.argmax(outputs)
        n_local = 10000
        X_local = inputs[best_idx] + np.random.normal(0, 0.05, (n_local, n_dim))
        X_local = np.clip(X_local, 0.001, 0.999)

        X_all = np.vstack([X_random, X_local])
        preds = model.predict(X_all)
        top_idx = np.argmax(preds)
        predicted_max = preds[top_idx]
        max_point = X_all[top_idx]

    return {
        "r_squared": r_squared,
        "predicted_max": predicted_max,
        "max_point": max_point,
        "model": model,
    }


def run_diagnostics(inputs, outputs, n_dim, func_num, gp_best_input):
    """
    Run polynomial diagnostic and compare with GP's best point.

    Prints a summary showing whether the polynomial agrees or
    disagrees with the GP about where the peak is.
    """
    # Skip if too few points for polynomial
    if len(outputs) < 10:
        print(f"  F{func_num}: too few points for diagnostic")
        return None

    # Choose degree based on data size
    n_features_d3 = int(math.factorial(n_dim + 3) /
                        (math.factorial(n_dim) * math.factorial(3)))
    if len(outputs) < n_features_d3 * 2:
        degree = 2
    else:
        degree = 3

    result = polynomial_diagnostic(inputs, outputs, n_dim, degree=degree)
    r2 = result["r_squared"]
    poly_max = result["max_point"]
    poly_val = result["predicted_max"]

    # Distance between polynomial max and GP best
    dist = np.linalg.norm(poly_max - gp_best_input)

    # Determine agreement
    if dist < 0.1:
        agreement = "AGREE"
    elif dist < 0.3:
        agreement = "SLIGHT DISAGREEMENT"
    else:
        agreement = "DISAGREE — worth investigating"

    print(f"  F{func_num} poly(deg={degree}): R²={r2:.3f} | "
          f"max={poly_val:.4e} at {np.round(poly_max, 3)} | "
          f"dist from GP best={dist:.3f} | {agreement}")

    return result