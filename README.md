# ML/AI Capstone Project - Black-Box Optimisation Challenge

## Project Overview

This capstone project is a black-box optimisation (BBO) challenge where the goal is to maximise the output of 8 unknown functions through iterative querying. Each week I submit one input point per function and receive the corresponding output. I cannot see the function equations, inspect gradients, or evaluate freely — one new data point per function per week across 13 rounds.

This mirrors real-world scenarios where evaluation is expensive: hyperparameter tuning of large models, drug discovery, or any setting where each experiment costs time or money. The core skill is making intelligent decisions under uncertainty with limited data.

For my career in computational biology, this connects directly to optimising experimental protocols and model hyperparameters on omics pipelines, where each evaluation involves significant compute and the search spaces are high-dimensional.

## Inputs and Outputs

Each function takes a vector of real-valued inputs in [0, 1], specified to six decimal places (internally clipped to [0.001, 0.999] to avoid degenerate boundary behaviour). Dimensionality varies: functions 1-2 are 2D, function 3 is 3D, functions 4-5 are 4D, function 6 is 5D, function 7 is 6D, function 8 is 8D. Submission format: x1-x2-x3-...-xn (e.g. 0.500000-0.750000 for a 2D function). The output is a single scalar. Starting data ranges from 10 to 40 points per function, growing by one per week.

## Challenge Objectives

Maximise the output of each function. Constraints: one query per function per week (13 rounds total), no access to function form or gradients, delayed feedback. The unknown structure means I cannot assume linearity or smoothness.

## Goals

- Find the global maximum of each function within a strict query budget
- Develop and adapt optimisation strategies based on observed results
- Balance exploration of untested regions against exploitation of known good areas
- Document the iterative decision-making process for reproducibility

## Key Technologies

- Python, NumPy, SciPy, scikit-learn
- Gaussian Process regression as surrogate model, with per-function Matern kernels (nu = 0.5 / 1.5 / 2.5) and tunable noise term (alpha)
- Acquisition functions: Expected Improvement, Upper Confidence Bound
- Latin Hypercube Sampling for space-filling exploration
- CMA-ES (evolutionary search on the GP predicted-mean surface) for stalled functions
- Ensemble candidate generation (UCB / EI / CMA-ES / polynomial), scored by GP mean
- Polynomial regression diagnostic to cross-check GP predictions
- Jupyter Notebooks for development and analysis

## Technical Approach

The strategy evolved through eight phases across the 13 rounds. Full round-by-round reasoning is in `Progress_Deck.md`.

| Phase | Rounds | Approach | Key insight |
|-------|--------|----------|-------------|
| Uniform baseline | 1 | Same GP + EI for all 8 | One strategy does not fit all (4/8 improved) |
| Adaptive routing | 2-3 | Per-function config (UCB / EI / LHS) | Match strategy to each function's character |
| Full exploration | 4 | Exploration-heavy everywhere | 0/8 by design — maps dead regions for later |
| Return to exploitation | 5 | Tight querying in good regions | Exploration data sharpens the GP |
| Kernel tuning | 6-7 | Matern 0.5 for narrow spikes, alpha tuning | Failures were wrong assumptions, not the surrogate |
| Noise modelling | 7-8 | alpha = 0.1 for noisy F2 | Same input, different output = noise to smooth through |
| CMA-ES | 9 | Evolutionary search on GP surface | F2 breakthrough (0.667 -> 0.762) |
| Ensemble + overrides | 10-13 | Multiple methods + model-free fallbacks | F3 and F6 breakthroughs; final precision squeeze |

## Results

All eight functions improved from their seeded baseline.

| Function | Dim | Initial best | Final best | Improvement |
|----------|-----|--------------|------------|-------------|
| 1 | 2D | 7.71e-16 | 0.0372 | ~10^14x |
| 2 | 2D | 0.611 | 0.762 | +25% |
| 3 | 3D | -0.035 | -0.008 | +77% |
| 4 | 4D | -4.026 | 0.563 | sign flip, +4.59 |
| 5 | 4D | 1088.9 | 8661.7 | +696% |
| 6 | 5D | -0.714 | -0.178 | +75% |
| 7 | 6D | 1.365 | 2.791 | +104% |
| 8 | 8D | 9.598 | 9.977 | +4% |

## Documentation

- [Datasheet](DATASHEET.md) — describes the query-history dataset: motivation, composition, collection, preprocessing, and intended/inappropriate uses.
- [Model Card](MODEL_CARD.md) — describes the optimisation approach: intended use, the strategy across all rounds, performance, assumptions, limitations, and ethical considerations.

## Project Structure

```
├── Analysis.ipynb            # Thin driver: FUNCTION_CONFIG, weekly data, run command
├── bbo/                      # Optimisation package
│   ├── data.py               # Load / append function data
│   ├── surrogates.py         # GP surrogate models
│   ├── acquisition.py        # EI, UCB, LHS
│   ├── optimizer.py          # Query generation, CMA-ES, ensemble
│   ├── diagnostics.py        # Polynomial cross-check
│   └── utils.py
├── Initial_data_points_starter/
│   └── initial_data/
│       ├── function_1/       # initial_inputs.npy, initial_outputs.npy
│       ├── ...
│       └── function_8/
├── Progress_Deck.md          # Weekly progress notes and decision log (all 13 rounds)
├── DATASHEET.md              # Dataset documentation
├── MODEL_CARD.md             # Model documentation
└── README.md
```
