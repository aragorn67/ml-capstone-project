# ML/AI Capstone Project - Black-Box Optimisation Challenge

## Project Overview

This capstone project is a black-box optimisation (BBO) challenge where the goal is to maximise the output of 8 unknown functions through iterative querying. Each week I submit one input point per function and receive the corresponding output. I cannot see the function equations, inspect gradients, or evaluate freely — one new data point per function per week across 13 rounds.

This mirrors real-world scenarios where evaluation is expensive: hyperparameter tuning of large models, drug discovery, or any setting where each experiment costs time or money. The core skill is making intelligent decisions under uncertainty with limited data.

For my career in computational biology, this connects directly to optimising experimental protocols and model hyperparameters on omics pipelines, where each evaluation involves significant compute and the search spaces are high-dimensional.

## Inputs and Outputs

Each function takes a vector of real-valued inputs in [0, 1], specified to six decimal places. Dimensionality varies: functions 1-2 are 2D, function 3 is 3D, functions 4-5 are 4D, function 6 is 5D, function 7 is 6D, function 8 is 8D. Submission format: x1-x2-x3-...-xn (e.g. 0.500000-0.750000 for a 2D function). The output is a single scalar. Starting data ranges from 10 to 40 points per function, growing by one per week.

## Challenge Objectives

Maximise the output of each function. Constraints: one query per function per week (13 rounds total), no access to function form or gradients, delayed feedback. The unknown structure means I cannot assume linearity or smoothness.

## Goals

- Find the global maximum of each function within a strict query budget
- Develop and adapt optimisation strategies based on observed results
- Balance exploration of untested regions against exploitation of known good areas
- Document the iterative decision-making process for reproducibility

## Key Technologies

- Python, NumPy, SciPy, scikit-learn
- Gaussian Process regression (Matern 2.5 kernel) as surrogate model
- Acquisition functions: Expected Improvement, Upper Confidence Bound
- Latin Hypercube Sampling for space-filling exploration
- Jupyter Notebooks for development and analysis

## Technical Approach

Round 1: Uniform GP + EI across all 8 functions. Result: 4/8 improved.

Round 2: Adaptive per-function strategy based on data character. LHS for function 1 (no GP signal), UCB with low kappa for functions 4 and 5 (exploit hot regions), increased EI exploration for function 3, more random restarts for high-dimensional functions. Result: 5/8 improved, function 5 nearly doubled (1089 to 2111).

Round 3: Kept round 2 configuration to validate consistency before further changes.

This section will be updated as the approach evolves through the remaining rounds.

## Project Structure

```
├── Analysis_1.ipynb          # Main notebook: data loading, GP fitting, query generation
├── Initial_data_points_starter/
│   └── initial_data/
│       ├── function_1/       # initial_inputs.npy, initial_outputs.npy
│       ├── function_2/
│       ├── ...
│       └── function_8/
├── week2_progress.md         # Weekly progress notes and decision log
└── README.md
```
