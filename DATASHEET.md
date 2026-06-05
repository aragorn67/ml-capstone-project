# Datasheet: BBO Capstone — Query History and Function Evaluations

Framework based on Gebru et al. (2021), *Datasheets for Datasets*, mapped onto the section structure required by the capstone activity.

---

## Motivation

**Why was this dataset created?**
It is the accumulated query history and corresponding function evaluations produced while running a Bayesian optimisation (BO) campaign against eight unknown ("black-box") objective functions of dimensionality 2D to 8D, over 13 weekly rounds. The dataset is both the *input to* and the *product of* the optimisation loop: at each round a surrogate is fit to all observed `(x, y)` pairs and used to choose the next queries, and the returned values are appended to the history. It is the model's only source of information about each function.

**What task does it support?**
Sample-efficient global optimisation of expensive, gradient-free functions — locating high-value regions of each function's domain using as few evaluations as possible.

**Who created it and for whom?**
Created by me as part of the Imperial Executive Education BBO capstone. Not commissioned externally; no funding body beyond the course context.

---

## Composition

**What does an instance represent?**
A single `(input, output)` pair: an input vector `x` in a function's domain and the scalar output `y` returned by the black-box oracle, recorded alongside the round it was submitted in and the strategy that generated it.

**What does the dataset contain?**
- Eight functions, F1–F8, with input dimensionality 2, 2, 3, 4, 4, 5, 6, 8.
- Inputs are real-valued vectors clipped to `[0.001, 0.999]` per dimension.
- Outputs are scalars from the oracle. No labels, classes, or derived targets.

**How large is it?**
Seeded from a course-provided `initial_data` set, then grown by a batch of roughly 12 queries per function per round across 13 rounds. `[INSERT: exact cumulative count per function if you want it precise — countable from the bbo/data files]`

**Format**
Per-function input–output records under `Initial_data_points_starter/initial_data`, loaded by `bbo.data.load_function_data`. The notebook's `new_data` dictionary holds each round's batch before it is appended.

**Are there gaps, noise, or irregularities?** Yes, and most are intrinsic to the method rather than data faults:
- Coverage is **non-uniform by design** — acquisition functions deliberately oversample promising regions, so data clusters there while large parts of each domain stay sparse or unobserved. It is not a representative sample of any domain.
- **F1** returned values indistinguishable from zero for the first two rounds (e.g. `1.2e-34`, `-1.0e-288`) until space-filling design surfaced a narrow responsive spike near `(0.439, 0.452)`. Early F1 records are uninformative.
- **F2 is genuinely noisy**: the same input `(0.7346, 0.8029)` returned `0.472`, `0.666`, `0.599`, and `0.434` across rounds. The dataset therefore contains stochastic, non-reproducible outputs for at least one function.
- **F6–F8** are severely undersampled relative to their volume (curse of dimensionality) — a few dozen points in up to an 8D cube.
- No corrupted records, but no guarantee the global optimum has been sampled or bracketed for any function.

**Is anything confidential or sensitive?** No. The functions are synthetic; no personal, proprietary, or sensitive content.

---

## Collection process

**How were the queries generated?**
Through an iterative BO loop, one batch per weekly round:
1. Fit a Gaussian Process (GP) surrogate per function to all observed data.
2. Optimise an acquisition function over the surrogate to propose next queries.
3. Submit to the oracle and record returned outputs.

Later rounds added non-GP generators (CMA-ES on the GP surface, an ensemble that scores candidates by GP mean, a polynomial diagnostic, and occasional model-free or manual overrides), but the collect-fit-propose cycle stayed the same.

**What strategy was used?**
A **per-function** strategy held in a `FUNCTION_CONFIG` dictionary, revised each round from observed behaviour. Strategies span `gp_ei`, `gp_ucb`, `lhs`, and `ensemble`, with per-function `kappa`, `xi`, `n_restarts`, `exploit_std`, `kernel_nu`, and `alpha`.

**Over what time frame?**
13 weekly rounds. `[INSERT: start/end dates]`

**Who was involved?**
Solely me. The oracle returns values automatically; there is no human labelling.

---

## Preprocessing and uses

**Were transformations applied?**
- Inputs clipped to `[0.001, 0.999]` per dimension before submission.
- GP outputs normalised at fit time (`normalize_y=True`) to handle differing output scales across functions.
- Per-function GP noise term `alpha` raised where appropriate (e.g. `0.1` for the noisy F2, `1e-4` to loosen overconfident fits) — a modelling choice, not a change to the recorded data.
- No imputation, deduplication, or filtering of recorded evaluations. The full raw history is retained because every past observation conditions the next decision; repeated queries at the same point are kept precisely because they reveal F2's noise.

**Intended uses**
- Driving and reproducing the BO loop for these eight functions.
- Post-hoc analysis of convergence, acquisition behaviour, noise, and per-function difficulty.
- A teaching record of how the strategy evolved across 13 rounds.

**Inappropriate uses**
- Treating it as a representative or i.i.d. sample of any domain (it is adversarially clustered by acquisition).
- Training a standalone model meant to generalise across a whole domain — the sampling bias would yield overconfident, locally-valid predictions.
- Inferring properties of unsampled regions, especially in F6–F8.

---

## Distribution and maintenance

**Where is it available?** Public BBO capstone GitHub repository: `[INSERT: repo URL]`.

**Terms of use** `[INSERT: licence — e.g. MIT for code; data course-provided, released for course use]`. The functions themselves are course-provided; only my query history and configuration are released here.

**Who maintains it?** Maintained by me for the duration of the capstone, updated each round; the progress notes track round-by-round changes.

**How are updates communicated?** Through commit history and the weekly progress notes. No long-term maintenance commitment beyond the programme.
