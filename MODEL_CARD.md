# Model Card: Adaptive Per-Function Bayesian Optimiser

Framework based on Mitchell et al. (2019), *Model Cards for Model Reporting*, mapped onto the section structure required by the capstone activity.

---

## Overview

- **Name:** Adaptive Per-Function Bayesian Optimiser (`FUNCTION_CONFIG`-routed, GP-based with CMA-ES and ensemble extensions)
- **Type:** Gaussian-Process surrogate Bayesian optimisation with per-function strategy routing; later augmented with evolutionary (CMA-ES) and ensemble candidate generation
- **Version:** v13 (final round). `[INSERT: commit hash if you want it pinned]`
- **Implementation:** Python; scikit-learn `GaussianProcessRegressor`, `cma`; organised as a `bbo/` package (`data.py`, `surrogates.py`, `acquisition.py`, `optimizer.py`, `diagnostics.py`, `utils.py`) driven by a thin `Analysis.ipynb` holding `FUNCTION_CONFIG` and the round's data.

---

## Intended use

**Suitable for**
- Sequential, sample-efficient optimisation of expensive black-box functions where evaluations are scarce and gradients unavailable.
- Low-to-moderate dimensionality (≈2–8D), continuous inputs, a single scalar objective.
- Settings where calibrated posterior uncertainty matters for managing the explore/exploit trade-off, and where per-instance hand-tuning is acceptable.

**Use cases to avoid**
- High-dimensional problems: vanilla GP inference is O(n³) and the search space grows exponentially. F8 (8D) was the hardest case and gained only ~4%.
- Cheap, fast-to-evaluate functions, where grid or evolutionary search alone suffices.
- Multi-objective or discrete/combinatorial problems without modification.
- Fully automated deployment: this approach relied on weekly human inspection and per-function reconfiguration. It is a human-in-the-loop method, not a turnkey optimiser.
- Anything safety-critical without independent validation — the surrogate's uncertainty is only as trustworthy as its kernel and noise assumptions, which were repeatedly wrong before being corrected.

---

## Details: strategy across the 13 rounds

The capstone ran 13 weekly rounds (the activity brief references "ten rounds"; the documented strategy below reflects the actual 13). The core loop — fit a per-function surrogate, optimise an acquisition function, submit a batch, append results — was constant. What evolved was *how candidates were generated per function*, governed by `FUNCTION_CONFIG` (strategy ∈ `gp_ei`/`gp_ucb`/`lhs`/`ensemble`, plus `kappa`, `xi`, `n_restarts`, `exploit_std`, `kernel_nu`, `alpha`).

How decisions were made and how the approach evolved:

- **Phase 1 — Uniform baseline (R1).** One strategy for all eight: GP with Matérn-2.5 + EI (`xi=0.01`). 4/8 improved. Confirmed that one strategy does not fit all, and that F1 had no usable GP signal.
- **Phase 2 — Adaptive routing (R2–3).** Introduced per-function config from observed behaviour: UCB (`kappa`) for exploitable functions (F4, F5), EI for those needing exploration, LHS for the signal-less F1. F5 doubled twice (1089→2111→4399); F1 finally produced signal (`0.0103`).
- **Phase 3 — Full exploration (R4).** All functions switched to exploration-heavy settings (high `kappa`/`xi`, LHS). 0/8 improved by design — the round's purpose was to map dead regions and boundaries so later exploitation could be sharper.
- **Phase 4 — Return to exploitation (R5).** Tight querying in confirmed high-value regions, using the boundary map from R4.
- **Phase 5 — Kernel tuning (R6–7).** Diagnosed that failures on F1/F3/F8 were not the surrogate but its assumptions. Switched F1 to Matérn-0.5 (its peak is a narrow spike, not smooth); loosened overconfident GPs via `alpha=1e-4`. F1 tripled; F8 broke a 5-round stall.
- **Phase 6 — Noise modelling (R7–8).** Diagnosed F2 as genuinely noisy (identical input returning 0.43–0.67). Raised `alpha=0.1` so the GP smoothed through noise instead of chasing a high-variance region.
- **Phase 7 — CMA-ES (R9).** Added CMA-ES optimising the GP's predicted mean for stalled functions, motivated by HPFSO's finding that CMA-ES was the single strongest method. F2 broke through (0.667→0.762) after six stuck rounds.
- **Phase 8 — Ensemble + model-free overrides (R10–13).** Generated candidates from UCB, EI, CMA-ES, and a polynomial diagnostic (when R²>0.8), scored by GP mean, best wins. When all model-based methods failed on F6 for eight rounds, a model-free weighted centroid of the top-3 points finally cracked it (−0.243→−0.178). Final rounds used tight exploitation plus manual overrides (e.g. F5 pushed to `0.999999` precision; F3 a UCB candidate when CMA-ES kept returning the same point).

---

## Performance

**Metric.** Best observed output per function. These are unknown functions with no available ground-truth optimum, so true regret is not computable; the primary signal is the incumbent best and whether each round improves it. Secondary signal: per-round win rate (functions improved / 8) and improvement over the seeded baseline.

**Cumulative results (end of challenge — all eight improved):**

| Function | Dim | Initial best | Final best | Improvement | Round achieved | Decisive technique |
|---|---|---|---|---|---|---|
| F1 | 2 | 7.71e-16 | 0.0372 | ~10^14× | R12 | Matérn-0.5 (narrow spike) + tight UCB |
| F2 | 2 | 0.611 | 0.762 | +25% | R9 | Noise modelling (`alpha=0.1`) → CMA-ES |
| F3 | 3 | −0.035 | −0.008 | +77% | R10 | Ensemble / UCB override |
| F4 | 4 | −4.026 | 0.563 | +4.59 units | R12 | Steady GP+UCB exploitation |
| F5 | 4 | 1088.9 | 8661.7 | +696% | R12 | Boundary-pushing UCB; max-precision inputs |
| F6 | 5 | −0.714 | −0.178 | +75% | R11 | Model-free weighted centroid |
| F7 | 6 | 1.365 | 2.791 | +104% | R12 | UCB + CMA-ES; seven consecutive improvements |
| F8 | 8 | 9.598 | 9.977 | +4% | R10 | `alpha=1e-4` to fix overconfidence |

**Reading the results**
- Largest relative gains on F5 (+696%) and F4 (sign flip, −4.03→+0.56).
- Hardest case was the highest-dimensional F8 (+4%), consistent with GP degradation in high dimensions.
- The exploration round (R4) scored 0/8 yet was prerequisite to several later gains — a metric of pure win-rate would have mislabelled it a failure.

---

## Assumptions and limitations

**Assumptions (several were violated and corrected mid-campaign)**
- The objective is well-modelled by a GP — but smoothness was wrong for F1 (needed Matérn-0.5) and noise was wrong for F2 (needed `alpha=0.1`).
- Evaluations are near-noiseless (`alpha=1e-6`) — false for F2.
- The optimum lies within the clipped cube `[0.001, 0.999]^d`.
- Stationarity: a single kernel/lengthscale describes the whole domain.

**Limitations and failure modes**
- Vanilla GP is O(n³) in observations and degrades in high dimensions; F8 is near the practical limit.
- A stationary kernel struggles with sharply varying smoothness across a domain.
- Greedy acquisition stalls in local optima (F2 stuck six rounds, F6 eight); mitigated by scheduled exploration, CMA-ES, ensembles, and model-free fallbacks rather than solved cleanly.
- The per-function routing is hand-tuned on observed behaviour — strong here, but not guaranteed to transfer to unseen functions, and reliant on weekly human judgement.
- Manual overrides (centroid, midpoint, max-precision inputs) improved results but are bespoke, not principled — the F3 midpoint experiment failed outright.

---

## Ethical considerations

The functions here are synthetic, so direct ethical risk is low. The transferable point concerns real BBO settings — bioprocess critical-quality-attribute optimisation, experimental protocol tuning — where each evaluation is costly and a recommended query has real consequences. There, documenting the surrogate's assumptions, its uncertainty calibration, and the explore/exploit policy is what lets others judge whether a recommended query is trustworthy.

This campaign is a concrete argument for that transparency: several gains came only *after* a wrong assumption was made explicit and corrected (F1's smoothness, F2's noise, F8's overconfidence). A model card that records where the model extrapolates (high posterior variance) versus interpolates (low variance near data), and which assumptions have already failed, lets a reader avoid repeating those mistakes — and lets anyone re-running the loop with the same `FUNCTION_CONFIG`, kernels, and seeds recover the same decisions and history.

**Would more detail improve this card?** Adding per-round seeds, hyperparameter trajectories, and acquisition-value logs would improve full reproducibility and is worth doing if the repo is meant as a reference implementation. For the capstone's purpose — communicating *how the approach decides* and *where it should and should not be trusted* — the current structure is sufficient: it states the decision rule, the assumptions (including the ones that broke), and the failure modes explicitly, which is what a reader needs to judge and adapt it.
