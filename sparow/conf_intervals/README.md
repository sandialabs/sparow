# Confidence Interval Estimation (Uncertainty Quantification) for Stochastic Programs

For stochastic programs, this module provides methods for estimating confidence intervals on an upper bound for the optimality gap of a given candidate first-stage solution.

It supports both a single-fidelity workflow for one model and a multifidelity workflow that uses approximate control variates. In the multifidelity case, the goal is to reduce the variance of the high-fidelity Monte Carlo estimator of the upper bound on the optimality gap.

## Main methods

Two algorithms are currently supported:

- **Standard Multiple Replications Procedure (MRP)**  
  For single fidelity workflows. Uses repeated independent scenario-batch replications to estimate an upper bound on the optimality gap and construct a confidence interval from the sample mean and sample variance of the replication outputs. The standard MRP procedure follows [Mak, Morton, and Wood (1999)] and [Bayraksan and Morton (2009)].

- **Approximate Control Variates Multiple Replications Procedure (ACV-MRP)**  
  For multifidelity workflows. Extends standard MRP by using a low-fidelity model as an approximate control variate. High-fidelity and low-fidelity replication outputs are evaluated on the same sampled scenario batches to induce correlation and reduce variance.

## Model interface

To make a new stochastic programming instance compatible with the confidence interval code, the problem module should expose the following factory functions:

- `get_sp_model_for_uq(...)` for **single-fidelity** workflows
  Returns a single-fidelity model wrapper for use with `StandardMRP` and exact finite-population gap evaluation.

- `get_model_ensemble_for_uq(...)` for **multifidelity** workflows  
  Returns a multifidelity ensemble for use with `ACVMRP` and PyApprox integration.

These should return objects compatible with the internal protocols defined in `protocols.py`.

In practice:

- `StandardMRP` expects **one model wrapper**
- `ACVMRP` expects **two model wrappers**: one high fidelity and one low fidelity

## Minimal usage: Standard MRP

```python
from sparow.conf_intervals.mrp_options import MRPOptions
from sparow.conf_intervals.standard_mrp import StandardMRP
from my_problem_module import get_sp_model_for_uq

model = get_sp_model_for_uq(
    model_name="MyModel",
    use_integer=False,
    seed=12345,
    with_replacement=True,
)

xhat = {
    # first-stage variable values
}

options = MRPOptions(
    n=50,
    m=10,
    alpha=0.05,
    seed=12345,
    with_replacement=True,
    solver_name="gurobi_direct",
    verbose=True,
)

mrp = StandardMRP(model=model, options=options)
results = mrp.run(xhat=xhat)

print(results)
```

## Minimal usage: ACV-MRP

```python
from sparow.conf_intervals.acv_mrp_options import ACVMRPOptions
from sparow.conf_intervals.acv_mrp import ACVMRP
from my_problem_module import get_model_ensemble_for_uq

ensemble = get_model_ensemble_for_uq(
    model_name="HF",
    use_integer=False,
    seed=12345,
    with_replacement=True,
    lf_model_type="classic",
)

hf_model = ensemble.high_fidelity_model()
lf_model = ensemble.low_fidelity_model()

xhat = {
    # first-stage variable values
}

options = ACVMRPOptions(
    n=50,
    m=10,
    M=5,
    alpha=0.05,
    seed=12345,
    with_replacement=True,
    solver_name="gurobi_direct",
    verbose=True,
)

acv = ACVMRP(
    hf_model=hf_model,
    lf_model=lf_model,
    options=options,
)

results = acv.run(xhat=xhat)

print(results)
```

## True finite-population benchmark

If the full finite scenario population is known and your stochastic programming instance is small enough to solve over all scenarios within a reasonable amount of time, the package also provides a helper to compute the exact finite-population optimal value, the value of xhat, and the resulting true optimality gap:

```python
from sparow.conf_intervals.evaluate_true_optimality_gap import TrueOptimalityGapEvaluator

true_gap_evaluator = TrueOptimalityGapEvaluator(
    model=model,
    solver_name="gurobi_direct",
)

true_gap_results = true_gap_evaluator.compute_true_gap(xhat=xhat)
print(true_gap_results)
```

## Notes

- The current implementation assumes minimization problems.
- The standard MRP interval is based on a Monte Carlo sample mean CLT and uses the sample standard error.
- The ACV-MRP interval uses a plug-in variance estimate for the control-variate-adjusted estimator.
- The low-fidelity model should be chosen so that it is both cheaper and sufficiently correlated with the high-fidelity Monte Carlo estimator.

## PyApprox Integration

The codebase also includes a PyApprox [Jakeman (2023)] integration for using multifidelity allocation tools to suggest high-fidelity and low-fidelity sample counts, which can then be translated into ACV-MRP parameters. This helps allocate finite computational budget between high-fidelity model evaluations and low-fidelity model evaluations, such that we achieve maximum variance reduction for the optimality gap estimator.

## References

- Mak, Wai-Kei, David P. Morton, and R. Kevin Wood. 1999. “Monte Carlo Bounding Techniques for Determining Solution Quality in Stochastic Programs.” *Operations Research Letters* 24 (1–2): 47–56.
- Bayraksan, Güzin, and David P. Morton. 2009. “Assessing Solution Quality in Stochastic Programs via Sampling.” In *Decision Technologies and Applications*, 102–122. INFORMS.
- Jakeman, J. D. 2023. “PyApprox: A software package for sensitivity analysis, Bayesian inference, optimal experimental design, and multi-fidelity uncertainty quantification and surrogate modeling.” *Environmental Modelling & Software* 170: 105825.

