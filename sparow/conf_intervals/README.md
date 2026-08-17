# Confidence Interval Estimation (Uncertainty Quantification) for Stochastic Programs

For stochastic programs, this module provides methods for estimating confidence intervals on an upper bound for the optimality gap of a given candidate first-stage solution.

It supports both single-fidelity estimation for one model and multifidelity estimation using approximate control variates. In the multifidelity setting, the goal is to reduce the variance of the high-fidelity optimality-gap upper-bound estimator.

## Main methods

Two algorithms are currently supported:

- **Standard Multiple Replications Procedure (MRP)**  
  For single fidelity workflows. Uses repeated independent scenario-batch replications to estimate an upper bound on the optimality gap and construct a confidence interval from the sample mean and sample variance of the replication outputs. The standard MRP procedure follows [Mak, Morton, and Wood (1999)] and [Bayraksan and Morton (2009)].

- **Approximate Control Variates Multiple Replications Procedure (ACV-MRP)**  
  For multifidelity workflows. Extends standard MRP by using a low-fidelity model as an approximate control variate. High- and low-fidelity replication outputs are evaluated on the same sampled scenario batches to induce correlation and reduce variance.

## Model interface

To make a new stochastic programming instance compatible with the confidence interval code, the problem module should expose the following factory functions:

- `get_sp_model_for_uq(...)` for **single-fidelity** workflows. Returns one model wrapper satisfying `StochasticProgramModelProtocol`, for use with StandardMRP and exact finite-population gap evaluation.

- `get_model_ensemble_for_uq(...)` for **multifidelity** workflows. Returns one ensemble satisfying `ModelEnsembleProtocol`, for use with ACVMRP and PyApprox integration.

These should return objects compatible with the internal protocols defined in `protocols.py`.

In practice:

- `StandardMRP` expects one `StochasticProgramModelProtocol` object.
- `ACVMRP` expects one `ModelEnsembleProtocol` object containing a high-fidelity and a low-fidelity model wrapper.

## Minimal usage: Standard MRP

```python
from sparow.conf_intervals.options import UQOptions
from sparow.conf_intervals.standard_mrp import StandardMRP

# This is the function you have to define for your problem instance
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

options = UQOptions(
    n=50,       # number of scenarios per replication batch
    m=10,       # number of replications
    alpha=0.05, # significance level for the confidence interval
    seed=12345, # random seed for reproducibility of algorithm run
    with_replacement=True, # when sampling scenario batches, whether to do it with or without replacement
    solver_name="gurobi_direct",
    verbose=True,
)

mrp = StandardMRP(model=model, options=options)
results = mrp.run(xhat=xhat)

print(results)
```

## Minimal usage: ACV-MRP

```python
from sparow.conf_intervals.options import UQOptions
from sparow.conf_intervals.acv_mrp import ACVMRP

# This is the function you have to define for your problem instance
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

options = UQOptions(
    n=100,            # number of scenarios per replication batch
    m=30,             # number of paired HF/LF replications
    M=10,             # number of additional LF-only replications
    alpha=0.05,       # significance level for the confidence interval
    seed=12345,       # random seed for reproducibility of algorithm run
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

If the full set of scenarios is known & finite, and your stochastic programming instance is small enough to solve over all scenarios within a reasonable amount of time, the package also provides a helper to compute the exact true optimal value, the value of xhat, and the resulting true optimality gap.

The true, underlying scenario distribution is referred to as the population distribution, and allows us to compute the true population quantities of interest for benchmarking:

```python
from sparow.conf_intervals.evaluate_true_optimality_gap import TrueOptimalityGapEvaluator

true_gap_evaluator = TrueOptimalityGapEvaluator(
    model=model,
    solver_name="gurobi_direct",
)

true_gap_results = true_gap_evaluator.compute_true_gap(xhat=xhat)
print(true_gap_results)
```

## Usage with `sparow/bin`

There are some shell scripts and plotting code located in sparow/bin that can be used for running numerical experiments for the UQ workflows. You can change the file paths to use any of the stochastic programming model instances from `sparow_examples` that have the necessary `get_sp_model_for_uq(...)` and  `get_model_ensemble_for_uq(...)` factory functions. You can also create and test your own custom stochastic programming models, if desired.

If you want to run the optional plotting scripts that generate figures from CSV
results, install the plotting dependencies as follows:

```bash
cd sparow
pip install -e ".[plot]"
cd ..
```

## Notes

- The current implementation assumes minimization. For a maximization problem, multiply the objective by −1 so it becomes an equivalent minimization problem. 
- The confidence intervals are built from the sample mean and sample standard error of the replication outputs, using a CLT-based approximation. Because the guarantee is asymptotic, the following quantities should be large enough that the replication-level gap estimates and variance estimates are reasonably stable:
    - the number of replications, m
    - the scenario batch size, n
    - for multifidelity approach, any additional low-fidelity evaluations, M

In the standard MRP setup, the Monte Carlo error in the sample mean shrinks at the usual rate $1/\sqrt{m}$. As $n \rightarrow \infty$, each replication output (i.e.- each optimality gap estimate) converges to its population counterpart. In the ACV-MRP setup, as $M \rightarrow \infty$, the error in estimating the low-fidelity mean used in the control variate converges to 0.

- The ACV-MRP interval uses a plug-in variance estimate for the control-variate-adjusted estimator. 
- The low-fidelity model should be chosen so that it is cheaper to evaluate and its replication-level gap estimates are sufficiently correlated with the high-fidelity replication-level gap estimates.

## PyApprox Integration

The codebase also includes a PyApprox [Jakeman (2023)] integration for using multifidelity allocation tools to suggest high-fidelity and low-fidelity sample counts, which can then be translated into ACV-MRP parameters. This helps allocate finite computational budget between high-fidelity model evaluations and low-fidelity model evaluations, such that we achieve maximum variance reduction for the optimality gap estimator within a pre-specified, finite computational budget. If you're interested in this, you will also need to install PyApprox, and note that, as of now, it requires Python >= 3.11: https://github.com/sandialabs/pyapprox/ 

## References

- Mak, Wai-Kei, David P. Morton, and R. Kevin Wood. 1999. “Monte Carlo Bounding Techniques for Determining Solution Quality in Stochastic Programs.” *Operations Research Letters* 24 (1–2): 47–56.
- Bayraksan, Güzin, and David P. Morton. 2009. “Assessing Solution Quality in Stochastic Programs via Sampling.” In *Decision Technologies and Applications*, 102–122. INFORMS.
- Jakeman, J. D. 2023. “PyApprox: A software package for sensitivity analysis, Bayesian inference, optimal experimental design, and multi-fidelity uncertainty quantification and surrogate modeling.” *Environmental Modelling & Software* 170: 105825.

