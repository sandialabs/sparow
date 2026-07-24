# Sparow Confidence Interval (CI) Module

This module implements confidence interval estimation procedures for stochastic programs, specifically the Multiple Replications Procedure (MRP) for the signle-fidelity case and Approximate Control Variate MRP (ACV-MRP) for the multifidelity case.

## Overview

The quantity of interest is an upper bound on the optimality gap of a candidate first-stage solution. The core algorithms are:

1. **Standard MRP** - Multiple Replications Procedure for single-fidelity models
2. **ACV-MRP** - Approximate Control Variate MRP for multifidelity models (extends Standard MRP)

## Core Components

### Algorithm Implementations

#### `standard_mrp.py`
Implements the standard Multiple Replications Procedure algorithm:
- Estimates the optimality gap upper bound for a candidate solution `xhat`
- Uses `m` independent replications, each with `n` sampled scenarios
- Core logic: sample scenarios → solve SAA → evaluate candidate → compute gap

#### `acv_mrp.py`
Implements the Approximate Control Variate MRP algorithm:
- **Extends** Standard MRP with control variates for variance reduction
- Uses both high-fidelity (HF) and low-fidelity (LF) models
- Runs `m` paired replications (HF+LF) + `M` additional LF-only replications
- Requires problem adapter to support multifidelity via `supports_acv()`

### Options and Configuration

#### `mrp_options.py`
Defines `MRPOptions` dataclass for standard MRP configuration.

#### `acv_mrp_options.py`
Defines `ACVMRPOptions` dataclass that extends `MRPOptions`.
### Problem Interface

#### `ci_problem_adapter.py`
Abstract base class `CIProblemAdapter` defines the interface between algorithms and problem-specific implementations:

**Required methods (must be implemented by subclasses):**
- `get_scenario_population()`: Return full scenario population
- `build_model_data(scenarios)`: Convert scenarios to model data dict
- `build_stochastic_program(model_data)`: Build Sparow stochastic program
- `first_stage_variable_order()`: Return ordered list of first-stage variables

**Optional methods (for multifidelity support):**
- `supports_acv()`: Return True if adapter supports ACV-MRP
- `set_active_fidelity(fidelity)`: Switch between HF/LF models
- `get_fidelity_levels()`: Return available fidelity levels (will be used in future for when you have multiple models of different fidelities)

**Provided utility methods:**
- `solve_extensive_form()`: Solve EF using Sparow's ExtensiveFormSolver
- `evaluate_first_stage_solution()`: Evaluate fixed candidate first stage solution
- `validate_scenario_population()`: Validate scenario format to ensure downstream code can use it
- Conversion utilities between dict/vector representations

### Supporting Utilities

#### `scenario_sampler.py`
`ScenarioSampler` class handles all of the scenario sampling logic:
- Draws iid Monte Carlo samples from finite scenario population
- Supports both with-replacement and without-replacement sampling
- Uses `SeedSequence` for reproducible random streams across replications
- We keep track of the scenarios we've drawn by their population ID/ index.

#### `evaluate_true_optimality_gap.py`
`TrueOptimalityGapEvaluator` computes exact optimality gap when full scenario population is available:
- Solves EF over entire population to get true optimal value
- Evaluates candidate solution on full population
- Computes exact gap for validation/benchmarking

### Command-Line Interface

#### `cli.py`
Main entry point with two operating modes:

**Single-run mode**
This is for a fixed set of experiment parameters: n, m, M.

**Grid-experiment mode:**
This is for trying different combinations in a parameter sweep: n, m, M.

Handles:
- Dynamic loading of problem-specific adapters via `get_ci_problem_adapter()`
- Candidate solution generation from scratch, or read-in solution from .npy file
- Result visualization via plot scripts

## Shell Scripts

### `run_single_mrp.sh`
Example script for running standard MRP on the farmers problem:
- Runs MRP with specified `n` and `m`
- Computes true optimality gap for validation

### `run_single_acvmrp.sh`
Example script for running ACV-MRP on a toy discrete facility location problem:
- WILL INCLUDE DISCRETE FACILITY LOCATION EXEMPLAR LATER
- Uses `--acv-mrp` flag to enable ACV-MRP mode

### `run_experiments.sh`
Grid experiment script for parameter sweep:
- Uses nested scenario sampling for fair comparison across parameters
- Generates CSV results and plots using the plotting scripts: plot_mrp_results.py or plot_acvmrp_results.py

## Example: Farmers Problem

The `sparow_examples.farmers.MRPfarmers` module provides a concrete implementation:

```python
# Basic 3-scenario problem from Birge & Louveaux
BasicAdapter = CIProblemAdapter(...)

# Advanced problem with interpolated yields (finite population)
AdvancedAdapter = CIProblemAdapter(...)
```

The advanced farmers problem:
- Creates Cartesian product of interpolated crop yields
- Has finite scenario population for exact gap computation
- Used as benchmark for MRP algorithm validation

## Extending for New Problems

To use MRP with a new stochastic programming problem:

1. **Create a problem adapter:**
   ```python
   from sparow.ci import CIProblemAdapter
   
   class MyProblemAdapter(CIProblemAdapter):
       def get_scenario_population(self):
           # Return list of scenario dicts
           pass
       
       def build_model_data(self, scenarios):
           # Convert to Sparow model_data format
           pass
           
       def build_stochastic_program(self, model_data):
           # Return Sparow stochastic_program object
           pass
           
       def first_stage_variable_order(self):
           # Return list of first-stage variable names
           pass
   ```

2. **Add factory function:**
   ```python
   def get_ci_problem_adapter(model_name=None, **kwargs):
       if model_name == "variant1":
           return MyProblemAdapter(model_name="variant1", ...)
       return MyProblemAdapter(...)  # default
   ```

## Future Extensions

The architecture supports upcoming features:

1. **PyApprox Integration:**
   - ACVMRPOptions includes `use_pyapprox` and `pyapprox_config` fields
   - Future PR will add PyApprox group ACV option

2. **Multifidelity Examples:**
   - ACV-MRP currently works with discrete facility location example
   - Future PR will add more ACV-MRP examples
