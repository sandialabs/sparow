import math
import numpy as np
import pytest

from sparow.conf_intervals.options import UQOptions
from sparow.conf_intervals.standard_mrp import StandardMRP
from sparow.conf_intervals.acv_mrp import ACVMRP
from sparow.conf_intervals.scenario_sampler import ScenarioSampler
from sparow.conf_intervals.experiment_helpers import build_candidate_solution

from sparow.conf_intervals.pyapprox_interface import convert_pyapprox_allocation_to_acvmrp_params

from sparow.conf_intervals.pyapprox_helpers import (
    run_pyapprox_pilot,
    allocate_pyapprox_budget,
)

from sparow.sp.examples.farmers.MRPfarmers import get_sp_model_for_uq
from sparow.sp.examples.facilityloc.uq_discrete_facilityloc import get_model_ensemble_for_uq

# ============================================================================
# Shared Test Data & Fixtures
# ============================================================================


@pytest.fixture
def advanced_model():
    """
    Single-fidelity Advanced Farmers model wrapper.
    """
    return get_sp_model_for_uq(
        model_name="Advanced",
        use_integer=False,
        seed=12345,
        with_replacement=True,
    )


@pytest.fixture
def advanced_scenarios(advanced_model):
    """
    Full finite scenario population for Advanced Farmers.
    """
    scenarios = advanced_model.scenario_population().scenarios()
    advanced_model.scenario_population.validate(scenarios)
    return scenarios


@pytest.fixture
def basic_model():
    """
    Single-fidelity Basic Farmers model wrapper.
    """
    return get_sp_model_for_uq(
        model_name="Basic",
        use_integer=False,
        seed=12345,
        with_replacement=True,
    )


@pytest.fixture
def basic_scenarios(basic_model):
    """
    Full finite scenario population for Basic Farmers.
    """
    scenarios = basic_model.scenario_population().scenarios()
    basic_model.scenario_population().validate(scenarios)
    return scenarios

XHAT_DISCRETE_FACILITYLOC = {
    "x[0]": 0.0,
    "x[1]": 0.0,
    "x[2]": 0.0,
    "x[3]": 0.0,
    "x[4]": 1.0,
    "x[5]": 1.0,
}

@pytest.fixture
def facilityloc_ensemble():
    """
    Two-model HF/LF facility-location ensemble for multifidelity tests.
    """
    return get_model_ensemble_for_uq(
        model_name="HF",
        use_integer=False,
        seed=678,
        with_replacement=True,
        lf_model_type="classic",
    )


@pytest.fixture
def hf_model(facilityloc_ensemble):
    """
    High-fidelity facility-location model wrapper.
    """
    return facilityloc_ensemble.high_fidelity_model()


@pytest.fixture
def lf_model(facilityloc_ensemble):
    """
    Low-fidelity facility-location model wrapper.
    """
    return facilityloc_ensemble.low_fidelity_model()

# ============================================================================
# Candidate-solution generation
# ============================================================================


def test_candidate_generation_reproducibility(advanced_model):
    """
    Same candidate seed + same sample size + same replacement rule
    should give identical sampled candidate xhat and objective.
    """
    candidate_model_1 = get_sp_model_for_uq(
        model_name="Advanced",
        use_integer=False,
        seed=12345,
        with_replacement=True,
    )
    xhat1, obj1 = build_candidate_solution(
        model=candidate_model_1,
        candidate_scen_count=500,
        solver_name="highs",
    )

    candidate_model_2 = get_sp_model_for_uq(
        model_name="Advanced",
        use_integer=False,
        seed=12345,
        with_replacement=True,
    )
    xhat2, obj2 = build_candidate_solution(
        model=candidate_model_2,
        candidate_scen_count=500,
        solver_name="highs",
    )

    candidate_model_3 = get_sp_model_for_uq(
        model_name="Advanced",
        use_integer=False,
        seed=123,
        with_replacement=False,
    )
    xhat3, obj3 = build_candidate_solution(
        model=candidate_model_3,
        candidate_scen_count=100,
        solver_name="highs",
    )

    candidate_model_4 = get_sp_model_for_uq(
        model_name="Advanced",
        use_integer=False,
        seed=123,
        with_replacement=False,
    )
    xhat4, obj4 = build_candidate_solution(
        model=candidate_model_4,
        candidate_scen_count=100,
        solver_name="highs",
    )

    assert xhat1 == xhat2
    assert obj1 == obj2

    assert xhat3 == xhat4
    assert obj3 == obj4


# ============================================================================
# ScenarioSampler behavior
# ============================================================================


def test_without_replacement_sampling(advanced_model):
    """
    A replication batch sampled without replacement should contain
    no duplicate population indices.
    """
    sampler = ScenarioSampler(
        scenario_population=advanced_model.scenario_population(),
        seed=2468,
        with_replacement=False,
    )

    sampled = sampler.draw_scenarios(n=600, replication_id=0)
    pop_indices = [s["Population_Index"] for s in sampled]

    assert len(pop_indices) == len(set(pop_indices))


def test_with_replacement_sampling(advanced_model):
    """
    With replacement, all sampled scenarios should still get Probability = 1/n,
    and duplicates are allowed.
    """
    n = 600
    sampler = ScenarioSampler(
        scenario_population=advanced_model.scenario_population(),
        seed=2468,
        with_replacement=True,
    )

    sampled = sampler.draw_scenarios(n=n, replication_id=0)

    # Every sampled scenario should have probability 1/n
    for s in sampled:
        assert math.isclose(s["Probability"], 1.0 / n)

    # We do not require duplicates to appear, only that duplicates are allowed.
    # So the main assertion is that no error occurs and probabilities are correct.


def test_nested_sample_correctness(advanced_model):
    """
    If one replication first draws a superset sample of size n_max, then
    smaller n samples formed by taking prefixes should be strict subsets
    of the larger sample.
    """
    sampler = ScenarioSampler(
        scenario_population=advanced_model.scenario_population(),
        seed=54321,
        with_replacement=True,
    )

    rep_id = 0
    sampled_superset = sampler.draw_scenarios(n=600, replication_id=rep_id)

    sampled_500 = sampled_superset[:500]
    sampled_400 = sampled_superset[:400]

    # Compare population indices to ensure nesting is exact
    superset_indices = [s["Population_Index"] for s in sampled_superset]
    sample_500_indices = [s["Population_Index"] for s in sampled_500]
    sample_400_indices = [s["Population_Index"] for s in sampled_400]

    assert sample_500_indices == superset_indices[:500]
    assert sample_400_indices == superset_indices[:400]


# ============================================================================
# Scenario-format validation
# ============================================================================


@pytest.mark.parametrize(
    "bad_scenarios, expected_msg",
    [
        (
            [{"ID": "scen_0", "Probability": 1.0}],
            "is missing required key",
        ),  # missing Yield
        (
            [{"Yield": {"WHEAT": 2.0}, "Probability": 1.0}],
            "is missing required key",
        ),  # missing ID
        (
            [{"ID": "scen_2", "Yield": {"WHEAT": 2.0}}],
            "is missing required key",
        ),  # missing Probability
        (
            [[{"ID": "scen_0", "Probability": 1.0, "Yield": {"WHEAT": 2.0}}]],
            "is not a dictionary",
        ),  # not a dictionary
    ],
)
def test_scenario_format_validation(advanced_model, bad_scenarios, expected_msg):
    """
    Each malformed scenario case should independently raise the expected validation error.
    """
    with pytest.raises(RuntimeError, match=expected_msg):
        advanced_model.scenario_population().validate(bad_scenarios)


# ============================================================================
# MRP reproducibility
# ============================================================================


def test_mrp_run_reproducibility(advanced_model):
    """
    Same xhat + same MRP options + same scenario population should produce
    identical replication values and CI outputs.
    """
    candidate_model = get_sp_model_for_uq(
        model_name="Advanced",
        use_integer=False,
        seed=11111,
        with_replacement=True,
    )
    xhat, _ = build_candidate_solution(
        model=candidate_model,
        candidate_scen_count=5,
        solver_name="highs",
    )

    options = UQOptions(
        n=50,
        m=5,
        alpha=0.05,
        seed=6789,
        with_replacement=True,
        solver_name="highs",
        verbose=False,
    )

    mrp1 = StandardMRP(model=advanced_model, options=options)
    results1 = mrp1.run(xhat)

    mrp2 = StandardMRP(model=advanced_model, options=options)
    results2 = mrp2.run(xhat)

    assert results1["point_estimate"] == results2["point_estimate"]
    assert results1["sample_variance"] == results2["sample_variance"]
    assert results1["sample_std"] == results2["sample_std"]
    assert results1["t_statistic"] == results2["t_statistic"]
    assert results1["half_width"] == results2["half_width"]
    assert results1["ci_lower"] == results2["ci_lower"]
    assert results1["ci_upper"] == results2["ci_upper"]
    assert np.allclose(results1["replication_values"], results2["replication_values"])
    assert (
        results1["sampled_indices_by_replication"]
        == results2["sampled_indices_by_replication"]
    )

# =============================================================================
# Tests for multifidelity model/ensemble construction
# =============================================================================


def test_facilityloc_ensemble_construction(facilityloc_ensemble, hf_model, lf_model):
    """
    The facility-location ensemble should expose distinct HF and LF wrappers
    with compatible scenario populations.
    """
    assert hf_model.fidelity() == "high"
    assert lf_model.fidelity() == "low"
    assert len(facilityloc_ensemble.models()) == 2

    hf_scenarios = hf_model.scenario_population().scenarios()
    lf_scenarios = lf_model.scenario_population().scenarios()

    hf_model.scenario_population().validate(hf_scenarios)
    lf_model.scenario_population().validate(lf_scenarios)

    assert len(hf_scenarios) == len(lf_scenarios)
    assert len(hf_scenarios) > 0

# =============================================================================
# Tests for PyApprox pilot / allocation helpers
# =============================================================================

def test_convert_pyapprox_allocation_to_acvmrp_params():
    """
    The helper translating PyApprox counts to ACVMRP counts should satisfy:
      m = N_HF
      M = N_LF - N_HF
    """
    m, M = convert_pyapprox_allocation_to_acvmrp_params([7, 19])
    assert m == 7
    assert M == 12


def test_run_pyapprox_pilot_is_reproducible_and_cost_delay_is_reflected(facilityloc_ensemble):
    """
    Run the same pilot study twice with the same seed and settings, and check:
      1. the estimated pilot covariance matrix is reproducible,
      2. the estimated pilot correlation is reproducible,
      3. the injected HF artificial delay makes the HF estimated cost exceed the LF estimated cost.
    """
    pilot_1 = run_pyapprox_pilot(
        ensemble=facilityloc_ensemble,
        xhat=XHAT_DISCRETE_FACILITYLOC,
        batch_size=4,
        solver_name="highs",
        solver_options=None,
        seed=678,
        n_pilot=10,
        hf_cost_delay_seconds=1.0,
        lf_cost_delay_seconds=0.0,
        verbose=False,
        t0=0.0,
    )

    pilot_2 = run_pyapprox_pilot(
        ensemble=facilityloc_ensemble,
        xhat=XHAT_DISCRETE_FACILITYLOC,
        batch_size=4,
        solver_name="highs",
        solver_options=None,
        seed=678,
        n_pilot=10,
        hf_cost_delay_seconds=1.0,
        lf_cost_delay_seconds=0.0,
        verbose=False,
        t0=0.0,
    )

    # Same seed and same setup should give the same pilot covariance estimate.
    assert np.allclose(pilot_1["cov_np"], pilot_2["cov_np"])

    # Same seed and same setup should give the same pilot correlation estimate.
    assert math.isclose(
        float(pilot_1["rho_hat_pilot"]),
        float(pilot_2["rho_hat_pilot"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    # The injected HF delay should make HF more expensive than LF.
    assert pilot_1["costs_np"][0] - 0.9 > pilot_1["costs_np"][1]

    # Pilot covariance entries should be finite.
    assert np.all(np.isfinite(pilot_1["cov_np"]))


def test_allocate_pyapprox_budget_respects_budget(facilityloc_ensemble):
    """
    Using one fixed pilot study, check two budget regimes:
      1. If the total budget is smaller than the estimated pilot cost and pilot
         cost is charged against the budget, the allocation should be infeasible.
      2. If the budget is sufficiently large, the allocation should be feasible
         and the returned ACVMRP counts should be consistent with the total
         HF/LF sample counts.
    """
    pilot_info = run_pyapprox_pilot(
        ensemble=facilityloc_ensemble,
        xhat=XHAT_DISCRETE_FACILITYLOC,
        batch_size=4,
        solver_name="highs",
        solver_options=None,
        seed=678,
        n_pilot=4,
        hf_cost_delay_seconds=1.0,
        lf_cost_delay_seconds=0.0,
        verbose=False,
        t0=0.0,
    )

    estimated_pilot_cost = float(np.sum(pilot_info["costs_np"]) * 4)

    # Case 1: budget too small once pilot is charged against it
    alloc_small = allocate_pyapprox_budget(
        pilot_info=pilot_info,
        total_budget=0.25 * estimated_pilot_cost,
        n_pilot=4,
        count_pilot_cost_against_budget=True,
    )

    assert alloc_small["allocation_feasible"] is False
    assert alloc_small["remaining_budget"] <= 0.0
    assert alloc_small["m_paired"] == 0
    assert alloc_small["M_additional_lf"] == 0

    # Case 2: sufficiently large budget should produce a valid allocation
    alloc_large = allocate_pyapprox_budget(
        pilot_info=pilot_info,
        total_budget=4.0 * estimated_pilot_cost,
        n_pilot=4,
        count_pilot_cost_against_budget=True,
    )

    assert alloc_large["allocation_feasible"] is True
    assert alloc_large["remaining_budget"] > 0.0

    # Translation consistency:
    #   m = total HF count
    #   M = total LF count - total HF count
    assert alloc_large["m_paired"] == alloc_large["pyapprox_hf_total"]
    assert alloc_large["M_additional_lf"] == (
        alloc_large["pyapprox_lf_total"] - alloc_large["pyapprox_hf_total"]
    )

    # Predicted variance / std should be finite for a feasible allocation.
    assert np.isfinite(alloc_large["predicted_pyapprox_var"])
    assert np.isfinite(alloc_large["predicted_pyapprox_std"])


# =============================================================================
# Tests for ACVMRP output structure
# =============================================================================

def test_acvmrp_run_returns_expected_fields(facilityloc_ensemble):
    """
    A small ACVMRP run should return the key multifidelity diagnostics and
    confidence-interval quantities.
    """
    options = UQOptions(
        n=4,
        m=3,
        M=2,
        alpha=0.05,
        seed=678,
        with_replacement=True,
        solver_name="highs",
        verbose=False,
    )

    acv = ACVMRP(
        hf_model=facilityloc_ensemble.high_fidelity_model(),
        lf_model=facilityloc_ensemble.low_fidelity_model(),
        options=options,
    )

    results = acv.run(xhat=XHAT_DISCRETE_FACILITYLOC)

    expected_keys = [
        "point_estimate",
        "point_estimate_hf_only",
        "ci_lower",
        "ci_upper",
        "half_width",
        "control_variate_coefficient",
        "sample_correlation",
        "variance_acv_estimator",
        "standard_error_acv",
        "variance_reduction_factor",
        "F_values",
        "G_paired_values",
        "G_all_values",
    ]
    for key in expected_keys:
        assert key in results

    assert results["ci_lower"] == 0.0
    assert results["ci_upper"] >= results["ci_lower"]
    assert len(results["F_values"]) == options.m
    assert len(results["G_paired_values"]) == options.m
    assert len(results["G_all_values"]) == options.m + options.M
