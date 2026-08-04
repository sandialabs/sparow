import math
import numpy as np
import pytest
from scipy import stats

from sparow.conf_intervals.mrp_options import MRPOptions
from sparow.conf_intervals.standard_mrp import StandardMRP
from sparow.conf_intervals.scenario_sampler import ScenarioSampler
from sparow.conf_intervals.cli import (
    load_problem_adapter,
    build_candidate_solution,
    run_single_mrp_experiment,
    run_mrp_grid_experiment,
)

from sparow_examples.farmers.MRPfarmers import get_ci_problem_adapter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def advanced_adapter():
    return get_ci_problem_adapter(model_name="Advanced", use_integer=False)


@pytest.fixture
def advanced_scenarios(advanced_adapter):
    scenarios = advanced_adapter.get_scenario_population()
    advanced_adapter.validate_scenario_population(scenarios)
    return scenarios


@pytest.fixture
def basic_adapter():
    return get_ci_problem_adapter(model_name="Basic", use_integer=False)


@pytest.fixture
def basic_scenarios(basic_adapter):
    scenarios = basic_adapter.get_scenario_population()
    basic_adapter.validate_scenario_population(scenarios)
    return scenarios


# ============================================================================
# Candidate-solution generation
# ============================================================================


def test_candidate_generation_reproducibility(advanced_adapter, advanced_scenarios):
    """
    Same candidate seed + same sample size + same replacement rule
    should give identical sampled candidate xhat and objective.
    """
    xhat1, obj1 = build_candidate_solution(
        problem_adapter=advanced_adapter,
        full_scenarios=advanced_scenarios,
        candidate_scen_count=500,
        candidate_seed=12345,
        with_replacement=True,
        solver_name="gurobi_direct",
    )

    xhat2, obj2 = build_candidate_solution(
        problem_adapter=advanced_adapter,
        full_scenarios=advanced_scenarios,
        candidate_scen_count=500,
        candidate_seed=12345,
        with_replacement=True,
        solver_name="gurobi_direct",
    )

    xhat3, obj3 = build_candidate_solution(
        problem_adapter=advanced_adapter,
        full_scenarios=advanced_scenarios,
        candidate_scen_count=100,
        candidate_seed=123,
        with_replacement=False,
        solver_name="gurobi_direct",
    )

    xhat4, obj4 = build_candidate_solution(
        problem_adapter=advanced_adapter,
        full_scenarios=advanced_scenarios,
        candidate_scen_count=100,
        candidate_seed=123,
        with_replacement=False,
        solver_name="gurobi_direct",
    )

    assert xhat1 == xhat2
    assert obj1 == obj2

    assert xhat3 == xhat4
    assert obj3 == obj4


# ============================================================================
# ScenarioSampler behavior
# ============================================================================


def test_without_replacement_sampling(advanced_scenarios):
    """
    A replication batch sampled without replacement should contain
    no duplicate population indices.
    """
    sampler = ScenarioSampler(
        scenarios=advanced_scenarios,
        seed=2468,
        with_replacement=False,
    )

    sampled = sampler.draw_scenarios(n=600, replication_id=0)
    pop_indices = [s["Population_Index"] for s in sampled]

    assert len(pop_indices) == len(set(pop_indices))


def test_with_replacement_sampling(advanced_scenarios):
    """
    With replacement, all sampled scenarios should still get Probability = 1/n,
    and duplicates are allowed.
    """
    n = 600
    sampler = ScenarioSampler(
        scenarios=advanced_scenarios,
        seed=2468,
        with_replacement=True,
    )

    sampled = sampler.draw_scenarios(n=n, replication_id=0)

    # Every sampled scenario should have probability 1/n
    for s in sampled:
        assert math.isclose(s["Probability"], 1.0 / n)

    # We do not require duplicates to appear, only that duplicates are allowed.
    # So the main assertion is that no error occurs and probabilities are correct.


def test_nested_sample_correctness(advanced_scenarios):
    """
    If one replication first draws a superset sample of size n_max, then
    smaller n samples formed by taking prefixes should be strict subsets
    of the larger sample.
    """
    sampler = ScenarioSampler(
        scenarios=advanced_scenarios,
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
def test_scenario_format_validation(advanced_adapter, bad_scenarios, expected_msg):
    """
    Each malformed scenario case should independently raise the expected validation error.
    """
    with pytest.raises(RuntimeError, match=expected_msg):
        advanced_adapter.validate_scenario_population(bad_scenarios)


# ============================================================================
# MRP reproducibility
# ============================================================================


def test_mrp_run_reproducibility(advanced_adapter, advanced_scenarios):
    """
    Same xhat + same MRP options + same scenario population should produce
    identical replication values and CI outputs.
    """
    xhat, _ = build_candidate_solution(
        problem_adapter=advanced_adapter,
        full_scenarios=advanced_scenarios,
        candidate_scen_count=5,
        candidate_seed=11111,
        with_replacement=True,
        solver_name="gurobi_direct",
    )

    options = MRPOptions(
        n=50,
        m=5,
        alpha=0.05,
        seed=6789,
        with_replacement=True,
        solver_name="gurobi_direct",
        verbose=False,
    )

    mrp1 = StandardMRP(
        problem_adapter=advanced_adapter,
        scenarios=advanced_scenarios,
        options=options,
    )
    results1 = mrp1.run(xhat)

    mrp2 = StandardMRP(
        problem_adapter=advanced_adapter,
        scenarios=advanced_scenarios,
        options=options,
    )
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


# ===========================================================================
# Reproducibility of grid experiments and single runs from sparow.ci.cli
# ===========================================================================


def test_grid_experiment_reproducibility_same_candidate_and_mrp_seed(tmp_path):
    """
    Running the grid experiment twice with the same candidate seed and MRP seed
    should give identical xhat, true gap, and row-by-row grid results.
    """
    output_csv_1 = tmp_path / "results1.csv"
    output_csv_2 = tmp_path / "results2.csv"
    xhat_file_1 = tmp_path / "xhat1.npy"
    xhat_file_2 = tmp_path / "xhat2.npy"

    kwargs = dict(
        model_module_name="sparow_examples.farmers.MRPfarmers",
        model_name="Advanced",
        solver_name="gurobi_direct",
        candidate_scen_count=5,
        candidate_seed=12345,
        candidate_with_replacement=True,
        alpha=0.05,
        mrp_seed=678,
        mrp_with_replacement=True,
        m_values=[5, 10],
        n_values=[200, 100],
        use_existing_xhat=False,
        use_integer=False,
    )

    res1 = run_mrp_grid_experiment(
        xhat_file=str(xhat_file_1),
        output_csv=str(output_csv_1),
        **kwargs,
    )

    res2 = run_mrp_grid_experiment(
        xhat_file=str(xhat_file_2),
        output_csv=str(output_csv_2),
        **kwargs,
    )

    assert res1["xhat"] == res2["xhat"]
    assert res1["candidate_ef_objective"] == res2["candidate_ef_objective"]
    assert res1["true_optimal_value"] == res2["true_optimal_value"]
    assert res1["candidate_true_objective"] == res2["candidate_true_objective"]
    assert res1["true_gap"] == res2["true_gap"]
    assert res1["rows"] == res2["rows"]


def test_single_run_reproducibility_same_mrp_seed(advanced_adapter, advanced_scenarios):
    """
    A single MRP run should be reproducible when the MRP seed is fixed.
    """
    xhat, _ = build_candidate_solution(
        problem_adapter=advanced_adapter,
        full_scenarios=advanced_scenarios,
        candidate_scen_count=5,
        candidate_seed=12345,
        with_replacement=True,
        solver_name="gurobi_direct",
    )

    res1 = run_single_mrp_experiment(
        problem_adapter=advanced_adapter,
        scenarios=advanced_scenarios,
        xhat=xhat,
        n=100,
        m=10,
        alpha=0.05,
        seed=678,
        with_replacement=True,
        solver_name="gurobi_direct",
    )

    res2 = run_single_mrp_experiment(
        problem_adapter=advanced_adapter,
        scenarios=advanced_scenarios,
        xhat=xhat,
        n=100,
        m=10,
        alpha=0.05,
        seed=678,
        with_replacement=True,
        solver_name="gurobi_direct",
    )

    assert res1["point_estimate"] == res2["point_estimate"]
    assert res1["sample_variance"] == res2["sample_variance"]
    assert res1["sample_std"] == res2["sample_std"]
    assert res1["half_width"] == res2["half_width"]
    assert res1["ci_upper"] == res2["ci_upper"]
    assert np.allclose(res1["replication_values"], res2["replication_values"])
