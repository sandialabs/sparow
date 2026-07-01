import pytest
import pyomo.environ as pyo

from sparow.sp.examples import (
    LF_newsvendor,
    HF_newsvendor,
    MFrandom_newsvendor,
    simple_newsvendor,
    single_scenario_newsvendor,
    simple_absolute_value,
    adjustable_absolute_value,
    AMPL_facilityloc,
    AMPL_facilityloc_Benders_Test,
)
from sparow.ef import ExtensiveFormSolver

from pyomo.common.dependencies import attempt_import

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

from sparow.benders import BendersSolver

import pyomo.opt
from pyomo.common import unittest

open_source_solver = set(pyomo.opt.check_available_solvers("highs"))
if len(open_source_solver) == 0:
    open_source_solver = set(pyomo.opt.check_available_solvers("glpk"))
# solvers = set(pyomo.opt.check_available_solvers("gurobi")) | open_source_solver
solvers = set(pyomo.opt.check_available_solvers("gurobi"))

persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers(
        # "appsi_highs",
        # "appsi_gurobi",
        "gurobi_persistent",
    )
)


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestBenders_NonPersistent:

    def test_abs(self, mip_solver):
        app = simple_absolute_value()
        solver = BendersSolver()
        solver.set_options(solver=mip_solver, subproblem_solver=mip_solver)

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_shifted_abs(self, mip_solver):
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            #    loglevel="DEBUG",
        )
        a_val = 1
        model_data = {
            "scenarios": [
                {"ID": 1, "LB": None, "UB": None},
            ],
        }
        app_data = dict(a=a_val, c=0, L=1, R=1)
        app = adjustable_absolute_value(
            local_app_data=app_data, local_model_data=model_data
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(a_val)

    def test_shifted_abs_2(self, mip_solver):
        solver = BendersSolver()
        solver.set_options(solver=mip_solver, subproblem_solver=mip_solver)
        a_vals = [1, -1, 3, 4]
        for a_val in a_vals:
            model_data = {
                "scenarios": [
                    {"ID": 1, "LB": None, "UB": None},
                ],
            }
            app_data = dict(a=a_val, c=0, L=1, R=1)
            app = adjustable_absolute_value(
                local_app_data=app_data, local_model_data=model_data
            )

            default_lower_eta = -1_000
            eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
            results = solver.solve(app.sp, eta_bounds_map)
            results_dict = results.to_dict()
            soln = next(iter(results_dict["solutions"].values()))

            obj_val = soln["objectives"][0]["value"]
            assert obj_val == pytest.approx(app.objective_value)
            assert app.unique_solution
            x = soln["variables"][0]["value"]
            assert x == pytest.approx(a_val)

    def test_single_scenario_newsvendor(self, mip_solver):
        app = single_scenario_newsvendor()
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            #    loglevel="DEBUG",
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        assert app.unique_solution
        obj_val = soln["objectives"][0]["value"]
        x = soln["variables"][0]["value"]
        print(f"{x=}, {obj_val=}")
        assert obj_val == pytest.approx(app.objective_value)
        assert x == pytest.approx(app.solution_values["x"])

    def test_simple_newsvendor(self, mip_solver):
        app = simple_newsvendor()
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            #    loglevel= "DEBUG",
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        print(eta_bounds_map.keys())
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])
        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)

    def test_facilityloc(self, mip_solver):
        app = AMPL_facilityloc()
        # solver = BendersSolver()
        # solver.set_options(solver=mip_solver,
        #                    subproblem_solver=mip_solver,
        #                    loglevel= "DEBUG",
        #                    )
        # default_lower_eta = -1_000
        # #not sure the s values here map to what the bundles actually expect
        # eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        # #app.sp.bundles are {'HF_High', 'HF_Low', 'HF_Medium'}
        # #so the eta's are getting those names

        # #the scenario keys for s are of the style ('HF', 'Low') as a tuple
        # #for each of the scenario models, we get a block definition like:
        # #s : Size=1, Index={('HF', 'High')}, Active=True
        # #so there appears to be a mismatch between bundles and scenario keys

        # #first iteration of subproblems is giving infeasible/unbounded error code
        # #need to print out first master solve results, master model, and subproblem model
        # results = solver.solve(app.sp, eta_bounds_map)
        # results_dict = results.to_dict()
        # obj_val = results_dict["solutions"][0]["objectives"][0]["value"]

        # assert obj_val == pytest.approx(app.objective_value)

        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            custom_b_upper="High",
            #    loglevel= "INFO",
            # loglevel="DEBUG",
        )
        default_lower_eta = -100_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        obj_val = results_dict["solutions"][0]["objectives"][0]["value"]

        assert obj_val == pytest.approx(app.objective_value)

    def test_facilitylo_benders_test(self, mip_solver):
        app = AMPL_facilityloc_Benders_Test()
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            custom_b_upper="High",
            #    loglevel= "INFO",
            # loglevel="DEBUG",
        )
        default_lower_eta = -100_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        obj_val = results_dict["solutions"][0]["objectives"][0]["value"]

        assert obj_val == pytest.approx(app.objective_value)


class TestBenders_Errors(unittest.TestCase):
    def test_allow_infeasible_subproblems(self):
        app = simple_absolute_value()
        solver = BendersSolver()
        solver.set_options(
            solver="glpk",
            subproblem_solver="glpk",
            is_persistent_solver=False,
            allow_infeasible_subproblems=True,
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        assert_text = "Must use a persistent solver to support feasibility cuts"
        with self.assertRaisesRegex(AssertionError, assert_text):
            results = solver.solve(app.sp, eta_bounds_map)


@pytest.mark.skipif(
    len(persistent_mip_solvers) == 0, reason="No persistent solvers available"
)
@unittest.pytest.mark.parametrize("mip_solver", persistent_mip_solvers)
class TestBenders_Persistent:

    def test_abs(self, mip_solver):
        app = simple_absolute_value()
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            is_persistent_solver=True,
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_shifted_abs(self, mip_solver):
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            is_persistent_solver=True,
        )
        a_val = 1
        model_data = {
            "scenarios": [
                {"ID": 1, "LB": None, "UB": None},
            ],
        }
        app_data = dict(a=a_val, c=0, L=1, R=1)
        app = adjustable_absolute_value(
            local_app_data=app_data, local_model_data=model_data
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(a_val)

    def test_shifted_abs_2(self, mip_solver):
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            is_persistent_solver=True,
        )
        a_vals = [1, -1, 3, 4]
        for a_val in a_vals:
            model_data = {
                "scenarios": [
                    {"ID": 1, "LB": None, "UB": None},
                ],
            }
            app_data = dict(a=a_val, c=0, L=1, R=1)
            app = adjustable_absolute_value(
                local_app_data=app_data, local_model_data=model_data
            )

            default_lower_eta = -1_000
            eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
            results = solver.solve(app.sp, eta_bounds_map)
            results_dict = results.to_dict()
            soln = next(iter(results_dict["solutions"].values()))

            obj_val = soln["objectives"][0]["value"]
            assert obj_val == pytest.approx(app.objective_value)
            assert app.unique_solution
            x = soln["variables"][0]["value"]
            assert x == pytest.approx(a_val)

    def test_single_scenario_newsvendor(self, mip_solver):
        app = single_scenario_newsvendor()
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            is_persistent_solver=True,
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        assert app.unique_solution
        obj_val = soln["objectives"][0]["value"]
        x = soln["variables"][0]["value"]
        print(f"{x=}, {obj_val=}")
        assert obj_val == pytest.approx(app.objective_value)
        assert x == pytest.approx(app.solution_values["x"])

    def test_simple_newsvendor(self, mip_solver):
        app = simple_newsvendor()
        solver = BendersSolver()
        solver.set_options(
            solver=mip_solver,
            subproblem_solver=mip_solver,
            is_persistent_solver=True,
        )

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])
        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
