import pytest
import pyomo.environ as pyo

from sparow.sp.examples import (
    LF_newsvendor,
    HF_newsvendor,
    MFrandom_newsvendor,
    simple_newsvendor,
    simple_absolute_value,
    adjustable_absolute_value,
)

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


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestBendersNewsvendor:

    def test_abs(self, mip_solver):
        app = simple_absolute_value()
        solver = BendersSolver()
        solver.set_options(solver=mip_solver, subproblem_solver=mip_solver)

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve_in_dev(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_shifted_abs(self, mip_solver):
        solver = BendersSolver()
        solver.set_options(solver=mip_solver, subproblem_solver=mip_solver)
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
        results = solver.solve_in_dev(app.sp, eta_bounds_map)
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
            results = solver.solve_in_dev(app.sp, eta_bounds_map)
            results_dict = results.to_dict()
            soln = next(iter(results_dict["solutions"].values()))

            obj_val = soln["objectives"][0]["value"]
            assert obj_val == pytest.approx(app.objective_value)
            assert app.unique_solution
            x = soln["variables"][0]["value"]
            assert x == pytest.approx(a_val)

    def Xtest_simple(self, mip_solver):
        app = simple_newsvendor()
        solver = BendersSolver()
        solver.set_options(solver=mip_solver, subproblem_solver=mip_solver)

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve_in_dev(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def Xtest_LF(self, mip_solver):
        app = LF_newsvendor()
        solver = BendersSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve_in_dev(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def Xtest_HF(self, mip_solver):
        app = HF_newsvendor()
        solver = BendersSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve_in_dev(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def Xtest_MFrandom(self, mip_solver):
        app = MFrandom_newsvendor()
        solver = BendersSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve_in_dev(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert not app.unique_solution
        # The optimal x value is not unique, so we don't test its value
