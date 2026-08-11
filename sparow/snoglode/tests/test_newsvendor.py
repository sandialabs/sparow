import pytest
import pyomo.environ as pyo

from sparow.sp.examples import (
    LF_newsvendor,
    HF_newsvendor,
    MFrandom_newsvendor,
    simple_newsvendor,
)
from sparow.snoglode import SnoglodeSolver

import pyomo.opt
from pyomo.common import unittest

open_source_solver = set(pyomo.opt.check_available_solvers("highs"))
if len(open_source_solver) == 0:
    open_source_solver = set(pyomo.opt.check_available_solvers("glpk"))
solvers = set(pyomo.opt.check_available_solvers("gurobi")) | open_source_solver


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestEFNewsvendor:

    def test_simple(self, mip_solver):
        app = simple_newsvendor()
        solver = SnoglodeSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_simple_return_EF(self, mip_solver):
        app = simple_newsvendor()
        solver = SnoglodeSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve_and_return_EF(app.sp)
        results_dict = results.solutions.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

        assert obj_val == pytest.approx(pyo.value(results.model.obj))
        assert x == pytest.approx(pyo.value(results.model.first_stage_variables[0]))

    def test_LF(self, mip_solver):
        app = LF_newsvendor()
        solver = SnoglodeSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_HF(self, mip_solver):
        app = HF_newsvendor()
        solver = SnoglodeSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_MFrandom(self, mip_solver):
        app = MFrandom_newsvendor()
        solver = SnoglodeSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert not app.unique_solution
        # The optimal x value is not unique, so we don't test its value
