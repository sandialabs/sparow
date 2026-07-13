import pytest
import pyomo.environ as pyo

from sparow.sp.examples import (
    simple_absolute_value,
    feasibility_included_absolute_value,
)
from sparow.ef import ExtensiveFormSolver

import pyomo.opt
from pyomo.common import unittest

open_source_solver = set(pyomo.opt.check_available_solvers("highs"))
if len(open_source_solver) == 0:
    open_source_solver = set(pyomo.opt.check_available_solvers("glpk"))
solvers = set(pyomo.opt.check_available_solvers("gurobi")) | open_source_solver


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestEFAbsoluteValue:

    def test_simple(self, mip_solver):
        app = simple_absolute_value()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_feasibility_variant(self, mip_solver):
        app = feasibility_included_absolute_value()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])
