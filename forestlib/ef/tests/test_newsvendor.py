import pytest
import pyomo.environ as pyo

from forestlib.sp.examples import (
    LF_newsvendor,
    HF_newsvendor,
    MFrandom_newsvendor,
    simple_newsvendor,
)
from forestlib.ef import ExtensiveFormSolver

import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("glpk", "gurobi"))
# solvers = ["glpk"] if "glpk" in solvers else ["gurobi"]


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestEFNewsvendor:

    def test_simple(self, mip_solver):
        sp = simple_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict[None]["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(76.5)
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(60.0)

    def test_simple_return_EF(self, mip_solver):
        sp = simple_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve_and_return_EF(sp)
        results_dict = results.solutions.to_dict()
        soln = next(iter(results_dict[None]["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(76.5)
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(60.0)

        assert obj_val == pytest.approx(pyo.value(results.model.obj))
        assert x == pytest.approx(pyo.value(results.model.rootx[0]))

    def test_LF(self, mip_solver):
        sp = LF_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict[None]["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(80.01)
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(72.0)

    def test_HF(self, mip_solver):
        sp = HF_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict[None]["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(82.335)
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(54.0)

    def test_MFrandom(self, mip_solver):
        sp = MFrandom_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict[None]["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(81.3525)
        # WEH - The optimal x value is not unique, so we don't test its value
        # x = soln["variables"][0]["value"]
        # assert x == pytest.approx(60.0)
