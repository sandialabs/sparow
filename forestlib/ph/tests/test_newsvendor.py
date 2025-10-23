import pytest

from forestlib.sp.examples import LF_newsvendor, HF_newsvendor, MFrandom_newsvendor, simple_newsvendor
from forestlib.ph import ProgressiveHedgingSolver

import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("glpk", "gurobi"))
solvers = ["glpk"] if "glpk" in solvers else ["gurobi"]


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestEFNewsvendor:

    def test_simple(self, mip_solver):
        sp = simple_newsvendor()
        solver = ProgressiveHedgingSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict['Finalized Last PH Solution']["solutions"].values()))

        x = soln["variables"][0]["value"]
        assert x == pytest.approx(60.0, 0.1)
        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(76.5, 0.1)

    def test_LF(self, mip_solver):
        sp = LF_newsvendor()
        solver = ProgressiveHedgingSolver()
        solver.set_options(solver=mip_solver)
        solver.set_options(max_iterations=1000)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict['Finalized Last PH Solution']["solutions"].values()))

        x = soln["variables"][0]["value"]
        assert x == pytest.approx(72.0, 0.1)
        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(80.01, 0.1)

    def test_HF(self, mip_solver):
        sp = HF_newsvendor()
        solver = ProgressiveHedgingSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict['Finalized Last PH Solution']["solutions"].values()))

        x = soln["variables"][0]["value"]
        assert x == pytest.approx(54.0, 1.0)
        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(82.335, 0.1)

    def test_MFrandom(self, mip_solver):
        sp = MFrandom_newsvendor()
        solver = ProgressiveHedgingSolver()
        solver.set_options(solver=mip_solver)
        solver.set_options(max_iterations=1000)
        results = solver.solve(sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict['Finalized Last PH Solution']["solutions"].values()))

        x = soln["variables"][0]["value"]
        assert x == pytest.approx(60.0,0.1)
        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(81.3525,0.1)

