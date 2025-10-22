import pytest
import pyomo.environ as pyo

from forestlib.solnpool import PoolManager
from forestlib.sp import stochastic_program
from forestlib.sp.examples import LF_newsvendor, HF_newsvendor, MFrandom_newsvendor
from forestlib.ef import ExtensiveFormSolver

import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("glpk", "gurobi"))
solvers = ["glpk"] if "glpk" in solvers else ["gurobi"]


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestEFNewsvendor:

    def test_LF(self, mip_solver):
        sp = LF_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        pool_manager = PoolManager()
        pool_manager.reset_solution_counter()
        results = solver.solve(sp)
        results_dict = results.to_dict()
        obj_val = results_dict[None]["solutions"][0]["objectives"][0]["value"]

        assert obj_val == pytest.approx(76.5)

    def test_HF(self, mip_solver):
        sp = HF_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        pool_manager = PoolManager()
        pool_manager.reset_solution_counter()
        results = solver.solve(sp)
        results_dict = results.to_dict()
        obj_val = results_dict[None]["solutions"][0]["objectives"][0]["value"]

        assert obj_val == pytest.approx(82.335)

    def test_MFrandom(self, mip_solver):
        sp = MFrandom_newsvendor()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        pool_manager = PoolManager()
        pool_manager.reset_solution_counter()
        results = solver.solve(sp)
        results_dict = results.to_dict()
        obj_val = results_dict[None]["solutions"][0]["objectives"][0]["value"]

        assert obj_val == pytest.approx(79.4775)

