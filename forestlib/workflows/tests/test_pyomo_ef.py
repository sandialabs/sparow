import pytest
import pyomo.environ as pyo

from forestlib.sp.examples import simple_newsvendor, MFpaired_newsvendor
from forestlib.ph import ProgressiveHedgingSolver
from forestlib.ef import ExtensiveFormSolver
from forestlib.workflows import initialize_EF, create_and_initialize_EF

import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("gurobi"))


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_initialize_EF(mip_solver):
    sp = simple_newsvendor()
    solver = ProgressiveHedgingSolver()
    solver.set_options(solver=mip_solver)
    solver.set_options(max_iterations=5)
    results = solver.solve(sp)

    # Double check that the solution value looks good
    results_dict = results.to_dict()
    soln = next(iter(results_dict["Finalized Last PH Solution"]["solutions"].values()))
    obj_val = soln["objectives"][0]["value"]
    assert obj_val == pytest.approx(76.5, 0.01)

    results.set_pool("Finalized Last PH Solution")
    soln = next(iter(results.solutions))

    # Check that 'resolve=False' does not compute the objective
    M = create_and_initialize_EF(sp, soln, resolve=False)
    assert pyo.value(M.obj, exception=False) is None

    # Check that 'resolve=True' computes the objective
    M = create_and_initialize_EF(sp, soln, resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(76.5, 0.01)


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_initialize_MF_EF(mip_solver):
    sp = MFpaired_newsvendor()
    solver = ExtensiveFormSolver()
    solver.set_options(solver=mip_solver)
    results = solver.solve(sp)

    # Double check that the solution value looks good
    results_dict = results.to_dict()
    soln = next(iter(results_dict[None]["solutions"].values()))
    obj_val = soln["objectives"][0]["value"]
    assert obj_val == pytest.approx(81.3525, 0.01)

    # results.set_pool(Finalized Last PH Solution")
    soln = next(iter(results.solutions))

    # Create EF with LF scenarios
    M = create_and_initialize_EF(sp, soln, model_fidelities=["LF"], resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(80.25, 0.01)

    # Create EF with LF scenarios
    M = create_and_initialize_EF(sp, soln, model_fidelities=["HF"], resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(82.455, 0.01)

    # Create EF with LF and HF scenarios
    M = create_and_initialize_EF(sp, soln, model_fidelities=["LF", "HF"], resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(81.3525, 0.01)

    # Check EF with the unspecified scenarios, which pulls-in all model fidelities
    M = create_and_initialize_EF(sp, soln, resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(81.3525, 0.01)
