import pytest
import pyomo.environ as pyo

from sparow.sp.examples import simple_newsvendor, MFpaired_newsvendor
from sparow.ph import ProgressiveHedgingSolver
from sparow.ef import ExtensiveFormSolver
from sparow.workflows import initialize_EF, create_and_initialize_EF

import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("gurobi", "highs"))


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_initialize_EF(mip_solver):
    app = simple_newsvendor()

    solver = ProgressiveHedgingSolver()
    solver.set_options(solver=mip_solver)
    solver.set_options(max_iterations=5)
    results = solver.solve(app.sp)

    # Double check that the solution value looks good
    results_dict = results.to_dict()
    import pprint

    pprint.pprint(results_dict)
    soln = next(iter(results_dict["solutions"].values()))
    obj_val = soln["objectives"][0]["value"]
    assert obj_val == pytest.approx(76.5, 0.01)

    results.activate("Finalized Last PH Solution")
    soln = next(iter(results.solutions))

    # Check that 'resolve=False' does not compute the objective
    M = create_and_initialize_EF(app.sp, soln, resolve=False)
    assert pyo.value(M.obj, exception=False) is None

    # Check that 'resolve=True' computes the objective
    M = create_and_initialize_EF(app.sp, soln, resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(76.5, 0.01)

    # Check that 'resolve=True' computes the correct first- and second-stage variables
    assert pyo.value(M.s[None, 1].x) == pytest.approx(60.93, 0.01)
    assert pyo.value(M.s[None, 2].x) == pytest.approx(60.93, 0.01)
    assert pyo.value(M.s[None, 3].x) == pytest.approx(60.93, 0.01)
    assert pyo.value(M.s[None, 4].x) == pytest.approx(60.93, 0.01)
    assert pyo.value(M.s[None, 5].x) == pytest.approx(60.93, 0.01)

    assert pyo.value(M.s[None, 1].y) == pytest.approx(65.53, 0.01)
    assert pyo.value(M.s[None, 2].y) == pytest.approx(61.02, 0.01)
    assert pyo.value(M.s[None, 3].y) == pytest.approx(77.53, 0.01)
    assert pyo.value(M.s[None, 4].y) == pytest.approx(86.53, 0.01)
    assert pyo.value(M.s[None, 5].y) == pytest.approx(92.53, 0.01)


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_initialize_MF_EF(mip_solver):
    app = MFpaired_newsvendor()
    solver = ExtensiveFormSolver()
    solver.set_options(solver=mip_solver)
    results = solver.solve(app.sp)

    # Double check that the solution value looks good
    results_dict = results.to_dict()
    soln = next(iter(results_dict["solutions"].values()))
    obj_val = soln["objectives"][0]["value"]
    assert obj_val == pytest.approx(81.3525, 0.01)

    # results.activate(Finalized Last PH Solution")
    soln = next(iter(results.solutions))

    # Create EF with LF scenarios
    M = create_and_initialize_EF(app.sp, soln, model_fidelities=["LF"], resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(80.25, 0.01)

    # Create EF with LF scenarios
    M = create_and_initialize_EF(app.sp, soln, model_fidelities=["HF"], resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(82.455, 0.01)

    # Create EF with LF and HF scenarios
    M = create_and_initialize_EF(
        app.sp, soln, model_fidelities=["LF", "HF"], resolve=True
    )
    assert pyo.value(M.obj, exception=False) == pytest.approx(81.3525, 0.01)

    # Check EF with the unspecified scenarios, which pulls-in all model fidelities
    M = create_and_initialize_EF(app.sp, soln, resolve=True)
    assert pyo.value(M.obj, exception=False) == pytest.approx(81.3525, 0.01)
