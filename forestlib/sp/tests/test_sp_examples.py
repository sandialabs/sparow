import pytest

from forestlib.sp.examples import (
    simple_newsvendor,
    LF_newsvendor,
    HF_newsvendor,
    MFrandom_newsvendor,
    MFpaired_newsvendor,
)

import pyomo.opt
import pyomo.environ as pyo
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("glpk", "gurobi"))
solvers = ["glpk"] if "glpk" in solvers else ["gurobi"]


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_sp_simple_newsvendor(mip_solver):
    app = simple_newsvendor()
    sp = app.sp

    assert sp.get_objective_coef(0) == 0

    assert set(sp.bundles.keys()) == {"1", "2", "3", "4", "5"}
    assert sp.bundles["1"].probability == 0.2

    #
    # Testing internal data structures
    #
    M1 = sp.create_subproblem("1")
    assert set(sp.int_to_FirstStageVar.keys()) == {"1"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    M2 = sp.create_subproblem("2")
    assert set(sp.int_to_FirstStageVar.keys()) == {"1", "2"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    #
    # Test subproblem solver logic
    #
    sp.solve(M1, solver=mip_solver)
    assert pyo.value(M1.s[None, 1].x) == pytest.approx(15.0)

    sp.solve(M2, solver=mip_solver)
    assert pyo.value(M2.s[None, 2].x) == pytest.approx(60.0)


class TestMFNewsVendor:
    """
    Test the multi-fidelity news vendor application

    See https://stoprog.org/sites/default/files/SPTutorial/TutorialSP.pdf
    """


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_sp_LF_newsvendor(mip_solver):
    app = LF_newsvendor()
    sp = app.sp

    assert set(sp.bundles.keys()) == {"LF_1", "LF_2", "LF_3", "LF_4", "LF_5"}
    assert sp.bundles["LF_1"].probability == 0.1

    #
    # Testing internal data structures
    #
    M1 = sp.create_subproblem("LF_1")
    assert set(sp.int_to_FirstStageVar.keys()) == {"LF_1"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    M2 = sp.create_subproblem("LF_2")
    assert set(sp.int_to_FirstStageVar.keys()) == {"LF_1", "LF_2"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    #
    # Test subproblem solver logic
    #
    sp.solve(M1, solver=mip_solver)
    assert pyo.value(M1.s["LF", "1"].x) == pytest.approx(15.0)

    sp.solve(M2, solver=mip_solver)
    assert pyo.value(M2.s["LF", "2"].x) == pytest.approx(60.0)


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_sp_HF_newsvendor(mip_solver):
    app = HF_newsvendor()
    sp = app.sp

    assert set(sp.bundles.keys()) == {"HF_1", "HF_2", "HF_3", "HF_4", "HF_5"}
    assert sp.bundles["HF_1"].probability == 0.05

    #
    # Testing internal data structures
    #
    M1 = sp.create_subproblem("HF_1")
    assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    M2 = sp.create_subproblem("HF_2")
    assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1", "HF_2"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    #
    # Test subproblem solver logic
    #
    sp.solve(M1, solver=mip_solver)
    assert pyo.value(M1.s["HF", "1"].x) == pytest.approx(9.0)

    sp.solve(M2, solver=mip_solver)
    assert pyo.value(M2.s["HF", "2"].x) == pytest.approx(40.0)


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_MFpaired(mip_solver):
    app = MFpaired_newsvendor()
    sp = app.sp

    assert set(sp.bundles.keys()) == {"1", "2", "3", "4", "5"}
    assert sp.bundles["1"].probability == 0.2

    #
    # Testing internal data structures
    #
    M1 = sp.create_subproblem("1")
    assert set(sp.int_to_FirstStageVar.keys()) == {"1"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    M2 = sp.create_subproblem("2")
    assert set(sp.int_to_FirstStageVar.keys()) == {"1", "2"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    #
    # Test subproblem solver logic
    #

    # Subproblem M1 has multiple solutions
    # sp.solve(M1, solver=mip_solver)
    # assert len(M1.s) == 2
    # print(f'{pyo.value(M1.s["HF", 1].x)=} {pyo.value(M1.s["LF", 1].x)=} {pyo.value(M1.s["HF", 1].y)=} {pyo.value(M1.s["LF", 1].y)=}')
    # assert pyo.value(M1.s["HF", 1].x) == pytest.approx(15.0)
    # assert pyo.value(M1.s["LF", 1].x) == pytest.approx(15.0)
    # assert pyo.value(M1.s["HF", 1].y) == pytest.approx(21.0)
    # assert pyo.value(M1.s["LF", 1].y) == pytest.approx(15.0)

    sp.solve(M2, solver=mip_solver)
    assert len(M2.s) == 2
    # print(f'{pyo.value(M2.s["HF", 2].x)=} {pyo.value(M2.s["LF", 2].x)=} {pyo.value(M2.s["HF", 2].y)=} {pyo.value(M2.s["LF", 2].y)=}')
    assert pyo.value(M2.s["HF", "2"].x) == pytest.approx(60.0)
    assert pyo.value(M2.s["LF", "2"].x) == pytest.approx(60.0)
    assert pyo.value(M2.s["HF", "2"].y) == pytest.approx(78.0)
    assert pyo.value(M2.s["LF", "2"].y) == pytest.approx(60.0)


@unittest.pytest.mark.parametrize("mip_solver", solvers)
def test_MFrandom(mip_solver):
    app = MFrandom_newsvendor()
    sp = app.sp

    assert sp.get_bundles() == {
        "HF_1": {
            "probability": 0.2,
            "scenarios": [("HF", "1"), ("LF", "1"), ("LF", "3")],
            "scenario_probability": {
                ("HF", "1"): 0.5,
                ("LF", "3"): 0.25,
                ("LF", "1"): 0.25,
            },
        },
        "HF_2": {
            "probability": 0.2,
            "scenarios": [("HF", "2"), ("LF", "1"), ("LF", "4")],
            "scenario_probability": {
                ("HF", "2"): 0.5,
                ("LF", "4"): 0.25,
                ("LF", "1"): 0.25,
            },
        },
        "HF_3": {
            "probability": 0.2,
            "scenarios": [("HF", "3"), ("LF", "1"), ("LF", "3")],
            "scenario_probability": {
                ("HF", "3"): 0.5,
                ("LF", "3"): 0.25,
                ("LF", "1"): 0.25,
            },
        },
        "HF_4": {
            "probability": 0.2,
            "scenarios": [("HF", "4"), ("LF", "4"), ("LF", "5")],
            "scenario_probability": {
                ("HF", "4"): 0.5,
                ("LF", "5"): 0.25,
                ("LF", "4"): 0.25,
            },
        },
        "HF_5": {
            "probability": 0.2,
            "scenarios": [("HF", "5"), ("LF", "4"), ("LF", "5")],
            "scenario_probability": {
                ("HF", "5"): 0.5,
                ("LF", "5"): 0.25,
                ("LF", "4"): 0.25,
            },
        },
    }

    assert set(sp.bundles.keys()) == {"HF_1", "HF_2", "HF_3", "HF_4", "HF_5"}
    assert sp.bundles["HF_1"].probability == 0.2

    #
    # Testing internal data structures
    #
    M1 = sp.create_subproblem("HF_1")
    assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    M2 = sp.create_subproblem("HF_2")
    assert set(sp.int_to_FirstStageVar.keys()) == {"HF_2", "HF_1"}
    assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

    #
    # Test subproblem solver logic
    #

    # Subproblem M1 has multiple solutions
    # sp.solve(M1, solver=mip_solver)
    # assert len(M1.s) == 3
    # assert set(M1.s.keys()) == {("HF", 1), ("LF", 3), ("LF", 4)}
    # assert pyo.value(M1.s["HF", 1].x) == pytest.approx(25.0)
    # assert pyo.value(M1.s["LF", 3].x) == pytest.approx(25.0)
    # assert pyo.value(M1.s["LF", 4].x) == pytest.approx(25.0)
    # assert pyo.value(M1.s["HF", 1].y) == pytest.approx(26.0)
    # assert pyo.value(M1.s["LF", 3].y) == pytest.approx(95.5)
    # assert pyo.value(M1.s["LF", 4].y) == pytest.approx(104.5)

    sp.solve(M2, solver=mip_solver)
    assert len(M2.s) == 3
    assert set(M2.s.keys()) == {("HF", "2"), ("LF", "1"), ("LF", "4")}
    assert pyo.value(M2.s["HF", "2"].x) == pytest.approx(40.0)
    assert pyo.value(M2.s["LF", "1"].x) == pytest.approx(40.0)
    assert pyo.value(M2.s["LF", "4"].x) == pytest.approx(40.0)
    assert pyo.value(M2.s["HF", "2"].y) == pytest.approx(70.0)
    assert pyo.value(M2.s["LF", "1"].y) == pytest.approx(42.5)
    assert pyo.value(M2.s["LF", "4"].y) == pytest.approx(97.0)
