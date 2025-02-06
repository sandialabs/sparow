# Test variable initialization logic for Pyomo models

import pytest
import pyomo.environ as pyo
from forestlib.sp import stochastic_program


@pytest.fixture
def sp1():
    def builder(*args, **kwargs):
        M = pyo.ConcreteModel()
        M.x = pyo.Var()
        M.y = pyo.Var()
        M.o = pyo.Objective(expr=M.x)
        return M

    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_model(model_builder=builder)
    sp.initialize_model(model_data=dict(scenarios=[dict(ID=1), dict(ID=2)]))
    return sp


@pytest.fixture
def sp2():
    def builder(*args, **kwargs):
        M = pyo.ConcreteModel()
        M.x = pyo.Var()
        M.y = pyo.Var(within=pyo.Binary)
        M.o = pyo.Objective(expr=M.x)
        return M

    sp = stochastic_program(first_stage_variables=["y", "x"])
    sp.initialize_model(model_builder=builder)
    sp.initialize_model(model_data=dict(scenarios=[dict(ID=1), dict(ID=2)]))
    return sp


class TestPHPyomo(object):

    def test_simple1(self, sp1):
        M = sp1.create_subproblem("1")

        assert sp1.varcuid_to_int == {pyo.ComponentUID("x"): 0}
        assert list(sp1.int_to_FirstStageVar["1"].keys()) == [0]
        assert sp1.shared_variables() == [0]
        assert sp1.get_objective_coef(0) == 0.5  # x

    def test_simple2(self, sp2):
        M = sp2.create_subproblem("1")

        assert sp2.varcuid_to_int == {
            pyo.ComponentUID("y"): 0,
            pyo.ComponentUID("x"): 1,
        }
        assert list(sorted(sp2.int_to_FirstStageVar["1"].keys())) == [0, 1]
        assert sp2.shared_variables() == [0, 1]
        assert sp2.get_objective_coef(0) == 0  # y
        assert sp2.get_objective_coef(1) == 0.5  # x

    def test_continuous1(self, sp1):
        M = sp1.create_subproblem("1")
        assert sp1.continuous_fsv()
        assert sp1._binary_or_integer_fsv == set()

    def test_continuous2(self, sp2):
        M = sp2.create_subproblem("1")
        assert not sp2.continuous_fsv()
        assert sp2._binary_or_integer_fsv == {0}
