# Test variable initialization logic for Pyomo models

import pyomo.environ as pyo
import pytest
from forestlib.ph.sp import stochastic_program


class TestPHPyomo(object):

    def test_simple1(self):
        def builder(**kwargs):
            M = pyo.ConcreteModel()
            M.x = pyo.Var()
            M.y = pyo.Var()
            M.o = pyo.Objective(expr=M.x)

        sp = stochastic_program(
            objective="o", first_stage_variables=["x"], model_builder=builder
        )
        sp.initialize_varmap(b=0, M=M)
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}
        assert list(sp.int_to_var[0].keys()) == [0]
        assert sp.shared_variables() == [0]

    def test_simple2(self):
        def builder(**kwargs):
            M = pyo.ConcreteModel()
            M.x = pyo.Var()
            M.y = pyo.Var()

        sp = stochastic_program(first_stage_variables=["y", "x"], model_builder=builder)
        sp.initialize_varmap(b=0, M=M)
        assert sp.varcuid_to_int == {pyo.ComponentUID("y"): 0, pyo.ComponentUID("x"): 1}
        assert list(sorted(sp.int_to_var[0].keys())) == [0, 1]
        assert sp.shared_variables() == [0, 1]
