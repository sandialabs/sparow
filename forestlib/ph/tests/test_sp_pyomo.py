# Test variable initiallization logic for Pyomo models

import pyomo.environ as pyo
import pytest
from forestlib.ph.sp import StochasticProgram_Pyomo

class TestPHPyomo(object):

    def test_simple1(self):
        M = pyo.ConcreteModel()
        M.x = pyo.Var()
        M.y = pyo.Var()

        sp = StochasticProgram_Pyomo(first_stage_variables=['x'])
        sp.initialize_varmap(b=0, M=M)
        assert sp.varcuid_to_int == {pyo.ComponentUID('x'):0}
        assert list(sp.int_to_var[0].keys()) == [0]
        assert sp.shared_variables() == [0]

    def test_simple2(self):
        M = pyo.ConcreteModel()
        M.x = pyo.Var()
        M.y = pyo.Var()

        sp = StochasticProgram_Pyomo(first_stage_variables=['y','x'])
        sp.initialize_varmap(b=0, M=M)
        assert sp.varcuid_to_int == {pyo.ComponentUID('y'):0, pyo.ComponentUID('x'):1}
        assert list(sorted(sp.int_to_var[0].keys())) == [0,1]
        assert sp.shared_variables() == [0,1]

        
