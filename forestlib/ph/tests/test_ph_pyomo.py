# Test variable initiallization logic for Pyomo models

import pyomo.environ as pyo
import pytest
from forestlib.ph.ph import ProgressiveHedgingSolver_Pyomo

class TestPHPyomo(object):

    def test_simple(self):
        M = pyo.ConcreteModel()
        M.x = pyo.Var()
        M.y = pyo.Var()

        solver = ProgressiveHedgingSolver_Pyomo(first_stage_variables=['x'])

        solver.initialize_varmap(b=0, M=M)
        print(solver.varcuid_to_int)
        print(solver.int_to_var)

        assert False

        
