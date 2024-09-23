import pyomo.environ as pyo
import pytest
from forestlib.ph.sp import StochasticProgram_Pyomo
import numpy as np

class TestFarmersEF(object):

    def test_root_model_solve(self):
        model = pyo.ConcreteModel()
        yields=[3, 3.6, 24]
        # Variables
        model.X = pyo.Var(["WHEAT", "CORN", "BEETS"], within=pyo.NonNegativeReals)
        model.Y = pyo.Var(["WHEAT", "CORN"], within=pyo.NonNegativeReals)
        model.W = pyo.Var(["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"],
                            within=pyo.NonNegativeReals,)

        # Objective function
        model.PLANTING_COST = 150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
        model.PURCHASE_COST = 238 * model.Y["WHEAT"] + 210 * model.Y["CORN"]
        model.SALES_REVENUE = (
        170 * model.W["WHEAT"] + 150 * model.W["CORN"]
        + 36 * model.W["BEETS_FAVORABLE"] + 10 * model.W["BEETS_UNFAVORABLE"]
        )
        model.OBJ = pyo.Objective(
        expr=model.PLANTING_COST + model.PURCHASE_COST - model.SALES_REVENUE,
        sense=pyo.minimize
        )

        # Constraints
        model.CONSTR= pyo.ConstraintList()

        model.CONSTR.add(pyo.summation(model.X) <= 500)
        model.CONSTR.add(
        yields[0] * model.X["WHEAT"] + model.Y["WHEAT"] - model.W["WHEAT"] >= 200
        )
        model.CONSTR.add(
        yields[1] * model.X["CORN"] + model.Y["CORN"] - model.W["CORN"] >= 240
        )
        model.CONSTR.add(
        yields[2] * model.X["BEETS"] - model.W["BEETS_FAVORABLE"] - model.W["BEETS_UNFAVORABLE"] >= 0
        )
        model.W["BEETS_FAVORABLE"].setub(6000)
        sp = StochasticProgram_Pyomo(first_stage_variables=['X','Y','W'])
        sp.solver='ipopt'
        sp.initialize_varmap(b=0, M=model)
        res=sp.solve(model,tee=True)
        assert np.isclose(pyo.value(model.OBJ), -167666.66764875906, rtol=1e-05, atol=1e-08)

    def test_EF_model_solve(self):
        pass
