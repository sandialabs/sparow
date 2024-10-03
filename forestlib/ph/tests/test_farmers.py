import random
import pyomo.environ as pyo
import pytest
from forestlib.ph import StochasticProgram_Pyomo
from forestlib.ph import ProgressiveHedgingSolver
import numpy as np

random.seed(923874938740938740)


def model_builder(scen, scen_args):
    model = pyo.ConcreteModel(scen["ID"])

    crops_multiplier = int(scen["crops_multiplier"])

    def crops_init(m):
        retval = []
        for i in range(crops_multiplier):
            retval.append("WHEAT" + str(i))
            retval.append("CORN" + str(i))
            retval.append("SUGAR_BEETS" + str(i))
        return retval

    model.CROPS = pyo.Set(initialize=crops_init)

    #
    # Parameters
    #

    model.TOTAL_ACREAGE = 500.0 * crops_multiplier

    def _scale_up_data(indict):
        outdict = {}
        for i in range(crops_multiplier):
            for crop in ["WHEAT", "CORN", "SUGAR_BEETS"]:
                outdict[crop + str(i)] = indict[crop]
        return outdict

    model.PriceQuota = _scale_up_data(
        {"WHEAT": 100000.0, "CORN": 100000.0, "SUGAR_BEETS": 6000.0}
    )

    model.SubQuotaSellingPrice = _scale_up_data(
        {"WHEAT": 170.0, "CORN": 150.0, "SUGAR_BEETS": 36.0}
    )

    model.SuperQuotaSellingPrice = _scale_up_data(
        {"WHEAT": 0.0, "CORN": 0.0, "SUGAR_BEETS": 10.0}
    )

    model.CattleFeedRequirement = _scale_up_data(
        {"WHEAT": 200.0, "CORN": 240.0, "SUGAR_BEETS": 0.0}
    )

    model.PurchasePrice = _scale_up_data(
        {"WHEAT": 238.0, "CORN": 210.0, "SUGAR_BEETS": 100000.0}
    )

    model.PlantingCostPerAcre = _scale_up_data(
        {"WHEAT": 150.0, "CORN": 230.0, "SUGAR_BEETS": 260.0}
    )

    #
    # Stochastic Data
    #
    def Yield_init(m, cropname):
        # yield as in "crop yield"
        crop_base_name = cropname.rstrip("0123456789")
        return scen["Yield"][crop_base_name] + random.uniform(
            0, 1
        )  # farmerstream.rand()

    model.Yield = pyo.Param(
        model.CROPS, within=pyo.NonNegativeReals, initialize=Yield_init, mutable=True
    )

    #
    # Variables
    #

    if scen_args.get('use_integer',True):
        model.DevotedAcreage = pyo.Var(
            model.CROPS,
            within=pyo.NonNegativeIntegers,
            bounds=(0.0, model.TOTAL_ACREAGE),
        )
    else:
        model.DevotedAcreage = pyo.Var(model.CROPS, bounds=(0.0, model.TOTAL_ACREAGE))

    model.QuantitySubQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantitySuperQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))

    #
    # Constraints
    #

    def ConstrainTotalAcreage_rule(model):
        return pyo.sum_product(model.DevotedAcreage) <= model.TOTAL_ACREAGE

    model.ConstrainTotalAcreage = pyo.Constraint(rule=ConstrainTotalAcreage_rule)

    def EnforceCattleFeedRequirement_rule(model, i):
        return (
            model.CattleFeedRequirement[i]
            <= (model.Yield[i] * model.DevotedAcreage[i])
            + model.QuantityPurchased[i]
            - model.QuantitySubQuotaSold[i]
            - model.QuantitySuperQuotaSold[i]
        )

    model.EnforceCattleFeedRequirement = pyo.Constraint(
        model.CROPS, rule=EnforceCattleFeedRequirement_rule
    )

    def LimitAmountSold_rule(model, i):
        return (
            model.QuantitySubQuotaSold[i]
            + model.QuantitySuperQuotaSold[i]
            - (model.Yield[i] * model.DevotedAcreage[i])
            <= 0.0
        )

    model.LimitAmountSold = pyo.Constraint(model.CROPS, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(model, i):
        return (0.0, model.QuantitySubQuotaSold[i], model.PriceQuota[i])

    model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

    # Stage-specific cost computations;

    def ComputeFirstStageCost_rule(model):
        return pyo.sum_product(model.PlantingCostPerAcre, model.DevotedAcreage)

    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model):
        expr = pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
        expr -= pyo.sum_product(model.SubQuotaSellingPrice, model.QuantitySubQuotaSold)
        expr -= pyo.sum_product(
            model.SuperQuotaSellingPrice, model.QuantitySuperQuotaSold
        )
        return expr

    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model):
        return -model.FirstStageCost - model.SecondStageCost

    #model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule)
    model.obj = pyo.Objective(rule=total_cost_rule)

    return model


FarmerSP = StochasticProgram_Pyomo(
    first_stage_variables=["DevotedAcreage[*]"], model_builder=model_builder
)

bundle_data = {
    "scenarios": [
        {
            "ID": "BelowAverageScenario",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "crops_multiplier": 1.0,
            "Probability": 0.3,
        },
        {
            "ID": "AverageScenario",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "crops_multiplier": 1.0,
            "Probability": 0.3,
        },
        {
            "ID": "AboveAverageScenario",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "crops_multiplier": 1.0,
            "Probability": 0.4,
        },
    ]
}
FarmerSP.initialize_bundles(bundle_data=bundle_data)

ph = ProgressiveHedgingSolver()
ph.solve(FarmerSP, max_iterations=10, solver='gurobi')


class TestFarmersEF(object):

    def test_root_model_solve(self):
        model = pyo.ConcreteModel()
        yields = [3, 3.6, 24]
        # Variables
        model.X = pyo.Var(["WHEAT", "CORN", "BEETS"], within=pyo.NonNegativeReals)
        model.Y = pyo.Var(["WHEAT", "CORN"], within=pyo.NonNegativeReals)
        model.W = pyo.Var(
            ["WHEAT", "CORN", "BEETS_FAVORABLE", "BEETS_UNFAVORABLE"],
            within=pyo.NonNegativeReals,
        )

        # Objective function
        model.PLANTING_COST = (
            150 * model.X["WHEAT"] + 230 * model.X["CORN"] + 260 * model.X["BEETS"]
        )
        model.PURCHASE_COST = 238 * model.Y["WHEAT"] + 210 * model.Y["CORN"]
        model.SALES_REVENUE = (
            170 * model.W["WHEAT"]
            + 150 * model.W["CORN"]
            + 36 * model.W["BEETS_FAVORABLE"]
            + 10 * model.W["BEETS_UNFAVORABLE"]
        )
        model.OBJ = pyo.Objective(
            expr=model.PLANTING_COST + model.PURCHASE_COST - model.SALES_REVENUE,
            sense=pyo.minimize,
        )

        # Constraints
        model.CONSTR = pyo.ConstraintList()

        model.CONSTR.add(pyo.summation(model.X) <= 500)
        model.CONSTR.add(
            yields[0] * model.X["WHEAT"] + model.Y["WHEAT"] - model.W["WHEAT"] >= 200
        )
        model.CONSTR.add(
            yields[1] * model.X["CORN"] + model.Y["CORN"] - model.W["CORN"] >= 240
        )
        model.CONSTR.add(
            yields[2] * model.X["BEETS"]
            - model.W["BEETS_FAVORABLE"]
            - model.W["BEETS_UNFAVORABLE"]
            >= 0
        )
        model.W["BEETS_FAVORABLE"].setub(6000)
        sp = StochasticProgram_Pyomo(first_stage_variables=["X", "Y", "W"])
        sp.solver = "ipopt"
        sp.initialize_varmap(b=0, M=model)
        res = sp.solve(model, tee=True)
        assert np.isclose(
            pyo.value(model.OBJ), -167666.66764875906, rtol=1e-05, atol=1e-08
        )

    def test_EF_model_solve(self):
        pass
