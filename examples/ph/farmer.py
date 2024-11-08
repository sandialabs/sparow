import random
import pyomo.environ as pyo
import pytest
from forestlib.ph import stochastic_program
from forestlib.ph import ProgressiveHedgingSolver
import numpy as np
from IPython import embed

random.seed(923874938740938740)


def model_builder(app_dta,scen, scen_args):
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

    if scen_args.get("use_integer", True):
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
        return model.FirstStageCost + model.SecondStageCost

    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule)

    return model


sp = stochastic_program(
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

testEF = True
testPH = True

if testPH:
    sp.initialize_bundles(
        bundle_data=bundle_data, bundle_scheme="single_scenario"
    )
    ph = ProgressiveHedgingSolver()
    ph.solve(sp, max_iterations=2, solver="gurobi", loglevel="DEBUG")

if testEF:
    sp.initialize_bundles(bundle_data=bundle_data, bundle_scheme="single_bundle")
    for b in list(sp.scenarios_in_bundle.keys()):
        print(b)
        EF_model = sp.create_EF(b=b)
        res = sp.solve(EF_model, solver="gurobi", solver_options={"tee": True})

        for s in sp.scenarios_in_bundle[b]:
            # print(pyo.value(EF_model.s[s].DevotedAcreage))
            EF_model.s[s].DevotedAcreage.pprint()
