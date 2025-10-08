import random
import pyomo.environ as pyo
import pytest
from forestlib.ph import stochastic_program
from forestlib.ph import ProgressiveHedgingSolver
import numpy as np

random.seed(923874938740938740)


def first_stage_builder(app_data, app_args):
    model = pyo.ConcreteModel()

    model.crops_multiplier = int(app_data["crops_multiplier"])

    def crops_init(m):
        retval = []
        for i in range(model.crops_multiplier):
            retval.append("WHEAT" + str(i))
            retval.append("CORN" + str(i))
            retval.append("SUGAR_BEETS" + str(i))
        return retval

    model.CROPS = pyo.Set(initialize=crops_init)

    #
    # Parameters
    #

    model.TOTAL_ACREAGE = 500.0 * model.crops_multiplier

    def _scale_up_data(indict):
        outdict = {}
        for i in range(model.crops_multiplier):
            for crop in ["WHEAT", "CORN", "SUGAR_BEETS"]:
                outdict[crop + str(i)] = indict[crop]
        return outdict

    model.PlantingCostPerAcre = _scale_up_data(
        {"WHEAT": 150.0, "CORN": 230.0, "SUGAR_BEETS": 260.0}
    )

    #
    # Variables
    #

    if app_args.get("use_integer", True):
        model.DevotedAcreage = pyo.Var(
            model.CROPS,
            within=pyo.NonNegativeIntegers,
            bounds=(0.0, model.TOTAL_ACREAGE),
        )
    else:
        model.DevotedAcreage = pyo.Var(model.CROPS, bounds=(0.0, model.TOTAL_ACREAGE))

    #
    # Constraints
    #

    def ConstrainTotalAcreage_(model):
        return pyo.sum_product(model.DevotedAcreage) <= model.TOTAL_ACREAGE

    model.ConstrainTotalAcreage = pyo.Constraint(rule=ConstrainTotalAcreage_)

    def ComputeFirstStageCost_(model):
        return pyo.sum_product(model.PlantingCostPerAcre, model.DevotedAcreage)

    #
    # Objective
    #

    model.TotalPlantingCost = pyo.Objective(rule=ComputeFirstStageCost_)

    return model


def second_stage_builder(M, scen, scen_args):

    model = pyo.ConcreteModel()

    #
    # Parameters
    #

    def _scale_up_data(indict):
        outdict = {}
        for i in range(M.crops_multiplier):
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
        M.CROPS, within=pyo.NonNegativeReals, initialize=Yield_init, mutable=True
    )

    #
    # Variables
    #

    model.QuantitySubQuotaSold = pyo.Var(M.CROPS, bounds=(0.0, None))
    model.QuantitySuperQuotaSold = pyo.Var(M.CROPS, bounds=(0.0, None))
    model.QuantityPurchased = pyo.Var(M.CROPS, bounds=(0.0, None))

    #
    # Constraints
    #

    def EnforceCattleFeedRequirement_(model, i):
        return (
            model.CattleFeedRequirement[i]
            <= (model.Yield[i] * M.DevotedAcreage[i])
            + model.QuantityPurchased[i]
            - model.QuantitySubQuotaSold[i]
            - model.QuantitySuperQuotaSold[i]
        )

    model.EnforceCattleFeedRequirement = pyo.Constraint(
        M.CROPS, rule=EnforceCattleFeedRequirement_
    )

    def LimitAmountSold_(model, i):
        return (
            model.QuantitySubQuotaSold[i]
            + model.QuantitySuperQuotaSold[i]
            - (model.Yield[i] * M.DevotedAcreage[i])
            <= 0.0
        )

    model.LimitAmountSold = pyo.Constraint(M.CROPS, rule=LimitAmountSold_)

    def EnforceQuotas_(model, i):
        return (0.0, model.QuantitySubQuotaSold[i], model.PriceQuota[i])

    model.EnforceQuotas = pyo.Constraint(M.CROPS, rule=EnforceQuotas_)

    #
    # Objective
    #

    def TotalProfit_(model):
        return (
            pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
            - pyo.sum_product(model.SubQuotaSellingPrice, model.QuantitySubQuotaSold)
            - pyo.sum_product(
                model.SuperQuotaSellingPrice, model.QuantitySuperQuotaSold
            )
        )

    model.TotalProfit = pyo.Objective(rule=TotalProfit_)

    return model


sp = stochastic_program(model_builder_list=[first_stage_builder, second_stage_builder])

app_data = {"crops_multiplier": 1.0}
sp.initialize_application(app_data=app_data)

bundle_data = {
    "scenarios": [
        {
            "ID": "BelowAverageScenario",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "Probability": 0.3,
        },
        {
            "ID": "AverageScenario",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "Probability": 0.3,
        },
        {
            "ID": "AboveAverageScenario",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "Probability": 0.4,
        },
    ]
}
sp.initialize_bundles(bundle_data=bundle_data)

ph = ProgressiveHedgingSolver()
ph.solve(sp, max_iterations=10, solver="gurobi", loglevel="DEBUG")
