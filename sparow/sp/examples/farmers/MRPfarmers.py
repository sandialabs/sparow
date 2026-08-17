# This file containts two versions of the farmer problem, used to benchmark the MRP
# algorithm's point estimates and confidence intervals:
#
# Basic farmers = Classic 3-scenario problem from Birge & Louveaux
# Advanced farmers = For each crop, do a linear interpolation between the BelowAverage
#                    scenario yield and the AboveAverage scenario yield from the Basic
#                    farmers problem. For each crop, select any one of these yields
#                    with equal probability. The full scenario distribution is a
#                    discrete uniform over all resulting scenario vectors.
#
# You can run this file as a script to write the full Advanced Farmers scenario population to file.
#
# NOTE: Advanced farmers intentionally has a finite set of possible scenarios,
# so that the Extensive Form can be solved over all possible scenarios to get the
# exact optimal value for computing the true optimality gap of any given
# candidate solution.

import pyomo.environ as pyo
import numpy as np
from sparow.sp import stochastic_program

from sparow.conf_intervals.scenario_population import FiniteScenarioPopulation
from sparow.conf_intervals.scenario_sampler import ScenarioSampler
from sparow.conf_intervals.sp_model_wrapper_for_uq import SPModelWrapperforUQ
from sparow.conf_intervals.protocols import StochasticProgramModelProtocol

import argparse
import json
import os

# ==== GLOBAL DATA =============================================================


class GlobalData:
    num_scens = 3  ### should be >= 3
    num_data_points = 10  # number of interpolated yield values per crop


if GlobalData.num_scens < 3:
    raise RuntimeError(f"Number of scenarios must be >= 3")

# ==== BASIC FARMERS SCENARIO DATA =============================================

Basic_scendata = {
    "scenarios": [
        {
            "ID": "scen_0",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "Probability": 1.0 / 3.0,
        },
        {
            "ID": "scen_1",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "Probability": 1.0 / 3.0,
        },
        {
            "ID": "scen_2",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "Probability": 1.0 / 3.0,
        },
    ]
}

# ==== ADVANCED FARMERS SCENARIO DATA ==========================================


class AdvancedScenario_dict(object):
    """
    Construct the full finite population set of scenarios for Advanced Farmers.

    For each crop, we linearly interpolate between the BelowAverage and
    AboveAverage yield values using num_data_points support points. The full
    scenario distribution is then the Cartesian product of those support
    points across the 3 crops, with equal probability assigned to each
    possible scenario vector.
    """

    def __init__(self, num_data_points):
        self.num_data_points = num_data_points

        self.wheat_support = np.linspace(2.0, 3.0, num_data_points)
        self.corn_support = np.linspace(2.4, 3.6, num_data_points)
        self.sugar_support = np.linspace(16.0, 24.0, num_data_points)

    def scenario_generator(self):
        """
        Final output is a dictionary with a single key-value pair.
        The key is "scenarios"
        The value is a list, called scen_dict_list. It contains one dictionary per possible
        population scenario. Each scenario's dictionary must contain "ID", set of yields, and
        "Probability".
        """
        total_scens = (
            self.num_data_points**3
        )  # there are 3 crops, each with num_data_points possible yields
        scen_prob = 1.0 / total_scens  # each scenario vector gets equal probability

        scen_id = 0  # naming convention: each scenario ID string ends in a number (population index)
        scen_dict_list = []

        for w in self.wheat_support:
            for c in self.corn_support:
                for s in self.sugar_support:
                    scen_dict_list.append(
                        {
                            "ID": f"scen_{scen_id}",
                            "Yield": {
                                "WHEAT": float(w),
                                "CORN": float(c),
                                "SUGAR_BEETS": float(s),
                            },
                            "Probability": scen_prob,
                        }
                    )
                    scen_id += 1

        return {"scenarios": scen_dict_list}


AdvancedScen_object = AdvancedScenario_dict(GlobalData.num_data_points)
Advanced_scendata = AdvancedScen_object.scenario_generator()


# ==== COMMON FARMERS MODEL BUILDER ============================================


def model_builder(data, args):
    """
    Common model builder for both Basic and Advanced farmers.

    The model structure is the same, only the scenario data (yield values
    and probabilities) differ.
    """

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.TOTAL_ACREAGE = 500.0

    def crops_init(m):
        return ["WHEAT", "CORN", "SUGAR_BEETS"]

    model.CROPS = pyo.Set(initialize=crops_init)

    def _data(indict):
        return {crop: indict[crop] for crop in ["WHEAT", "CORN", "SUGAR_BEETS"]}

    model.PriceQuota = _data(
        {"WHEAT": 100000.0, "CORN": 100000.0, "SUGAR_BEETS": 6000.0}
    )

    model.SubQuotaSellingPrice = _data(  # favorable selling prices
        {"WHEAT": 170.0, "CORN": 150.0, "SUGAR_BEETS": 36.0}
    )

    model.SuperQuotaSellingPrice = _data(  # unfavorable selling prices
        {"WHEAT": 0.0, "CORN": 0.0, "SUGAR_BEETS": 10.0}
    )

    model.CattleFeedRequirement = _data(  # right hand sides of demand constraints
        {"WHEAT": 200.0, "CORN": 240.0, "SUGAR_BEETS": 0.0}
    )

    model.PurchasePrice = (
        _data(  # purchasing costs.... cannot purchase sugar beets, so use dummy value
            {"WHEAT": 238.0, "CORN": 210.0, "SUGAR_BEETS": 100000.0}
        )
    )

    model.PlantingCostPerAcre = _data(  # planting costs
        {"WHEAT": 150.0, "CORN": 230.0, "SUGAR_BEETS": 260.0}
    )

    ### STOCHASTIC DATA
    def Yield_init(m, cropname):
        return data["Yield"][cropname]

    model.Yield = pyo.Param(
        model.CROPS,
        within=pyo.NonNegativeReals,
        initialize=Yield_init,
        mutable=True,
    )

    ### VARIABLES
    if args.get("use_integer", False):  # stage-1 vars integer
        model.DevotedAcreage = pyo.Var(
            model.CROPS,
            within=pyo.NonNegativeIntegers,
            bounds=(0.0, model.TOTAL_ACREAGE),
        )
    else:
        model.DevotedAcreage = pyo.Var(  # stage-1 vars continuous
            model.CROPS,
            bounds=(0.0, model.TOTAL_ACREAGE),
        )

    model.QuantitySubQuotaSold = pyo.Var(
        model.CROPS, bounds=(0.0, None)
    )  # qnty sold at favorable price
    model.QuantitySuperQuotaSold = pyo.Var(
        model.CROPS, bounds=(0.0, None)
    )  # qnty sold at unfavorable price
    model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))  # qnty purchased

    ### CONSTRAINTS
    def ConstrainTotalAcreage_rule(model):
        return sum(model.DevotedAcreage[c] for c in model.CROPS) <= model.TOTAL_ACREAGE

    model.ConstrainTotalAcreage = pyo.Constraint(rule=ConstrainTotalAcreage_rule)

    def EnforceCattleFeedRequirement_rule(model, c):
        return model.CattleFeedRequirement[c] <= (
            model.Yield[c] * model.DevotedAcreage[c]
            + model.QuantityPurchased[c]
            - model.QuantitySubQuotaSold[c]
            - model.QuantitySuperQuotaSold[c]
        )

    model.EnforceCattleFeedRequirement = pyo.Constraint(
        model.CROPS, rule=EnforceCattleFeedRequirement_rule
    )

    def LimitAmountSold_rule(model, c):
        return (
            model.QuantitySubQuotaSold[c]
            + model.QuantitySuperQuotaSold[c]
            - model.Yield[c] * model.DevotedAcreage[c]
        ) <= 0.0

    model.LimitAmountSold = pyo.Constraint(model.CROPS, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(model, c):
        return (0.0, model.QuantitySubQuotaSold[c], model.PriceQuota[c])

    model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

    ### OBJECTIVE
    def ComputeFirstStageCost_rule(model):
        return sum(
            model.PlantingCostPerAcre[c] * model.DevotedAcreage[c] for c in model.CROPS
        )

    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model):
        expr = sum(
            model.PurchasePrice[c] * model.QuantityPurchased[c] for c in model.CROPS
        )
        expr -= sum(
            model.SubQuotaSellingPrice[c] * model.QuantitySubQuotaSold[c]
            for c in model.CROPS
        )
        expr -= sum(
            model.SuperQuotaSellingPrice[c] * model.QuantitySuperQuotaSold[c]
            for c in model.CROPS
        )
        return expr

    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model):
        return model.FirstStageCost + model.SecondStageCost

    model.Total_Cost_Objective = pyo.Objective(
        rule=total_cost_rule,
        sense=pyo.minimize,
    )

    return model


# ==== MODEL DATA ===============================================================
app_data = {}
model_data = {
    "Basic": Basic_scendata,
    "Advanced": Advanced_scendata,
}

# ==== STOCHASTIC PROGRAM CONSTRUCTORS =========================================


def Basic_farmers():
    sp = stochastic_program(first_stage_variables=["DevotedAcreage[*]"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="Basic",
        model_data=model_data["Basic"],
        model_builder=model_builder,
    )
    return sp


def Advanced_farmers():
    sp = stochastic_program(first_stage_variables=["DevotedAcreage[*]"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="Advanced",
        model_data=model_data["Advanced"],
        model_builder=model_builder,
    )
    return sp


# =====================================================================
# Single-fidelity model-wrapper interface for confidence-interval code
# =====================================================================


def get_sp_model_for_uq(
    model_name="Advanced",
    use_integer=False,
    seed=12345,
    with_replacement=True,
) -> StochasticProgramModelProtocol:
    """
    Build one single-fidelity stochastic-program model wrapper for
    uncertainty quantification (UQ).

    This is the single-model entry point for StandardMRP algorithm.

    Returns
    -------
    StochasticProgramModelProtocol
        A model wrapper that owns:
          - the finite scenario population,
          - the scenario sampler,
          - the model builder,
          - the first-stage variable metadata,
          - and the replication-level solve/evaluate logic.
    """
    if model_name == "Basic":
        scenario_data = Basic_scendata
    elif model_name == "Advanced":
        scenario_data = Advanced_scendata
    else:
        raise ValueError(f"Unknown farmer model_name: {model_name}")

    # The scenario population object owns the full finite set of scenarios
    # and validation logic. We do not need vector encoding yet for StandardMRP.
    scenario_population = FiniteScenarioPopulation(
        scenarios=scenario_data["scenarios"],
        required_scenario_keys=["Yield"],
        scenario_vector_keys=[],
    )

    # The sampler is separate from the model wrapper, so sampling logic
    # stays reusable across different models and algorithms.
    scenario_sampler = ScenarioSampler(
        scenario_population=scenario_population,
        seed=seed,
        with_replacement=with_replacement,
    )

    # The underlying farmer model can optionally use integer first-stage variables.
    local_app_data = dict(app_data)
    local_app_data["use_integer"] = use_integer

    first_stage_vars = ["DevotedAcreage[*]"]
    first_stage_order = [
        "DevotedAcreage[WHEAT]",
        "DevotedAcreage[CORN]",
        "DevotedAcreage[SUGAR_BEETS]",
    ]

    model = SPModelWrapperforUQ(
        name=model_name,
        fidelity="high",  # single-fidelity case, so treat this as the primary model
        scenario_population=scenario_population,
        scenario_sampler=scenario_sampler,
        model_builder=model_builder,
        app_data=local_app_data,
        first_stage_variables=first_stage_vars,
        first_stage_variable_order=first_stage_order,
    )

    # Runtime-check the returned object against the protocol
    if not isinstance(model, StochasticProgramModelProtocol):
        raise RuntimeError(
            f"Object returned by get_sp_model_for_uq(...) for model_name={model_name} "
            "does not satisfy StochasticProgramModelProtocol."
        )

    return model


# =================================================================
# Write the scenario data to file for use in the confidence interval
# estimation command-line interface (sparow.conf_intervals.cli)
# =================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Write the full Advanced Farmers scenario population to a file "
        "that sparow.conf_intervals.cli --scenario-file can read."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path ending in .json or .npy",
    )
    args = parser.parse_args()

    # For writing the scenario file, we only need the finite scenario list.
    model = get_sp_model_for_uq(
        model_name="Advanced",
        use_integer=False,
        seed=12345,
        with_replacement=True,
    )

    scenarios = model.scenario_population().scenarios()
    model.scenario_population().validate(scenarios)

    outpath = os.path.abspath(args.output)
    (
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.dirname(outpath)
        else None
    )

    if outpath.endswith(".json"):
        with open(outpath, "w") as f:
            print(f"Writing population scenarios to file: {outpath}")
            json.dump({"scenarios": scenarios}, f, indent=2)
    elif outpath.endswith(".npy"):
        print(f"Writing population scenarios to file: {outpath}")
        np.save(outpath, {"scenarios": scenarios}, allow_pickle=True)
    else:
        raise ValueError("Output file must end with .json or .npy")

    print(f"Wrote {len(scenarios)} scenarios to: {outpath}")
    print(f"Use this with: --scenario-file {outpath}")


if __name__ == "__main__":
    main()
