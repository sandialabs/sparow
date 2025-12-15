import random
import math
from munch import Munch
import pyomo.environ as pyo
from sparow.sp import stochastic_program

random.seed(923874938740938740)


#
# Global data for the HF model:
#
class GlobalData:
    num_plots = 20
    num_scens = 3  ### should be >= 3


if GlobalData.num_scens < 3:
    raise RuntimeError(f"Number of scenarios must be >= 3")

#
# list of possible per-plot scenarios for LF model:
#
LF_scendata = {
    "scenarios": [
        {
            "ID": "scen_0",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "Probability": 0.3,
        },
        {
            "ID": "scen_1",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "Probability": 0.3,
        },
        {
            "ID": "scen_2",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "Probability": 0.4,
        },
    ]
}

#
# list of possible per-plot scenarios for HF model:
#
HF_scendata = {
    "scenarios": [
        {
            "ID": "BBB",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "Probability": 0.027,
        },
        {
            "ID": "VVV",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "Probability": 0.027,
        },
        {
            "ID": "AAA",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "Probability": 0.064,
        },
        {
            "ID": "BBV",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 20.0},
            "Probability": 0.027,
        },
        {
            "ID": "BBA",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 24.0},
            "Probability": 0.036,
        },
        {
            "ID": "BVB",
            "Yield": {"WHEAT": 2.0, "CORN": 3.0, "SUGAR_BEETS": 16.0},
            "Probability": 0.027,
        },
        {
            "ID": "BAB",
            "Yield": {"WHEAT": 2.0, "CORN": 3.6, "SUGAR_BEETS": 16.0},
            "Probability": 0.036,
        },
        {
            "ID": "VBB",
            "Yield": {"WHEAT": 2.5, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "Probability": 0.027,
        },
        {
            "ID": "ABB",
            "Yield": {"WHEAT": 3.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "Probability": 0.036,
        },
        {
            "ID": "BVV",
            "Yield": {"WHEAT": 2.0, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "Probability": 0.027,
        },
        {
            "ID": "AVV",
            "Yield": {"WHEAT": 3.0, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "Probability": 0.036,
        },
        {
            "ID": "VBV",
            "Yield": {"WHEAT": 2.5, "CORN": 2.4, "SUGAR_BEETS": 20.0},
            "Probability": 0.027,
        },
        {
            "ID": "VAV",
            "Yield": {"WHEAT": 2.5, "CORN": 3.6, "SUGAR_BEETS": 20.0},
            "Probability": 0.036,
        },
        {
            "ID": "VVB",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 16.0},
            "Probability": 0.027,
        },
        {
            "ID": "VVA",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 24.0},
            "Probability": 0.036,
        },
        {
            "ID": "AAB",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 16.0},
            "Probability": 0.048,
        },
        {
            "ID": "AAV",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 20.0},
            "Probability": 0.048,
        },
        {
            "ID": "ABA",
            "Yield": {"WHEAT": 3.0, "CORN": 2.4, "SUGAR_BEETS": 24.0},
            "Probability": 0.048,
        },
        {
            "ID": "AVA",
            "Yield": {"WHEAT": 3.0, "CORN": 3.0, "SUGAR_BEETS": 24.0},
            "Probability": 0.048,
        },
        {
            "ID": "BAA",
            "Yield": {"WHEAT": 2.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "Probability": 0.048,
        },
        {
            "ID": "VAA",
            "Yield": {"WHEAT": 2.5, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "Probability": 0.048,
        },
        {
            "ID": "BVA",
            "Yield": {"WHEAT": 2.0, "CORN": 3.0, "SUGAR_BEETS": 24.0},
            "Probability": 0.036,
        },
        {
            "ID": "BAV",
            "Yield": {"WHEAT": 2.0, "CORN": 3.6, "SUGAR_BEETS": 20.0},
            "Probability": 0.036,
        },
        {
            "ID": "VBA",
            "Yield": {"WHEAT": 2.5, "CORN": 2.4, "SUGAR_BEETS": 24.0},
            "Probability": 0.036,
        },
        {
            "ID": "VAB",
            "Yield": {"WHEAT": 2.5, "CORN": 3.6, "SUGAR_BEETS": 16.0},
            "Probability": 0.036,
        },
        {
            "ID": "AVB",
            "Yield": {"WHEAT": 3.0, "CORN": 3.0, "SUGAR_BEETS": 16.0},
            "Probability": 0.036,
        },
        {
            "ID": "ABV",
            "Yield": {"WHEAT": 3.0, "CORN": 2.4, "SUGAR_BEETS": 20.0},
            "Probability": 0.036,
        },
    ]
}


#
# create the LF_data/HF_data dicts using per-plot scenarios
#
class LFScenario_dict(object):

    def __init__(self, LF_scendata):
        self.LF_scendata = LF_scendata

    def scenario_generator(self):
        self.scen_dict_list = []
        self.scen_dict = {"scenarios": self.scen_dict_list}

        """
        creating a dictionary with a single key, "scenarios", 
        with entry "scen_dict_list" which is a list that contains the scenario info. 
            each entry of the list contains an ID corresponding to the scenario, the list of 
            items from LF_scendata contained in the scenario, and the probability
        """
        for s in range(3):
            self.scen_dict_list.append(
                {"ID": f"scen_{s}", "list_IDs": [], "Probability": None}
            )

        ### always samples the three original (LF) scenarios:
        self.scen_dict_list[0]["list_IDs"].append(
            {
                "ID": "BBB",
                "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
                "Probability": 0.027,
            }
        )
        self.scen_dict_list[0]["Probability"] = 0.027
        self.scen_dict_list[1]["list_IDs"].append(
            {
                "ID": "VVV",
                "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
                "Probability": 0.027,
            }
        )
        self.scen_dict_list[1]["Probability"] = 0.027
        self.scen_dict_list[2]["list_IDs"].append(
            {
                "ID": "AAA",
                "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
                "Probability": 0.064,
            }
        )
        self.scen_dict_list[2]["Probability"] = 0.064

        ### normalize scenario probabilities
        norm_factor = sum(self.scen_dict_list[s]["Probability"] for s in range(3))
        for s in range(3):
            self.scen_dict_list[s]["Probability"] /= norm_factor

        return self.scen_dict


class HFScenario_dict(object):

    def __init__(self, HF_scendata):
        self.HF_scendata = HF_scendata

    def scenario_generator(self, num_plots, num_scens):
        self.scen_dict_list = []
        self.scen_dict = {"scenarios": self.scen_dict_list}

        """
        creating a dictionary with a single key, "scenarios", 
        with entry "scen_dict_list" which is a list that contains the scenario info. 
            each entry of the list contains an ID corresponding to the scenario, the list of 
            items from HF_scendata contained in the scenario, and the probability
        """
        for s in range(num_scens):
            self.scen_dict_list.append(
                {"ID": f"scen_{s}", "list_IDs": [], "Probability": None}
            )

        ### the scenarios are randomly sampled:
        for s in range(num_scens):
            self.scen_dict_list[s]["list_IDs"].extend(
                random.choices(self.HF_scendata["scenarios"], k=num_plots)
            )
            scen_probs = []
            for k in range(num_plots):
                scen_probs.append(self.scen_dict_list[s]["list_IDs"][k]["Probability"])
            self.scen_dict_list[s]["Probability"] = math.prod(scen_probs)

        ### remove redundant scenarios (not the most elegant solution... but pretty sure it works):
        for u in range(num_scens):
            for s in range(u, num_scens):
                if (
                    self.scen_dict_list[s]["list_IDs"]
                    == self.scen_dict_list[u]["list_IDs"]
                ):
                    redundant_scen = self.scen_dict_list[s]["list_IDs"]
                    self.scen_dict_list[s]["list_IDs"] = []
                    new_scen = random.choices(
                        self.HF_scendata["scenarios"], k=num_plots
                    )
                    while new_scen == redundant_scen:
                        new_scen = random.choices(
                            self.HF_scendata["scenarios"], k=num_plots
                        )
                    self.scen_dict_list[s]["list_IDs"].extend(new_scen)
                    scen_probs = []
                    for k in range(num_plots):
                        scen_probs.append(
                            self.scen_dict_list[s]["list_IDs"][k]["Probability"]
                        )
                    self.scen_dict_list[s]["Probability"] = math.prod(scen_probs)

        ### normalize scenario probabilities
        norm_factor = sum(
            self.scen_dict_list[s]["Probability"] for s in range(num_scens)
        )
        for s in range(num_scens):
            self.scen_dict_list[s]["Probability"] /= norm_factor

        return self.scen_dict


#
# construct model_data:
#
HFScen_object = HFScenario_dict(HF_scendata)
HF_data = HFScen_object.scenario_generator(GlobalData.num_plots, GlobalData.num_scens)

app_data = {"num_plots": GlobalData.num_plots}
model_data = {"LF": LF_scendata, "HF": HF_data}
# print(HF_data)


#
# Construct LF farmers problem model:
#
def LF_model_builder(data, args):
    num_plots = GlobalData.num_plots
    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.TOTAL_ACREAGE = 500.0
    model.PLOTS = pyo.Set(initialize=[j for j in range(num_plots)])

    def crops_init(m):
        retval = []
        # for j in range(num_plots):
        retval.append("WHEAT")
        retval.append("CORN")
        retval.append("SUGAR_BEETS")
        return retval

    model.CROPS = pyo.Set(initialize=crops_init)

    def _scale_up_data(indict):
        outdict = {}
        # for j in range(num_plots):
        for crop in ["WHEAT", "CORN", "SUGAR_BEETS"]:
            outdict[crop] = indict[crop]
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

    ### STOCHASTIC DATA
    def Yield_init(m, cropname):
        crop_base_name = cropname.rstrip("0123456789")
        return data["Yield"][crop_base_name]

    model.Yield = pyo.Param(
        model.CROPS, within=pyo.NonNegativeReals, initialize=Yield_init, mutable=True
    )

    ### VARIABLES
    if args.get("use_integer", True):
        model.DevotedAcreage = pyo.Var(
            model.CROPS,
            model.PLOTS,
            within=pyo.NonNegativeIntegers,
            bounds=(0.0, model.TOTAL_ACREAGE / num_plots),
        )
    else:
        model.DevotedAcreage = pyo.Var(
            model.CROPS, model.PLOTS, bounds=(0.0, model.TOTAL_ACREAGE / num_plots)
        )

    model.QuantitySubQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantitySuperQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))

    ### CONSTRAINTS
    def ConstrainPerPlotAcreage_rule(model):
        return (
            sum(
                sum(model.DevotedAcreage[c, j] for j in model.PLOTS)
                for c in model.CROPS
            )
            <= model.TOTAL_ACREAGE
        )

    model.ConstrainPerPlotAcreage = pyo.Constraint(rule=ConstrainPerPlotAcreage_rule)

    if len(model.PLOTS) > 1:

        def ConsistentPerPlotAcreage_rule(model):
            return (
                0,
                sum(
                    sum(
                        model.DevotedAcreage[c, j]
                        for j in range(1, len(model.PLOTS) - 1)
                    )
                    for c in model.CROPS
                ),
                0,
            )

        model.ConsistentPerPlotAcreage = pyo.Constraint(
            rule=ConsistentPerPlotAcreage_rule
        )

    def EnforceCattleFeedRequirement_rule(model, c):
        return model.CattleFeedRequirement[c] <= model.Yield[c] * model.DevotedAcreage[
            c, 0
        ] + model.QuantityPurchased[c] - (
            model.QuantitySubQuotaSold[c] + model.QuantitySuperQuotaSold[c]
        )

    model.EnforceCattleFeedRequirement = pyo.Constraint(
        model.CROPS, rule=EnforceCattleFeedRequirement_rule
    )

    def LimitAmountSold_rule(model, c):
        return (
            model.QuantitySubQuotaSold[c]
            + model.QuantitySuperQuotaSold[c]
            - model.Yield[c] * model.DevotedAcreage[c, 0]
        ) <= 0.0

    model.LimitAmountSold = pyo.Constraint(model.CROPS, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(model, c):
        return (0.0, model.QuantitySubQuotaSold[c], model.PriceQuota[c])

    model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

    ### OBJECTIVE
    def ComputeFirstStageCost_rule(model):
        return sum(
            model.PlantingCostPerAcre[c] * model.DevotedAcreage[c, 0]
            for c in model.CROPS
        )

    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model):
        expr = pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
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

    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return model


#
# Construct HF farmers problem model:
#
def model_builder(data, args):
    num_plots = GlobalData.num_plots
    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.TOTAL_ACREAGE = 500.0
    model.PLOTS = pyo.Set(initialize=[j for j in range(num_plots)])

    def crops_init(m):
        retval = []
        retval.append("WHEAT")
        retval.append("CORN")
        retval.append("SUGAR_BEETS")
        return retval

    model.CROPS = pyo.Set(initialize=crops_init)

    def _scale_up_data(indict):
        outdict = {}
        for crop in ["WHEAT", "CORN", "SUGAR_BEETS"]:
            outdict[crop] = indict[crop]
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

    ### STOCHASTIC DATA
    def Yield_init(m, cropname, plot):  ### per-plot crop yields
        crop_base_name = cropname.rstrip("0123456789")
        return data["list_IDs"][plot]["Yield"][crop_base_name]

    model.Yield = pyo.Param(
        model.CROPS,
        model.PLOTS,
        within=pyo.NonNegativeReals,
        initialize=Yield_init,
        mutable=True,
    )

    ### VARIABLES
    if args.get("use_integer", True):
        model.DevotedAcreage = pyo.Var(
            model.CROPS,
            model.PLOTS,
            within=pyo.NonNegativeIntegers,
            bounds=(0.0, model.TOTAL_ACREAGE / num_plots),
        )
    else:
        model.DevotedAcreage = pyo.Var(
            model.CROPS, model.PLOTS, bounds=(0.0, model.TOTAL_ACREAGE / num_plots)
        )

    model.QuantitySubQuotaSold = pyo.Var(model.CROPS, model.PLOTS, bounds=(0.0, None))
    model.QuantitySuperQuotaSold = pyo.Var(model.CROPS, model.PLOTS, bounds=(0.0, None))
    model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))

    ### CONSTRAINTS
    def ConstrainPerPlotAcreage_rule(model, j):
        return (
            sum(model.DevotedAcreage[c, j] for c in model.CROPS)
            <= model.TOTAL_ACREAGE / num_plots
        )

    model.ConstrainPerPlotAcreage = pyo.Constraint(
        model.PLOTS, rule=ConstrainPerPlotAcreage_rule
    )

    def EnforceCattleFeedRequirement_rule(model, c):
        return model.CattleFeedRequirement[c] <= sum(
            model.Yield[c, j] * model.DevotedAcreage[c, j] for j in model.PLOTS
        ) + model.QuantityPurchased[c] - sum(
            model.QuantitySubQuotaSold[c, j] + model.QuantitySuperQuotaSold[c, j]
            for j in model.PLOTS
        )

    model.EnforceCattleFeedRequirement = pyo.Constraint(
        model.CROPS, rule=EnforceCattleFeedRequirement_rule
    )

    def LimitAmountSold_rule(model, c, j):
        return (
            model.QuantitySubQuotaSold[c, j]
            + model.QuantitySuperQuotaSold[c, j]
            - model.Yield[c, j] * model.DevotedAcreage[c, j]
            <= 0.0
        )

    model.LimitAmountSold = pyo.Constraint(
        model.CROPS, model.PLOTS, rule=LimitAmountSold_rule
    )

    def EnforceQuotas_rule(model, c):
        return (
            0.0,
            sum(model.QuantitySubQuotaSold[c, j] for j in model.PLOTS),
            model.PriceQuota[c],
        )

    model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

    ### OBJECTIVE
    def ComputeFirstStageCost_rule(model):
        return sum(
            model.PlantingCostPerAcre[c]
            * sum(model.DevotedAcreage[c, j] for j in model.PLOTS)
            for c in model.CROPS
        )

    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model):
        expr = pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
        expr -= sum(
            model.SubQuotaSellingPrice[c]
            * sum(model.QuantitySubQuotaSold[c, j] for j in model.PLOTS)
            for c in model.CROPS
        )
        expr -= sum(
            model.SuperQuotaSellingPrice[c]
            * sum(model.QuantitySuperQuotaSold[c, j] for j in model.PLOTS)
            for c in model.CROPS
        )
        return expr

    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model):
        return model.FirstStageCost + model.SecondStageCost

    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return model


#
# options to solve, LF, HF, or MF:
#


def HF_farmers():
    sp = stochastic_program(first_stage_variables=["DevotedAcreage[*,*]"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=model_builder
    )
    return Munch(sp=sp)


def LF_farmers():
    sp = stochastic_program(first_stage_variables=["DevotedAcreage[*,*]"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_model_builder
    )
    return Munch(sp=sp)


def MFrandom_farmers():
    sp = stochastic_program(first_stage_variables=["DevotedAcreage[*,*]"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=model_builder
    )
    sp.initialize_model(
        name="LF",
        model_data=model_data["LF"],
        model_builder=LF_model_builder,
        default=False,
    )
    sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 2.0, "LF": 1.0},
    )
    return Munch(sp=sp)
