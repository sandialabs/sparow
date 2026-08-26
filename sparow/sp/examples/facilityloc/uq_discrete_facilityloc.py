import pyomo.environ as pyo
import itertools
import math
import random
import numpy as np
from pathlib import Path

from sparow.conf_intervals.scenario_population import FiniteScenarioPopulation
from sparow.conf_intervals.scenario_sampler import ScenarioSampler
from sparow.conf_intervals.sp_model_wrapper_for_uq import SPModelWrapperforUQ
from sparow.conf_intervals.model_ensemble import ModelEnsemble
from sparow.conf_intervals.protocols import (
    StochasticProgramModelProtocol,
    ModelEnsembleProtocol,
)

import argparse
import json
import os

"""
FACILITY LOCATION
    - HF model is a MIP (first-stage binary variables, second-stage BINARY AND CONTINUOUS variables)
        - Each facility can support a fixed number of customers (binary vars in the second stage)
    - LF model relaxes the second stage binary vars
        - LF scenarios are the same as HF
        - Constraint ensuring logic between z and y is taken out so that y takes on continuous values
* Scenarios are generated using linear interpolation between low and high demand values for each city, 
  creating a discrete uniform distribution over all resulting scenarios
* The number of scenarios is controlled by the shell script's N parameter, not by app_data
* You can run this file as a script to write the full scenario population to a .json or .npy file
"""

app_data = {"n": 6, "t": 4}  # number of facilities & customers
app_data["f"] = [
    260000,
    275000,
    270000,
    285000,
    320000,
    340000,
]  # fixed costs for opening facilities (facility 4 and facility 5 are expensive)
app_data["c"] = [
    [4200.0, 5200.0, 12500.0, 18000.0],  # facility 0 good for cust 0,1
    [4600.0, 4800.0, 11800.0, 17500.0],  # facility 1 also good for cust 0,1
    [12800.0, 12000.0, 4100.0, 5600.0],  # facility 2 good for cust 2,3
    [13500.0, 12600.0, 4500.0, 5100.0],  # facility 3 also good for cust 2,3
    [7600.0, 7900.0, 7800.0, 8200.0],  # facility 4 is expensive
    [9000.0, 9400.0, 9100.0, 9600.0],  # facility 5 is expensive
]  # servicing costs
app_data["k"] = [2500, 2500, 2500, 2500, 2500, 2500]  # facility capacity
app_data["s"] = [2, 2, 2, 2, 2, 2]  # max number of customers each facility can service
app_data["a"] = [
    [3900.0, 5000.0, 12000.0, 17000.0],
    [4300.0, 4700.0, 11400.0, 16800.0],
    [12200.0, 11600.0, 3900.0, 5200.0],
    [12900.0, 12100.0, 4300.0, 4800.0],
    [7000.0, 7300.0, 7100.0, 7600.0],
    [8600.0, 9000.0, 8700.0, 9200.0],
]  # transportation costs

BASE_DIR = Path(__file__).resolve().parent  # path to directory that contains this file

bigM_path = BASE_DIR / "bigM.txt"
with open(bigM_path, "r") as file:  # read in big-M value from bigM.txt
    bigM_str = file.read()
app_data["bigM"] = float(bigM_str)


# ==== SCENARIO DATA ===========================================================

# Define low and high demand values for each city (customer)
low_demands = [180.0, 500.0, 140.0, 40.0]
high_demands = [700.0, 1400.0, 650.0, 260.0]


class FacilityLocationScenarioData(object):
    """
    Construct the full finite population set of facility-location scenarios.

    For each customer, linearly interpolate between the low and high demand
    values using num_data_points support points. The full
    scenario distribution is then the Cartesian product of those support
    points across the customers, with equal probability assigned to each
    possible scenario vector.
    """

    def __init__(self, num_data_points):
        self.num_data_points = num_data_points

        self.demand_supports = []
        for low, high in zip(low_demands, high_demands):
            self.demand_supports.append(np.linspace(low, high, num_data_points))

    def scenario_generator(self):
        """
        Final output is a dictionary with a single key-value pair.
        The key is "scenarios"
        The value is a list, called scen_dict_list. It contains one dictionary per possible
        population scenario. Each scenario's dictionary must contain "ID", demands, and
        "Probability".
        """
        total_scens = self.num_data_points ** len(self.demand_supports)
        scen_prob = 1.0 / total_scens  # each scenario vector gets equal probability

        scen_id = 0  # naming convention: each scenario ID string ends in a number (population index)
        scen_dict_list = []

        # Use itertools.product to get cartesian product
        for demand_tuple in itertools.product(*self.demand_supports):
            scen_dict_list.append(
                {
                    "ID": f"scen_{scen_id}",
                    "Demand": [float(d) for d in demand_tuple],
                    "Probability": scen_prob,
                }
            )
            scen_id += 1

        return {"scenarios": scen_dict_list}


HFScenarioObject = FacilityLocationScenarioData(num_data_points=5)
LFScenarioObject = FacilityLocationScenarioData(num_data_points=5)

HF_scendata = HFScenarioObject.scenario_generator()
LF_scendata = LFScenarioObject.scenario_generator()


# ==== MODEL DATA ===============================================================

# This is a multi-model container:
# stores scenario datasets for each model
scenario_data_by_model = {
    "HF": HF_scendata,
    "LF": LF_scendata,
}

# ==== MODEL BUILDERS ===========================================================


def LF_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]
    bigM = data["bigM"]
    s = data["s"]
    a = data["a"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.y = pyo.Var(
        model.N, model.T, domain=[0, 1]
    )  # y[i, j] in [0,1] if customer j's demand is met by facility i (RELAXED VAR)
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = volume of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]
        # sum of demand met by all facilities for customer j is greater than demand from customer j

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))
        # sum of production from all facilities is greater than sum of total demand from all customers

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]
        # volume of demand met is less than capacity for each facility. this constraint also ensures logic between x, z

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    def OpenFacilities_rule(model, i, j):
        return model.y[i, j] <= model.x[i]
        # facility i needs to be open to fulfill customer j's demand w/ facility i

    model.OpenFacilities = pyo.Constraint(model.N, model.T, rule=OpenFacilities_rule)

    ### COMMENTING THIS OUT SO THAT y TAKES ON CONTINUOUS VALUES ###
    # def LogicFacilities_rule(model, i, j):
    #    return model.z[i, j] <= bigM*model.y[i, j]
    # if facility i doesn't meet customer j's demand, volume of demand met by i for j is 0
    # model.LogicFacilities = pyo.Constraint(model.N, model.T, rule=LogicFacilities_rule)

    def CustomersPerFacility_rule(model, i):
        return sum(model.y[i, j] for j in range(t)) <= s[i]
        # limit on the number of customers serviced by facility j

    model.CustomersPerFacility = pyo.Constraint(model.N, rule=CustomersPerFacility_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        # cost of fulfilling total volume of demand
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
        # cost of transporting goods from facility i to customer j
        expr += sum(sum(a[i][j] * model.y[i, j] for j in range(t)) for i in range(n))
        # cost of opening facilities
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


def HF_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]
    bigM = data["bigM"]
    s = data["s"]
    a = data["a"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.y = pyo.Var(
        model.N, model.T, within=pyo.Binary
    )  # y[i, j] == 1 if customer j's demand is met by facility i
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = volume of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]
        # sum of demand met by all facilities for customer j is greater than demand from customer j

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))
        # sum of production from all facilities is greater than sum of total demand from all customers

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]
        # volume of demand met is less than capacity for each facility. this constraint also ensures logic between x, z

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    def OpenFacilities_rule(model, i, j):
        return model.y[i, j] <= model.x[i]
        # facility i needs to be open to fulfill customer j's demand w/ facility i

    model.OpenFacilities = pyo.Constraint(model.N, model.T, rule=OpenFacilities_rule)

    def LogicFacilities_rule(model, i, j):
        return model.z[i, j] <= bigM * model.y[i, j]
        # if facility i doesn't meet customer j's demand, volume of demand met by i for j is 0

    model.LogicFacilities = pyo.Constraint(model.N, model.T, rule=LogicFacilities_rule)

    def CustomersPerFacility_rule(model, i):
        return sum(model.y[i, j] for j in range(t)) <= s[i]
        # limit on the number of customers serviced by facility j

    model.CustomersPerFacility = pyo.Constraint(model.N, rule=CustomersPerFacility_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        # cost of fulfilling total volume of demand
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
        # cost of transporting goods from facility i to customer j
        expr += sum(sum(a[i][j] * model.y[i, j] for j in range(t)) for i in range(n))
        # cost of opening facilities
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


# =====================================================================
# Single-fidelity model-wrapper interface for confidence-interval code
# =====================================================================


def get_sp_model_for_uq(
    model_name="HF",
    use_integer=False, # dummy compatibility argument; TODO: replace with more flexible kwargs handling
    seed=12345,
    with_replacement=True,
) -> StochasticProgramModelProtocol:
    """
    Build one confidence-interval-facing stochastic-program wrapper.

    Returns
    -------
    StochasticProgramModelProtocol
        One model wrapper that owns its scenario population, sampler,
        model builder, and first-stage metadata.
    """
    if model_name == "HF":
        scenario_data = scenario_data_by_model["HF"]
        model_builder = HF_builder
        fidelity = "high"
    elif model_name == "LF":
        scenario_data = scenario_data_by_model["LF"]
        model_builder = LF_builder
        fidelity = "low"
    else:
        raise ValueError(f"Unknown facility location model_name: {model_name}")

    # The scenario population object stores the finite list of scenarios and
    # validates their native SPAROW formatting.
    scenario_population = FiniteScenarioPopulation(
        scenarios=scenario_data["scenarios"],
        required_scenario_keys=["Demand"],
        scenario_vector_keys=["Demand"],
    )

    # The sampler is kept separate from the model wrapper so that sampling
    # logic remains reusable across different algorithms.
    scenario_sampler = ScenarioSampler(
        scenario_population=scenario_population,
        seed=seed,
        with_replacement=with_replacement,
    )

    first_stage_vars = ["x"]
    first_stage_order = [f"x[{i}]" for i in range(app_data["n"])]

    model = SPModelWrapperforUQ(
        name=model_name,
        fidelity=fidelity,
        scenario_population=scenario_population,
        scenario_sampler=scenario_sampler,
        model_builder=model_builder,
        app_data=app_data,
        first_stage_variables=first_stage_vars,
        first_stage_variable_order=first_stage_order,
    )

    if not isinstance(model, StochasticProgramModelProtocol):
        raise RuntimeError(
            f"Object returned by get_sp_model_for_uq(...) for model_name={model_name} "
            "does not satisfy StochasticProgramModelProtocol."
        )

    return model


# =====================================================================
# Multifidelity ensemble interface for ACV-MRP and PyApprox
# =====================================================================


def get_model_ensemble_for_uq(
    model_name="HF",
    use_integer=False,
    seed=12345,
    with_replacement=True,
    lf_model_type="classic",
) -> ModelEnsembleProtocol:
    """
    Build a two-model HF/LF ensemble for multifidelity workflows.

    This is the standard entry point for ACV-MRP and PyApprox integration.

    NOTE: model_name is included for interface consistency.
    The ensemble always contains both the HF and LF facility-location models.

    NOTE: lf_model_type is included for interface consistency.
    It is ignored here because facility location currently has only one option for the LF model.

    Returns
    -------
    ModelEnsembleProtocol
        Ensemble with:
          - model 0 = HF facility-location model
          - model 1 = LF facility-location model
    """
    hf_model = get_sp_model_for_uq(
        model_name="HF",
        use_integer=use_integer,
        seed=seed,
        with_replacement=with_replacement,
    )

    lf_model = get_sp_model_for_uq(
        model_name="LF",
        use_integer=use_integer,
        seed=seed,
        with_replacement=with_replacement,
    )

    ensemble = ModelEnsemble([hf_model, lf_model])

    if not isinstance(ensemble, ModelEnsembleProtocol):
        raise RuntimeError(
            "Object returned by get_model_ensemble_for_uq(...) does not satisfy "
            "ModelEnsembleProtocol."
        )

    return ensemble


# =================================================================
# Write the scenario data to file
# =================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Write the full facility location scenario population to a file "
    )
    parser.add_argument(
        "--output", required=True, help="Output file path ending in .json or .npy"
    )
    parser.add_argument(
        "--num-data-points",
        type=int,
        default=10,
        help="Number of interpolated demand values per customer (default: 10)",
    )
    args = parser.parse_args()

    scenario_object = FacilityLocationScenarioData(args.num_data_points)
    scenario_data = scenario_object.scenario_generator()
    scen_dict_list = scenario_data["scenarios"]
    print(f"\n ==== Number of population scenarios: {len(scen_dict_list)} === \n")

    # Scenarios are the same for the LF and HF models
    scenario_population = FiniteScenarioPopulation(
        scenarios=scenario_data["scenarios"],
        required_scenario_keys=["Demand"],
        scenario_vector_keys=["Demand"],
    )

    scenario_population.validate()
    scenarios = scenario_population.scenarios()

    outpath = os.path.abspath(args.output)
    (
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.dirname(outpath)
        else None
    )

    if outpath.endswith(".json"):
        with open(outpath, "w") as f:
            json.dump({"scenarios": scenarios}, f, indent=2)
    elif outpath.endswith(".npy"):
        np.save(outpath, {"scenarios": scenarios}, allow_pickle=True)
    else:
        raise ValueError("Output file must end with .json or .npy")

    print(f"Wrote {len(scenarios)} scenarios to: {outpath}")
    print(f"Use this with: --scenario-file {outpath}")


if __name__ == "__main__":
    main()
