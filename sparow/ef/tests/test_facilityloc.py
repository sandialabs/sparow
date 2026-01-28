import pytest
import pyomo.environ as pyo

from or_topas.solnpool import PoolManager
from sparow.sp import stochastic_program
from sparow.sp.examples import AMPL_facilityloc
from sparow.ef import ExtensiveFormSolver

import pyomo.opt
from pyomo.common import unittest

solvers = set(pyomo.opt.check_available_solvers("glpk", "gurobi", "highs"))


"""
Note that this test just ensures our extensive form solution matches AMPL's!! It does not test PH.
* Problem data adapted from https://ampl.com/colab/notebooks/ampl-development-tutorial-26-stochastic-capacitated-facility-location-problem.html#problem-description
"""


@unittest.pytest.mark.parametrize("mip_solver", solvers)
class TestFacilityLoc:

    def test_facilityloc_old(self, mip_solver):
        app_data = {"n": 3, "t": 4}  # number of facilities & customers
        app_data["f"] = [400000, 200000, 600000]  # fixed costs for opening facilities
        app_data["c"] = [
            [5739.725, 6539.725, 8650.40, 22372.1125],
            [6055.05, 6739.055, 8050.40, 21014.225],
            [8650.40, 7539.055, 4539.72, 15024.325],
        ]  # servicing costs
        app_data["k"] = [1550, 650, 1750]  # facility capacity

        customer_demand = {
            "San_Antonio_TX": [450, 650, 887],
            "Dallas_TX": [910, 1134, 1456],
            "Jackson_MS": [379, 416, 673],
            "Birmingham_AL": [91, 113, 207],
        }
        demand_levels = ["Low", "Medium", "High"]

        # mapping of demand levels to their corresponding values
        demand_value_mapping = {"Low": 0, "Medium": 1, "High": 2}
        # mapping of demand levels to their corresponding probabilities
        demand_prob_mapping = {"Low": 0.25, "Medium": 0.5, "High": 0.25}

        LF_scenarios = [
            ("Low", "Low", "Low", "Low"),
            ("Medium", "Medium", "Medium", "Medium"),
            ("High", "High", "High", "High"),
        ]
        LFscens_list = []  # list of LF scenarios
        for lscen_idx, lscen in enumerate(LF_scenarios):
            scen = demand_levels[lscen_idx]
            LFscens_list.append(
                {
                    "ID": f"{scen}",
                    "Demand": [
                        customer_demand[key][demand_value_mapping[scen]]
                        for key in customer_demand.keys()
                    ],
                    "Probability": demand_prob_mapping[scen],
                }
            )

        model_data = {"HF": {"scenarios": LFscens_list}}

        def HF_builder(data, args):
            n = data["n"]
            t = data["t"]
            f = data["f"]
            c = data["c"]
            k = data["k"]

            ### STOCHASTIC DATA
            d = data["Demand"]

            model = pyo.ConcreteModel(data["ID"])

            ### PARAMETERS
            model.N = pyo.Set(initialize=[i for i in range(n)])
            model.T = pyo.Set(initialize=[j for j in range(t)])

            ### VARIABLES
            model.x = pyo.Var(
                model.N, within=pyo.Binary
            )  # x[i] == 1 if facility i is open
            model.z = pyo.Var(
                model.N, model.T, within=pyo.NonNegativeReals
            )  # z[i, j] = proportion of customer j's demand met by facility i

            ### CONSTRAINTS
            def MeetDemand_rule(model, j):
                return sum(model.z[i, j] for i in range(n)) >= d[j]

            model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

            def SufficientProduction_rule(model):
                return sum(k[i] * model.x[i] for i in range(n)) >= sum(
                    d[j] for j in range(t)
                )

            model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

            def Capacity_rule(
                model, i
            ):  # note this constraint also ensures logic between x, z
                return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]

            model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

            ### OBJECTIVE
            def Obj_rule(model):
                expr = sum(
                    sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n)
                )
                expr += sum(f[i] * model.x[i] for i in range(n))
                return expr

            model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

            return model

        sp = stochastic_program(first_stage_variables=["x"])
        sp.initialize_application(app_data=app_data)
        sp.initialize_model(
            name="HF", model_data=model_data["HF"], model_builder=HF_builder
        )
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        pool_manager = PoolManager()
        pool_manager.solution_counter = 0
        results = solver.solve(sp)
        results_dict = results.to_dict()
        obj_val = results_dict["solutions"][0]["objectives"][0]["value"]

        assert obj_val == pytest.approx(16758018.59625)

    def test_facilityloc(self, mip_solver):
        app = AMPL_facilityloc()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        pool_manager = PoolManager()
        pool_manager.solution_counter = 0
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        obj_val = results_dict["solutions"][0]["objectives"][0]["value"]

        assert obj_val == pytest.approx(app.objective_value)
