import pprint
import pytest

# from IPython import embed
import random
import numpy as np

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
from mpisppy.opt.ph import PH

from sparow.sp import stochastic_program
from sparow.ph import ProgressiveHedgingSolver
from pyomo.opt import check_available_solvers

gurobi_available = len(check_available_solvers("gurobi", "appsi_gurobi")) == 2

verbose = False
if verbose:
    loglevel = "DEBUG"
else:
    loglevel = "WARN"
iters = 7

random.seed(923874938740938740)


def build_model_mpi(yields):
    model = pyo.ConcreteModel()

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

    return model


def scenario_creator(scenario_name):
    if scenario_name == "good":
        yields = [3, 3.6, 24]
    elif scenario_name == "average":
        yields = [2.5, 3, 20]
    elif scenario_name == "bad":
        yields = [2, 2.4, 16]
    else:
        raise ValueError("Unrecognized scenario name")

    model = build_model_mpi(yields)
    sputils.attach_root_node(model, model.PLANTING_COST, [model.X])
    model._mpisppy_probability = 1.0 / 3
    return model


# Test_MPI_EF = False
#
# if Test_MPI_EF:
#    options = {"solver": "gurobi"}
#    all_scenario_names = ["good", "average", "bad"]
#    ef = ExtensiveForm(options, all_scenario_names, scenario_creator)
#    results = ef.solve_extensive_form(tee=True)
#
#    objval = ef.get_objective_value()
#    print("EXTENSIVE FORM OBJECTIVE")
#    print(f"{objval:.1f}")


# options = {
#    "solver_name": "gurobi",
#    "PHIterLimit": 10,
#    "defaultPHrho": 10,
#    "convthresh": 1e-3,
#    "verbose": True,
#    "display_progress": True,
#    "display_timing": False,
#    "iter0_solver_options": dict(),
#    "iterk_solver_options": dict(),
# }

# all_scenario_names = ["good", "average", "bad"]
# ph = PH(
#    options,
#   all_scenario_names,
#    scenario_creator,
# )


# ph.get_root_solution()
# results = ph.ph_main()


def model_builder(scen, scen_args):
    model = pyo.ConcreteModel(scen["ID"])
    if scen["ID"] == "BelowAverageScenario":
        yields = [2, 2.4, 16]
    elif scen["ID"] == "AverageScenario":
        yields = [2.5, 3, 20]
    elif scen["ID"] == "AboveAverageScenario":
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

    return model


# FarmerSP = stochastic_program(first_stage_variables=["X[*]"])
# FarmerSP.initialize_model(model_builder=model_builder)

model_data = {
    "scenarios": [
        {
            "ID": "BelowAverageScenario",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "crops_multiplier": 1.0,
            "Probability": 0.3333333333333333,
        },
        {
            "ID": "AverageScenario",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "crops_multiplier": 1.0,
            "Probability": 0.3333333333333333,
        },
        {
            "ID": "AboveAverageScenario",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "crops_multiplier": 1.0,
            "Probability": 0.3333333333333333,
        },
    ]
}


class TestSolverAgainstMPISPPY(object):
    @pytest.mark.skipif(not gurobi_available, reason="gurobi not installed")
    def test_EF_model_solve(self):
        options = {"solver": "gurobi"}
        all_scenario_names = ["good", "average", "bad"]
        ef = ExtensiveForm(options, all_scenario_names, scenario_creator)
        results = ef.solve_extensive_form(tee=False)
        ef_objval = ef.get_objective_value()
        if verbose:
            print("EF VALUE")
            print(ef_objval)

        FarmerSP = stochastic_program(first_stage_variables=["X[*]"])
        FarmerSP.initialize_model(model_data=model_data, model_builder=model_builder)
        ph = ProgressiveHedgingSolver()
        results = ph.solve(
            FarmerSP,
            max_iterations=iters - 1,
            solver="gurobi",
            loglevel=loglevel,
            default_rho=10,
        )

        soln = results.last_solution
        ph_objval = soln.objective().value
        if verbose:
            print("PH VALUE")
            print(ph_objval)
        # assert abs(ef_objval - ph_objval) < 1e-5
        pass

    @pytest.mark.skipif(not gurobi_available, reason="gurobi not installed")
    def test_PH_model_solve_obj(self):
        options = {
            "solver_name": "gurobi",
            "PHIterLimit": iters,
            "defaultPHrho": 10,
            "convthresh": 1e-3,
            "verbose": verbose,
            "display_progress": verbose,
            "display_timing": False,
            "iter0_solver_options": dict(),
            "iterk_solver_options": dict(),
        }
        all_scenario_names = ["good", "average", "bad"]
        ph = PH(options, all_scenario_names, scenario_creator)
        results = ph.ph_main()
        mpi_objval = ph.Eobjective()

        FarmerSP = stochastic_program(first_stage_variables=["X[*]"])
        FarmerSP.initialize_model(model_data=model_data, model_builder=model_builder)
        ph = ProgressiveHedgingSolver()
        results = ph.solve(
            FarmerSP,
            max_iterations=iters - 1,
            solver="gurobi",
            loglevel=loglevel,
            default_rho=10,
        )

        soln = results.last_solution
        ph_objval = soln.objective().value
        if verbose:
            print("mpi_objval:", mpi_objval)
            print("ph_objval: ", ph_objval)
        assert abs((mpi_objval - ph_objval) / ph_objval) < 1e-3

    @pytest.mark.skipif(not gurobi_available, reason="gurobi not installed")
    def test_PH_model_solve_xbar(self):
        if verbose:
            print("=" * 60)
            print("Running MPISPPY")
            print("=" * 60)

        options = {
            "solver_name": "gurobi",
            "PHIterLimit": iters,
            "defaultPHrho": 10,
            "convthresh": 1e-3,
            "verbose": verbose,
            "display_progress": verbose,
            "display_timing": False,
            "iter0_solver_options": dict(),
            "iterk_solver_options": dict(),
        }
        all_scenario_names = ["good", "average", "bad"]

        ph = PH(options, all_scenario_names, scenario_creator)
        results = ph.ph_main()
        xbar_list = []
        for index in [("ROOT", 0), ("ROOT", 1), ("ROOT", 2)]:
            xbar_list.append(
                pyo.value(ph.local_scenarios["good"]._mpisppy_model.xbars[index])
            )
        xbar_mpi = np.array(xbar_list)

        if verbose:
            print("=" * 60)
            print("Running Forestlib.ph")
            print("=" * 60)

        FarmerSP = stochastic_program(first_stage_variables=["X[*]"])
        FarmerSP.initialize_model(model_data=model_data, model_builder=model_builder)
        ph = ProgressiveHedgingSolver()
        results = ph.solve(
            FarmerSP,
            max_iterations=iters - 1,
            solver="gurobi",
            loglevel=loglevel,
            default_rho=10,
        )
        soln = results.last_solution
        xbar_ph = np.array([var.value for var in reversed(soln.variables())])

        if verbose:
            print("xbar_mpi:", xbar_mpi)
            print("xbar_ph:", xbar_ph)
        assert np.allclose(xbar_ph, xbar_mpi)

    @pytest.mark.skipif(not gurobi_available, reason="gurobi not installed")
    def test_PH_model_solve_w_scen(self):
        options = {
            "solver_name": "gurobi",
            "PHIterLimit": iters,
            "defaultPHrho": 10,
            "convthresh": 1e-3,
            "verbose": verbose,
            "display_progress": verbose,
            "display_timing": False,
            "iter0_solver_options": dict(),
            "iterk_solver_options": dict(),
        }
        all_scenario_names = ["good", "average", "bad"]
        ph = PH(options, all_scenario_names, scenario_creator)
        results = ph.ph_main()
        w_bar_mpisppy = []
        for s in all_scenario_names:
            row = []
            for index in [("ROOT", 0), ("ROOT", 1), ("ROOT", 2)]:
                row.append(pyo.value(ph.local_scenarios[s]._mpisppy_model.W[index]))
            w_bar_mpisppy.append(row)

        FarmerSP = stochastic_program(first_stage_variables=["X[*]"])
        FarmerSP.initialize_model(model_data=model_data, model_builder=model_builder)
        ph = ProgressiveHedgingSolver()
        results = ph.solve(
            FarmerSP,
            max_iterations=iters - 1,
            solver="gurobi",
            loglevel=loglevel,
            default_rho=10,
        )
        soln = results.last_solution

        scenarios = ["AboveAverageScenario", "AverageScenario", "BelowAverageScenario"]
        data = []
        for scenario in scenarios:
            temp = []
            for i in range(2, -1, -1):
                temp.append(soln.variable(i).suffix.w[scenario])
            data.append(temp)
        array2 = np.array(data)

        array1 = np.array(w_bar_mpisppy)
        assert np.allclose(array1, array2)
