import pytest
import pyomo.environ as pyo
from forestlib.ph import stochastic_program


#
# Data for a simple newsvendor example
#
app_data = dict(c=1.0, b=1.5, h=0.1)
bundle_data = {
    "scenarios": [
        {"ID": 1, "d": 15},
        {"ID": 2, "d": 60},
        {"ID": 3, "d": 72},
        {"ID": 4, "d": 78},
        {"ID": 5, "d": 82},
    ]
}


#
# Function that constructs a newsvendor model
# including a single second stage
#
def model_builder(app_data, scen_data, args):
    b = app_data["b"]
    c = app_data["c"]
    h = app_data["h"]
    d = scen_data["d"]

    M = pyo.ConcreteModel(scen_data["ID"])

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.o = pyo.Objective(expr=M.y)

    return M


#
# Function that constructs the first stage of a
# newsvendor model
#
def first_stage(M, app_data, args):
    M.x = pyo.Var(within=pyo.NonNegativeReals)


#
# Function that constructs the second stage of a
# newsvendor model
#
def second_stage(M, S, app_data, scen_data, args):
    b = app_data["b"]
    c = app_data["c"]
    h = app_data["h"]
    d = scen_data["d"]

    S.y = pyo.Var()

    S.o = pyo.Objective(expr=S.y)
    S.greater = pyo.Constraint(expr=S.y >= (c - b) * M.x + b * d)
    S.less = pyo.Constraint(expr=S.y >= (c + h) * M.x - h * d)


class TestNewsVendor:
    """
    Test the news vendor application

    See https://stoprog.org/sites/default/files/SPTutorial/TutorialSP.pdf
    """

    def test_single_builder(self):
        sp = stochastic_program(
            first_stage_variables=["x"], model_builder=model_builder
        )
        sp.initialize_application(app_data=app_data)
        sp.initialize_bundles(bundle_data=bundle_data)

        assert set(sp.bundles.keys()) == {"1", "2", "3", "4", "5"}
        assert sp.bundle_probability["1"] == 0.2

        #
        # Testing internal data structures
        #
        M1 = sp.create_subproblem("1")
        assert set(sp.int_to_FirstStageVar.keys()) == {"1"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        M2 = sp.create_subproblem("2")
        assert set(sp.int_to_FirstStageVar.keys()) == {"1", "2"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        #
        # Test subproblem solver logic
        #
        sp.solve(M1, solver="glpk")
        assert pyo.value(M1.s[1].x) == 15.0

        sp.solve(M2, solver="glpk")
        assert pyo.value(M2.s[2].x) == 60.0

    def test_multistage_builder(self):
        sp = stochastic_program(model_builder_list=[first_stage, second_stage])
        sp.initialize_application(app_data=app_data)
        sp.initialize_bundles(bundle_data=bundle_data)

        assert set(sp.bundles.keys()) == {"1", "2", "3", "4", "5"}
        assert sp.bundle_probability["1"] == 0.2

        #
        # Testing internal data structures
        #
        M1 = sp.create_subproblem("1")
        assert set(sp.int_to_FirstStageVar.keys()) == {"1"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        M2 = sp.create_subproblem("2")
        assert set(sp.int_to_FirstStageVar.keys()) == {"1", "2"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        #
        # Test subproblem solver logic
        #
        M1.pprint()
        sp.solve(M1, solver="glpk")
        assert pyo.value(M1.x) == 15.0

        sp.solve(M2, solver="glpk")
        assert pyo.value(M2.x) == 60.0
