import pytest
import pyomo.environ as pyo
from forestlib.ph import stochastic_program


#
# Data for a simple newsvendor example
#
app_data = dict(c=1.0, b=1.5, h=0.1)
model_data = {
    "LF": {
        "scenarios": [
            {"ID": 1, "d": 15},
            {"ID": 2, "d": 60},
            {"ID": 3, "d": 72},
            {"ID": 4, "d": 78},
            {"ID": 5, "d": 82},
        ]
    },
    "HF": {
        "data": {"B": 0.9},
        "scenarios": [
            {"ID": 1, "d": 15, "C": 1.4},
            {"ID": 2, "d": 60, "C": 1.3},
            {"ID": 3, "d": 72, "C": 1.2},
            {"ID": 4, "d": 78, "C": 1.1},
            {"ID": 5, "d": 82, "C": 1.0},
        ],
    },
}


#
# Function that constructs a newsvendor model
# including a single second stage
#
def LF_builder(data, args):
    b = data["b"]
    c = data["c"]
    h = data["h"]
    d = data["d"]

    M = pyo.ConcreteModel(data["ID"])

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.o = pyo.Objective(expr=M.y)

    return M


def HF_builder(data, args):
    b = data["b"]
    B = data["B"]
    c = data["c"]
    C = data["C"]
    h = data["h"]
    d = data["d"]

    M = pyo.ConcreteModel(data["ID"])

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.greaterX = pyo.Constraint(expr=M.y >= (C - B) * M.x + B * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.o = pyo.Objective(expr=M.y)

    return M


class TestMFNewsVendor:
    """
    Test the multi-fidelity news vendor application

    See https://stoprog.org/sites/default/files/SPTutorial/TutorialSP.pdf
    """

    def test_LF_builder(self):
        sp = stochastic_program(first_stage_variables=["x"])
        sp.initialize_application(app_data=app_data)
        sp.initialize_model(
            name="LF", model_builder=LF_builder, model_data=model_data["LF"]
        )

        assert set(sp.bundles.keys()) == {"LF_1", "LF_2", "LF_3", "LF_4", "LF_5"}
        assert sp.bundles["LF_1"].probability == 0.2

        #
        # Testing internal data structures
        #
        M1 = sp.create_subproblem("LF_1")
        assert set(sp.int_to_FirstStageVar.keys()) == {"LF_1"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        M2 = sp.create_subproblem("LF_2")
        assert set(sp.int_to_FirstStageVar.keys()) == {"LF_1", "LF_2"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        #
        # Test subproblem solver logic
        #
        sp.solve(M1, solver="glpk")
        assert pyo.value(M1.s["LF", 1].x) == 15.0

        sp.solve(M2, solver="glpk")
        assert pyo.value(M2.s["LF", 2].x) == 60.0

    def test_HF_builder(self):
        sp = stochastic_program(first_stage_variables=["x"])
        sp.initialize_application(app_data=app_data)
        sp.initialize_model(
            name="HF", model_builder=HF_builder, model_data=model_data["HF"]
        )

        assert set(sp.bundles.keys()) == {"HF_1", "HF_2", "HF_3", "HF_4", "HF_5"}
        assert sp.bundles["HF_1"].probability == 0.2

        #
        # Testing internal data structures
        #
        M1 = sp.create_subproblem("HF_1")
        assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        M2 = sp.create_subproblem("HF_2")
        assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1", "HF_2"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        #
        # Test subproblem solver logic
        #
        sp.solve(M1, solver="glpk")
        assert pyo.value(M1.s["HF", 1].x) == 9.0

        sp.solve(M2, solver="glpk")
        assert pyo.value(M2.s["HF", 2].x) == 40.0

    def test_MF_builder1(self):
        sp = stochastic_program(first_stage_variables=["x"])
        sp.initialize_application(app_data=app_data)
        sp.initialize_model(
            name="HF", model_data=model_data["HF"], model_builder=HF_builder
        )
        sp.initialize_model(
            name="LF",
            model_data=model_data["LF"],
            model_builder=LF_builder,
            default=False,
        )
        sp.initialize_bundles(scheme="mf_paired")

        assert set(sp.bundles.keys()) == {"1", "2", "3", "4", "5"}
        assert sp.bundles["1"].probability == 0.2

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
        assert len(M1.s) == 2
        assert pyo.value(M1.s["HF", 1].x) == 15.0
        assert pyo.value(M1.s["LF", 1].x) == 15.0
        assert pyo.value(M1.s["HF", 1].y) == 21.0
        assert pyo.value(M1.s["LF", 1].y) == 15.0

        sp.solve(M2, solver="glpk")
        assert len(M2.s) == 2
        assert pyo.value(M2.s["HF", 2].x) == 60.0
        assert pyo.value(M2.s["LF", 2].x) == 60.0
        assert pyo.value(M2.s["HF", 2].y) == 78.0
        assert pyo.value(M2.s["LF", 2].y) == 60.0

    def test_MF_builder2(self):
        sp = stochastic_program(first_stage_variables=["x"])
        sp.initialize_application(app_data=app_data)
        sp.initialize_model(
            name="HF", model_data=model_data["HF"], model_builder=HF_builder
        )
        sp.initialize_model(
            name="LF",
            model_data=model_data["LF"],
            model_builder=LF_builder,
            default=False,
        )
        sp.initialize_bundles(scheme="mf_random_nested", LF=2, seed=1234567890)

        assert sp.get_bundles() == {
            "HF_1": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 1): 0.3333333333333333,
                    ("LF", 3): 0.3333333333333333,
                    ("LF", 4): 0.3333333333333333,
                },
                "scenarios": [("HF", 1), ("LF", 3), ("LF", 4)],
            },
            "HF_2": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 2): 0.3333333333333333,
                    ("LF", 1): 0.3333333333333333,
                    ("LF", 3): 0.3333333333333333,
                },
                "scenarios": [("HF", 2), ("LF", 1), ("LF", 3)],
            },
            "HF_3": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 3): 0.3333333333333333,
                    ("LF", 4): 0.3333333333333333,
                    ("LF", 5): 0.3333333333333333,
                },
                "scenarios": [("HF", 3), ("LF", 4), ("LF", 5)],
            },
            "HF_4": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 4): 0.3333333333333333,
                    ("LF", 2): 0.3333333333333333,
                    ("LF", 3): 0.3333333333333333,
                },
                "scenarios": [("HF", 4), ("LF", 2), ("LF", 3)],
            },
            "HF_5": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 5): 0.3333333333333333,
                    ("LF", 1): 0.3333333333333333,
                    ("LF", 2): 0.3333333333333333,
                },
                "scenarios": [("HF", 5), ("LF", 1), ("LF", 2)],
            },
        }

        assert set(sp.bundles.keys()) == {"HF_1", "HF_2", "HF_3", "HF_4", "HF_5"}
        assert sp.bundles["HF_1"].probability == 0.2

        #
        # Testing internal data structures
        #
        M1 = sp.create_subproblem("HF_1")
        assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        M2 = sp.create_subproblem("HF_2")
        assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1", "HF_2"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        #
        # Test subproblem solver logic
        #
        sp.solve(M1, solver="glpk")
        assert len(M1.s) == 3
        assert set(M1.s.keys()) == {("HF", 1), ("LF", 3), ("LF", 4)}
        assert pyo.value(M1.s["HF", 1].x) == 25.0
        assert pyo.value(M1.s["LF", 3].x) == 25.0
        assert pyo.value(M1.s["LF", 4].x) == 25.0
        assert pyo.value(M1.s["HF", 1].y) == 26.0
        assert pyo.value(M1.s["LF", 3].y) == 95.5
        assert pyo.value(M1.s["LF", 4].y) == 104.5

        sp.solve(M2, solver="glpk")
        assert len(M2.s) == 3
        assert set(M2.s.keys()) == {("HF", 2), ("LF", 1), ("LF", 3)}
        assert pyo.value(M2.s["HF", 2].x) == 15.0
        assert pyo.value(M2.s["LF", 1].x) == 15.0
        assert pyo.value(M2.s["LF", 3].x) == 15.0
        assert pyo.value(M2.s["HF", 2].y) == 82.5
        assert pyo.value(M2.s["LF", 1].y) == 15.0
        assert pyo.value(M2.s["LF", 3].y) == 100.5

    def test_MF_builder3(self):
        sp = stochastic_program(first_stage_variables=["x"])
        sp.initialize_application(app_data=app_data)
        sp.initialize_model(
            name="HF", model_data=model_data["HF"], model_builder=HF_builder
        )
        sp.initialize_model(
            name="LF",
            model_data=model_data["LF"],
            model_builder=LF_builder,
            default=False,
        )
        sp.initialize_bundles(scheme="mf_random_nested", LF=2, seed=1234567890, model_weight={"HF":2.0, "LF":1.0})

        assert sp.get_bundles() == {
            "HF_1": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 1): 0.5,
                    ("LF", 3): 0.25,
                    ("LF", 4): 0.25,
                },
                "scenarios": [("HF", 1), ("LF", 3), ("LF", 4)],
            },
            "HF_2": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 2): 0.5,
                    ("LF", 1): 0.25,
                    ("LF", 3): 0.25,
                },
                "scenarios": [("HF", 2), ("LF", 1), ("LF", 3)],
            },
            "HF_3": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 3): 0.5,
                    ("LF", 4): 0.25,
                    ("LF", 5): 0.25,
                },
                "scenarios": [("HF", 3), ("LF", 4), ("LF", 5)],
            },
            "HF_4": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 4): 0.5,
                    ("LF", 2): 0.25,
                    ("LF", 3): 0.25,
                },
                "scenarios": [("HF", 4), ("LF", 2), ("LF", 3)],
            },
            "HF_5": {
                "probability": 0.2,
                "scenario_probability": {
                    ("HF", 5): 0.5,
                    ("LF", 1): 0.25,
                    ("LF", 2): 0.25,
                },
                "scenarios": [("HF", 5), ("LF", 1), ("LF", 2)],
            },
        }

        assert set(sp.bundles.keys()) == {"HF_1", "HF_2", "HF_3", "HF_4", "HF_5"}
        assert sp.bundles["HF_1"].probability == 0.2

        #
        # Testing internal data structures
        #
        M1 = sp.create_subproblem("HF_1")
        assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        M2 = sp.create_subproblem("HF_2")
        assert set(sp.int_to_FirstStageVar.keys()) == {"HF_1", "HF_2"}
        assert sp.varcuid_to_int == {pyo.ComponentUID("x"): 0}

        #
        # Test subproblem solver logic
        #
        sp.solve(M1, solver="glpk")
        assert len(M1.s) == 3
        assert set(M1.s.keys()) == {("HF", 1), ("LF", 3), ("LF", 4)}
        assert pyo.value(M1.s["HF", 1].x) == 25.0
        assert pyo.value(M1.s["LF", 3].x) == 25.0
        assert pyo.value(M1.s["LF", 4].x) == 25.0
        assert pyo.value(M1.s["HF", 1].y) == 26.0
        assert pyo.value(M1.s["LF", 3].y) == 95.5
        assert pyo.value(M1.s["LF", 4].y) == 104.5

        sp.solve(M2, solver="glpk")
        assert len(M2.s) == 3
        assert set(M2.s.keys()) == {("HF", 2), ("LF", 1), ("LF", 3)}
        assert pyo.value(M2.s["HF", 2].x) == 40.0
        assert pyo.value(M2.s["LF", 1].x) == 40.0
        assert pyo.value(M2.s["LF", 3].x) == 40.0
        assert pyo.value(M2.s["HF", 2].y) == 70.0
        assert pyo.value(M2.s["LF", 1].y) == 42.5
        assert pyo.value(M2.s["LF", 3].y) == 88.0
