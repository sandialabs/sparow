import pyomo.environ as pyo
from forestlib.sp import stochastic_program

#
# Data for a simple newsvendor example
#
app_data = dict(c=1.0, b=1.5, h=0.1)
model_data = {
    "LF": {
        "scenarios": [
            {"ID": "1", "d": 15, "Probability": 0.1},
            {"ID": "2", "d": 60, "Probability": 0.2},
            {"ID": "3", "d": 72, "Probability": 0.1},
            {"ID": "4", "d": 78, "Probability": 0.3},
            {"ID": "5", "d": 82, "Probability": 0.3},
        ]
    },
    "HF": {
        "data": {"B": 0.9},
        "scenarios": [
            {"ID": "1", "d": 15, "C": 1.4, "Probability": 0.05},
            {"ID": "2", "d": 60, "C": 1.3, "Probability": 0.4},
            {"ID": "3", "d": 72, "C": 1.2, "Probability": 0.1},
            {"ID": "4", "d": 78, "C": 1.1, "Probability": 0.35},
            {"ID": "5", "d": 82, "C": 1.0, "Probability": 0.1},
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


def HF_newsvendor():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    return sp


def LF_newsvendor():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder
    )
    return sp


def MFrandom_newsvendor():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(
        scheme="mf_random",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 2.0, "LF": 1.0},
    )
    return sp


def MFpaired_newsvendor():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )
    sp.initialize_bundles(scheme="mf_paired")
    return sp
