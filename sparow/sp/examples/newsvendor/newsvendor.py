from munch import Munch
import pyomo.environ as pyo
from sparow.sp import stochastic_program

#
# Data for a simple newsvendor example
#
app_data = dict(c=1.0, b=1.5, h=0.1)
model_data = {
    "scenarios": [
        {"ID": 1, "d": 15},
        {"ID": 2, "d": 60},
        {"ID": 3, "d": 72},
        {"ID": 4, "d": 78},
        {"ID": 5, "d": 82},
    ],
}


#
# Function that constructs a newsvendor model
# including a single second stage
#
def builder(data, args):
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


def simple_newsvendor():
    """
    Newsvendor example adapted from

    A Tutorial on Stochastic Programming
    Alexander Shapiro∗ and Andy Philpott†
    March 21, 2007
    https://www.epoc.org.nz/papers/ShapiroTutorialSP.pdf
    """
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(model_data=model_data, model_builder=builder)
    return Munch(
        sp=sp,
        objective_value=76.5,
        unique_solution=True,
        solution_values={"x": 60.0, "s[None,1].x": 15.0, "s[None,2].x": 60.0},
    )
