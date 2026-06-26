from munch import Munch
import pyomo.environ as pyo
from sparow.sp import stochastic_program

#
# Data for a simple modified absolute value example
#
app_data = dict(a=0, c=0, L=1, R=1)
model_data = {
    "scenarios": [
        {"ID": 1, "LB": None, "UB": None},
    ],
}

modified_app_data_example = dict(a=3, c=0, L=1, R=1, LB=-5, UB=5)
modified_model_data_example = {
    "scenarios": [
        {"ID": 1, "LB": -5, "UB": 5},
    ],
}

modified_app_data = dict(a=3, c=0, L=1, R=1)
modified_model_data = {
    "scenarios": [
        {"ID": 1, "LB": -5, "UB": 5},
    ],
}


def builder(data, args):
    r"""
    This subproblem implements the following function

    Q(x) =  max{R(x-a), -L(x-a)} if x \in [LB, UB]
            +\infty              if x \not \in [LB,UB]

    The subproblem that implements this is:
    Q(x) = min_{y >= 0} R*y[R] + L*y[L]
            s.t.    -y[R] + y[L] == a - x
                    y[UB_Slack] == UB - x
                    y[LB_Slack] == -LB + x

    Note that this is only LP representable when -L <= R
    This subproblem should result in the following cuts:
    Opt Cuts:
    \theta >= R(x-a)
    \theta >= -L(x-a)
    Corresponding to dual vertices [R, 0, 0]' and [-L, 0, 0]'

    Feas Cuts:
    0 >= x - UB
    0 >= -x + LB
    Corresponding to extreme rays [0, -1, 0]' and [0, 0, -1]
    """

    a = data["a"]
    c = data.get("c", 0)
    L = data["L"]
    R = data["R"]
    LB = data.get("LB", None)
    UB = data.get("UB", None)
    constant_offset = data.get("constant_offset", 0)

    m = pyo.ConcreteModel(data["ID"])
    m.x = pyo.Var()

    y_indices = ["Right", "Left"]
    if LB is not None:
        y_indices.append("LB_Slack")
    if UB is not None:
        y_indices.append("UB_Slack")
    m.y_indices = pyo.Set(initialize=y_indices)
    m.y = pyo.Var(m.y_indices, bounds=(0, None))

    # objective
    m.obj = pyo.Objective(expr=c * m.x + R * m.y["Right"] + L * m.y["Left"] + constant_offset)

    # vertex constriant
    m.vertex_cons = pyo.Constraint(expr=-m.y["Right"] + m.y["Left"] == a - m.x)

    # optional lower bound constraint
    if LB is not None:
        m.lb_cons = pyo.Constraint(expr=m.y["LB_Slack"] - m.x == -LB)

    if UB is not None:
        m.ub_cons = pyo.Constraint(expr=m.y["UB_Slack"] + m.x == UB)

    return m

def builder_testing(data, args):
    """
    Adds integrality to x and a couple bound constraints
    Meant for testing benders methods
    """

    a = data["a"]
    c = data.get("c", 0)
    L = data["L"]
    R = data["R"]
    LB = data.get("LB", None)
    UB = data.get("UB", None)

    m = pyo.ConcreteModel(data["ID"])
    m.x = pyo.Var(domain=pyo.Integers)

    y_indices = ["Right", "Left"]
    if LB is not None:
        y_indices.append("LB_Slack")
    if UB is not None:
        y_indices.append("UB_Slack")
    m.y_indices = pyo.Set(initialize=y_indices)
    m.y = pyo.Var(m.y_indices, bounds=(0, None))

    # objective
    m.obj = pyo.Objective(expr=c * m.x + R * m.y["Right"] + L * m.y["Left"])

    # vertex constriant
    m.vertex_cons = pyo.Constraint(expr=-m.y["Right"] + m.y["Left"] == a - m.x)

    # x constraint
    m.x_lower = pyo.Constraint(expr = -m.x <= 5)
    m.x_upper = pyo.Constraint(expr = m.x <= 7)

    # optional lower bound constraint
    if LB is not None:
        m.lb_cons = pyo.Constraint(expr=m.y["LB_Slack"] - m.x == -LB)

    if UB is not None:
        m.ub_cons = pyo.Constraint(expr=m.y["UB_Slack"] + m.x == UB)

    return m


def simple_absolute_value():
    r"""
    Adapted from Modified_Absolute_Value problem in OR-TOPAS
    Implements the following function:

    Q(x) =  max{R(x-a), -L(x-a)} if x \in [LB, UB]
            +\infty              if x \not \in [LB,UB]

    The problem that implements this is:
    Q(x) = min_{y >= 0} R*y[R] + L*y[L]
            s.t.    -y[R] + y[L] == a - x
                    y[UB_Slack] == UB - x
                    y[LB_Slack] == -LB + x

    We use the overall optimization problem:
    min_{y>=0,x} c*x + R*y[R] + L*y[L]
            s.t.    -y[R] + y[L] == a - x
                    y[UB_Slack] == UB - x
                    y[LB_Slack] == -LB + x

    Note that this is only LP representable when -L <= R.
    We can make this default to the modified abs problem with c=0.

    Note that when c = 0, x^* = a, when a \in [LB,UB].
    When a \not \in \[LB,UB], x^* = proj_{[LB,UB]}(a).
    If a \in [LB,UB] and c=0 then z^* = 0.
    If a \not \in [LB,UB] and c=0 then z^* = max{R(proj_{[LB,UB]}(a)-a), -L(proj_{[LB,UB]}(a)-a)}.



    Benders cut context:
    This subproblem should result in the following cuts:
    Opt Cuts:
    \theta >= R(x-a)
    \theta >= -L(x-a)
    Corresponding to dual vertices [R, 0, 0]' and [-L, 0, 0]'

    Feas Cuts:
    0 >= x - UB
    0 >= -x + LB
    Corresponding to extreme rays [0, -1, 0]' and [0, 0, -1]
    """
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(model_data=model_data, model_builder=builder)
    return Munch(
        sp=sp,
        objective_value=0,
        unique_solution=True,
        solution_values={
            "x": app_data["a"],
        },
    )


def feasibility_included_absolute_value():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=modified_app_data)
    sp.initialize_model(model_data=modified_model_data, model_builder=builder)
    return Munch(
        sp=sp,
        objective_value=0,
        unique_solution=True,
        solution_values={
            "x": modified_app_data["a"],
        },
    )

def absolute_value_testing_version():
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(model_data=model_data, model_builder=builder_testing)
    return Munch(
        sp=sp,
        objective_value=0,
        unique_solution=True,
        solution_values={
            "x": app_data["a"],
        },
    )

def adjustable_absolute_value(*,local_app_data, local_model_data):
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=local_app_data)
    sp.initialize_model(model_data=local_model_data, model_builder=builder)
    return Munch(
        sp=sp,
        objective_value=0,
        unique_solution=True,
        solution_values={
            "x": app_data["a"],
        },
    )
