import pyomo.environ as pyo
from or_topas.solnpool.solution import Solution
from or_topas.solnpool.solution import VariableInfo
from or_topas.solnpool.solution import ObjectiveInfo
from or_topas.solnpool.solnpool import PoolCounter
from or_topas.solnpool.solnpool import PoolManager
from or_topas.solnpool.solnpool import PoolPolicy
from or_topas.solnpool.solnpool import SolutionPoolBase


def constrain_EF_model(
    *, sp, M, first_stage_variables, fraction_same, filter_zeros=True
):
    #
    # Add a constraint that at least `fraction_same` of the specified first-stage-variables
    # match the given values.
    #
    assert (
        fraction_same >= 0 and fraction_same <= 1.0
    ), f"Unexpected value: {fraction_same=}"
    if fraction_same <= 1e-3:
        return M

    # Filter zero values
    if filter_zeros:
        first_stage_variables = {
            k: v for k, v in first_stage_variables.items() if v > 0.0
        }
        assert (
            len(first_stage_variables) > 0
        ), f"No non-zero first-stage-variables are specified"
    else:
        assert len(first_stage_variables) > 0, f"No first-stage-variables are specified"

    # Check that the specified first-stage-variables are in the model
    var = {name: M.rootx[i] for i, name in sp.int_to_FirstStageVarName.items()}
    for name in first_stage_variables:
        assert name in var, f"Missing variable {name} in model first stage variables"

    if fraction_same >= 1 - 1e-3:
        for name, value in first_stage_variables.items():
            var[name].fix(value)
        return M

    # Add a block of constraints
    M.EFmod = pyo.Block()
    M.EFmod.A = list(first_stage_variables.keys())
    M.EFmod.x = pyo.Var(M.EFmod.A, domain=pyo.Binary)

    M.EFmod.c = pyo.ConstraintList()
    for name, value in first_stage_variables.items():
        # If x[name] is one, then var[name] == value
        M.EFmod.c.add(var[name] - value <= 1 - M.EFmod.x[name])
        M.EFmod.c.add(value - var[name] <= 1 - M.EFmod.x[name])

    # The fraction of matching variables is >= fraction_same
    M.EFmod.c_lim = pyo.Constraint(
        expr=sum(M.EFmod.x[i] for i in M.EFmod.x) >= len(M.EFmod.x) * fraction_same
    )


def _sparow_as_solution(*args, **kwargs):
    return SparowSolution(*args, **kwargs)


class SparowSolution(Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ## TODO: 2nd stage solution as dict mapping scenario name to Solution obj


class SparowPoolManager(
    PoolManager, VariableInfo, ObjectiveInfo, PoolCounter, SolutionPoolBase
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_variable(*args, **kwargs):
        return VariableInfo(*args, **kwargs)

    @staticmethod
    def create_objective(*args, **kwargs):
        return ObjectiveInfo(*args, **kwargs)

    def add_pool(
        self, *, name=None, policy=PoolPolicy.keep_best, as_solution=None, **kwds
    ):
        if as_solution is None:
            as_solution = _sparow_as_solution
        return PoolManager.add_pool(
            self, name=name, policy=policy, as_solution=as_solution, **kwds
        )
