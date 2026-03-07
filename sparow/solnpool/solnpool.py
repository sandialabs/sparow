from or_topas.solnpool.solution import Solution, VariableInfo, ObjectiveInfo
from or_topas.solnpool.solnpool import PoolManager, PoolPolicy


def _sparow_as_solution(*args, **kwargs):
    return SparowSolution(*args, **kwargs)


def create_variable(*args, **kwargs):
    return VariableInfo(*args, **kwargs)


def create_objective(*args, **kwargs):
    return ObjectiveInfo(*args, **kwargs)


class SparowSolution(Solution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ## TODO: 2nd stage solution as dict mapping scenario name to Solution obj


class SparowPoolManager(PoolManager, VariableInfo, ObjectiveInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_pool(
        self, *, name=None, policy=PoolPolicy.keep_best, as_solution=None, **kwds
    ):
        if as_solution is None:
            as_solution = _sparow_as_solution
        return PoolManager.add_pool(
            self, name=name, policy=policy, as_solution=as_solution, **kwds
        )
