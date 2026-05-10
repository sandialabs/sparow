from or_topas.solnpool.solution import Solution, VariableInfo, ObjectiveInfo
from or_topas.solnpool.solnpool import PoolManager, PoolPolicy
from typing import Any, Callable


def _sparow_as_solution(*args: Any, **kwargs: Any) -> "SparowSolution":
    """
    Create a SparowSolution instance.

    Parameters
    ----------
    *args : tuple
        Positional arguments for the SparowSolution constructor.
    **kwargs : dict
        Keyword arguments for the SparowSolution constructor.

    Returns
    -------
    SparowSolution
        The created SparowSolution instance.
    """
    return SparowSolution(*args, **kwargs)


def create_variable(*args: Any, **kwargs: Any) -> VariableInfo:
    """
    Create a VariableInfo instance.

    Parameters
    ----------
    *args : tuple
        Positional arguments for the VariableInfo constructor.
    **kwargs : dict
        Keyword arguments for the VariableInfo constructor.

    Returns
    -------
    VariableInfo
        The created VariableInfo instance.
    """
    return VariableInfo(*args, **kwargs)


def create_objective(*args: Any, **kwargs: Any) -> ObjectiveInfo:
    """
    Create an ObjectiveInfo instance.

    Parameters
    ----------
    *args : tuple
        Positional arguments for the ObjectiveInfo constructor.
    **kwargs : dict
        Keyword arguments for the ObjectiveInfo constructor.

    Returns
    -------
    ObjectiveInfo
        The created ObjectiveInfo instance.
    """
    return ObjectiveInfo(*args, **kwargs)


class SparowSolution(Solution):
    """
    A class to represent a solution in the context of Sparow.

    Attributes
    ----------
    Attributes inherited from Solution class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    ## TODO: 2nd stage solution as dict mapping scenario name to Solution obj


class SparowPoolManager(PoolManager, VariableInfo, ObjectiveInfo):
    """
    A class to manage a pool of solutions in the context of Sparow.

    Attributes
    ----------
    Attributes inherited from PoolManager, VariableInfo, and ObjectiveInfo classes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def add_pool(
        self,
        *,
        name: str | None = None,
        policy: PoolPolicy = PoolPolicy.keep_best,
        as_solution: Callable[..., Any] | None = None,
        **kwds: Any
    ) -> Any:
        """
        Add a pool to the manager.

        Parameters
        ----------
        name : str, optional
            Name of the pool.
        policy : PoolPolicy, optional
            Policy for managing the pool (default is PoolPolicy.keep_best).
        as_solution : callable, optional
            Function to create solution objects (default is _sparow_as_solution).
        **kwds : dict
            Additional keyword arguments.

        Returns
        -------
        object
            The added pool.
        """
        if as_solution is None:
            as_solution = _sparow_as_solution
        return PoolManager.add_pool(
            self, name=name, policy=policy, as_solution=as_solution, **kwds
        )
