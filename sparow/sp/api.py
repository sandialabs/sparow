from .sp_pyomo import (
    # StochasticProgram_Pyomo_MultistageBuilder,
    StochasticProgram_Pyomo_NamedBuilder,
)


def stochastic_program(
    *, model_builder_list=None, first_stage_variables=None, aml="pyomo"
):
    """
    Create a stochastic program using the specified modeling framework.

    Parameters
    ----------
    model_builder_list : list, optional
        A list of functions used to construct the model. Not currently supported.
    first_stage_variables : list, optional
        A list of strings that denote the first-stage variables in the model.
    aml : str, optional
        The modeling framework used to construct the model (default is "pyomo").

    Returns
    -------
    StochasticProgram_Pyomo_NamedBuilder
        An instance of the stochastic program builder.

    Raises
    ------
    RuntimeError
        If the specified AML is not supported or if model_builder_list is provided.
    """
    if aml == "pyomo":
        if model_builder_list is not None:
            raise RuntimeError("No support for multi-stage models right now")
        else:
            return StochasticProgram_Pyomo_NamedBuilder(
                first_stage_variables=first_stage_variables
            )

    else:
        raise RuntimeError(f"AML {aml} is not currently supported.")
