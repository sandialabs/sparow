from .sp_pyomo import (
    StochasticProgram_Pyomo_MultistageBuilder,
    StochasticProgram_Pyomo_NamedBuilder,
)


def stochastic_program(
    *, model_builder_list=None, first_stage_variables=None, aml="pyomo"
):
    """
    aml - The modeling framework used to construct the model.

    model_builder_list - A list of functions used to construct the model.

    first_stage_variables - A list of strings that denote the first-stage variables in the model.
        This is only used if model_builder is specified.
    """
    if aml == "pyomo":
        if model_builder_list is not None:
            return StochasticProgram_Pyomo_MultistageBuilder(
                model_builder_list=model_builder_list
            )
        else:
            return StochasticProgram_Pyomo_NamedBuilder(
                first_stage_variables=first_stage_variables
            )

    else:
        raise RuntimeError(f"AML {aml} is not currently supported.")
