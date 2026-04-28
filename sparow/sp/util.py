import pyomo.environ as pyo


def constrain_EF_model(
    *, sp, M, first_stage_variables, fraction_same, filter_zeros=True
):
    """
    Add constraints to ensure a fraction of first-stage variables match specified values.

    Parameters
    ----------
    sp : StochasticProgram
        The stochastic program instance.
    M : pyomo.ConcreteModel
        The Pyomo model to constrain.
    first_stage_variables : dict
        Dictionary of first-stage variable names and their values.
    fraction_same : float
        Fraction of variables that must match the specified values (0 <= fraction_same <= 1).
    filter_zeros : bool, optional
        Whether to filter out zero values from first_stage_variables (default is True).

    Returns
    -------
    pyomo.ConcreteModel
        The constrained model.

    Raises
    ------
    AssertionError
        If fraction_same is outside [0, 1], or if no variables are specified.
    """
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
