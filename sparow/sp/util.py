import pyomo.environ as pyo

class try_import:

    def __init__(self):
        self.success = False

    def __bool__(self):
        return self.success

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is ImportError or exc_type is ModuleNotFoundError:
            # Catch import errors
            return True
        elif exc_type is not None:
            # Propogate other exceptions
            return False

        # No import errors, so return
        self.success = True
        return True

    def __str__(self):
        return str(self.success)


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
