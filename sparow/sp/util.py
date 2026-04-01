import pyomo.environ as pyo
import pyomo


def relax_second_stage(sp, M, *, relax_dict):

    for b in M.s.index_set():
        key = b[-1] if isinstance(b, tuple) else b
        if relax_dict[key]:
            print(
                "------------------Relaxing Noncontinuous Variables----------------------"
            )
            block = M.s[b]
            var_data_obj = {v.name: v for name, v in sp._first_stage_variables(M=block)}

            for v in block.component_objects(pyo.Var, active=True):
                if not isinstance(v, pyomo.core.base.var.IndexedVar):
                    if v.name in var_data_obj:
                        pass
                    else:
                        if not isinstance(v, pyomo.core.base.var.IndexedVar):
                            if v.domain is pyo.Binary:
                                v.domain = pyo.Reals
                                v.bounds = (0, 1)
                            elif v.domain is pyo.Integers:
                                v.domain = pyo.Reals
                            elif v.domain is pyo.Boolean:
                                v.domain = pyo.Reals
                                v.bounds = (0, 1)
                            elif v.domain is pyo.NonNegativeIntegers:
                                v.domain = pyo.NonNegativeReals
                            elif v.domain is pyo.NegativeIntegers:
                                v.domain = pyo.NegativeReals
                            elif v.domain is pyo.NonPositiveIntegers:
                                v.domain = pyo.NonPositiveReals

                else:
                    for _v in v.index_set():
                        if (v[_v].domain is pyo.Binary) and v[
                            _v
                        ].name not in var_data_obj:
                            v[_v].domain = pyo.Reals
                            v[_v].bounds = (0, 1)
                        elif (v[_v].domain is pyo.Integers) and v[
                            _v
                        ].name not in var_data_obj:
                            v[_v].domain = pyo.Reals
                        elif (v[_v].domain is pyo.Boolean) and v[
                            _v
                        ].name not in var_data_obj:
                            v[_v].domain = pyo.Reals
                            v[_v].bounds = (0, 1)
                        elif (v[_v].domain is pyo.NonNegativeIntegers) and v[
                            _v
                        ].name not in var_data_obj:
                            v[_v].domain = pyo.NonNegativeReals
                        elif (v[_v].domain is pyo.NegativeIntegers) and v[
                            _v
                        ].name not in var_data_obj:
                            v[_v].domain = pyo.NegativeReals
                        elif (v[_v].domain is pyo.NonPositiveIntegers) and v[
                            _v
                        ].name not in var_data_obj:
                            v[_v].domain = pyo.NonPositiveReals

    return M


def constrain_EF_model(
    *, sp, M, first_stage_variables, fraction_same, filter_zeros=True
):
    #
    # Add a constraint that at least `fraction_same` of the specified first-stage-variables
    # match the given values.
    #
    assert (
        fraction_same >= 0 and fraction_same <= 1.0
    ), f"Unexpected value: {fraction_same}"
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
