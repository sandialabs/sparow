def initialize_EF(sp, model, solution, resolve=True):
    # NOTE - we could solve each subproblem separately, but that
    #           wouldn't compute the objective
    if resolve:
        for i in sp.shared_variables():
            model.first_stage_variables[i].fix(solution.variable(i).value)

        sp.solve(model)

        for i in sp.shared_variables():
            model.first_stage_variables[i].unfix()

    else:
        for i in sp.shared_variables():
            model.first_stage_variables[i].set_value(solution.variable(i).value)

    #
    # Set the value of the 'other' scenario variables, if they are provided.
    # The compact repn only optimizes over the first-stage variables in the 'first' scenario,
    # so the pyomo first-stage variables used in the other scenarios need to be explicitly
    # set here.
    #
    if model.scenario_varmap:
        for i in sp.shared_variables():
            for var in model.scenario_varmap[i]:
                var.set_value(solution.variable(i).value)

    return model


def create_and_initialize_EF(sp, solution, model_fidelities=None, resolve=True):
    M = sp.create_EF(
        cache_bundles=False, model_fidelities=model_fidelities, compact_repn=True
    )
    return initialize_EF(sp, M, solution, resolve=resolve)
