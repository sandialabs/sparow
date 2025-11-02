def initialize_EF(sp, model, solution, resolve=True):
    # NOTE - we could solve each subproblem separately, but that
    #           wouldn't compute the objective
    if resolve:
        for i in sp.shared_variables():
            model.rootx[i].fix(solution.variable(i).value)

        sp.solve(model)

        for i in sp.shared_variables():
            model.rootx[i].unfix()

    else:
        for i in sp.shared_variables():
            model.rootx[i].set_value(solution.variable(i).value)

    return model


def create_and_initialize_EF(sp, solution, model_fidelities=None, resolve=True):
    M = sp.create_EF(cache_bundles=False, model_fidelities=model_fidelities)
    return initialize_EF(sp, M, solution, resolve=resolve)
