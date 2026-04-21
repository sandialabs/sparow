from . import bundling_functions


def initialize_bundles(
    *,
    scheme=None,
    models=None,
    default_model=None,
    model_data=None,
    scenario_data=None,
    **kwargs,
):
    if scenario_data is None:
        scenario_data = {}
    if model_data is None:
        model_data = {}

    if scheme == None:
        scheme = "single_scenario"
    if models == None:
        models = [default_model] + list(
            sorted(model for model in scenario_data.keys() if model != default_model)
        )
    else:
        for name in models:
            assert name in scenario_data

    assert len(models) > 0, "Cannot initialize bundles without model data"
    if "model_weight" in kwargs:
        model_weight = kwargs["model_weight"]
    else:
        model_weight = {
            model: mdata.get("_model_weight_", 1.0)
            for model, mdata in model_data.items()
        }

    if model_weight:
        return bundling_functions.BundleObj(
            data=scenario_data,
            models=models,
            model_weight=model_weight,
            scheme=scheme,
            bundle_args=kwargs,
        )
    else:
        return bundling_functions.BundleObj(
            data=scenario_data,
            models=models,
            scheme=scheme,
            bundle_args=kwargs,
        )
